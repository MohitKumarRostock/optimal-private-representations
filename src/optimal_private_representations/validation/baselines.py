"""Focused synthetic checks for the baseline wrappers; not a DP certificate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from ..experiments.models import (
    METHODS, fit_model, analytic_risk, population_oracle,
    fit_finite_private_task_learner, fractional_histogram,
)


class RecordingRNG:
    def __init__(self, seed):
        self.base = np.random.default_rng(seed)
        self.calls = []

    def laplace(self, loc, scale, size=None):
        value = self.base.laplace(loc, scale, size=size)
        self.calls.append((float(scale), size, np.array(value, copy=True)))
        return value


class NegativeNoise:
    def laplace(self, loc, scale, size=None):
        return -100.0 if size is None else np.full(size, -100.0)


def main(output_path=None):
    assertions = 0
    def check(condition):
        nonlocal assertions
        assert condition
        assertions += 1

    data_rng = np.random.default_rng(73015)
    cells, n = 7, 31
    # The last two declared cells are unseen; labels need not be binary.
    states = data_rng.integers(cells - 2, size=n)
    labels = data_rng.random(n)
    p = np.arange(1, cells + 1, dtype=float)
    p /= p.sum()
    m = np.array([.15, .85, .25, .75, .35, .65, .5])
    maximum_risk_identity_error = 0.0
    minimum_oracle_excess = np.inf
    fit_cases = 0
    for epsilon in (0.0, np.log(2), np.log(4)):
        oracle = population_oracle(p, m, epsilon, second=m)
        for alpha in (0.0, 0.2, 1.0, 2.0):
            for method in METHODS:
                result = fit_model(method, states, labels, cells, epsilon, alpha,
                                   np.random.default_rng(41))
                fit_cases += 1
                q, g = result["q"], result["decoder"]
                check(q.shape == (cells, result["k"]) and g.shape == (result["k"],))
                check(np.allclose(q.sum(axis=1), 1.0, atol=1e-14, rtol=0))
                check(np.all(q >= 0) and np.all(g >= -1e-14) and np.all(g <= 1 + 1e-14))
                check(np.all(q.max(axis=0) <= np.exp(epsilon) * q.min(axis=0) + 1e-14))
                value = analytic_risk(p, m, m, q, g)
                direct = float(np.sum(p[:, None] * q * (
                    m[:, None] * (1.0 - g[None, :]) ** 2
                    + (1.0 - m[:, None]) * g[None, :] ** 2)))
                maximum_risk_identity_error = max(maximum_risk_identity_error, abs(value - direct))
                check(abs(value - direct) <= 5e-15)
                check(abs(value - analytic_risk(p, m, float(p @ m), q, g)) <= 5e-15)
                excess = value - oracle["risk"]
                minimum_oracle_excess = min(minimum_oracle_excess, excess)
                check(excess >= -1e-12)
                if alpha == 0 and method != "nonprivate_fit":
                    check(np.array_equal(g, [.5]) and np.array_equal(q, np.ones((cells, 1))))

    # Verify the recorded uncapped implementation and same first noise draw.
    epsilon, alpha = np.log(2), 0.8
    wrapped_rng, source_rng = RecordingRNG(819), RecordingRNG(819)
    wrapped = fit_model("attaining", states, labels, cells, epsilon, alpha, wrapped_rng)
    source = fit_finite_private_task_learner(
        states, labels, cells=cells, epsilon_query=epsilon, alpha_train=alpha, rng=source_rng)
    check(np.array_equal(wrapped["q"], source.probabilities(np.arange(cells))))
    check(np.array_equal(wrapped["decoder"], source.decoder))
    no_mean_rng = RecordingRNG(819)
    fit_model("no_mean_calibration", states, labels, cells, epsilon, alpha, no_mean_rng)
    check(np.array_equal(wrapped_rng.calls[0][2], no_mean_rng.calls[0][2]))
    N = n - n // 2
    check([x[0] for x in wrapped_rng.calls] == [2/alpha, 4/(N*alpha), 2/(N*alpha)])
    check([x[0] for x in no_mean_rng.calls] == [2/alpha, 2/(N*alpha)])
    for method, scales in (
        ("direct_histogram", [2/(n*alpha)]),
        ("private_mean", [1/(n*alpha)]),
        ("public_order", [4/(N*alpha), 2/(N*alpha)]),
    ):
        recorder = RecordingRNG(89)
        fit_model(method, states, labels, cells, epsilon, alpha, recorder)
        check([x[0] for x in recorder.calls] == scales)
        for _, shape, _ in recorder.calls:
            check(shape is None or shape == (cells, 2))

    # Public-order ablation ignores the pilot block, including its labels.
    changed_states, changed_labels = states.copy(), labels.copy()
    changed_states[:n//2] = cells - 1
    changed_labels[:n//2] = 1.0 - changed_labels[:n//2]
    original = fit_model("public_order", states, labels, cells, epsilon, alpha, np.random.default_rng(123))
    changed = fit_model("public_order", changed_states, changed_labels, cells, epsilon, alpha, np.random.default_rng(123))
    check(np.array_equal(original["q"], changed["q"]))
    check(np.array_equal(original["decoder"], changed["decoder"]))

    # All-clipped histogram repair is public and produces a valid full channel.
    for method in ("direct_histogram", "no_mean_calibration"):
        repaired = fit_model(method, states, labels, cells, epsilon, alpha, NegativeNoise())
        check(repaired["decoder"].shape == (1,) and
              np.allclose(repaired["decoder"], [.5], rtol=0, atol=1e-15))
        check(np.array_equal(repaired["q"], np.ones((cells, 1))))

    # nonprivate_fit must remain a fitting comparator even when alpha=0.
    nonprivate = fit_model("nonprivate_fit", np.zeros(n, dtype=int), np.ones(n),
                           cells, epsilon, 0.0, np.random.default_rng(17))
    check(np.array_equal(nonprivate["decoder"], [1.0]))
    check(nonprivate["training_private"] is False)

    # Independent exact known-law check: fair binary task at epsilon=log(2).
    exact = population_oracle([.5, .5, 0, 0], [0, 1, .3, .8], np.log(2), second=[0, 1, .3, .8])
    check(exact["k"] == 2)
    check(abs(exact["risk"] - 2/9) <= 1e-14)

    # The new all-sample histogram/mean releases have the stated sensitivities.
    alphabet = [(h, y) for h in range(3) for y in (0.0, .3, 1.0)]
    max_hist_sensitivity = max_mean_sensitivity = 0.0
    neighboring_pairs = 0
    for h0, y0 in alphabet:
        for h1, y1 in alphabet:
            s0, s1 = np.array([h0, 0, 1]), np.array([h1, 0, 1])
            yv0, yv1 = np.array([y0, .2, .7]), np.array([y1, .2, .7])
            hist0 = fractional_histogram(s0, yv0, 3, normalize=True)
            hist1 = fractional_histogram(s1, yv1, 3, normalize=True)
            dh = float(np.abs(hist0 - hist1).sum())
            dm = abs(float(yv0.mean() - yv1.mean()))
            max_hist_sensitivity = max(max_hist_sensitivity, dh)
            max_mean_sensitivity = max(max_mean_sensitivity, dm)
            neighboring_pairs += 1
            check(dh <= 2/3 + 1e-14 and dm <= 1/3 + 1e-14)

    output = {
        "status": "passed", "assertions": assertions,
        "fit_cases": fit_cases, "neighboring_pairs": neighboring_pairs,
        "maximum_risk_identity_error": maximum_risk_identity_error,
        "minimum_oracle_excess": float(minimum_oracle_excess),
        "fair_binary_oracle_risk": exact["risk"],
        "fair_binary_oracle_output_count": exact["k"],
        "max_normalized_histogram_sensitivity_at_n3": max_hist_sensitivity,
        "max_mean_sensitivity_at_n3": max_mean_sensitivity,
        "scope": "Focused algebra/implementation checks, not numerical proof of DP or minimaxity.",
    }
    from ..io import prepare_output
    path = Path(output_path) if output_path else Path("outputs/validation/baseline_checks.json")
    prepare_output(path.parent)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
