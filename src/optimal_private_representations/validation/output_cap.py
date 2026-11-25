"""Independent finite-instance checks for the privacy-dependent output-count cap.

Runs the capped and original searches on the same moments, exhaustively
enumerates small ordered partitions, and compares fitted learners with the
same synthetic random seeds. It does not test or certify differential privacy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .. import learner
from ..calibration import fit_ordered_private_dp, output_count_cap
from ..legacy import calibration as original
from ..io import prepare_output


def exhaustive_objective(p, q, scores, epsilon):
    """Enumerate every consecutive partition, without the DP recurrence."""
    levels = sorted(set(float(x) for x in scores[p > 0]))
    c, mu = math.expm1(epsilon), float(q.sum())
    best = mu * mu
    for mask in range(1 << max(0, len(levels) - 1)):
        thresholds = [levels[j + 1] for j in range(len(levels) - 1)
                      if mask & (1 << j)]
        bins = np.searchsorted(thresholds, scores, side="right")
        k = len(thresholds) + 1
        mass = np.array([p[bins == j].sum() for j in range(k)])
        moment = np.array([q[bins == j].sum() for j in range(k)])
        decoder = (mu + c * moment) / (1 + c * mass)
        if np.any(np.diff(decoder) < -1e-14):
            continue
        objective = float(np.sum((mu + c * moment) ** 2 /
                                 (1 + c * mass)) / (k + c))
        best = max(best, objective)
    return best


def main(output_path=None):
    rng = np.random.default_rng(2026112509)
    epsilons = [0.0, math.log(1.25), math.log(math.sqrt(3)),
                math.log(2), math.log(3), math.log(4)]
    comparisons = exhaustive_checks = 0
    max_search_error = max_exhaustive_error = 0.0
    for cells in [2, 3, 4, 6, 8, 10, 16, 32, 64]:
        for repetition in range(8):
            weights = rng.integers(1, 20, size=cells).astype(float)
            if repetition % 3 == 0:
                weights[::3] = 0
            p = weights / weights.sum()
            q = p * (rng.integers(0, 11, size=cells) / 10)
            scores = (rng.permutation(cells).astype(float) if repetition % 2
                      else rng.integers(0, max(2, cells // 2), size=cells).astype(float))
            for epsilon in epsilons:
                capped = fit_ordered_private_dp(p, q, scores, epsilon)
                full = original.fit_ordered_private_dp(p, q, scores, epsilon)
                error = abs(capped["explained_second_moment"] -
                            full["explained_second_moment"])
                max_search_error = max(max_search_error, error)
                assert error < 3e-13
                assert capped["k"] <= output_count_cap(epsilon, cells)
                assert output_count_cap(epsilon, cells) <= 9
                comparisons += 1
                if cells <= 10:
                    exact = exhaustive_objective(p, q, scores, epsilon)
                    error = abs(capped["explained_second_moment"] - exact)
                    max_exhaustive_error = max(max_exhaustive_error, error)
                    assert error < 3e-13
                    exhaustive_checks += 1

    # The known three-bin example must remain available at epsilon=log(4).
    p = np.array([0.3, 0.2, 0.2, 0.3])
    m = np.array([0.0, 0.5, 0.5, 1.0])
    three = fit_ordered_private_dp(p, p * m, m, math.log(4))
    assert three["k"] == 3

    fitted_comparisons = 0
    max_model_risk_error = 0.0
    saved_optimizer = learner.fit_ordered_private_dp
    try:
        for cells in [4, 16, 32]:
            p = rng.dirichlet(np.ones(cells) / 2)
            m = rng.uniform(0, 1, size=cells)
            for n in [1, 32, 128, 512]:
                states = rng.choice(cells, n, p=p)
                labels = rng.binomial(1, m[states])
                for epsilon in [0.0, math.log(2), math.log(4)]:
                    for alpha in [0.0, 0.25, 1.0]:
                        kwargs = dict(cells=cells, epsilon_query=epsilon,
                                      alpha_train=alpha)
                        learner.fit_ordered_private_dp = saved_optimizer
                        capped = learner.fit_finite_private_task_learner(
                            states, labels, rng=np.random.default_rng(1749), **kwargs)
                        learner.fit_ordered_private_dp = original.fit_ordered_private_dp
                        full = learner.fit_finite_private_task_learner(
                            states, labels, rng=np.random.default_rng(1749), **kwargs)
                        def risk(model):
                            channel = model.probabilities(np.arange(cells))
                            return float(np.sum(p[:, None] * channel *
                                (m[:, None] * (1 - model.decoder[None, :]) ** 2 +
                                 (1 - m[:, None]) * model.decoder[None, :] ** 2)))
                        error = abs(risk(capped) - risk(full))
                        max_model_risk_error = max(max_model_risk_error, error)
                        assert error < 3e-13
                        fitted_comparisons += 1
    finally:
        learner.fit_ordered_private_dp = saved_optimizer
    result = {
        "optimizer_comparisons": comparisons,
        "exhaustive_partition_checks": exhaustive_checks,
        "same_seed_fitted_learner_comparisons": fitted_comparisons,
        "max_optimizer_objective_difference": max_search_error,
        "max_exhaustive_objective_difference": max_exhaustive_error,
        "max_population_risk_difference": max_model_risk_error,
        "three_bin_example_retained": True,
        "timing_claim": "No new runtime measurements were made.",
        "privacy_scope": "Mathematical privacy proof retained; numerical privacy not certified.",
    }
    target = Path(output_path) if output_path else Path("outputs/validation/output_cap_verification.json")
    prepare_output(target.parent)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
