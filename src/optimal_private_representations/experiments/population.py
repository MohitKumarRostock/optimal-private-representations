"""Exact finite-law representation studies with independent float64 checks.

Every set partition of each alphabet (at most four states) is enumerated
using Fraction arithmetic, including nonconsecutive partitions. The best
randomized-response partition is compared with the independent ordered DP.
The scalar squared-loss structural reduction justifies the private oracle;
finite enumeration does not prove that reduction or a general privacy claim.
Exact rational laws and exp(epsilon) values remain separate from float64
inputs. Small positive exact deficiencies are retained below float resolution.
No Monte Carlo fitting, sampling, or rate estimation is performed here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


from ..legacy import calibration as _dp_module
from ..legacy.calibration import fit_ordered_private_dp
from ..io import prepare_output

DP_PATH = Path(_dp_module.__file__)


def set_partitions(n: int) -> Iterable[tuple[int, ...]]:
    """Canonical restricted-growth strings; one per unlabeled set partition."""
    if n < 1:
        raise ValueError("The alphabet must be nonempty")

    def extend(prefix: tuple[int, ...], largest: int):
        if len(prefix) == n:
            yield prefix
            return
        for label in range(largest + 2):
            yield from extend(prefix + (label,), max(largest, label))

    yield from extend((0,), 0)


@dataclass(frozen=True)
class ExactOracle:
    phi: F
    risk: F
    mu: F
    k: int
    partition: tuple[int, ...]
    decoder: tuple[F, ...]
    all_optimal_partitions: tuple[tuple[int, ...], ...]
    n_partitions: int
    strict_suboptimality_gap: F | None


def exhaustive_rational_oracle(
    p: tuple[F, ...], means: tuple[F, ...], r: F, second_moment: F,
) -> ExactOracle:
    """Optimize all RR partitions using exact matrices and Bayes decoding.

    This implementation does not import any formulas, intervals, score
    ordering, states, or backtracking rules from the DP.
    """
    if not 1 <= len(p) <= 4 or len(p) != len(means):
        raise ValueError("This independent check is restricted to d <= 4")
    if sum(p) != 1 or min(p) <= 0 or r < 1:
        raise ValueError("Require positive probabilities summing to one and r>=1")
    if min(means) < 0 or max(means) > 1:
        raise ValueError("Require conditional means in [0,1]")
    mu = sum(pi * mi for pi, mi in zip(p, means))
    candidates = []
    for partition in set_partitions(len(p)):
        k = max(partition) + 1
        q = tuple(tuple((r if label == j else F(1)) / (r + k - 1)
                        for j in range(k)) for label in partition)
        mass = tuple(sum(p[i] * q[i][j] for i in range(len(p)))
                     for j in range(k))
        first = tuple(sum(p[i] * means[i] * q[i][j] for i in range(len(p)))
                      for j in range(k))
        decoder = tuple(first[j] / mass[j] for j in range(k))
        phi = sum(mass[j] * (decoder[j] - mu) ** 2 for j in range(k))
        candidates.append((phi, k, partition, decoder))
    best_phi = max(c[0] for c in candidates)
    optimal = [c for c in candidates if c[0] == best_phi]
    # Prefer fewer outputs, then lexicographic labels, when the optimum ties.
    best = min(optimal, key=lambda c: (c[1], c[2]))
    lower_values = [c[0] for c in candidates if c[0] < best_phi]
    return ExactOracle(
        phi=best_phi, risk=second_moment - mu * mu - best_phi, mu=mu,
        k=best[1], partition=best[2], decoder=best[3],
        all_optimal_partitions=tuple(c[2] for c in optimal),
        n_partitions=len(candidates),
        strict_suboptimality_gap=(best_phi - max(lower_values)) if lower_values else None,
    )


def float_population_oracle(p, means, epsilon: float, second_moment: float):
    """Inherited float64 DP plus stable, centered evaluation of its channel."""
    p = np.asarray(p, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    # The exact rational law sums to one.  Remove only conversion roundoff.
    p = p / p.sum()
    fitted = fit_ordered_private_dp(p, p * means, means, epsilon)
    mu = float(p @ means)
    mass = p @ fitted["q"]
    phi = float(np.sum(mass * (fitted["decoder"] - mu) ** 2))
    q = fitted["q"]
    ratio = float(np.max(np.max(q, axis=0) / np.min(q, axis=0)))
    return {
        "phi": phi, "risk": float(second_moment - mu * mu - phi),
        "k": int(fitted["k"]), "partition": fitted["bins"].tolist(),
        "decoder": fitted["decoder"].tolist(), "q": q.tolist(),
        "row_sum_error": float(np.max(np.abs(q.sum(axis=1) - 1))),
        "max_column_ratio": ratio,
    }


def coarsen(p: tuple[F, ...], means: tuple[F, ...], groups: tuple[int, ...]):
    labels = sorted(set(groups))
    if labels != list(range(len(labels))) or len(groups) != len(p):
        raise ValueError("Use consecutive, zero-based group labels")
    coarse_p = tuple(sum(pi for pi, gi in zip(p, groups) if gi == j) for j in labels)
    coarse_mean = tuple(sum(pi * mi for pi, mi, gi in zip(p, means, groups) if gi == j)
                        / coarse_p[j] for j in labels)
    error = tuple(mi - coarse_mean[gi] for mi, gi in zip(means, groups))
    mse = sum(pi * ei ** 2 for pi, ei in zip(p, error))
    l1 = sum(pi * abs(ei) for pi, ei in zip(p, error))
    linf = max(abs(ei) for ei in error)
    return coarse_p, coarse_mean, mse, l1, linf


def jensen_gap(p, means, groups, r, codebook):
    """Exact hinge penalty of a fixed codebook under the given coarsening."""
    pt, mt, *_ = coarsen(p, means, groups)
    gamma = (r - 1) / (len(codebook) + r - 1)
    gap = F(0)
    for lo, hi in itertools.pairwise(codebook):
        b = (lo + hi) / 2
        h = sum(pi * max(mi - b, F(0)) for pi, mi in zip(p, means))
        t = sum(pi * max(mi - b, F(0)) for pi, mi in zip(pt, mt))
        gap += 2 * gamma * (hi - lo) * (h - t)
    return gap


def fstrings(values):
    return [str(v) for v in values]


def json_cell(value):
    return json.dumps(value, separators=(",", ":"))


def evaluate_law(case_id, p, means, groups, r, second_moment, target_model):
    """Evaluate one explicitly recorded law and actual deterministic coarsening."""
    pt, mt, mse, l1, linf = coarsen(p, means, groups)
    h_exact = exhaustive_rational_oracle(p, means, r, second_moment)
    t_exact = exhaustive_rational_oracle(pt, mt, r, second_moment)
    epsilon = math.log(float(r))
    h_float = float_population_oracle([float(x) for x in p], [float(x) for x in means],
                                      epsilon, float(second_moment))
    t_float = float_population_oracle([float(x) for x in pt], [float(x) for x in mt],
                                      epsilon, float(second_moment))
    exact_gap = h_exact.phi - t_exact.phi
    if exact_gap < 0:
        raise AssertionError(f"Information increased under coarsening: {case_id}")
    float_gap = h_float["phi"] - t_float["phi"]
    h_error = abs(h_float["phi"] - float(h_exact.phi))
    t_error = abs(t_float["phi"] - float(t_exact.phi))
    # This conservative resolution marker concerns subtracting fitted-oracle
    # values, not exact Fraction output.  It never changes the raw result.
    resolution = 64 * np.finfo(float).eps * max(1.0, float(second_moment))
    tolerance = 2e-13
    if max(h_error, t_error) > tolerance:
        raise AssertionError(f"DP/exhaustive disagreement at {case_id}: {h_error}, {t_error}")
    for result in (h_float, t_float):
        if result["row_sum_error"] > 2e-14 or result["max_column_ratio"] > float(r) * (1 + 2e-14):
            raise AssertionError(f"Numerical RR feasibility failure at {case_id}")
    a = ((r - 1) / (r + 1)) ** 2
    if exact_gap > a / 4 or exact_gap > a * l1:
        raise AssertionError(f"Exact population upper-bound check failed: {case_id}")
    row = {
        "case_id": case_id, "target_model": target_model,
        "epsilon": epsilon, "exp_epsilon_exact": str(r), "a": float(a), "a_exact": str(a),
        "input_states": len(p), "statistic_states": len(pt),
        "probabilities_exact": json_cell(fstrings(p)),
        "conditional_means_exact": json_cell(fstrings(means)),
        "coarsening": json_cell(groups),
        "statistic_probabilities_exact": json_cell(fstrings(pt)),
        "statistic_means_exact": json_cell(fstrings(mt)),
        "target_second_moment": float(second_moment),
        "target_second_moment_exact": str(second_moment),
        "mse": float(mse), "mse_exact": str(mse),
        "l1": float(l1), "l1_exact": str(l1), "linf": float(linf),
        "risk_h_reference": float(h_exact.risk), "risk_t_reference": float(t_exact.risk),
        "risk_h_exact": str(h_exact.risk), "risk_t_exact": str(t_exact.risk),
        "phi_h_reference": float(h_exact.phi), "phi_t_reference": float(t_exact.phi),
        "phi_h_exact": str(h_exact.phi), "phi_t_exact": str(t_exact.phi),
        "deficiency_reference": float(exact_gap), "deficiency_exact": str(exact_gap),
        "deficiency_exact_status": "exact_zero" if exact_gap == 0 else "exact_positive",
        "risk_h_float64": h_float["risk"], "risk_t_float64": t_float["risk"],
        "phi_h_float64": h_float["phi"], "phi_t_float64": t_float["phi"],
        "deficiency_float64": float_gap,
        "deficiency_float64_resolution": resolution,
        "deficiency_float64_resolved": abs(float_gap) > resolution,
        "k_h_reference": h_exact.k, "k_t_reference": t_exact.k,
        "k_h_float64": h_float["k"], "k_t_float64": t_float["k"],
        "partition_h_reference": json_cell(h_exact.partition),
        "partition_t_reference": json_cell(t_exact.partition),
        "optimal_partitions_h_exact": json_cell(h_exact.all_optimal_partitions),
        "optimal_partitions_t_exact": json_cell(t_exact.all_optimal_partitions),
        "decoder_h_exact": json_cell(fstrings(h_exact.decoder)),
        "decoder_t_exact": json_cell(fstrings(t_exact.decoder)),
        "partition_h_float64": json_cell(h_float["partition"]),
        "partition_t_float64": json_cell(t_float["partition"]),
        "decoder_h_float64": json_cell(h_float["decoder"]),
        "decoder_t_float64": json_cell(t_float["decoder"]),
        "partitions_checked_h": h_exact.n_partitions, "partitions_checked_t": t_exact.n_partitions,
        "dp_exhaustive_phi_error_h": h_error, "dp_exhaustive_phi_error_t": t_error,
        "dp_exhaustive_check_pass": True,
        "max_rr_row_sum_error": max(h_float["row_sum_error"], t_float["row_sum_error"]),
        "max_rr_column_ratio": max(h_float["max_column_ratio"], t_float["max_column_ratio"]),
    }
    return row, h_exact, t_exact


def three_point_study():
    p, means, r = (F(1, 3),) * 3, (F(0), F(1, 2), F(1)), F(2)
    maps = {
        "identity": (0, 1, 2), "sufficient_left": (0, 1, 1),
        "sufficient_right": (0, 0, 1), "boundary_crossing": (1, 0, 1),
        "constant": (0, 0, 0),
    }
    expected_gaps = {"identity": F(0), "sufficient_left": F(0),
                     "sufficient_right": F(0), "boundary_crossing": F(1, 80),
                     "constant": F(1, 80)}
    rows = []
    for name, groups in maps.items():
        row, h, t = evaluate_law(f"three_{name}", p, means, groups, r, F(5, 12), "Y=M(H)")
        assert h.risk == F(37, 240) and h.phi - t.phi == expected_gaps[name]
        j_left = jensen_gap(p, means, groups, r, (F(3, 8), F(3, 5)))
        j_right = jensen_gap(p, means, groups, r, (F(2, 5), F(5, 8)))
        pt, *_ = coarsen(p, means, groups)
        row.update({"statistic": name, "j_left": float(j_left), "j_left_exact": str(j_left),
                    "j_right": float(j_right), "j_right_exact": str(j_right),
                    "statistic_entropy_nats": -sum(float(x) * math.log(float(x)) for x in pt)})
        rows.append(row)
    by_name = {row["statistic"]: row for row in rows}
    assert by_name["sufficient_left"]["mse_exact"] == "1/24"
    assert by_name["sufficient_left"]["j_left_exact"] == "0"
    assert by_name["sufficient_left"]["j_right_exact"] == "1/1600"
    assert by_name["boundary_crossing"]["j_left_exact"] == "39/1600"
    return rows


def exact_four_state_study():
    rows = []
    asymptotic_coefficient = F(35, 324)
    for power in range(3, 31):
        t = F(1, 2 ** power)
        p = ((1 - t) / 2, t / 2, t / 2, (1 - t) / 2)
        means = (F(0), F(1, 2) - t, F(1, 2) + t, F(1))
        ey2 = sum(pi * mi ** 2 for pi, mi in zip(p, means))
        row, h, coarse = evaluate_law(f"exact_t_2pow_minus_{power}", p, means,
                                      (0, 1, 1, 2), F(2), ey2, "Y=M(H)")
        phi_h_formula = (1 - t + 2 * t ** 2) ** 2 / 36
        phi_t_formula = (1 - t) ** 2 / (4 * (9 - t ** 2))
        gap_formula = t ** 2 * (35 - 34*t + 31*t**2 + 4*t**3 - 4*t**4) / (36 * (9 - t**2))
        assert h.phi == phi_h_formula and coarse.phi == phi_t_formula
        assert h.phi - coarse.phi == gap_formula and F(row["mse_exact"]) == t ** 3
        ratio = gap_formula / t ** 2
        row.update({"t": float(t), "t_exact": str(t), "dyadic_power": power,
                    "deficiency_over_mse_two_thirds": float(ratio),
                    "deficiency_over_t_squared_exact": str(ratio),
                    "asymptotic_coefficient": float(asymptotic_coefficient),
                    "asymptotic_coefficient_exact": str(asymptotic_coefficient),
                    "coefficient_relative_error": float(abs(ratio / asymptotic_coefficient - 1)),
                    "exact_formula_check_pass": True})
        rows.append(row)
    return rows


def capped_parameters(a: F, v: F):
    """Rationalize the source construction while preserving its inequalities.

    Cube/square roots only choose a dyadic rational just below the real root.
    All subsequent probability laws and all reference oracles are exact.
    """
    if v < 0:
        raise ValueError("v must be nonnegative")
    if v == 0:
        return F(1, 4), F(0), "zero_budget"
    if a > 0 and v <= a * a / 64:
        wf = float(v / (a * a)) ** (1 / 3)
        w = F.from_float(wf)
        while a * a * w ** 3 > v:
            wf = math.nextafter(wf, 0.0)
            w = F.from_float(wf)
        h = a * w
        regime = "small_budget_two_thirds_construction"
    elif v <= F(1, 64):
        w = F(1, 4)
        hf = math.sqrt(float(v / w))
        h = F.from_float(hf)
        while w * h ** 2 > v:
            hf = math.nextafter(hf, 0.0)
            h = F.from_float(hf)
        regime = "middle_budget_square_root_construction"
    else:
        w = h = F(1, 4)
        regime = "saturated_construction"
    assert 0 < w <= F(1, 4) and 0 < h <= F(1, 4)
    assert w * h * h <= v and h >= a * w
    return w, h, regime


def gamma(a: float, v: float):
    if a == 0 or v == 0:
        return 0.0, "zero"
    branches = {"constant": a, "square_root": a * math.sqrt(v),
                "two_thirds": (a * v) ** (2 / 3)}
    name = min(branches, key=branches.get)
    return branches[name], name


def modulus_grid_study():
    rows = []
    # r, rather than a rounded epsilon, is the exact primitive of each law.
    ratios = [F(1), F(51, 50), F(21, 20), F(11, 10), F(5, 4), F(2), F(4)]
    generic_budgets = {F(str(float(x))) for x in np.geomspace(1e-16, 0.25, 121)}
    generic_budgets.update({F(0), F(1, 64), F(1, 4), F(1), F(4)})
    for r in ratios:
        a = ((r - 1) / (r + 1)) ** 2
        budgets = set(generic_budgets)
        if a > 0:
            budgets.update({a*a/6400, a*a/64, a*a, min(F(1, 4), 100*a*a)})
        for index, v in enumerate(sorted(budgets)):
            w, h, regime = capped_parameters(a, v)
            p = ((1 - w) / 2, w / 2, w / 2, (1 - w) / 2)
            means = (F(1, 4), F(1, 2)-h, F(1, 2)+h, F(3, 4))
            row, full, coarse = evaluate_law(f"modulus_r_{r.numerator}_{r.denominator}_{index:03d}",
                                             p, means, (0, 1, 1, 2), r, F(1, 2),
                                             "Y|H~Bernoulli(M(H))")
            expected_phi_t = a * (1 - w) ** 2 / (16 * (1 - a * w * w))
            assert coarse.phi == expected_phi_t
            exact_gap = full.phi - coarse.phi
            comparator_phi = a * ((1 - w) / 4 + w * h) ** 2
            source_lower = F(37, 120) * a * w * h
            if a > 0 and v > 0:
                assert full.phi >= comparator_phi and exact_gap >= source_lower
            elif a == 0 or v == 0:
                assert exact_gap == 0
            g_budget, g_branch = gamma(float(a), float(v))
            g_actual, _ = gamma(float(a), row["mse"])
            upper_actual = min(float(a)/4, float(a)*math.sqrt(row["mse"]),
                               6*(float(a)*row["mse"])**(2/3))
            if g_budget:
                assert float(exact_gap) >= (37/1920)*g_budget*(1-1e-12)
            assert float(exact_gap) <= upper_actual*(1+1e-12) + 1e-30
            row.update({"budget_v": float(v), "budget_v_exact": str(v),
                        "w": float(w), "h": float(h), "w_exact": str(w), "h_exact": str(h),
                        "construction_regime": regime, "gamma_branch": g_branch,
                        "gamma_budget": g_budget, "gamma_actual_mse": g_actual,
                        "deficiency_over_gamma_budget": float(exact_gap)/g_budget if g_budget else None,
                        "deficiency_over_gamma_actual_mse": float(exact_gap)/g_actual if g_actual else None,
                        "theorem_lower_budget": (37/1920)*g_budget,
                        "theorem_upper_actual_mse": upper_actual,
                        "source_feasible_phi_h": float(comparator_phi),
                        "source_feasible_phi_h_exact": str(comparator_phi),
                        "source_lower_bound": float(source_lower), "source_lower_bound_exact": str(source_lower),
                        "source_phi_h_comparator_globally_optimal": full.phi == comparator_phi,
                        "source_phi_t_formula_check_pass": True,
                        "source_lower_bound_check_pass": True,
                        "actual_mse_within_budget_exact": F(row["mse_exact"]) <= v})
            rows.append(row)
    return rows


def write_csv(path: Path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path):
    output_dir = prepare_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    studies = {"geometry_three_point": three_point_study(),
               "geometry_exact_four_state": exact_four_state_study(),
               "geometry_modulus_grid": modulus_grid_study()}
    for name, rows in studies.items():
        write_csv(output_dir / f"{name}.csv", rows)
    all_rows = [row for rows in studies.values() for row in rows]
    positive_grid = [r for r in studies["geometry_modulus_grid"] if r["gamma_budget"] > 0]
    last_exact = studies["geometry_exact_four_state"][-1]
    manifest = {
        "schema_version": "1.0", "experiment": "Population representation studies",
        "deterministic": True, "random_seed": None, "numpy_version": np.__version__,
        "source_provenance": {
            "three_point": "Exact three-point examples with reoptimization; see docs/methods.md",
            "modulus": "Four-state actual-coarsening modulus construction; see docs/methods.md",
            "four_state_coefficient": "Exact four-state coarsening family with coefficient 35/324",
            "float_dp": "optimal_private_representations.legacy.calibration",
        },
        "float_dp_sha256": hashlib.sha256(DP_PATH.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "exact_reference_method": "Fraction arithmetic over every canonical set partition (Bell numbers 1,2,5,15); explicit RR matrices and Bayes decoders",
        "oracle_scope": "Scalar squared-loss RR-on-bins reduction justifies the population oracle; enumeration itself covers all RR partitions",
        "floating_method": "Inherited float64 ordered DP; centered explained variance used for evaluation",
        "file_row_counts": {f"{name}.csv": len(rows) for name, rows in studies.items()},
        "checks": {
            "total_laws": len(all_rows), "all_dp_exhaustive_checks_pass": True,
            "max_dp_exhaustive_phi_absolute_error": max(max(r["dp_exhaustive_phi_error_h"], r["dp_exhaustive_phi_error_t"]) for r in all_rows),
            "dp_exhaustive_absolute_tolerance": 2e-13,
            "max_rr_row_sum_error": max(r["max_rr_row_sum_error"] for r in all_rows),
            "all_exact_three_point_values_pass": True,
            "all_exact_four_state_formulas_pass": True,
            "all_exact_modulus_coarse_oracle_formulas_pass": True,
            "all_exact_source_lower_bound_checks_pass": True,
            "all_actual_mse_budget_checks_pass": True,
            "exact_positive_but_float64_unresolved_rows": sum(r["deficiency_exact_status"] == "exact_positive" and not r["deficiency_float64_resolved"] for r in all_rows),
            "smallest_t": last_exact["t"],
            "coefficient_at_smallest_t": last_exact["deficiency_over_mse_two_thirds"],
            "coefficient_limit_exact": "35/324",
            "coefficient_relative_error_at_smallest_t": last_exact["coefficient_relative_error"],
            "minimum_deficiency_over_gamma_budget": min(r["deficiency_over_gamma_budget"] for r in positive_grid),
            "maximum_deficiency_over_gamma_budget": max(r["deficiency_over_gamma_budget"] for r in positive_grid),
        },
        "interpretation_limits": [
            "All results are population calculations, not sample minimax evidence or experiments on learned releases.",
            "Exact references apply to the recorded rational laws and rational r; DP inputs use float64 approximations.",
            "Do not replace unresolved float64 gaps by an assertion of exact zero; use the exact reference/status columns.",
            "Modulus rows distinguish budget_v from actual mse. The source family saturates at actual mse=1/64.",
            "Gamma's formal constant branch begins at budget_v>=1. Any genuine conditional-mean MSE is at most 1/4, so this is a budget plateau, not an empirically attainable MSE regime.",
            "The feasible H comparator in the modulus proof need not be optimal; both unrestricted population oracles are optimized here.",
            "The curves are checks and illustrations of established formulas; numerical agreement is not a proof of a minimax theorem or historical novelty.",
        ],
        "columns": {
            "*_exact": "Exact rational scalar, or JSON array of exact rational strings.",
            "*_reference": "Float64 rendering of an independently enumerated exact rational oracle quantity, intended for plotting.",
            "*_float64": "Raw inherited DP result; no clipping of negative/zero numerical deficiency.",
            "deficiency_float64_resolved": "Conservative absolute-resolution diagnostic; the rational status remains authoritative for these laws.",
            "k_*_reference": "Fewest outputs among tied exact RR optima. Multiple optimizers are retained in optimal_partitions_*_exact.",
        },
    }
    (output_dir / "geometry_checks.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["checks"], indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/population/data"))
    args = parser.parse_args()
    run(args.output_dir.resolve())
