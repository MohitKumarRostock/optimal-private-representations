"""Post-design population diagnostic: optimal RR need not be binary.

Run: python output_count_experiment.py

The law, identified in the notes-result selection design check, has four
states, probabilities (3/10,1/5,1/5,3/10), and conditional Bernoulli means
(0,1/2,1/2,1).  This diagnostic was added after the main geometry studies
selected two outputs throughout their positive-budget grid.  It addresses
that specific interpretation risk; it is not a Monte Carlo experiment.

Exact Fraction enumeration over all set partitions is imported from the
population study.  The best binary value is independently evaluated over
every one- or two-cell partition, including nonconsecutive partitions.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

from .population import exhaustive_rational_oracle, set_partitions
from ..io import prepare_output


HERE = Path(__file__).resolve().parent


def best_at_most_two(p, means, r):
    mu = sum(pi * mi for pi, mi in zip(p, means))
    candidates = []
    for partition in set_partitions(len(p)):
        k = max(partition) + 1
        if k > 2:
            continue
        # Construct the channel explicitly, then Bayes-decode each column.
        q = [[(r if label == j else F(1)) / (r + k - 1) for j in range(k)]
             for label in partition]
        mass = [sum(p[i] * q[i][j] for i in range(len(p))) for j in range(k)]
        first = [sum(p[i] * means[i] * q[i][j] for i in range(len(p))) for j in range(k)]
        decoder = tuple(first[j] / mass[j] for j in range(k))
        phi = sum(mass[j] * (decoder[j] - mu) ** 2 for j in range(k))
        candidates.append((phi, k, partition, decoder))
    maximum = max(x[0] for x in candidates)
    best = min((x for x in candidates if x[0] == maximum), key=lambda x: (x[1], x[2]))
    return best, tuple(x[2] for x in candidates if x[0] == maximum), len(candidates)


def run(output_dir):
    output_dir = prepare_output(output_dir)
    p = (F(3, 10), F(1, 5), F(1, 5), F(3, 10))
    means = (F(0), F(1, 2), F(1, 2), F(1))
    second_moment, variance = F(1, 2), F(1, 4)
    ratios = (F(1), F(5, 4), F(3, 2), F(2), F(3), F(4))
    rows = []
    for r in ratios:
        full = exhaustive_rational_oracle(p, means, r, second_moment)
        binary, binary_optima, n_binary = best_at_most_two(p, means, r)
        binary_phi, binary_k, binary_partition, binary_decoder = binary
        binary_risk = variance - binary_phi
        gap = full.phi - binary_phi
        assert gap >= 0 and full.n_partitions == 15 and n_binary == 8
        rows.append({
            "exp_epsilon_exact": str(r), "epsilon": math.log(float(r)),
            "probabilities_exact": json.dumps([str(x) for x in p]),
            "conditional_means_exact": json.dumps([str(x) for x in means]),
            "target_model": "Y|H~Bernoulli(M(H))", "target_variance_exact": str(variance),
            "optimal_k": full.k, "best_at_most_two_k": binary_k,
            "optimal_partition": json.dumps(full.partition),
            "optimal_partitions_exact": json.dumps(full.all_optimal_partitions),
            "optimal_decoder_exact": json.dumps([str(x) for x in full.decoder]),
            "best_at_most_two_partition": json.dumps(binary_partition),
            "best_at_most_two_optimal_partitions": json.dumps(binary_optima),
            "best_at_most_two_decoder_exact": json.dumps([str(x) for x in binary_decoder]),
            "phi_unrestricted_exact": str(full.phi), "phi_at_most_two_exact": str(binary_phi),
            "risk_unrestricted_exact": str(full.risk), "risk_at_most_two_exact": str(binary_risk),
            "binary_restriction_gap_exact": str(gap),
            "phi_unrestricted": float(full.phi), "phi_at_most_two": float(binary_phi),
            "risk_unrestricted": float(full.risk), "risk_at_most_two": float(binary_risk),
            "binary_restriction_gap": float(gap),
            "partitions_checked_unrestricted": full.n_partitions,
            "partitions_checked_at_most_two": n_binary,
            "exact_enumeration_check_pass": True,
        })
    last = rows[-1]
    assert last["optimal_k"] == 3
    assert F(last["phi_unrestricted_exact"]) == F(27, 760)
    assert F(last["phi_at_most_two_exact"]) == F(81, 2356)
    assert F(last["binary_restriction_gap_exact"]) == F(27, 23560)
    assert [row["optimal_k"] for row in rows] == [1, 2, 2, 2, 2, 3]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "output_count.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {
        "schema_version": "1.0", "status": "Post-design interpretation diagnostic",
        "purpose": "Prevent the main family's two-output optima from suggesting that binary RR always suffices.",
        "provenance": "Four-state output-count diagnostic added after the original study design; see docs/experiments.md.",
        "method": "Exact Fraction enumeration of all 15 set partitions and all 8 partitions with at most two cells; no Monte Carlo.",
        "unrestricted_oracle_justification": "The scalar squared-loss RR-on-bins reduction guarantees an optimal RR partition; enumeration includes every partition of this four-state law.",
        "all_checks_pass": True,
        "exp_epsilon_4_conclusion": {
            "optimal_k": 3, "optimal_phi_exact": "27/760", "best_binary_phi_exact": "81/2356",
            "optimal_risk_exact": "163/760", "best_binary_risk_exact": "127/589",
            "binary_restriction_gap_exact": "27/23560",
            "binary_restriction_gap": float(F(27, 23560)),
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "population_module_sha256": hashlib.sha256((HERE / "population.py").read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": rows,
        "limits": [
            "The claim about at most two outputs in this diagnostic is restricted to the RR-partition family.",
            "This exact finite-law comparison does not characterize the optimal output count for every law or privacy budget.",
            "The unrestricted oracle uses three outputs at r=4; this suffices to refute a general binary-RR-sufficiency interpretation.",
        ],
    }
    (output_dir / "output_count.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["exp_epsilon_4_conclusion"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/population/data"))
    args = parser.parse_args()
    run(args.output_dir.resolve())
