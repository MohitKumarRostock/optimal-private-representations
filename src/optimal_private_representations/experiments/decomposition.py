"""Post-run diagnostic replay of existing recorded near-tie fits.

This decomposition was added after inspecting the original learning curves;
it is not a preregistered additional study. It regenerates exactly the data and
fitting noise of the 20 near-tie configurations already in learning_raw.csv.
It adds no independent fitting samples or Monte Carlo replicates.

For each existing fitted channel q, Bayes-redecode under the known synthetic
law to split total oracle excess into channel excess plus decoder error.
Summaries give equal weight to the two fixed sign strata, using their within-
stratum variances and Welch--Satterthwaite pointwise 95% Monte Carlo intervals.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import t as student_t

from .models import analytic_risk, fit_model, population_oracle
from .learning import law_arrays
from ..io import prepare_output
from ..provenance import validate_protocol_sources


METHODS = ("attaining", "direct_histogram", "nonprivate_fit")
POST_RUN_STATUS = (
    "Post-run diagnostic decomposition of existing fits, added after inspection "
    "of the original learning curves; not preregistered and not new sampling."
)


def canonical_assignment(q):
    """Label the high-probability RR bin by its first state occurrence."""
    raw = np.argmax(q, axis=1)
    labels = {}
    answer = []
    for value in raw:
        value = int(value)
        if value not in labels:
            labels[value] = len(labels)
        answer.append(labels[value])
    return tuple(answer)


def source_assignment(sign):
    if sign not in (-1, 1):
        raise ValueError("near-tie source requires sign -1 or +1")
    return (0, 1, 1, 1) if sign > 0 else (0, 0, 0, 1)


def classify_partition(q, sign):
    assignment = canonical_assignment(q)
    if q.shape[1] == 1:
        category = "constant"
    elif q.shape[1] > 2:
        category = "multibin"
    elif assignment == source_assignment(sign):
        category = "binary_correct"
    elif assignment == source_assignment(-sign):
        category = "binary_wrong"
    else:
        category = "binary_other"
    return category, assignment


def redecode(p, m, q):
    masses = p @ q
    if np.any(masses <= 0):
        raise RuntimeError("The returned RR channel has a zero-mass output")
    decoder = ((p * m) @ q) / masses
    return decoder, analytic_risk(p, m, m, q, decoder), masses


def source_channel(assignment, epsilon):
    bins = np.asarray(assignment, dtype=int)
    contrast = math.expm1(epsilon)
    return (1 + contrast * (bins[:, None] == np.arange(2))) / (2 + contrast)


def balanced_interval(minus, plus):
    """Equal-sign mean and variance; Welch interval for fixed sign strata."""
    minus, plus = np.asarray(minus, float), np.asarray(plus, float)
    if len(minus) != len(plus) or len(minus) < 2:
        raise ValueError("Require two balanced sign strata with at least two fits each")
    means = [float(minus.mean()), float(plus.mean())]
    variance_of_means = [float(np.var(x, ddof=1) / len(x)) for x in (minus, plus)]
    variance_sum = sum(variance_of_means)
    mean = sum(means) / 2
    se = math.sqrt(variance_sum) / 2
    if variance_sum > 0:
        denominator = sum(v * v / (len(x) - 1)
                          for v, x in zip(variance_of_means, (minus, plus)))
        df = variance_sum * variance_sum / denominator
        half_width = float(student_t.ppf(.975, df)) * se
    else:
        # A zero observed variance is reported as such, not a probability theorem.
        df, half_width = math.inf, 0.0
    return {
        "mean": mean, "se": se, "ci_low": mean - half_width,
        "ci_high": mean + half_width, "welch_df": df,
        "minus_mean": means[0], "plus_mean": means[1],
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/decomposition/data"))
    args = parser.parse_args(argv)
    data = prepare_output(args.data_dir)
    protocol_path, original_path = data / "learning_protocol.json", data / "learning_raw.csv"
    protocol = json.loads(protocol_path.read_text())
    configs = [c for c in protocol["configurations"]
               if c["study"] == "near_tie" and c["family"] == "near_tie"]
    repetitions = int(protocol["replicates_per_configuration"])
    if len(configs) != 20 or repetitions < 2:
        raise RuntimeError("Expected all 20 near-tie configurations and at least two fits per sign/method")
    if any(c["d"] != 4 or c["sign"] not in (-1, 1) for c in configs):
        raise RuntimeError("Unexpected near-tie source family")
    config_map = {c["config_id"]: c for c in configs}
    validate_protocol_sources(protocol)

    original = {}
    all_original_rows = 0
    with original_path.open() as f:
        for row in csv.DictReader(f):
            all_original_rows += 1
            if row["config_id"] not in config_map or row["method"] not in METHODS:
                continue
            key = (row["config_id"], int(row["replicate"]), row["method"])
            if key in original:
                raise RuntimeError("Duplicate existing fit " + str(key))
            original[key] = row
    expected_keys = {(c["config_id"], rep, method)
                     for c in configs for rep in range(repetitions) for method in METHODS}
    if set(original) != expected_keys:
        raise RuntimeError("The original file does not contain exactly the expected replay fits")

    fields = [
        "config_id", "n", "d", "c", "amplitude", "sign", "alpha", "epsilon",
        "replicate", "method", "risk", "original_risk", "replay_risk_error",
        "oracle_risk", "population_redecoded_risk", "total_excess", "channel_excess",
        "decoder_error", "decoder_squared_distance", "decomposition_identity_error",
        "decoder_identity_error", "normalized_total_excess", "normalized_channel_excess",
        "normalized_decoder_error", "a", "k", "canonical_assignment", "partition_category",
        "binary_correct", "binary_wrong", "binary_other", "constant", "multibin",
        "source_wrong_gap", "normalized_source_wrong_gap", "fitted_decoder", "population_decoder",
    ]
    raw_path = data / "near_tie_decomposition_raw.csv"
    temporary_raw = raw_path.with_suffix(".csv.tmp")
    rows = []
    maxima = defaultdict(float)
    minimum_channel_excess, minimum_decoder_error = math.inf, math.inf
    started = time.perf_counter()
    with temporary_raw.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for config in configs:
            p, m = law_arrays(config)
            oracle = population_oracle(p, m, config["epsilon"], second=m)
            V = oracle["risk"]
            correct_q = source_channel(source_assignment(config["sign"]), config["epsilon"])
            wrong_q = source_channel(source_assignment(-config["sign"]), config["epsilon"])
            _, correct_risk, _ = redecode(p, m, correct_q)
            _, wrong_risk, _ = redecode(p, m, wrong_q)
            if oracle["k"] != 2 or abs(correct_risk - V) > 2e-13:
                raise RuntimeError("Expected source partition is not population optimal")
            a = math.tanh(config["epsilon"] / 2) ** 2
            normalization = math.sqrt(config["n"]) / a
            for rep in range(repetitions):
                seed = np.random.SeedSequence([
                    protocol["master_seed"], int(config["config_id"]), rep])
                data_seed, noise_seed = seed.spawn(2)
                data_rng = np.random.default_rng(data_seed)
                h = data_rng.choice(config["d"], size=config["n"], p=p)
                y = data_rng.binomial(1, m[h]).astype(float)
                for method in METHODS:
                    fit = fit_model(method, h, y, cells=config["d"],
                                    epsilon_query=config["epsilon"], alpha_train=config["alpha"],
                                    rng=np.random.default_rng(noise_seed))
                    q, g = fit["q"], fit["decoder"]
                    risk = analytic_risk(p, m, m, q, g)
                    old = original[(config["config_id"], rep, method)]
                    replay_error = risk - float(old["risk"])
                    maxima["replay_risk_error"] = max(maxima["replay_risk_error"], abs(replay_error))
                    maxima["oracle_replay_error"] = max(maxima["oracle_replay_error"], abs(V - float(old["oracle_risk"])))
                    if (abs(replay_error) > 5e-13 or abs(V - float(old["oracle_risk"])) > 5e-13
                            or int(fit["k"]) != int(old["k"])):
                        raise RuntimeError("Replay mismatch " + str((config["config_id"], rep, method)))
                    gP, riskP, masses = redecode(p, m, q)
                    total, channel, decoder = risk - V, riskP - V, risk - riskP
                    squared = float(masses @ (g - gP) ** 2)
                    identity_error = total - (channel + squared)
                    decoder_error = decoder - squared
                    maxima["decomposition_identity_error"] = max(maxima["decomposition_identity_error"], abs(identity_error))
                    maxima["decoder_identity_error"] = max(maxima["decoder_identity_error"], abs(decoder_error))
                    minimum_channel_excess = min(minimum_channel_excess, channel)
                    minimum_decoder_error = min(minimum_decoder_error, decoder)
                    if channel < -5e-13 or decoder < -5e-13 or abs(identity_error) > 5e-13:
                        raise RuntimeError("Bayes-risk decomposition failed")
                    category, assignment = classify_partition(q, config["sign"])
                    if category == "binary_correct" and abs(channel) > 5e-13:
                        raise RuntimeError("Correct source partition has positive channel excess")
                    if category == "binary_wrong" and abs(channel - (wrong_risk - V)) > 5e-13:
                        raise RuntimeError("Wrong source partition mismatch")
                    row = {key: config[key] for key in
                           ("config_id", "n", "d", "amplitude", "sign", "alpha", "epsilon")}
                    row.update(
                        c=config["amplitude"], replicate=rep, method=method,
                        risk=risk, original_risk=float(old["risk"]), replay_risk_error=replay_error,
                        oracle_risk=V, population_redecoded_risk=riskP, total_excess=total,
                        channel_excess=channel, decoder_error=decoder,
                        decoder_squared_distance=squared, decomposition_identity_error=identity_error,
                        decoder_identity_error=decoder_error,
                        normalized_total_excess=normalization * total,
                        normalized_channel_excess=normalization * channel,
                        normalized_decoder_error=normalization * decoder,
                        a=a, k=int(fit["k"]), canonical_assignment="".join(map(str, assignment)),
                        partition_category=category, source_wrong_gap=wrong_risk - V,
                        normalized_source_wrong_gap=normalization * (wrong_risk - V),
                        fitted_decoder=json.dumps(g.tolist(), separators=(",", ":")),
                        population_decoder=json.dumps(gP.tolist(), separators=(",", ":")),
                    )
                    for name in ("binary_correct", "binary_wrong", "binary_other", "constant", "multibin"):
                        row[name] = int(category == name)
                    writer.writerow(row)
                    rows.append(row)
            f.flush()
            print(f'Replayed {config["config_id"]}: n={config["n"]}, c={config["amplitude"]}, '
                  f'sign={config["sign"]}; {len(rows)} existing fits, {time.perf_counter()-started:.1f}s', flush=True)
    temporary_raw.replace(raw_path)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["n"], row["c"], row["method"])].append(row)
    summaries = []
    metrics = [
        "risk", "population_redecoded_risk", "total_excess", "channel_excess", "decoder_error",
        "normalized_total_excess", "normalized_channel_excess", "normalized_decoder_error", "k",
        "binary_correct", "binary_wrong", "binary_other", "constant", "multibin",
    ]
    for (n, c, method), group in sorted(groups.items()):
        minus = [r for r in group if r["sign"] == -1]
        plus = [r for r in group if r["sign"] == 1]
        if len(minus) != repetitions or len(plus) != repetitions:
            raise RuntimeError("Unbalanced or incomplete replay group")
        item = dict(n=n, c=c, amplitude=c, method=method,
                    config_minus=minus[0]["config_id"], config_plus=plus[0]["config_id"],
                    replicates_per_sign=repetitions, replayed_fits=len(group),
                    alpha=group[0]["alpha"], epsilon=group[0]["epsilon"], a=group[0]["a"],
                    source_wrong_gap_mean=float(np.mean([r["source_wrong_gap"] for r in group])),
                    normalized_source_wrong_gap_mean=float(np.mean([r["normalized_source_wrong_gap"] for r in group])))
        for metric in metrics:
            for key, value in balanced_interval([r[metric] for r in minus], [r[metric] for r in plus]).items():
                item[metric + "_" + key] = value
        for category in ("binary_correct", "binary_wrong", "binary_other", "constant", "multibin"):
            item[category + "_count"] = sum(r[category] for r in group)
        if sum(item[categ + "_count"] for categ in
               ("binary_correct", "binary_wrong", "binary_other", "constant", "multibin")) != len(group):
            raise RuntimeError("Partition categories do not cover the fits")
        summaries.append(item)
    summary_path = data / "near_tie_decomposition_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    metadata = {
        "status": "passed", "analysis_status": POST_RUN_STATUS,
        "configurations_replayed": len(configs), "replicates_per_configuration": repetitions,
        "methods": list(METHODS), "replayed_fits": len(rows), "summary_groups": len(summaries),
        "original_study_distinct_models": all_original_rows, "additional_independent_models": 0,
        "maximum_errors": dict(maxima), "minimum_channel_excess": minimum_channel_excess,
        "minimum_decoder_error": minimum_decoder_error,
        "intervals": (
            "Pointwise approximate 95% Monte Carlo intervals for equally weighted sign strata. "
            "Mean=(mean_-+mean_+)/2; SE=sqrt(s_-^2/R+s_+^2/R)/2; "
            "Welch-Satterthwaite degrees of freedom. No multiplicity correction. "
            "Indicator intervals use the same moment approximation; zero observed variance "
            "does not establish an exact zero population probability."
        ),
        "identity": "R(q,g)-V = R(q,g_P)-V + sum_z (p@q)[z]*(g[z]-g_P[z])^2",
        "decoder": "g_P=((p*m)@q)/(p@q)",
        "normalization": "sqrt(n)/tanh(epsilon/2)^2",
        "partition_canonicalization": "Relabel each row's highest-probability bin by first state occurrence.",
        "correct_binary_source_partition": {"sign_positive": [0,1,1,1], "sign_negative": [0,0,0,1]},
        "retained_other_cases": ["constant", "binary_other", "multibin"],
        "elapsed_seconds": time.perf_counter() - started,
        "hashes": {
            "learning_protocol": digest(protocol_path), "learning_raw": digest(original_path),
            "experiment_models": digest(Path(__file__).parent / "models.py"),
            "original_runner": digest(Path(__file__).parent / "learning.py"),
            "decomposition_script": digest(Path(__file__)),
            "decomposition_raw": digest(raw_path), "decomposition_summary": digest(summary_path),
        },
    }
    (data / "near_tie_decomposition_integrity.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
