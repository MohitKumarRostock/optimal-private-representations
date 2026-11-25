"""Verify frozen checksums, recompute summaries, and replay selected fits.

The checks inspect recorded synthetic evidence. They do not establish a
privacy theorem, certify machine arithmetic, or rerun the full simulation.
"""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from ..io import find_reference_dir, repository_root
from ..provenance import validate_protocol_sources
from ..experiments.learning import configurations, law_arrays
from ..experiments.models import analytic_risk, fit_model, population_oracle


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, *, atol=2e-12, label="value"):
    actual, expected = float(actual), float(expected)
    if actual == expected:
        return 0.0
    if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    return abs(actual - expected)


def verify_hashes(reference_dir):
    reference = find_reference_dir(reference_dir)
    candidates = [reference.parent.parent / "provenance/reference_data_manifest.json"]
    root = repository_root()
    if root:
        candidates.append(root / "provenance/reference_data_manifest.json")
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError("The reference-data hash manifest is required for validation")
    manifest = json.loads(manifest_path.read_text())
    checked = 0
    for relative, expected in manifest["files"].items():
        path = reference / Path(relative).name
        contents = path.read_bytes()
        if len(contents) != expected["bytes"] or hashlib.sha256(contents).hexdigest() != expected["sha256"]:
            raise AssertionError(f"Frozen reference hash mismatch: {path.name}")
        if "row_count" in expected:
            rows = read_csv(path)
            if len(rows) != expected["row_count"] or list(rows[0]) != expected["columns"]:
                raise AssertionError(f"Frozen reference schema mismatch: {path.name}")
        checked += 1
    return {"files_checked": checked, "manifest": str(manifest_path)}


def moments(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    half = float(student_t.ppf(0.975, len(values)-1) * se)
    return mean, se, mean-half, mean+half


def check_learning_tables(reference_dir):
    reference = find_reference_dir(reference_dir)
    protocol = json.loads((reference / "learning_protocol.json").read_text())
    source_status = validate_protocol_sources(protocol)
    if protocol["configurations"] != configurations():
        raise AssertionError("The packaged configuration grid differs from the recorded protocol")
    raw = read_csv(reference / "learning_raw.csv")
    groups = defaultdict(list)
    index = {}
    for row in raw:
        identity = (row["config_id"], int(row["replicate"]), row["method"])
        if identity in index:
            raise AssertionError(f"Duplicate raw fit: {identity}")
        index[identity] = row
        groups[(row["config_id"], row["method"])].append(row)
        close(float(row["risk"])-float(row["oracle_risk"]), row["excess"], label="excess identity")
        if float(row["excess"]) < -2e-11:
            raise AssertionError("A raw fit beats its recorded population oracle")
    expected = {(c["config_id"], rep, method)
                for c in protocol["configurations"]
                for rep in range(protocol["replicates_per_configuration"])
                for method in protocol["methods"]}
    if set(index) != expected:
        raise AssertionError("The raw fitting table is incomplete or has unexpected records")

    summaries = read_csv(reference / "learning_summary.csv")
    max_summary_error = 0.0
    if len(summaries) != len(groups):
        raise AssertionError("Incorrect number of method summary rows")
    for summary in summaries:
        rows = groups[(summary["config_id"], summary["method"])]
        for metric in ["risk", "excess", "k"]:
            actual = moments([float(row[metric]) for row in rows])
            for suffix, value in zip(["mean", "se", "ci_low", "ci_high"], actual):
                max_summary_error = max(max_summary_error, close(
                    value, summary[f"{metric}_{suffix}"], label=f"summary {metric}_{suffix}"))
        close(np.mean([int(row["k"]) == 1 for row in rows]), summary["constant_output_fraction"])
        close(np.mean([int(row["missing_cells"]) for row in rows]), summary["mean_missing_cells"])

    pairs = read_csv(reference / "paired_comparisons.csv")
    if len(pairs) != len(protocol["configurations"])*(len(protocol["methods"])-1):
        raise AssertionError("Incorrect number of paired comparisons")
    max_paired_error = 0.0
    for pair in pairs:
        rows = groups[(pair["config_id"], pair["method"])]
        differences = [float(row["risk"])-float(index[(row["config_id"],int(row["replicate"]),"attaining")]["risk"])
                       for row in rows]
        for suffix, value in zip(["mean", "se", "ci_low", "ci_high"], moments(differences)):
            max_paired_error = max(max_paired_error, close(value, pair[suffix], label=f"paired {suffix}"))
    return {"raw_fits":len(raw), "unique_fits":len(index), "summaries_recomputed":len(summaries),
            "paired_comparisons_recomputed":len(pairs), "max_summary_error":max_summary_error,
            "max_paired_error":max_paired_error, "source_hash_status":source_status}


def check_decomposition(reference_dir):
    reference = find_reference_dir(reference_dir)
    original = {(r["config_id"],r["replicate"],r["method"]):r
                for r in read_csv(reference / "learning_raw.csv")}
    raw = read_csv(reference / "near_tie_decomposition_raw.csv")
    groups = defaultdict(list)
    identities = set()
    maximum = 0.0
    for row in raw:
        identity = (row["config_id"],row["replicate"],row["method"])
        if identity in identities:
            raise AssertionError("Duplicate decomposition record")
        identities.add(identity)
        close(row["risk"], original[identity]["risk"], label="replayed risk")
        close(row["oracle_risk"], original[identity]["oracle_risk"], label="replayed oracle")
        maximum = max(maximum, close(float(row["channel_excess"])+float(row["decoder_squared_distance"]),
                                     row["total_excess"], label="Bayes decomposition"))
        close(row["decoder_error"],row["decoder_squared_distance"],label="decoder orthogonality")
        if float(row["channel_excess"]) < -5e-13 or float(row["decoder_error"]) < -5e-13:
            raise AssertionError("Negative decomposition component beyond numerical tolerance")
        groups[(int(row["n"]),float(row["c"]),row["method"])].append(row)
    summaries = read_csv(reference / "near_tie_decomposition_summary.csv")
    metrics = ["risk","population_redecoded_risk","total_excess","channel_excess","decoder_error",
               "normalized_total_excess","normalized_channel_excess","normalized_decoder_error","k",
               "binary_correct","binary_wrong","binary_other","constant","multibin"]
    maximum_summary = 0.0
    for summary in summaries:
        rows = groups[(int(summary["n"]),float(summary["c"]),summary["method"])]
        for metric in metrics:
            strata = [np.array([float(r[metric]) for r in rows if int(r["sign"])==sign]) for sign in [-1,1]]
            if len(strata[0]) != len(strata[1]) or len(strata[0]) < 2:
                raise AssertionError("Unbalanced decomposition strata")
            mean = float((strata[0].mean()+strata[1].mean())/2)
            variance = [float(x.var(ddof=1)/len(x)) for x in strata]
            se = math.sqrt(sum(variance))/2
            df = sum(variance)**2/sum(v*v/(len(x)-1) for v,x in zip(variance,strata)) if sum(variance)>0 else math.inf
            half = float(student_t.ppf(.975,df))*se if se else 0.0
            for suffix,value in [("mean",mean),("se",se),("ci_low",mean-half),("ci_high",mean+half),
                                 ("welch_df",df),("minus_mean",strata[0].mean()),("plus_mean",strata[1].mean())]:
                maximum_summary=max(maximum_summary,close(value,summary[f"{metric}_{suffix}"],label="decomposition summary"))
    if len(groups) != len(summaries) or len(raw) != 15360:
        raise AssertionError("Unexpected recorded decomposition dimensions")
    return {"recorded_rows":len(raw),"summary_groups_recomputed":len(summaries),
            "max_decomposition_identity_error":maximum,"max_summary_error":maximum_summary}


def replay_selected_fits(reference_dir):
    """Replay 48 fits spanning all six methods, endpoints and law families."""
    reference = find_reference_dir(reference_dir)
    protocol = json.loads((reference / "learning_protocol.json").read_text())
    validate_protocol_sources(protocol)
    selected = {"000","008","014","021","025","028","030","049"}
    originals = {(r["config_id"],r["method"]):r for r in read_csv(reference/"learning_raw.csv")
                 if r["config_id"] in selected and int(r["replicate"])==0}
    count, maximum = 0, 0.0
    for config in protocol["configurations"]:
        if config["config_id"] not in selected:
            continue
        p,m = law_arrays(config)
        seed = np.random.SeedSequence([protocol["master_seed"],int(config["config_id"]),0])
        data_seed, noise_seed = seed.spawn(2)
        rng = np.random.default_rng(data_seed)
        states = rng.choice(config["d"],size=config["n"],p=p)
        labels = rng.binomial(1,m[states]).astype(float)
        oracle = population_oracle(p,m,config["epsilon"],second=m)
        for method in protocol["methods"]:
            fitted=fit_model(method,states,labels,cells=config["d"],epsilon_query=config["epsilon"],
                             alpha_train=config["alpha"],rng=np.random.default_rng(noise_seed))
            risk=analytic_risk(p,m,m,fitted["q"],fitted["decoder"])
            old=originals[(config["config_id"],method)]
            maximum=max(maximum,close(risk,old["risk"],atol=5e-13,label="selected replay"))
            close(oracle["risk"],old["oracle_risk"],atol=5e-13,label="selected oracle")
            if fitted["k"] != int(old["k"]):
                raise AssertionError("Selected replay output count differs")
            count += 1
    return {"replayed_fits":count,"configurations":sorted(selected),"replicate":0,
            "max_risk_error":maximum,"runtime_columns_compared":False}


def run(reference_dir):
    return {"hashes":verify_hashes(reference_dir),
            "learning":check_learning_tables(reference_dir),
            "decomposition":check_decomposition(reference_dir),
            "selected_replay":replay_selected_fits(reference_dir)}

