"""Reproduce the finite-law learning simulation with the recorded uncapped algorithms.

All records are simulated. Fixed seeds reproduce the public research protocol.
Risk integrates target and query randomness analytically. The joint collection
of fitted experimental models is not a private release for confidential data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from ..io import prepare_output
from ..provenance import legacy_source_hashes

METHODS = (
    "attaining", "direct_histogram", "private_mean",
    "no_mean_calibration", "public_order", "nonprivate_fit",
)
MASTER_SEED = 2026112507


def bit_reverse_indices(d):
    if d < 1 or d & (d - 1):
        raise ValueError("bit-reversal law uses a power-of-two alphabet")
    width = max(1, (d - 1).bit_length())
    return np.array([int(format(i, f"0{width}b")[::-1], 2) for i in range(d)])


def law_arrays(config):
    d = config["d"]
    family = config["family"]
    if family == "permuted_gradient":
        p = np.full(d, 1.0 / d)
        m = np.linspace(0.05, 0.95, d)[bit_reverse_indices(d)]
    elif family == "rare_low_mean":
        p = (np.arange(1, d + 1, dtype=float)) ** -1.3
        p /= p.sum()
        m = np.linspace(0.02, 0.38, d)[bit_reverse_indices(d)]
    elif family == "near_tie":
        p = np.array([1/3, 1/6, 1/6, 1/3])
        theta = config["sign"] * config["amplitude"] / np.sqrt(config["n"])
        m = np.array([0.25, 0.5 + theta, 0.5 + theta, 0.75])
    else:
        raise ValueError(family)
    return p, m


def configurations():
    configs = []
    def add(study, family="permuted_gradient", **updates):
        item = dict(study=study, family=family, n=2048, d=16,
                    alpha=1.0, epsilon=float(np.log(4)),
                    amplitude=0.0, sign=0)
        item.update(updates)
        item["config_id"] = f"{len(configs):03d}"
        configs.append(item)
    for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        add("sample_size", n=n)
    for alpha in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]:
        add("training_budget", alpha=alpha)
    for epsilon in [0.0, 0.1, 0.25, 0.5, float(np.log(2)), 1.0, float(np.log(4))]:
        add("query_budget", epsilon=epsilon)
    for d in [4, 8, 16, 32, 64]:
        add("alphabet_size", d=d)
    for n in [128, 512, 2048, 8192]:
        add("mean_calibration", family="rare_low_mean", d=32, n=n,
            alpha=0.25, epsilon=float(np.log(2)))
    for n in [256, 1024, 4096, 16384, 65536]:
        for amplitude in [0.5, 1.0]:
            for sign in [-1, 1]:
                add("near_tie", family="near_tie", d=4, n=n,
                    epsilon=float(np.log(2)), amplitude=amplitude, sign=sign)
    return configs


def rate_scale(config):
    a = math.tanh(config["epsilon"] / 2) ** 2
    kappa = math.tanh(config["alpha"] / 2)
    n, d = config["n"], config["d"]
    t = n * kappa
    gamma = min(a, a * math.sqrt(d/n), (a*d/n) ** (2/3))
    return (1/n + a/math.sqrt(n) + gamma
            + (min(1, 1/t**2) if t else 1)
            + a*(min(1, d/t) if t else 1))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simulation/data"))
    parser.add_argument("--replicates", type=int, default=256)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.replicates < 2:
        raise ValueError("At least two replicates required for Monte Carlo uncertainty.")
    out = prepare_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    protocol = dict(
        title="Finite-law private representation learning evidence",
        date="2026-11-25", master_seed=MASTER_SEED,
        replicates_per_configuration=args.replicates,
        methods=list(METHODS), configurations=configs,
        law_definition="Bernoulli target conditional on the declared input state.",
        risk="Exact population squared risk integrates label and query randomness.",
        coupling="Same data and same initial synthetic RNG seed across methods within a replicate.",
        interval="Pointwise 95% Student-t Monte Carlo intervals for expected risk; paired intervals for method differences.",
        interpretation="Selected laws, including explicit n-dependent local alternatives. No empirical minimax or universal privacy proof.",
        source_hashes=legacy_source_hashes(),
        implementation="packaged legacy uncapped learner; recorded seed arithmetic retained",
        environment=dict(python=sys.version, numpy=np.__version__,
                         platform=platform.system(), machine=platform.machine()),
    )
    protocol_path = out / "learning_protocol.json"
    if args.resume and protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        for key in ["master_seed", "replicates_per_configuration", "configurations", "methods", "source_hashes"]:
            if old[key] != protocol[key]:
                raise RuntimeError("Resume protocol mismatch: " + key)
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    if args.protocol_only:
        print(f"Registered {len(configs)} configurations, {args.replicates} replicates, {len(METHODS)} methods.", flush=True)
        return

    from .models import fit_model, analytic_risk, population_oracle
    fields = ["config_id", "study", "family", "n", "d", "alpha", "epsilon",
              "amplitude", "sign", "replicate", "method", "risk", "oracle_risk",
              "excess", "k", "oracle_k", "fit_seconds", "a", "rate_scale",
              "target_mean", "missing_cells"]
    raw_path = out / "learning_raw.csv"
    done = set()
    if args.resume and raw_path.exists():
        with raw_path.open() as f:
            for row in csv.DictReader(f):
                done.add((row["config_id"], int(row["replicate"]), row["method"]))
    mode = "a" if done else "w"
    started = time.perf_counter()
    count = 0
    with raw_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done:
            writer.writeheader()
        for config in configs:
            p, m = law_arrays(config)
            oracle = population_oracle(p, m, config["epsilon"], second=m)
            oracle_risk = analytic_risk(p, m, m, oracle["q"], oracle["decoder"])
            for rep in range(args.replicates):
                seed = np.random.SeedSequence([MASTER_SEED, int(config["config_id"]), rep])
                data_seed, noise_seed = seed.spawn(2)
                rng = np.random.default_rng(data_seed)
                h = rng.choice(config["d"], size=config["n"], p=p)
                y = rng.binomial(1, m[h]).astype(float)
                missing = config["d"] - len(np.unique(h))
                for method in METHODS:
                    if (config["config_id"], rep, method) in done:
                        continue
                    noise = np.random.default_rng(noise_seed)
                    start_fit = time.perf_counter()
                    fit = fit_model(method, h, y, cells=config["d"],
                                    epsilon_query=config["epsilon"],
                                    alpha_train=config["alpha"], rng=noise)
                    fit_seconds = time.perf_counter() - start_fit
                    risk = analytic_risk(p, m, m, fit["q"], fit["decoder"])
                    excess = risk - oracle_risk
                    if excess < -2e-11:
                        raise RuntimeError(("fitted risk beats population oracle", config, method, rep, excess))
                    row = {key: config[key] for key in ["config_id","study","family","n","d","alpha","epsilon","amplitude","sign"]}
                    row.update(replicate=rep, method=method, risk=risk,
                               oracle_risk=oracle_risk, excess=excess, k=fit["k"],
                               oracle_k=oracle["k"], fit_seconds=fit_seconds,
                               a=math.tanh(config["epsilon"]/2)**2,
                               rate_scale=rate_scale(config), target_mean=float(p @ m),
                               missing_cells=missing)
                    writer.writerow(row)
                    count += 1
            f.flush()
            print(f'Completed {config["config_id"]}: {config["study"]}, n={config["n"]}, d={config["d"]}; {count} new fits, {time.perf_counter()-started:.1f}s', flush=True)
    (out/"learning_run.json").write_text(json.dumps(dict(
        new_fits=count, elapsed_seconds=time.perf_counter()-started,
        resumed=args.resume, protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest()), indent=2)+"\n")


if __name__ == "__main__":
    main()
