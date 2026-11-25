"""Separate, serial implementation benchmarks on public simulated data."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import platform
import time
import tracemalloc

import numpy as np
from .models import fit_model
from .learning import law_arrays
from ..io import prepare_output


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark/data"))
    ap.add_argument("--repeats", type=int, default=9)
    args = ap.parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    out = prepare_output(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    rows = []
    for d in [4, 8, 16, 32, 64, 128]:
        p, m = law_arrays(dict(family="permuted_gradient", d=d))
        rng = np.random.default_rng(np.random.SeedSequence([2026112508,d]))
        states = rng.choice(d, size=8192, p=p)
        labels = rng.binomial(1,m[states]).astype(float)
        for method in ["attaining", "direct_histogram"]:
            def fit(rep):
                return fit_model(method,states,labels,cells=d,
                                 epsilon_query=float(np.log(4)),alpha_train=1,
                                 rng=np.random.default_rng(np.random.SeedSequence([2026112509,d,rep])))
            warm = fit(1000)
            del warm
            for rep in range(args.repeats):
                start = time.perf_counter()
                model = fit(rep)
                elapsed = time.perf_counter()-start
                rows.append(dict(d=d,n=8192,method=method,replicate=rep,
                                 seconds=elapsed,k=model["k"],
                                 peak_traced_bytes=""))
                del model
            # Trace allocations in a separate run, never during timing.
            gc.collect()
            tracemalloc.start()
            model = fit(2000)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append(dict(d=d,n=8192,method=method,replicate="memory",
                             seconds="",k=model["k"],peak_traced_bytes=peak))
            del model
            print(f"Benchmarked d={d}, {method}", flush=True)
    with (out/"runtime_raw.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metadata=dict(
        implementation="legacy uncapped search; not timings of the capped public API",
        repeats=args.repeats,n=8192,alpha=1,epsilon=float(np.log(4)),
        methods=["attaining","direct_histogram"],alphabet_sizes=[4,8,16,32,64,128],
        timing="fit call only, excluding data generation, one discarded warm-up",
        memory="peak Python/NumPy traced allocations in a separate fitting call; not process RSS",
        scheduling="Run serially after the learning Monte Carlo and population computation finish.",
        python=platform.python_version(),numpy=np.__version__,system=platform.system(),
        machine=platform.machine(),logical_cpus=os.cpu_count())
    (out/"runtime_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__=="__main__":
    main()
