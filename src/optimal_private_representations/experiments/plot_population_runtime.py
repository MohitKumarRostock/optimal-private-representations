"""Figures from exact population references and separate serial benchmarks."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .summaries import COLOR,LABEL,save
from ..io import prepare_output


def main(data_dir, figure_dir):
    data=prepare_output(data_dir);figs=prepare_output(figure_dir)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
                         "axes.labelsize":9,"legend.fontsize":8,"pdf.fonttype":42,
                         "ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    with (data/"geometry_exact_four_state.csv").open() as f:exact=list(csv.DictReader(f))
    with (data/"geometry_modulus_grid.csv").open() as f:grid=list(csv.DictReader(f))
    fig,axs=plt.subplots(1,2,figsize=(10,4.0),layout="constrained")
    x=np.array([float(r["mse"]) for r in exact]);y=np.array([float(r["deficiency_reference"]) for r in exact])
    order=np.argsort(x);x=x[order];y=y[order]
    axs[0].loglog(x,y,"o-",color="#2166AC",ms=3,lw=1.4,label="Reoptimized oracle gap")
    axs[0].loglog(x,(35/324)*x**(2/3),"--",color="#D17A22",lw=1.4,label=r"$(35/324)v^{2/3}$")
    axs[0].set(title=r"A  Exact four-state family at $\varepsilon=\log 2$",
               xlabel="Actual reconstruction MSE v",ylabel="Private-task deficiency")
    axs[0].legend(frameon=False,loc="upper left")
    choices=[("51/50","#2166AC"),("21/20","#D17A22"),("11/10","#26876D"),("2","#A05AA6"),("4","#454B54")]
    for value,color in choices:
        rows=sorted([r for r in grid if r["exp_epsilon_exact"]==value and float(r["budget_v"])>0],key=lambda r:float(r["budget_v"]))
        x=[float(r["budget_v"]) for r in rows]
        y=[float(r["deficiency_over_gamma_budget"]) for r in rows]
        label=r"$e^\varepsilon="+value+"$"
        axs[1].semilogx(x,y,color=color,lw=1.4,label=label)
    axs[1].axhline(37/1920,color="black",ls=":",lw=1,label="Proved lower constant")
    axs[1].axvline(1/64,color="#7B8794",ls="--",lw=.9)
    axs[1].text(1/64,.46,"Family MSE saturates",rotation=90,ha="right",va="top",fontsize=7,color="#454B54")
    axs[1].set(title="B  Lower-family comparison across privacy budgets",
               xlabel="Allowed reconstruction MSE budget v",
               ylabel=r"$\Delta_\varepsilon(T)/\Gamma_a(v)$",ylim=(0,.48))
    axs[1].legend(frameon=False,ncol=2,loc="lower left",bbox_to_anchor=(0,.10),fontsize=7)
    for ax in axs:ax.grid(True,which="major",color="#DCE1E6",lw=.5)
    save(fig,figs,"representation_geometry")

    with (data/"runtime_raw.csv").open() as f:raw=list(csv.DictReader(f))
    groups=defaultdict(list);memory={}
    for r in raw:
        key=(int(r["d"]),r["method"])
        if r["replicate"]=="memory":memory[key]=int(r["peak_traced_bytes"])
        else:groups[key].append(float(r["seconds"]))
    summary=[]
    for (d,method),v in sorted(groups.items()):
        summary.append(dict(d=d,method=method,median_seconds=float(np.median(v)),
                            q25_seconds=float(np.quantile(v,.25)),q75_seconds=float(np.quantile(v,.75)),
                            min_seconds=min(v),max_seconds=max(v),repeats=len(v),
                            peak_traced_bytes=memory[(d,method)]))
    with (data/"runtime_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for method in ["attaining","direct_histogram"]:
        vals=[r for r in summary if r["method"]==method]
        x=[r["d"] for r in vals];y=[r["median_seconds"] for r in vals]
        axs[0].plot(x,y,"o-",color=COLOR[method],label=LABEL[method],ms=3)
        axs[0].fill_between(x,[r["q25_seconds"] for r in vals],[r["q75_seconds"] for r in vals],color=COLOR[method],alpha=.12)
        axs[1].plot(x,[r["peak_traced_bytes"]/2**20 for r in vals],"o-",color=COLOR[method],label=LABEL[method],ms=3)
    for ax in axs:
        ax.set_xscale("log",base=2);ax.set_yscale("log");ax.grid(True,which="major",color="#DCE1E6",lw=.5)
        ax.set_xlabel("Declared input states d")
    axs[0].set(title="A  Serial reference runtime",ylabel="Median fitting time (seconds)")
    axs[1].set(title="B  Separate allocation trace",ylabel="Peak traced fitting allocations (MiB)")
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    save(fig,figs,"runtime_and_memory")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main("outputs/reproduction/data", "outputs/reproduction/figures")
