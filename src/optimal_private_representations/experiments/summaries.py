"""Pointwise Monte Carlo summaries and publication figures for the recorded finite-law studies."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
from ..io import prepare_output

METHODS = ["attaining","direct_histogram","private_mean","no_mean_calibration","public_order","nonprivate_fit"]
LABEL = dict(attaining="Pilot and calibration",direct_histogram="Private histogram",
             private_mean="Private constant mean",no_mean_calibration="Without separate mean",
             public_order="Fixed index order",nonprivate_fit="Fit without training privacy")
COLOR = dict(attaining="#2166AC",direct_histogram="#D17A22",private_mean="#26876D",
             no_mean_calibration="#A05AA6",public_order="#D34B4B",nonprivate_fit="#454B54")
STYLE = dict(attaining="-",direct_histogram="-",private_mean="-.",
             no_mean_calibration="--",public_order=":",nonprivate_fit="--")


def interval(values):
    v=np.asarray(values,float);r=len(v);mean=float(np.mean(v))
    se=float(np.std(v,ddof=1)/np.sqrt(r)) if r>1 else 0.0
    half=float(student_t.ppf(.975,r-1)*se) if r>1 else 0.0
    return mean,se,mean-half,mean+half


def save(fig, folder, name):
    fig.savefig(folder/(name+".pdf"),bbox_inches="tight")
    fig.savefig(folder/(name+".png"),dpi=220,bbox_inches="tight")
    plt.close(fig)


def curve(ax,summary,study,xkey,methods,metric="excess",logx=False,logy=False):
    for method in methods:
        vals=sorted([r for r in summary if r["study"]==study and r["method"]==method],key=lambda r:r[xkey])
        x=np.array([r[xkey] for r in vals]);y=np.array([r[metric+"_mean"] for r in vals])
        lo=np.array([r[metric+"_ci_low"] for r in vals]);hi=np.array([r[metric+"_ci_high"] for r in vals])
        ax.plot(x,y,label=LABEL[method],color=COLOR[method],ls=STYLE[method],marker="o",ms=3,lw=1.55)
        ax.fill_between(x,np.maximum(lo,1e-9) if logy else lo,hi,color=COLOR[method],alpha=.12,linewidth=0)
    if logx:ax.set_xscale("log",base=2)
    if logy:ax.set_yscale("log")
    ax.grid(True,which="major",color="#DCE1E6",lw=.5)


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,default=Path("outputs/reproduction/data"))
    ap.add_argument("--figure-dir",type=Path,default=Path("outputs/reproduction/figures"))
    args=ap.parse_args(argv);data=prepare_output(args.data_dir);figs=prepare_output(args.figure_dir)
    protocol=json.loads((data/"learning_protocol.json").read_text())
    numeric=["n","d","alpha","epsilon","amplitude","sign","replicate","risk","oracle_risk",
             "excess","k","oracle_k","fit_seconds","a","rate_scale","target_mean","missing_cells"]
    with (data/"learning_raw.csv").open() as f:raw=list(csv.DictReader(f))
    for row in raw:
        for key in numeric:row[key]=float(row[key])
    expected=len(protocol["configurations"])*protocol["replicates_per_configuration"]*len(METHODS)
    if len(raw)!=expected:raise RuntimeError(f"Incomplete raw table: {len(raw)} vs {expected}")
    groups=defaultdict(list);unique=set()
    for row in raw:
        identity=(row["config_id"],row["replicate"],row["method"])
        if identity in unique:raise RuntimeError("Duplicate fit "+str(identity))
        unique.add(identity);groups[(row["config_id"],row["method"])].append(row)
    summary=[];paired=[]
    for (cid,method),rows in sorted(groups.items()):
        if len(rows)!=protocol["replicates_per_configuration"]:raise RuntimeError("Group size")
        item={key:rows[0][key] for key in ["config_id","study","family","n","d","alpha","epsilon","amplitude","sign","method","oracle_risk","oracle_k","a","rate_scale","target_mean"]}
        item["replicates"]=len(rows)
        for metric in ["risk","excess","k"]:
            mean,se,lo,hi=interval([r[metric] for r in rows])
            for suffix,val in [("mean",mean),("se",se),("ci_low",lo),("ci_high",hi)]:item[metric+"_"+suffix]=val
        item["constant_output_fraction"]=float(np.mean([r["k"]==1 for r in rows]))
        item["mean_missing_cells"]=float(np.mean([r["missing_cells"] for r in rows]))
        summary.append(item)
        if method!="attaining":
            ref={r["replicate"]:r["risk"] for r in groups[(cid,"attaining")]}
            mean,se,lo,hi=interval([r["risk"]-ref[r["replicate"]] for r in rows])
            paired.append(dict(config_id=cid,study=rows[0]["study"],method=method,
                               comparison="method minus attaining",mean=mean,se=se,
                               ci_low=lo,ci_high=hi,replicates=len(rows)))
    for filename,rows in [("learning_summary.csv",summary),("paired_comparisons.csv",paired)]:
        with (data/filename).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
                         "axes.labelsize":9,"legend.fontsize":8,"pdf.fonttype":42,
                         "ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    main_methods=["attaining","direct_histogram","private_mean","nonprivate_fit"]
    fig,axs=plt.subplots(2,2,figsize=(10,7.2),layout="constrained")
    curve(axs[0,0],summary,"sample_size","n",main_methods,logx=True,logy=True)
    axs[0,0].set(title="A  Sample size",xlabel="Fitting records n",ylabel="Expected oracle excess")
    curve(axs[0,1],summary,"alphabet_size","d",main_methods,logx=True,logy=True)
    axs[0,1].set(title="B  Input alphabet size",xlabel="Declared input states d",ylabel="Expected oracle excess")
    curve(axs[1,0],summary,"training_budget","alpha",main_methods)
    axs[1,0].set(title="C  Training privacy budget",xlabel=r"Training budget $\alpha$",ylabel="Expected oracle excess")
    curve(axs[1,1],summary,"query_budget","epsilon",main_methods)
    axs[1,1].set(title="D  Query privacy budget",xlabel=r"Query budget $\varepsilon$",ylabel="Expected oracle excess")
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    save(fig,figs,"learning_curves")

    fig,axs=plt.subplots(1,2,figsize=(10,3.9),layout="constrained")
    ablation=["attaining","no_mean_calibration","public_order","direct_histogram","private_mean"]
    curve(axs[0],summary,"sample_size","n",ablation,logx=True,logy=True)
    axs[0].set(title="A  Restricting the input order",xlabel="Fitting records n",ylabel="Expected oracle excess")
    curve(axs[1],summary,"mean_calibration","n",ablation,metric="risk",logx=True)
    vals=sorted([r for r in summary if r["study"]=="mean_calibration" and r["method"]=="attaining"],key=lambda r:r["n"])
    axs[1].axhline(vals[0]["oracle_risk"],color="#454B54",ls=":",lw=1.4,label="Population private oracle")
    axs[1].set(title="B  Uneven cells and a low target mean",xlabel="Fitting records n",ylabel="Expected squared prediction risk")
    handles,labels=axs[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    save(fig,figs,"ablation_curves")

    fig,axs=plt.subplots(1,2,figsize=(10,3.9),layout="constrained")
    curve(axs[0],summary,"query_budget","epsilon",main_methods,metric="risk")
    vals=sorted([r for r in summary if r["study"]=="query_budget" and r["method"]=="attaining"],key=lambda r:r["epsilon"])
    axs[0].plot([r["epsilon"] for r in vals],[r["oracle_risk"] for r in vals],
                color="black",ls=":",lw=1.5,label="Population private oracle")
    axs[0].set(title="A  Absolute risk and the moving oracle",xlabel=r"Query budget $\varepsilon$",ylabel="Expected squared prediction risk")
    axs[0].legend(frameon=False,fontsize=7,loc="lower left")
    near_summary=[]
    for n in [256,1024,4096,16384,65536]:
        for method in ["attaining","direct_histogram","nonprivate_fit"]:
            # Each displayed amplitude uses both predeclared signs, equally.
            rows=[r for r in raw if r["study"]=="near_tie" and r["n"]==n and r["amplitude"]==1 and r["method"]==method]
            strata=[np.array([r["excess"]*np.sqrt(n)/r["a"] for r in rows if r["sign"]==sign]) for sign in [-1,1]]
            means=[float(v.mean()) for v in strata]
            variances=[float(v.var(ddof=1)/len(v)) for v in strata]
            mean=.5*sum(means);se=.5*np.sqrt(sum(variances))
            df=sum(variances)**2/sum(v*v/(len(s)-1) for v,s in zip(variances,strata))
            half=float(student_t.ppf(.975,df)*se);lo=mean-half;hi=mean+half
            near_summary.append(dict(n=n,method=method,mean=mean,se=se,lo=lo,hi=hi,replicates=len(rows),df=df))
    for method in ["attaining","direct_histogram","nonprivate_fit"]:
        vals=[r for r in near_summary if r["method"]==method]
        x=[r["n"] for r in vals];y=[r["mean"] for r in vals]
        axs[1].plot(x,y,color=COLOR[method],ls=STYLE[method],marker="o",ms=3,label=LABEL[method])
        axs[1].fill_between(x,[r["lo"] for r in vals],[r["hi"] for r in vals],color=COLOR[method],alpha=.12)
    axs[1].set_xscale("log",base=2)
    axs[1].set(title="B  Local alternatives near a partition tie",xlabel="Fitting records n",
               ylabel=r"$\sqrt{n}\,\mathbb{E}[\mathrm{excess}]/a$")
    axs[1].legend(frameon=False,fontsize=7)
    for ax in axs:ax.grid(True,color="#DCE1E6",lw=.5)
    save(fig,figs,"risk_and_local_alternatives")
    with (data/"near_tie_display_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(near_summary[0]));w.writeheader();w.writerows(near_summary)
    integrity=dict(expected_rows=expected,observed_rows=len(raw),unique_fits=len(unique),
                   groups=len(groups),replicates_per_group=protocol["replicates_per_configuration"],
                   minimum_excess=min(r["excess"] for r in raw),maximum_risk=max(r["risk"] for r in raw),
                   intervals="Pointwise 95% Student-t Monte Carlo intervals, no multiplicity correction.",
                   paired_sign="Positive difference means the comparator has higher risk than attaining.")
    (data/"summary_integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
    print(json.dumps(integrity,indent=2))


if __name__=="__main__":
    main()
