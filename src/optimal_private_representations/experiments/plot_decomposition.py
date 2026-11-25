"""Plot the transparently post-run decomposition of the existing near-tie fits."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .summaries import COLOR,LABEL,save
from ..io import prepare_output


def main(data_dir, figure_dir):
    data=Path(data_dir);figs=prepare_output(figure_dir)
    with (data/"near_tie_decomposition_summary.csv").open() as f:
        rows=[r for r in csv.DictReader(f) if float(r["c"])==1]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
                         "axes.labelsize":9,"legend.fontsize":8,"pdf.fonttype":42,
                         "ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,2,figsize=(10,3.9),layout="constrained")
    vals=sorted([r for r in rows if r["method"]=="attaining"],key=lambda r:int(r["n"]))
    for metric,label,color,style in [
        ("normalized_total_excess","Total excess","#2166AC","-"),
        ("normalized_channel_excess","Channel component","#26876D","-"),
        ("normalized_decoder_error","Decoder component","#D17A22","--")]:
        x=[int(r["n"]) for r in vals];y=[float(r[metric+"_mean"]) for r in vals]
        axs[0].plot(x,y,style,color=color,label=label,marker="o",ms=3)
        axs[0].fill_between(x,[float(r[metric+"_ci_low"]) for r in vals],
                           [float(r[metric+"_ci_high"]) for r in vals],color=color,alpha=.12)
    for method in ["attaining","direct_histogram","nonprivate_fit"]:
        vals=sorted([r for r in rows if r["method"]==method],key=lambda r:int(r["n"]))
        x=[int(r["n"]) for r in vals];metric="normalized_channel_excess"
        axs[1].plot(x,[float(r[metric+"_mean"]) for r in vals],color=COLOR[method],
                    label=LABEL[method],marker="o",ms=3)
        axs[1].fill_between(x,[float(r[metric+"_ci_low"]) for r in vals],
                           [float(r[metric+"_ci_high"]) for r in vals],color=COLOR[method],alpha=.12)
    axs[0].set_title("A  Pilot and calibration error components")
    axs[1].set_title("B  Channel components across learners")
    for ax in axs:
        ax.set_xscale("log",base=2);ax.set_xlabel("Fitting records n")
        ax.set_ylabel(r"$\sqrt{n}/a$ times expected error")
        ax.grid(True,color="#DCE1E6",lw=.5);ax.legend(frameon=False,fontsize=8)
    save(fig,figs,"near_tie_decomposition")


if __name__=="__main__":
    main("outputs/reproduction/data", "outputs/reproduction/figures")
