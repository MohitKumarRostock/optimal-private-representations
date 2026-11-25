"""Generate reported numeric tables from the immutable simulation summaries."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from ..io import prepare_output


def main(data_dir, table_dir):
    data=prepare_output(data_dir);out=prepare_output(table_dir)
    with (data/"learning_summary.csv").open() as f:rows=list(csv.DictReader(f))
    index={(r["config_id"],r["method"]):r for r in rows}
    order=[("attaining","Pilot and calibration"),("direct_histogram","Private histogram"),
           ("private_mean","Private constant mean"),("no_mean_calibration","Without separate mean"),
           ("public_order","Fixed index order"),("nonprivate_fit","Fit without training privacy")]
    lines=[r"\begin{tabular}{@{}lrr@{}}",r"\toprule",
           r"Method & Balanced graded task & Uneven low-mean task\\",r"\midrule"]
    for method,label in order:
        vals=[]
        for cid in ["004","028"]:
            r=index[(cid,method)];mean=1e3*float(r["excess_mean"])
            half=1e3*(float(r["excess_ci_high"])-float(r["excess_ci_low"]))/2
            vals.append("$"+f"{mean:.3f}\\pm {half:.3f}"+"$")
        lines.append(label+" & "+" & ".join(vals)+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (out/"main_comparison.tex").write_text("\n".join(lines)+"\n")
    with (data/"runtime_summary.csv").open() as f:runtime=list(csv.DictReader(f))
    lines=[r"\begin{tabular}{@{}rrr@{}}",r"\toprule",
           r"Input states & Median fit time (ms) & Peak traced allocation (MiB)\\",r"\midrule"]
    for r in runtime:
        if r["method"]=="attaining":
            lines.append(f'{r["d"]} & {1e3*float(r["median_seconds"]):.3f} & {int(r["peak_traced_bytes"])/2**20:.3f}'+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (out/"runtime.tex").write_text("\n".join(lines)+"\n")
    with (data/"paired_comparisons.csv").open() as f:pairs=list(csv.DictReader(f))
    selected=dict(
        main_configuration_ids=["004","028"],
        balanced_oracle=float(index[("004","attaining")]["oracle_risk"]),
        low_mean_oracle=float(index[("028","attaining")]["oracle_risk"]),
        low_target_mean=float(index[("028","attaining")]["target_mean"]),
        paired_results=[r for r in pairs if r["config_id"] in ["004","026","028","029"]
                        and r["method"] in ["direct_histogram","no_mean_calibration","private_mean"]])
    (data/"reported_results.json").write_text(json.dumps(selected,indent=2)+"\n")


if __name__=="__main__":
    main("outputs/reproduction/data", "outputs/reproduction/tables")
