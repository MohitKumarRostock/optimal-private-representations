"""Command-line workflows for the self-contained research repository."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil

from .io import find_reference_dir, prepare_output


def write_report(path, report):
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


def copy_reference(reference, output):
    data=prepare_output(output/"data",reference)
    for source in reference.iterdir():
        if source.is_file() and source.suffix in {".csv",".json"}:
            shutil.copy2(source,data/source.name)
    return data


def parser():
    main=argparse.ArgumentParser(description=__doc__)
    sub=main.add_subparsers(dest="command",required=True)
    descriptions={
        "smoke":"Check imports, exact examples, and a small private fit.",
        "validate":"Verify frozen references, finite checks, and 48 selected fit replays.",
        "reproduce":"Rebuild summaries, exact geometry, publication figures, and tables without a full simulation.",
        "simulate":"Run the recorded 50-configuration simulation using uncapped algorithms.",
        "summarize":"Analyze a completed new simulation without changing its input data.",
        "population":"Run exact finite-law geometry and output-count calculations.",
        "decompose":"Explicitly replay existing local-alternative fits and decompose their risk.",
        "benchmark":"Measure the original uncapped implementation on new serial runs.",
    }
    for command,description in descriptions.items():
        child=sub.add_parser(command,help=description,description=description)
        child.add_argument("--output-dir",type=Path,default=Path("outputs")/command)
        if command in {"validate","reproduce","decompose"}:
            child.add_argument("--reference-dir",type=Path,default=None,
                               help="Frozen data directory; defaults to data/reference in the checkout.")
        if command=="simulate":
            child.add_argument("--replicates",type=int,default=256)
            child.add_argument("--resume",action="store_true")
            child.add_argument("--protocol-only",action="store_true")
        if command=="summarize":
            child.add_argument("--data-dir",type=Path,required=True,
                               help="Directory containing learning_raw.csv and learning_protocol.json.")
        if command=="benchmark":
            child.add_argument("--repeats",type=int,default=9)
    return main


def main(argv=None):
    args=parser().parse_args(argv)
    reference=find_reference_dir(args.reference_dir) if hasattr(args,"reference_dir") else None
    output=prepare_output(args.output_dir,reference)
    if args.command=="smoke":
        from .validation.smoke import run
        write_report(output/"smoke.json",run())
    elif args.command=="validate":
        from .validation import finite,baselines,output_cap
        from .validation.reference import run
        with redirect_stdout(io.StringIO()):
            report={"status":"passed","reference":run(reference),
                    "finite":finite.run_checks(),
                    "baselines":baselines.main(output/"baseline_checks.json"),
                    "output_cap":output_cap.main(output/"output_cap_verification.json")}
        (output/"finite_checks.json").write_text(json.dumps(report["finite"],indent=2)+"\n")
        write_report(output/"validation.json",report)
    elif args.command=="reproduce":
        from .experiments import population,output_count,publication,tables
        from .validation.reference import verify_hashes
        hashes=verify_hashes(reference)
        data=copy_reference(reference,output)
        with redirect_stdout(io.StringIO()):
            geometry=population.run(data)
            output_count.run(data)
            publication.render(data,output/"figures")
            tables.main(data,output/"tables")
        write_report(output/"reproduction.json",{
            "status":"passed","reference_verification":hashes,
            "geometry_combinations":geometry["checks"]["total_laws"],
            "generated_figure_pairs":6,"generated_tables":2,
            "new_learning_fits":0,"new_benchmarks":0,
            "decomposition":"Plot uses recorded post-run decomposition; no replay performed.",
            "output_dir":str(output)})
    elif args.command=="population":
        from .experiments import population,output_count
        data=prepare_output(output/"data")
        population.run(data)
        output_count.run(data)
    elif args.command=="simulate":
        from .experiments.learning import main as simulate
        options=["--output-dir",str(output/"data"),"--replicates",str(args.replicates)]
        if args.resume: options.append("--resume")
        if args.protocol_only: options.append("--protocol-only")
        simulate(options)
    elif args.command=="summarize":
        from .experiments.publication import render_learning
        source=args.data_dir.expanduser().resolve()
        output=prepare_output(output,source)
        data=prepare_output(output/"data",source)
        for filename in ["learning_raw.csv","learning_protocol.json"]:
            shutil.copy2(source/filename,data/filename)
        render_learning(data,output/"figures")
        write_report(output/"analysis.json",{
            "status":"passed","source_data":str(source),"inputs_modified":False,
            "new_learning_fits":0,"generated_figure_pairs":3,
            "summaries":["learning_summary.csv","paired_comparisons.csv","near_tie_display_summary.csv"],
            "output_dir":str(output)})
    elif args.command=="decompose":
        from .experiments import decomposition
        from .experiments.plot_decomposition import main as plot
        from .experiments.publication import publication_save
        from .experiments import plot_decomposition
        data=copy_reference(reference,output)
        decomposition.main(["--data-dir",str(data)])
        previous=plot_decomposition.save
        try:
            plot_decomposition.save=publication_save
            plot(data,output/"figures")
        finally:
            plot_decomposition.save=previous
    elif args.command=="benchmark":
        from .experiments.benchmarks import main as benchmark
        benchmark(["--output-dir",str(output/"data"),"--repeats",str(args.repeats)])

