import argparse
import json
from pathlib import Path
from .campaign import plan


def main():
    parser = argparse.ArgumentParser(description="LIBERO-PRO eight-condition training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--output-root", default="runs")
    p = sub.add_parser("download")
    p.add_argument("--output", type=Path, default=Path("weights/v8-original"))
    p = sub.add_parser("convert")
    p.add_argument("--input", type=Path, default=Path("weights/v8-original"))
    p.add_argument("--output", type=Path, default=Path("weights/v8-pytorch"))
    p.add_argument("--merged", type=Path, help="Reuse a completed merge after interrupted conversion")
    p = sub.add_parser("train")
    p.add_argument("--condition", required=True)
    p.add_argument("--model", default="weights/v8-pytorch")
    p.add_argument("--output-root", default="runs")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--micro-batch-size", type=int, help="Per-GPU PPO micro batch; global batch stays fixed")
    p = sub.add_parser("evaluate")
    p.add_argument("--condition", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--baseline", default="weights/v8-pytorch")
    p.add_argument("--split", choices=["validation", "test"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("run-eight")
    p.add_argument("--model", default="weights/v8-pytorch")
    p.add_argument("--output-root", default="runs")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--micro-batch-size", type=int, default=128)
    p = sub.add_parser("select")
    p.add_argument("--condition", required=True)
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("export")
    p.add_argument("--condition", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--baseline", default="weights/v8-pytorch")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("aggregate")
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan(args.output_root), indent=2) + "\n")
        print(args.output)
    elif args.command == "download":
        from .checkpoint import download_baseline
        from .runtime import ROOT
        baseline = json.loads((ROOT / "configs/baseline.json").read_text())
        print(download_baseline(baseline["destination"], baseline["destination_revision"], args.output))
    elif args.command == "convert":
        from .conversion import convert
        convert(args.input, args.output, args.merged)
    elif args.command == "train":
        from .training import train
        train(args.condition, args.model, args.output_root, args.iterations, args.smoke, args.dry_run,
              args.resume, args.micro_batch_size)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(args.condition, args.model, args.baseline, args.split, args.output, args.device)
    elif args.command == "select":
        from .evaluation import select_best
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(select_best(args.reports, args.condition), indent=2) + "\n")
    elif args.command == "export":
        from .export import export_checkpoint
        print(export_checkpoint(args.condition, args.model, args.baseline, args.output))
    elif args.command == "aggregate":
        from .evaluation import aggregate
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(aggregate(args.reports), indent=2) + "\n")
    elif args.command == "run-eight":
        from .orchestrate import run_eight
        run_eight(args.model, args.output_root, args.iterations, args.micro_batch_size)
