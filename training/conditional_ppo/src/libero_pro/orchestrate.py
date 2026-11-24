"""Sequential eight-policy campaign; release GPU memory between each subprocess."""
import json
from pathlib import Path
import subprocess
import sys
from .campaign import conditions
from .evaluation import select_best, aggregate


def run_eight(model, output_root, iterations, micro_batch_size=128):
    root, model = Path(output_root).resolve(), Path(model).resolve()
    if root.exists() and any(root.glob("*/launch.json")):
        raise FileExistsError("An existing campaign needs explicit per-condition --resume")
    if iterations < 1:
        raise ValueError("Iterations must be positive")
    root.mkdir(parents=True, exist_ok=True)
    def run(*args):
        subprocess.run([sys.executable, "-m", "libero_pro", *map(str, args)], check=True)
    selected_reports, baseline_reports, exports = [], [], []
    for c in conditions():
        run("train", "--condition", c.name, "--model", model, "--output-root", root, "--iterations", iterations,
            "--micro-batch-size", micro_batch_size)
        run_dir = root / c.name
        checkpoints = sorted(run_dir.glob("checkpoints/global_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        if not checkpoints:
            raise FileNotFoundError(f"No PPO checkpoints for {c.name}")
        validation_paths = sorted((run_dir / "evaluation").glob("validation_*.json"),
                                  key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
        if len(validation_paths) != iterations + 1:
            raise ValueError(f"Missing per-iteration validation for {c.name}")
        selected = select_best(validation_paths, c.name)
        if not Path(selected["model_path"]).is_dir():
            raise FileNotFoundError("The selected policy must have been retained")
        (run_dir / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
        exported = root / "final" / c.name
        run("export", "--condition", c.name, "--model", selected["model_path"], "--baseline", model, "--output", exported)
        exports.append(str(exported))
        for label, candidate in (("baseline", model), ("selected", exported)):
            report = run_dir / "evaluation" / f"test_{label}.json"
            run("evaluate", "--condition", c.name, "--model", candidate, "--baseline", model,
                "--split", "test", "--output", report)
            (baseline_reports if label == "baseline" else selected_reports).append(report)
    before, after = aggregate(baseline_reports), aggregate(selected_reports)
    result = dict(baseline=before, selected=after, delta=after["success_rate"] - before["success_rate"], exports=exports)
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(root / "comparison.json")
