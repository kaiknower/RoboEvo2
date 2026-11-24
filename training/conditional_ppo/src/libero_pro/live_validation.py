"""Blocking validation after each PPO iteration, using the synchronized rollout actor."""
import csv
import json
from pathlib import Path
import random
import shutil
import time

from .evaluation import evaluate, select_best


def evaluate_live_policy(policy, condition, model_path, baseline, output, device, smoke=False):
    import numpy as np
    import torch
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    training = policy.training
    devices = list(range(torch.cuda.device_count())) if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            policy.eval()
            return evaluate(condition, model_path, baseline, "validation", output,
                            device=device, policy=policy, smoke=smoke)
    finally:
        policy.train(training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def run_directory(runner):
    logger = runner.cfg.runner.logger
    return Path(logger.log_path).resolve() / logger.experiment_name


def validation_report(runner, model_path):
    step = runner.global_step
    settings = runner.cfg.runner.libero_pro_validation
    smoke = getattr(settings, "smoke", False)
    prefix = "smoke_validation" if smoke else "validation"
    output = run_directory(runner) / "evaluation" / f"{prefix}_{step:04d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        report = json.loads(output.read_text()) if smoke else select_best([output], settings.condition)
        if smoke:
            validate_smoke_report(report, settings.condition)
        if Path(report["model_path"]).resolve() != Path(model_path).resolve():
            raise ValueError("Existing validation report belongs to another checkpoint")
        return report
    partial = output.with_suffix(".jsonl")
    if partial.exists():
        partial.rename(partial.with_name(f"{partial.stem}.interrupted_{time.time_ns()}.jsonl"))
    started = time.monotonic()
    results = runner.rollout.evaluate_libero_pro(
        condition=settings.condition, model_path=str(model_path),
        baseline=settings.baseline, output=str(output), smoke=smoke).wait()
    reports = [r for r in results if r is not None]
    if len(reports) != 1:
        raise RuntimeError("Exactly rank 0 must return the validation report")
    report = reports[0]
    report.update(iteration=step, evaluation_seconds=time.monotonic() - started)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if smoke:
        validate_smoke_report(report, settings.condition)
    else:
        select_best([output], settings.condition)
    return report


def validate_smoke_report(report, condition):
    if (report.get("protocol") != "libero_pro_ppo_smoke_v1"
            or report.get("condition") != condition or report.get("split") != "validation"
            or report.get("complete") is not False or report.get("smoke") is not True
            or report.get("episodes") != 1 or report.get("action_cap") != 10
            or report.get("checks_passed") is not True or "success_rate" in report):
        raise ValueError("Invalid smoke evaluation report")


def record_curve_and_retain(runner):
    if getattr(runner.cfg.runner.libero_pro_validation, "smoke", False):
        metrics = {"smoke/evaluation_passed": 1}
        runner.metric_logger.log(data=metrics, step=runner.global_step)
        return metrics
    root = run_directory(runner)
    paths = sorted((root / "evaluation").glob("validation_*.json"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    best = select_best(paths, runner.cfg.runner.libero_pro_validation.condition)
    reports = [json.loads(p.read_text()) for p in paths]
    baseline = reports[0]["success_rate"]
    fields = ["iteration", "success_rate", "delta_vs_baseline", "reward_coverage", "evaluation_seconds"]
    with (root / "validation_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for p, report in zip(paths, reports):
            writer.writerow(dict(iteration=int(p.stem.rsplit("_", 1)[-1]),
                success_rate=report["success_rate"], delta_vs_baseline=report["success_rate"] - baseline,
                reward_coverage=report["reward_coverage"], evaluation_seconds=report.get("evaluation_seconds", 0)))
    (root / "selection.json").write_text(json.dumps(best, indent=2) + "\n")
    latest = root / "checkpoints" / f"global_step_{runner.global_step}"
    keep = {latest.resolve(), Path(best["model_path"]).resolve()}
    for checkpoint in (root / "checkpoints").glob("global_step_*"):
        if checkpoint.is_dir() and checkpoint.resolve() not in keep:
            shutil.rmtree(checkpoint)
    current = reports[-1]
    metrics = {f"eval/{key}": current[key] for key in ("success_rate", "reward_coverage")}
    metrics["eval/delta_vs_baseline"] = current["success_rate"] - baseline
    metrics["eval/best_success_rate"] = best["success_rate"]
    runner.metric_logger.log(data=metrics, step=runner.global_step)
    return metrics


def initialize_validation(runner):
    runner.update_rollout_weights()
    if runner.global_step == 0:
        model_path = runner.cfg.runner.libero_pro_validation.baseline
    else:
        model_path = run_directory(runner) / "checkpoints" / f"global_step_{runner.global_step}"
    validation_report(runner, model_path)
    record_curve_and_retain(runner)


def validate_iteration(runner):
    runner._save_checkpoint()
    runner.update_rollout_weights()
    checkpoint = run_directory(runner) / "checkpoints" / f"global_step_{runner.global_step}"
    validation_report(runner, checkpoint)
    return record_curve_and_retain(runner)
