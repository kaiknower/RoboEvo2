"""Create a local repair manifest from explicitly selected native failure cases."""
import argparse
import hashlib
import json
from pathlib import Path

from lharness.campaign import load_config
from repair_failed_case import load_source


def make_targets(config_path, source_cases):
    config = load_config(config_path)
    reference = Path(config["reference_output"]).resolve()
    targets, seen = [], set()
    for source in source_cases:
        source = Path(source).resolve(strict=True)
        if not source.is_relative_to(reference):
            raise ValueError(f"Case must be inside reference_output: {source}")
        raw = (source / "outcome.json").read_bytes()
        row = json.loads(raw)
        identity = {key: row[key] for key in ("group", "condition", "task_id", "init_id", "seed")}
        key = tuple(identity.values())
        if (row.get("status") != "completed" or row.get("protocol") != config["protocol"]
                or row["group"] not in config["checkpoint_groups"]
                or row["task_id"] not in config["task_ids"] or row["init_id"] not in config["init_ids"]
                or row["seed"] != config["seed"] + row["task_id"] * 1000 + row["init_id"]
                or key in seen):
            raise ValueError(f"Case is outside the configured cohort or duplicated: {source}")
        target = dict(identity, source_case=str(source), source_outcome=row,
                      source_outcome_sha256=hashlib.sha256(raw).hexdigest(),
                      source_actions_sha256=hashlib.sha256(
                          (source / "actions_proprioception.npz").read_bytes()).hexdigest())
        load_source(target, config)
        for camera in config["cameras"]:
            evidence = source / "coding/request_01" / f"{camera}_rgb.png"
            if not evidence.is_file():
                raise FileNotFoundError(evidence)
        targets.append(target)
        seen.add(key)
    return {"schema": "libero_pro_release_repair_targets_v1", "targets": targets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("cases", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(load_config(args.config)["reference_output"]).resolve()):
        parser.error("Output must remain outside reference_output")
    text = json.dumps(make_targets(args.config, args.cases), indent=2, allow_nan=False)
    with args.output.open("x") as stream:
        stream.write(text + "\n")


if __name__ == "__main__":
    main()
