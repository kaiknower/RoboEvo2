"""Run a sequence of matched, failure-driven harness and policy updates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from .evidence import curate, select_repairs, task_sampling

INITIAL = Path(__file__).parent / "initial" / "state.json"
ROUTES = ("code", "code_icl", "code_policy")


def load(path):
    return json.loads(Path(path).read_text())


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(encode(value) + b"\n")
        stage = Path(stream.name)
    stage.replace(path)


class Campaign:
    def __init__(self, config, workspace):
        self.config = config
        self.root = workspace.resolve()
        self.commands = config["commands"]
        self.cases = {split: config[split] for split in ("train", "development")}
        if not isinstance(config["initial_policy"], str) or not config["initial_policy"].strip():
            raise ValueError("Supply a demonstration-trained initial policy checkpoint")
        if type(config["seed"]) is not int or type(config["workers"]) is not int or config["workers"] < 1:
            raise ValueError("Supply an integer seed and a positive worker count")
        if not math.isfinite(config["gamma"]) or config["gamma"] <= 0:
            raise ValueError("Sampling gamma must be positive")
        ids, starts_by_split = set(), {}
        for split, cases in self.cases.items():
            if not cases:
                raise ValueError(f"Empty {split} cohort")
            starts = set()
            starts_by_split[split] = starts
            for case in cases:
                if case["split"] != split or not all(case.get(k) for k in ("id", "task", "goal", "conditions")):
                    raise ValueError("Each case needs a split, ID, task, goal and fixed execution conditions")
                key = (case["task"], digest(case["initial_state"]), digest(case["conditions"]), case["seed"])
                if case["id"] in ids or key in starts:
                    raise ValueError("Repeated case or reset across the configured cohorts")
                ids.add(case["id"])
                starts.add(key)
        train_starts = {(t, i) for t, i, _, _ in starts_by_split["train"]}
        if any((t, i) in train_starts for t, i, _, _ in starts_by_split["development"]):
            raise ValueError("Training and development share a task and initial state")
        demos = config["demonstrations"]
        if any(d.get("split") != "train" or not d.get("task") or not d.get("trajectory") for d in demos):
            raise ValueError("D0 must contain training demonstrations with tasks and trajectory references")
        curate([], {}, previous={"demonstrations": demos, "pairs": []})
        if any(d["task"] not in {c["task"] for c in self.cases["train"]} for d in demos):
            raise ValueError("Every demonstration task needs measured training rollouts")
        self.demos_by_task = {task: [d for d in demos if d["task"] == task]
                              for task in {c["task"] for c in self.cases["train"]}}
        if any(not any(d["goal"] == c["goal"] for d in self.demos_by_task[c["task"]])
               for c in self.cases["train"]):
            raise ValueError("Each training goal needs a matching D0 demonstration for Code+ICL")
        for stage in ("rollout", "repair", "integrate", "train", "evaluate"):
            command = self.commands[stage]
            if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
                raise ValueError(f"Expected an argv array for {stage}")

    def call(self, stage, **payload):
        payload = dict(stage=stage, round=self.round, **payload)
        job = self.directory / stage / digest(payload)
        receipt = job / "completed.json"
        if receipt.exists():
            return load(receipt)
        job.mkdir(parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=job))
        request, response = attempt / "request.json", attempt / "response.json"
        save(request, {**payload, "workdir": str(attempt)})
        command = list(self.commands[stage])
        if command[0] == "{python}":
            command[0] = sys.executable
        with (attempt / "worker.log").open("w") as log:
            subprocess.run([*command, str(request), str(response)], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        result = load(response)
        if not isinstance(result, dict):
            raise ValueError(f"Invalid {stage} response in {attempt}")
        if stage in ("rollout", "evaluate", "repair") and type(result.get("success")) is not bool:
            raise RuntimeError(f"Unjudged native outcome in {attempt}")
        save(receipt, result)
        return result

    def parallel(self, fn, items):
        with ThreadPoolExecutor(max_workers=self.config["workers"]) as workers:
            return list(workers.map(fn, items))

    def evaluation(self, policy, harness, cases=None):
        def measure(case):
            row = self.call("evaluate", case=case, policy=policy, harness=harness,
                            mode="hybrid" if harness else "standalone")
            if (row["case_id"], row["policy"], row["harness"]) != (case["id"], policy, harness):
                raise ValueError("Development measurement does not match the requested execution")
            return row["success"]
        return sum(self.parallel(measure, self.cases["development"] if cases is None else cases))

    def advance(self, state):
        self.round = state["round"] + 1
        self.directory = self.root / f"round-{self.round:04}"
        final = self.directory / "selected.json"
        if final.exists():
            return load(final)
        policy, harness = state["policy"], state["harness"]
        def rollout(case):
            row = self.call("rollout", case=case, policy=policy, harness=None, mode="standalone")
            if (row["case_id"], row["task"], row["goal"], row["split"]) != (
                    case["id"], case["task"], case["goal"], "train"):
                raise ValueError("Rollout receipt belongs to a different training case")
            return row

        cases = self.cases["train"]
        outcomes = self.parallel(rollout, cases)
        if len({r["id"] for r in outcomes}) != len(outcomes):
            raise ValueError("Duplicate rollout identity")
        rates = {}
        for task in {row["task"] for row in outcomes}:
            group = [row["success"] for row in outcomes if row["task"] == task]
            rates[task] = sum(group) / len(group)
        failures = [(case, row) for case, row in zip(cases, outcomes) if not row["success"]]

        def repair(job):
            case, failure, route = job
            rng = random.Random(digest([self.config["seed"], self.round, case["id"]]))
            matches = [d for d in self.demos_by_task[case["task"]] if d["goal"] == case["goal"]]
            demo = rng.choice(matches) if route == "code_icl" else None
            history = [e for e in state.get("experiences", []) if e["task"] == case["task"]][-6:]
            row = self.call("repair", case=case, failure=failure, harness=harness, route=route,
                            policy=policy if route == "code_policy" else None, demo=demo,
                            prior_experiences=history)
            if row["failure_id"] != failure["id"] or row["route"] != route:
                raise ValueError("Route response belongs to a different branch")
            return row

        attempts = self.parallel(repair, [(c, f, route) for c, f in failures for route in ROUTES])
        winners = select_repairs(outcomes, attempts)
        pool = curate(outcomes, winners, previous=state["pool"])
        save(self.directory / "pool.json", pool)
        candidate = harness
        if failures:
            merge = self.call("integrate", harness=harness, selected=winners, attempts=attempts,
                              failures=[row for _, row in failures],
                              regression_cases=[case for case, row in zip(cases, outcomes) if row["success"]])
            if merge.get("parent") != harness or type(merge.get("compatible")) is not bool:
                raise ValueError("Integration receipt needs the frozen parent and compatibility verdict")
            if merge["compatible"]:
                proposed = merge["harness"]
                if not isinstance(proposed, str) or not proposed:
                    raise ValueError("Missing integrated harness reference")
                targets = [case for case, row in zip(cases, outcomes)
                           if row["success"] or row["id"] in winners]
                if proposed != harness and targets and self.evaluation(policy, proposed, targets) == len(targets):
                    candidate = proposed
        weights = {name: task_sampling(rows, rates, self.config["gamma"]) if rows else {}
                   for name, rows in pool.items()}
        update = self.call("train", policy=policy, reference=policy, pool=pool, task_probabilities=weights,
                           cpd_enabled=bool(pool["pairs"]), ewc=True,
                           fisher_demonstrations=state["pool"]["demonstrations"])
        candidate_policy = update["policy"]
        if update.get("reference") != policy or not isinstance(candidate_policy, str) or not candidate_policy:
            raise ValueError("Training receipt needs a candidate and its round-start reference")
        old_score = self.evaluation(policy, None)
        new_score = self.evaluation(candidate_policy, None) if candidate_policy != policy else old_score
        selected_policy = candidate_policy if new_score > old_score else policy
        old_hybrid = self.evaluation(selected_policy, harness)
        new_hybrid = self.evaluation(selected_policy, candidate) if candidate != harness else old_hybrid
        selected_harness = candidate if new_hybrid > old_hybrid else harness
        public_lessons = load(selected_harness)["memory"].get("general", [])
        experiences = list(state.get("experiences", []))
        for row in attempts:
            entry = {"task": row["task"], "route": row["route"]}
            if row.get("guidance") and row["guidance"] in public_lessons:
                entry["guidance"] = row["guidance"]
            if entry in experiences:
                experiences.remove(entry)
            experiences.append(entry)
        selected = dict(round=self.round, policy=selected_policy, pool=pool,
                        experiences=experiences, harness=selected_harness,
                        development=dict(count=len(self.cases["development"]),
                                         old_standalone=old_score, candidate_standalone=new_score,
                                         old_hybrid=old_hybrid, candidate_hybrid=new_hybrid))
        save(final, selected)
        return selected

    def run(self, rounds):
        if rounds < 1:
            raise ValueError("Rounds must be positive")
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = {"config": self.config, "initial_harness": digest(load(INITIAL))}
            manifest = self.root / "campaign.json"
            if manifest.exists() and load(manifest) != identity:
                raise ValueError("Campaign configuration or seed harness changed")
            save(manifest, identity)
            state = dict(round=0, policy=self.config["initial_policy"], harness=str(INITIAL.resolve()),
                         pool={"demonstrations": self.config["demonstrations"], "pairs": []})
            for _ in range(rounds):
                state = self.advance(state)
            return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--check", action="store_true", help="Validate configuration on CPU")
    args = parser.parse_args()
    runner = Campaign(load(args.config), args.workspace)
    if args.check:
        print("Configuration validated")
    else:
        state = runner.run(args.rounds)
        print(json.dumps({k: v for k, v in state.items() if k not in {"pool", "experiences"}}, indent=2))


if __name__ == "__main__":
    main()
