"""Training-side evidence selection for a RoboEvo² round.

The native environment adapter owns the verdicts, capture, replay and evidence
references. Execution IDs are immutable. A successful row carries the original
goal and a trajectory reference to synchronized observations and applied actions.
State and observation IDs at the start of a continuation refer to its measured
state and policy input. The conditions ID binds the reset/scene, verifier,
remaining execution budget and control protocol under which branches are compared.
Scoring and held-out trajectories must use separate splits.

A failed policy episode can supply verified intermediate anchors. Each anchor
identifies its actual rejected action chunk at the recorded fork. Native adapters
compute action_sha256 on the applied actions (see action_digest). actions points
to that chunk, not the entire file containing it. This allows two different
file layouts to represent the same action target.
"""
from collections import defaultdict
from hashlib import sha256
import math
import re


ROUTES = ("code", "code_icl", "code_policy")
BOUNDARY = ("start_state", "observation", "conditions")
CONTROL = ("control_interface", "control_hz", "action_horizon")


def action_digest(actions):
    """Content identity of a finite [ticks, control dimensions] action target."""
    import numpy as np

    block = np.ascontiguousarray(actions, dtype="<f4")
    if block.ndim != 2 or not block.size or not np.isfinite(block).all():
        raise ValueError("Expected a finite, nonempty action matrix")
    block = np.where(block == 0, np.float32(0), block).astype("<f4", copy=False)
    header = f"float32-le:{block.shape[0]}x{block.shape[1]}\n".encode()
    return sha256(header + block.tobytes()).hexdigest()


def normalized_jerk(timestamps, positions, task_scale, grid_dt, window=7):
    """Dimensionless translational jerk measured on a uniform physical-time grid.

    positions has shape [time, active arm, xyz], or [time, xyz] for one arm.
    All routes use the same physical sampling interval, fixed task length scale
    and local cubic fit window. Each fit uses the nearest contiguous samples.
    """
    import numpy as np

    t = np.asarray(timestamps, dtype=np.float64)
    xyz = np.asarray(positions, dtype=np.float64)
    if xyz.ndim == 2:
        xyz = xyz[:, None, :]
    if (t.ndim != 1 or xyz.ndim != 3 or xyz.shape[0] != len(t)
            or not xyz.shape[1] or xyz.shape[-1] != 3 or not np.isfinite(xyz).all()
            or not np.isfinite(t).all() or not np.all(np.diff(t) > 0)):
        raise ValueError("Invalid measured arm positions or timestamps")
    if type(window) is not int or not 4 <= window <= len(t):
        raise ValueError("Insufficient measured positions for the cubic fit window")
    if not all(math.isfinite(x) and x > 0 for x in (grid_dt, task_scale)):
        raise ValueError("The time grid interval and task length scale must be positive")
    t = t - t[0]
    duration = t[-1]
    grid = np.linspace(0, duration, max(1, math.ceil(duration / grid_dt)) + 1)
    data = xyz.reshape(len(t), -1)
    jerk2 = []
    for point in grid:
        left = np.searchsorted((t[:-window] + t[window:]) * 0.5, point)
        selection = slice(left, left + window)
        u = (t[selection] - point) / grid_dt
        coefficients = np.linalg.lstsq(
            np.vander(u, N=4, increasing=True), data[selection], rcond=None
        )[0]
        jerk = 6.0 * coefficients[3] / grid_dt ** 3
        jerk2.append(float(np.dot(jerk, jerk)))
    energy = np.sum(np.diff(grid) * (np.array(jerk2[1:]) + jerk2[:-1]) * 0.5)
    score = float(duration ** 5 * energy / (xyz.shape[1] * task_scale ** 2))
    if not math.isfinite(score):
        raise ValueError("Motion jerk is not finite")
    return score


def _required_strings(row, names):
    if any(not isinstance(row.get(name), str) or not row[name].strip() for name in names):
        raise ValueError(f"Missing execution evidence ({', '.join(names)})")


def _validate(row):
    if row.get("source") == "demonstration":
        raise ValueError("D0 records enter only through the initial or previous data pool")
    _required_strings(row, ("id", "task", "goal"))
    if row.get("split") != "train":
        raise ValueError("Round updates require training-side native evidence")
    if "success" not in row or (row["success"] is not None and type(row["success"]) is not bool):
        raise ValueError("The native success verdict must be true, false or unknown")
    if row["success"] is not None:
        _required_strings(row, (*BOUNDARY, "trajectory"))


def _identity_map(rows, *, allow_demos=False):
    result = {}
    for row in rows:
        if allow_demos and row.get("source") == "demonstration":
            _required_strings(row, ("id", "task", "goal", "trajectory"))
            if row.get("split") != "train":
                raise ValueError("D0 demonstrations require a training-side source")
        else:
            _validate(row)
        if row["id"] in result and result[row["id"]] != row:
            raise ValueError(f"Conflicting evidence for execution {row['id']}")
        result[row["id"]] = row
    return result


def select_repairs(policy_rows, route_rows):
    """Choose the lowest-jerk verified native success per failure (paper Eq. 1).

    All successful alternatives for a failure must use a common comparison
    boundary. A failure without a success has no winner; keep its raw attempts.
    Native successes with insufficient measured motion remain in the raw attempt
    ledger. Explicitly flagged unrankable routes cannot win without a score.
    Exact ties break by route order for deterministic publication.
    """
    policies = _identity_map(policy_rows)
    candidates = defaultdict(dict)
    for attempt in _identity_map(route_rows).values():
        route = attempt.get("route")
        failed = policies.get(attempt.get("failure_id"))
        if (failed is None or failed["success"] is not False or route not in ROUTES
                or failed["task"] != attempt["task"] or failed["goal"] != attempt["goal"]
                or attempt["id"] in policies):
            raise ValueError("Repair must identify a native policy failure on the same task and goal")
        if attempt["success"] is not True:
            continue
        _required_strings(attempt, ("program",))
        if attempt.get("rank_eligible") is False:
            _required_strings(attempt, ("rank_reason",))
            continue
        jerk = attempt.get("jerk")
        if type(jerk) not in (int, float) or not math.isfinite(jerk) or jerk < 0:
            raise ValueError("A successful repair needs a measured, nonnegative jerk")
        options = candidates[failed["id"]]
        if route in options:
            raise ValueError("Each route submits at most one verified successful candidate")
        if options:
            other = next(iter(options.values()))
            if any(other[key] != attempt[key] for key in BOUNDARY):
                raise ValueError("Cannot rank repairs with different starting conditions")
        options[route] = attempt
    return {failure: min(options.values(), key=lambda row: (row["jerk"], ROUTES.index(row["route"])))
            for failure, options in candidates.items()}


def _chunk(row):
    digest = row.get("action_sha256")
    return (isinstance(digest, str) and bool(re.fullmatch(r"[0-9a-f]{64}", digest))
            and isinstance(row.get("actions"), str) and bool(row["actions"].strip())
            and isinstance(row.get("control_interface"), str) and bool(row["control_interface"].strip())
            and type(row.get("control_hz")) in (int, float) and math.isfinite(row["control_hz"])
            and row["control_hz"] > 0 and type(row.get("action_horizon")) is int
            and row["action_horizon"] > 0)


def _pair(failed, correct):
    if not _chunk(correct):
        return None
    anchors = [failed, *(row for row in failed.get("anchors", []) if row.get("verified") is True)]
    for anchor in anchors:
        if (not _chunk(anchor) or any(anchor.get(k) != correct[k] for k in (*BOUNDARY, *CONTROL))
                or correct["action_sha256"] == anchor["action_sha256"]):
            continue
        return {**{k: correct[k] for k in ("task", "goal", *BOUNDARY, *CONTROL)},
                "positive": correct["actions"], "positive_sha256": correct["action_sha256"],
                "negative": anchor["actions"], "negative_sha256": anchor["action_sha256"],
                "positive_id": correct["id"], "negative_id": failed["id"],
                "positive_trajectory": correct["trajectory"]}
    return None


def curate(policy_rows, winners, previous=None):
    """Accumulate native successes and verified, same-boundary preference pairs.

    Supply an initial data bank or an earlier curate() result as previous.
    D0 demonstrations in that trusted bank carry id, task, goal, split='train',
    source='demonstration' and trajectory. A native verdict is optional for D0. Repeated
    input rows are idempotent; contrary claims for an existing ID are errors.
    """
    policies = _identity_map(policy_rows)
    accepted = select_repairs(policies.values(), winners.values())
    if accepted != winners:
        raise ValueError("Winner keys must match their original failed policy execution")
    previous = previous or {"demonstrations": [], "pairs": []}
    demonstrations = _identity_map(previous["demonstrations"], allow_demos=True)
    if any(r.get("success") is not True and r.get("source") != "demonstration"
           for r in demonstrations.values()):
        raise ValueError("Replay demonstrations require native success")
    positive = [r for r in policies.values() if r["success"] is True and r.get("steps") != 0]
    positive += list(accepted.values())
    demonstrations = _identity_map([*demonstrations.values(), *positive], allow_demos=True)
    pairs = {}
    fresh_pairs = [_pair(policies[k], value) for k, value in accepted.items()]
    for pair in [*previous["pairs"], *(p for p in fresh_pairs if p is not None)]:
        key = (pair["positive_id"], pair["negative_id"], pair["start_state"], pair["observation"])
        positive_row = demonstrations.get(pair["positive_id"])
        if (positive_row is None or positive_row.get("source") == "demonstration"
                or (key in pairs and pairs[key] != pair)):
            raise ValueError("A replay pair must identify one consistent admitted correction")
        pairs[key] = pair
    return {"demonstrations": list(demonstrations.values()), "pairs": list(pairs.values())}


def task_sampling(pool_rows, policy_success_rates, gamma=1.0):
    """Equation 4: independent half-uniform, half-failure distributions per pool."""
    if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Failure-weight exponent gamma must be finite and positive")
    tasks = sorted({r["task"] for r in pool_rows})
    if not tasks:
        return {}
    weights = {}
    for task in tasks:
        rate = policy_success_rates.get(task)
        if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError(f"Expected a measured success rate in [0, 1] for {task}")
        weights[task] = (1.0 - rate) ** gamma
    total = sum(weights.values())
    if total == 0:
        return dict.fromkeys(tasks, 1.0 / len(tasks))
    return {task: 0.5 / len(tasks) + 0.5 * weight / total for task, weight in weights.items()}


def sample_batch(pool_rows, distribution, batch_size, rng):
    """Sample a task using Eq. 4, then a record uniformly within that task."""
    by_task = defaultdict(list)
    for row in pool_rows:
        by_task[row["task"]].append(row)
    if type(batch_size) is not int or batch_size < 0:
        raise ValueError("Batch size must be a nonnegative integer")
    if set(distribution) != set(by_task):
        raise ValueError("Sampling distribution must cover exactly the tasks present")
    if batch_size == 0:
        return []
    tasks = sorted(by_task)
    weights = [distribution[t] for t in tasks]
    if not tasks or not all(math.isfinite(w) and w >= 0 for w in weights) or not math.isclose(sum(weights), 1):
        raise ValueError("Expected a normalized task distribution")
    sampled = rng.choices(tasks, weights=weights, k=batch_size)
    return [rng.choice(by_task[t]) for t in sampled]
