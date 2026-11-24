"""Compile seed skills to public robot calls. The simulator owns termination."""
from __future__ import annotations

import json
from pathlib import Path


def initial_state():
    """Return an independent H0 for a new run."""
    return json.loads(Path(__file__).with_name("state.json").read_text(encoding="utf-8"))


def _argument(kind, value):
    if kind == "camera":
        valid = isinstance(value, str) and value in {"agentview", "robot0_eye_in_hand", "sideview"}
    elif kind == "pixel":
        valid = (isinstance(value, (list, tuple)) and len(value) == 2
                 and all(type(v) is int and v >= 0 for v in value))
    elif kind in {"offset_m", "rotation_rad"}:
        bound = 0.15 if kind == "offset_m" else 0.5
        valid = (isinstance(value, (list, tuple)) and len(value) == 3 and
                 all(type(v) in (int, float) and -bound <= v <= bound for v in value))
    elif kind == "steps":
        valid = type(value) is int and 1 <= value <= 80
    elif kind == "gripper":
        valid = type(value) in (int, float) and -1 <= value <= 1
    elif kind == "instruction":
        valid = isinstance(value, str) and 0 < len(value.strip()) <= 400 and value.isprintable()
    else:
        raise ValueError(f"Unknown skill argument type {kind}")
    if not valid:
        raise ValueError(f"Invalid {kind} argument")
    return list(value) if isinstance(value, tuple) else value


def compile_skill(state, route, name, arguments):
    """Resolve typed arguments to a bounded program; no Python source is evaluated."""
    if route not in state["routing"]["routes"]:
        raise ValueError("Unknown solution route")
    spec = state["skills"][name]
    if not isinstance(arguments, dict) or arguments.keys() - spec["arguments"].keys():
        raise ValueError("Unexpected skill arguments")
    bound = {}
    for key, parameter in spec["arguments"].items():
        if key not in arguments and "default" not in parameter:
            raise ValueError(f"Missing skill argument {key}")
        bound[key] = _argument(parameter["type"], arguments.get(key, parameter.get("default")))
    operations = []
    moved = False
    for step in spec["procedure"]:
        operation = step["op"]
        if operation not in PRIMITIVES or (operation == "vla_act" and route != "code_policy"):
            raise ValueError(f"{operation} is unavailable to route {route}")
        if operation == "pixel_target" and moved:
            raise ValueError("Motion requires a newly grounded pixel")
        moved = moved or operation != "observe"
        expected = PRIMITIVES[operation]
        if set(step) - {"op"} != expected.keys():
            raise ValueError(f"Invalid parameters for primitive {operation}")
        resolved = {"op": operation}
        for key, kind in expected.items():
            value = step[key]
            if isinstance(value, str) and value.startswith("$"):
                value = bound[value[1:]]
            resolved[key] = _argument(kind, value)
        operations.append(resolved)
    return operations


PRIMITIVES = {
    "observe": {},
    "pixel_target": {"camera": "camera", "pixel": "pixel", "offset": "offset_m",
                     "steps": "steps", "gripper": "gripper"},
    "move_delta": {"delta": "offset_m", "rotation": "rotation_rad", "steps": "steps", "gripper": "gripper"},
    "gripper": {"value": "gripper", "steps": "steps"},
    "vla_act": {"prompt": "instruction", "steps": "steps"},
}


def execute_skill(controller, state, route, name, arguments):
    """Execute on a trusted public controller and return its last public view.

    Pixel anchors must come from the current view. Each seed motion that can
    change that view ends with a new observation. The external evaluator may
    terminate execution on any control step.
    """
    operations = compile_skill(state, route, name, arguments)
    if sum(op.get("steps", 0) for op in operations) > controller.get_budget()["remaining_steps"]:
        raise ValueError("Skill exceeds remaining action budget")
    observed = None
    for step in operations:
        operation = step["op"]
        if operation == "observe":
            observed = controller.observe()
        else:
            getattr(controller, operation)(**{k: v for k, v in step.items() if k != "op"})
    return observed
