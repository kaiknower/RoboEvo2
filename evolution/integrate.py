"""Build a candidate skill, memory and routing snapshot from winning branch edits."""
import argparse
import copy
import json
from pathlib import Path
import re
import unicodedata

from .initial.skills import PRIMITIVES, _argument
from .run import load, save


_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
_COORDINATES = re.compile(
    rf"{_NUMBER}\s*[,;]\s*{_NUMBER}|[\[\(]\s*{_NUMBER}\s+{_NUMBER}(?:\s+{_NUMBER})?\s*[\]\)]|"
    rf"\b(?:x|y|z|xyz|xy|uv|row|column|pixel|world\s+(?:position|coordinate))"
    rf"\s*(?:=|:|at|is)\s*{_NUMBER}|\bpixel\s*\(?{_NUMBER}\s*(?:x|by|\s)\s*{_NUMBER}", re.I)
_PATH = re.compile(
    r"[/\\][\w.~-]|[a-z][a-z0-9+.-]*://|"
    r"\b[\w.-]+\.(?:jsonl?|ya?ml|csv|npy|npz|hdf5?|h5|sqlite3?|png|jpe?g|mp4|log|py|pkl|safetensors)\b", re.I)
_CASE_ID = re.compile(
    r"(?:\b|_)(?:episode|instance|seed|case|task|test|reset)(?:[\s_:#=.-]*\d+|"
    r"[\s_:-]+(?:id|index|identifier|number)[\s_:#=.-]+[a-z0-9]+|"
    r"[_-][a-f0-9]{6,})(?:\b|_)|(?:\b|_)(?:episode|seed|instance)_[a-z0-9]+", re.I)
_PRIVATE = re.compile(
    r"\b(?:ground truth|(?:simulator|simulation|internal|native|oracle|private) "
    r"(?:state|verdict|flag|predicate|success|done|contact|pose|reward)|"
    r"(?:is|was|check|get|has) (?:grasped|attached|done|success|contact)|"
    r"(?:contact|grasp|attachment|success|done|reward|termination) (?:flag|label|predicate|state|signal|bit)|"
    r"(?:read|query|get|use|access) (?:the )?(?:true |world )?(?:object pose|door joint state|attachment state)|"
    r"qpos|qvel|mjdata|simstate)\b")
_PRIVATE_FIELD = re.compile(r"\b(?:object_pose|object_world_position|door_joint_state|is_grasped|is_attached|"
                           r"contact_state|native_success|success_flag|grasp_flag)\b|"
                           r"(?:环境|仿真器|模拟器)真值|(?:读取|查询)(?:内部)?(?:接触标记|抓取标志|原生成功标记)", re.I)
_PRIVATE_NAME = re.compile(r"(?:^|_)(?:success|verdict|done|reward|grasped|attached|object_pose|"
                           r"(?:task|episode|case|seed|init|instance)_id)(?:$|_)")
_ARGUMENT_TYPES = {kind for params in PRIMITIVES.values() for kind in params.values()}
_FIELDS = {"scope", "use_when", "completion", "arguments", "procedure"}


def _generic_text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 400 or not value.isprintable():
        raise ValueError("Persistent prose must be a short readable description")
    text = unicodedata.normalize("NFKC", value)
    if _COORDINATES.search(text) or _PATH.search(text) or _CASE_ID.search(text):
        raise ValueError("Keep concrete coordinates, evidence paths and episode IDs out of persistent text")
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).lower()
    words = re.sub(r"[\W_]+", " ", words)
    if _PRIVATE.search(words) or _PRIVATE_FIELD.search(text):
        raise ValueError("Persistent text must describe publicly observable conditions")


def validate_name(name):
    if not isinstance(name, str) or not _NAME.fullmatch(name) or _PRIVATE_NAME.search(name):
        raise ValueError("Use a generic lower-case skill or argument name")
    _generic_text(name)


def validate_guidance(lesson):
    if not isinstance(lesson, dict) or set(lesson) != {"condition", "instruction", "skill"}:
        raise ValueError("Guidance needs an observable condition, instruction and named skill")
    validate_name(lesson["skill"])
    _generic_text(lesson["condition"])
    _generic_text(lesson["instruction"])


def validate_skill(spec, *, stored=True):
    allowed = _FIELDS | ({"dependencies", "revision"} if stored else set())
    if not isinstance(spec, dict) or not _FIELDS <= spec.keys() <= allowed:
        raise ValueError("Unexpected skill definition fields")
    for field in ("scope", "use_when", "completion"):
        _generic_text(spec[field])
    if "revision" in spec and (type(spec["revision"]) is not int or spec["revision"] < 0):
        raise ValueError("Invalid stored skill revision")
    args, steps = spec["arguments"], spec["procedure"]
    if not isinstance(args, dict) or not isinstance(steps, list) or not steps:
        raise ValueError("Skill edits require a typed procedure")
    for name, arg in args.items():
        validate_name(name)
        if not isinstance(arg, dict) or not {"type"} <= arg.keys() <= {"type", "default"}:
            raise ValueError("Unexpected argument fields")
        kind = arg["type"]
        if not isinstance(kind, str) or kind not in _ARGUMENT_TYPES:
            raise ValueError("Unsupported public argument type")
        if "default" in arg:
            if kind in {"pixel", "camera", "instruction"}:
                raise ValueError("Ground visual and language arguments at invocation")
            _argument(kind, arg["default"])

    used = set()
    moved = False
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("op"), str):
            raise ValueError("Expected a public control primitive")
        op = step["op"]
        if op not in PRIMITIVES or set(step) != {"op", *PRIMITIVES[op]}:
            raise ValueError("Unknown or unbounded public primitive")
        if op == "pixel_target" and moved:
            raise ValueError("Pixel targets must be grounded from the live pre-motion view")
        moved = moved or op != "observe"
        for key, kind in PRIMITIVES[op].items():
            value = step[key]
            if isinstance(value, str) and value.startswith("$"):
                name = value[1:]
                if name not in args or args[name]["type"] != kind:
                    raise ValueError("Skill argument and primitive types differ")
                used.add(name)
            else:
                if kind in {"pixel", "camera", "instruction"}:
                    raise ValueError("Bind visible targets and policy instructions at invocation")
                _argument(kind, value)
    if used != args.keys():
        raise ValueError("Each declared argument must be consumed by a public primitive")
    if "dependencies" in spec:
        deps = spec["dependencies"]
        if (not isinstance(deps, list) or not all(isinstance(k, str) for k in deps)
                or set(deps) != {step["op"] for step in steps}):
            raise ValueError("Dependencies must match the public primitives in the procedure")


def merge(request):
    parent = request["harness"]
    state = copy.deepcopy(load(parent))
    proposals = {}
    conflicts = set()
    guidance = []
    for winner in request["selected"].values():
        patch = winner.get("skill_edits", {})
        for name, spec in patch.items():
            validate_name(name)
            validate_skill(spec, stored=False)
            if name in proposals and proposals[name] != spec:
                conflicts.add(name)
            else:
                proposals[name] = spec
        lesson = winner.get("guidance")
        if lesson:
            validate_guidance(lesson)
            guidance.append(lesson)

    if not proposals and not guidance:
        return {"parent": parent, "compatible": True, "harness": parent, "deferred_conflicts": []}
    for name in sorted(proposals.keys() - conflicts):
        state["skills"][name] = {**proposals[name],
                                 "revision": state["skills"].get(name, {}).get("revision", -1) + 1}
    lessons = state["memory"].setdefault("general", [])
    routing = state["routing"].setdefault("guidance", [])
    for lesson in guidance:
        name = lesson["skill"]
        if name in state["skills"] and name not in conflicts and lesson not in lessons:
            lessons.append(lesson)
            rule = {"condition": lesson["condition"], "skill": name}
            if rule not in routing:
                routing.append(rule)
    state["round"] = request["round"]
    target = Path(request["workdir"]) / "harness.json"
    save(target, state)
    return {"parent": parent, "compatible": True, "harness": str(target),
            "deferred_conflicts": sorted(conflicts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    save(args.response, merge(load(args.request)))
