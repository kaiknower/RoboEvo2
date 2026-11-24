"""Bounded observation-only recovery plans; no simulator handle or native score.

The evaluator may raise its private BaseException to stop an action chunk.
BaseException is deliberately not caught: the private signal never becomes actor
feedback. A public journal is flushed before each primitive so interrupted
execution retains the proposed plan without revealing why the evaluator stopped.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .coding import CAMERAS, CodingPlanner, public_observation
from .controller import (CavityRouteAlignmentError, CAVITY_MAX_HEIGHT_DIFFERENCE_M,
                         CAVITY_MIN_HORIZONTAL_DISTANCE_M, validate_local_policy_prompt)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "libero.public_plan.v2"
LEGACY_SCHEMA = "libero.public_plan.v1"
PHASE_RECIPES = frozenset({"bound_transfer", "bound_drawer_pull", "bound_surface_stroke", "bound_cavity_place"})
BOUND_RECIPES = PHASE_RECIPES | {"bound_grasp", "bound_place"}
LEGACY_POSITION_GUARDED_RECIPES = frozenset({"grasp_observed_target", "place_held_object"})
BOUND_RECIPE_POSITION_ERROR_LIMIT_M = .04
BOUND_RECIPE_REPLAN_MESSAGE = (
    "Public end-effector feedback does not confirm this visual waypoint within "
    "the 0.04 m gross position-error limit. Remaining plan actions were skipped "
    "with the current gripper command preserved; use fresh images to replan. "
    "This check does not establish a grasp or placement.")
FORBIDDEN = re.compile(
    r"(?:\b(?:import|exec|eval|open|subprocess|os|sys)\s*\(|\bimport\s+[A-Za-z_]\w*|"
    r"\bfrom\s+[A-Za-z_][\w.]*\s+import\b|"
    r"env\s*\.|sim\s*\.|__|\b(?:task_id|taskid|seed|init_id|init_state|"
    r"world_xyz|target_xyz|goal_xyz|body_xpos|geom_xpos|object_states_dict|"
    r"parsed_problem|goal_state|reward|oracle|privileged|native_success|"
    r"check_success|attachment|contacts)\b)", re.IGNORECASE)


_VALIDATION_REASONS = {
    "Pixel target requires a fresh visual plan after motion": (
        "pixel_target_after_steps",
        "pixel_target must precede every step-advancing operation, including gripper, "
        "run_policy, move_delta, and another pixel_target. observe within the same plan "
        "does not rebind pixels. Use gripper=-1 inside pixel_target to open while "
        "approaching, or use an approved grasp recipe when its visual preconditions hold. "
        "If a separate opening is needed, end that plan with next=reobserve and choose "
        "the pixel in a new plan from fresh images. Alternatively use the v2 bound_grasp recipe, "
        "which binds the current pixel before opening within the same planning call."),
    "Plan contains unsupported or missing fields": (
        "unsupported_fields", "Use only the required fields of the selected public operation or recipe."),
    "Plan number exceeds its allowed bound": (
        "numeric_bound", "Use finite numbers: delta/offset components within +/-0.15 meters, "
        "rotation within +/-0.5 radians, and gripper within [-1,1]. Booleans are not numbers."),
    "Plan vector must contain three values": (
        "vector_shape", "Provide exactly three numeric values for each delta, rotation, and offset vector."),
    "Primitive steps must be an integer from 1 to 80": (
        "step_bound", "Set each primitive's steps to an integer from 1 to 80."),
    "Unknown public recipe": (
        "unknown_recipe", "Choose a recipe ID from the supplied frozen skills catalog."),
    "Oversized public plan": (
        "plan_size", "Keep the complete JSON plan within 32768 UTF-8 bytes."),
    "Duplicate plan key": (
        "duplicate_key", "Use each JSON object key only once."),
    "Plan contains forbidden content": (
        "unsupported_content", "Return only the public JSON grammar and a brief visual hypothesis. "
        "Do not include executable source, environment access, identifiers, or absolute-coordinate targets."),
    "Unknown public plan schema": (
        "schema", "Set schema to libero.public_plan.v2; v1 primitive plans are also accepted."),
    "Invalid visual hypothesis": (
        "hypothesis", "Use a hypothesis string of at most 2000 characters."),
    "Choose operations or one approved recipe": (
        "plan_form", "Provide either operations or one recipe with arguments, never both."),
    "Arguments require an approved recipe": (
        "recipe_arguments", "Use arguments only with a recipe; direct operations contain their own fields."),
    "A plan requires 1 to 12 bounded operations": (
        "operation_count", "Provide between 1 and 12 operations."),
    "Unknown public continuation": (
        "continuation", "Set next to reobserve or finish."),
    "Operations must be objects": (
        "operation_type", "Each operations entry must be a JSON object with an op field."),
    "Unknown current public camera": (
        "camera", "Choose a camera that is present in the currently supplied public observation."),
    "Pixel is outside the supplied current frame": (
        "pixel_bounds", "Use integer pixel [x,y] coordinates inside the supplied camera's original width and height."),
    "Unknown public operation": (
        "operation", "Use only observe, run_policy, vla_act, move_delta, gripper, pixel_target, or move_bound_target with a current visual target."),
    "Plan exceeds the remaining action budget": (
        "remaining_budget", "Reduce the sum of all operation steps to the supplied remaining_steps or less."),
    "Unknown current visual target": (
        "visual_target", "Define the named visual target from a current camera pixel in this plan's visual_targets."),
    "Visual targets require the v2 schema": (
        "visual_schema", "Use libero.public_plan.v2 for bound visual targets or bound_grasp/bound_place recipes."),
    "Local policy prompt must be nonempty single-line text of at most400 characters": (
        "local_policy_prompt", "Use a short single-line public instruction for one local manipulation attempt; no private identifiers or stop predicates."),
    "Local policy call must end the plan's action sequence": (
        "local_policy_reobserve", "Put any bounded code staging before vla_act. After vla_act only observe is permitted; the next original revision must inspect fresh views before further actions."),
}


def _validation_feedback(error):
    """Map local grammar errors to fixed public text, never echo error content."""
    if isinstance(error, json.JSONDecodeError):
        code, correction = "invalid_json", "Return one valid JSON object without Markdown or surrounding text."
    elif type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str:
        code, correction = _VALIDATION_REASONS.get(error.args[0], (
            "invalid_plan", "Correct the proposal to match the supplied public bounded JSON grammar."))
    else:
        code, correction = "invalid_plan", "Correct the proposal to match the supplied public bounded JSON grammar."
    return {"code": code, "correction": correction}


def _keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or not set(required) <= set(value):
        raise ValueError("Plan contains unsupported or missing fields")


def _number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Plan number exceeds its allowed bound")
    return value


def _vector(value, bound):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Plan vector must contain three values")
    return [_number(x, -bound, bound) for x in value]


def _steps(value):
    if type(value) is not int or not 1 <= value <= 80:
        raise ValueError("Primitive steps must be an integer from 1 to 80")
    return value


def _phase_steps(arguments, defaults):
    supplied = arguments.get("phase_steps", {})
    _keys(supplied, defaults)
    return {phase: _steps(supplied.get(phase, count)) for phase, count in defaults.items()}


def _bound_move(target, offset, steps, gripper, rotation=None):
    return {"op": "move_bound_target", "target": target, "offset": offset,
            "rotation": [0, 0, 0] if rotation is None else rotation,
            "steps": steps, "gripper": gripper}


def _relative_move(delta, steps, gripper, rotation=None):
    return {"op": "move_delta", "delta": delta,
            "rotation": [0, 0, 0] if rotation is None else rotation,
            "steps": steps, "gripper": gripper}


def _phase_recipe(name, arguments):
    """Public phase skeletons. Every phase below passes the ordinary primitive validator."""
    if isinstance(arguments, dict):
        for field in ("rotation", "carry_rotation", "stroke_rotation"):
            if field in arguments:
                _vector(arguments[field], .5)
    if name == "bound_cavity_place":
        _keys(arguments,
              ("entrance_camera", "entrance_pixel", "interior_camera", "interior_pixel",
               "entrance_offset", "interior_offset", "rotation", "phase_steps"),
              ("entrance_camera", "entrance_pixel", "interior_camera", "interior_pixel",
               "entrance_offset", "interior_offset"))
        steps = _phase_steps(arguments, dict(align=40, insert=40, release=12, exit=40))
        return [
            _bound_move("entrance", arguments["entrance_offset"], steps["align"], 1,
                        arguments.get("rotation")),
            _bound_move("interior", arguments["interior_offset"], steps["insert"], 1),
            {"op": "gripper", "value": -1, "steps": steps["release"]},
            _bound_move("entrance", arguments["entrance_offset"], steps["exit"], -1),
            {"op": "observe"},
        ]
    if name == "bound_transfer":
        _keys(arguments,
              ("source_camera", "source_pixel", "support_camera", "support_pixel",
               "contact_offset", "place_offset", "source_approach_offset", "support_approach_offset",
               "lift_delta", "retreat_delta", "rotation", "carry_rotation", "phase_steps"),
              ("source_camera", "source_pixel", "support_camera", "support_pixel",
               "contact_offset", "place_offset"))
        steps = _phase_steps(arguments, dict(open=12, approach=40, descend=40, close=12,
                                            lift=20, carry=40, lower=40, release=12, retreat=20))
        return [
            {"op": "gripper", "value": -1, "steps": steps["open"]},
            _bound_move("source", arguments.get("source_approach_offset", [0, 0, .12]),
                        steps["approach"], -1, arguments.get("rotation")),
            _bound_move("source", arguments["contact_offset"], steps["descend"], -1),
            {"op": "gripper", "value": 1, "steps": steps["close"]},
            _relative_move(arguments.get("lift_delta", [0, 0, .06]), steps["lift"], 1),
            _bound_move("support", arguments.get("support_approach_offset", [0, 0, .12]),
                        steps["carry"], 1, arguments.get("carry_rotation")),
            _bound_move("support", arguments["place_offset"], steps["lower"], 1),
            {"op": "gripper", "value": -1, "steps": steps["release"]},
            _relative_move(arguments.get("retreat_delta", [0, 0, .06]), steps["retreat"], -1),
            {"op": "observe"},
        ]
    common = ("camera", "pixel", "approach_offset", "contact_offset", "stroke",
              "rotation", "stroke_rotation", "retreat_delta", "phase_steps")
    required = ("camera", "pixel", "approach_offset", "contact_offset", "stroke")
    if name == "bound_drawer_pull":
        _keys(arguments, common, required)
        steps = _phase_steps(arguments, dict(open=12, approach=40, contact=40, close=12,
                                            stroke=40, release=12, retreat=20))
        return [
            {"op": "gripper", "value": -1, "steps": steps["open"]},
            _bound_move("anchor", arguments["approach_offset"], steps["approach"], -1,
                        arguments.get("rotation")),
            _bound_move("anchor", arguments["contact_offset"], steps["contact"], -1),
            {"op": "gripper", "value": 1, "steps": steps["close"]},
            _relative_move(arguments["stroke"], steps["stroke"], 1, arguments.get("stroke_rotation")),
            {"op": "gripper", "value": -1, "steps": steps["release"]},
            _relative_move(arguments.get("retreat_delta", [0, 0, .06]), steps["retreat"], -1),
            {"op": "observe"},
        ]
    _keys(arguments, common + ("gripper",), required)
    steps = _phase_steps(arguments, dict(prepare=12, approach=40, contact=40, stroke=40, retreat=20))
    gripper = arguments.get("gripper", 1)
    return [
        {"op": "gripper", "value": gripper, "steps": steps["prepare"]},
        _bound_move("anchor", arguments["approach_offset"], steps["approach"], gripper,
                    arguments.get("rotation")),
        _bound_move("anchor", arguments["contact_offset"], steps["contact"], gripper),
        _relative_move(arguments["stroke"], steps["stroke"], gripper, arguments.get("stroke_rotation")),
        _relative_move(arguments.get("retreat_delta", [0, 0, .06]), steps["retreat"], gripper),
        {"op": "observe"},
    ]


def _recipe(name, arguments):
    if name in PHASE_RECIPES:
        return _phase_recipe(name, arguments)
    if name in {"bound_grasp", "bound_place"}:
        grasping = name == "bound_grasp"
        offset_key = "contact_offset" if grasping else "offset"
        _keys(arguments, ("camera", "pixel", offset_key, "approach_offset", "rotation", "approach_steps"),
              ("camera", "pixel", offset_key))
        approach_steps = _steps(arguments.get("approach_steps", 40))
        prefix = [{"op": "gripper", "value": -1, "steps": 12}] if grasping else []
        return prefix + [
            {"op": "move_bound_target", "target": "anchor", "offset": arguments.get("approach_offset", [0, 0, .12]),
             "rotation": arguments.get("rotation", [0, 0, 0]), "steps": approach_steps, "gripper": -1 if grasping else 1},
            {"op": "move_bound_target", "target": "anchor", "offset": arguments[offset_key],
             "rotation": [0, 0, 0], "steps": 40, "gripper": -1 if grasping else 1},
            {"op": "gripper", "value": 1 if grasping else -1, "steps": 12},
            {"op": "move_delta", "delta": [0, 0, .06], "rotation": [0, 0, 0],
             "steps": 20, "gripper": 1 if grasping else -1}, {"op": "observe"},
        ]
    if name == "inspect":
        _keys(arguments, ())
        return [{"op": "observe"}]
    if name == "policy_continuation":
        _keys(arguments, ("steps",), ("steps",))
        return [{"op": "run_policy", "steps": arguments["steps"]}, {"op": "observe"}]
    if name == "relative_clearance":
        _keys(arguments, ("delta", "rotation", "steps", "gripper"), ("delta", "rotation", "steps", "gripper"))
        return [dict(arguments, op="move_delta"), {"op": "observe"}]
    if name in {"grasp_observed_target", "place_held_object"}:
        offset_key = "contact_offset" if name == "grasp_observed_target" else "offset"
        _keys(arguments, ("camera", "pixel", offset_key), ("camera", "pixel", offset_key))
        grasping = name == "grasp_observed_target"
        return [
            {"op": "pixel_target", "camera": arguments["camera"], "pixel": arguments["pixel"],
             "offset": arguments[offset_key], "steps": 40, "gripper": -1 if grasping else 1},
            {"op": "gripper", "value": 1 if grasping else -1, "steps": 12},
            {"op": "move_delta", "delta": [0, 0, 0.06], "rotation": [0, 0, 0],
             "steps": 20, "gripper": 1 if grasping else -1}, {"op": "observe"},
        ]
    raise ValueError("Unknown public recipe")


def validate_plan(raw, observation, *, remaining_steps=400):
    """Parse a closed JSON grammar; never evaluate or import generated content."""
    if isinstance(raw, str):
        if len(raw.encode()) > 32768:
            raise ValueError("Oversized public plan")
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate plan key")
                result[key] = value
            return result
        raw = json.loads(raw, object_pairs_hook=unique_pairs)
    encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode()) > 32768 or FORBIDDEN.search(encoded):
        raise ValueError("Plan contains forbidden content")
    _keys(raw, ("schema", "hypothesis", "operations", "recipe", "arguments", "next", "visual_targets"), ("schema",))
    if raw["schema"] not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError("Unknown public plan schema")
    hypothesis = raw.get("hypothesis", "")
    if not isinstance(hypothesis, str) or len(hypothesis) > 2000:
        raise ValueError("Invalid visual hypothesis")
    if ("recipe" in raw) == ("operations" in raw):
        raise ValueError("Choose operations or one approved recipe")
    if "arguments" in raw and "recipe" not in raw:
        raise ValueError("Arguments require an approved recipe")
    operations = _recipe(raw["recipe"], raw.get("arguments", {})) if "recipe" in raw else raw["operations"]
    definitions = raw.get("visual_targets", [])
    if raw.get("recipe") in BOUND_RECIPES:
        if definitions:
            raise ValueError("Bound recipes define their own visual target")
        arguments = raw["arguments"]
        if raw["recipe"] == "bound_cavity_place":
            definitions = [{"name": name, "camera": arguments[name + "_camera"],
                            "pixel": arguments[name + "_pixel"]} for name in ("entrance", "interior")]
        elif raw["recipe"] == "bound_transfer":
            definitions = [{"name": name, "camera": arguments[name + "_camera"],
                            "pixel": arguments[name + "_pixel"]} for name in ("source", "support")]
        else:
            definitions = [{"name": "anchor", "camera": arguments["camera"], "pixel": arguments["pixel"]}]
    if definitions and raw["schema"] != SCHEMA:
        raise ValueError("Visual targets require the v2 schema")
    if not isinstance(definitions, list) or len(definitions) > 4:
        raise ValueError("At most four current visual targets are allowed")
    targets, names = [], set()
    for definition in definitions:
        _keys(definition, ("name", "camera", "pixel"), ("name", "camera", "pixel"))
        name, camera, pixel = definition["name"], definition["camera"], definition["pixel"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) or name in names:
            raise ValueError("Visual target names must be unique short identifiers")
        if camera not in CAMERAS or camera not in observation:
            raise ValueError("Unknown current public camera")
        height, width = observation[camera]["images"]["rgb"].shape[:2]
        if (not isinstance(pixel, list) or len(pixel) != 2 or any(type(x) is not int for x in pixel)
                or not 0 <= pixel[0] < width or not 0 <= pixel[1] < height):
            raise ValueError("Pixel is outside the supplied current frame")
        names.add(name)
        targets.append(dict(name=name, camera=camera, pixel=list(pixel)))
    if not isinstance(operations, list) or not 1 <= len(operations) <= 12:
        raise ValueError("A plan requires 1 to 12 bounded operations")
    next_action = raw.get("next", "reobserve")
    if next_action not in ("reobserve", "finish"):
        raise ValueError("Unknown public continuation")
    validated, total = [], 0
    pixel_used = False
    moved = False
    for supplied in operations:
        if not isinstance(supplied, dict):
            raise ValueError("Operations must be objects")
        op = supplied.get("op")
        if op == "observe":
            _keys(supplied, ("op",), ("op",))
            validated.append({"op": op})
            continue
        if op == "run_policy":
            _keys(supplied, ("op", "steps"), ("op", "steps"))
            item = {"op": op, "steps": _steps(supplied["steps"])}
        elif op == "vla_act":
            _keys(supplied, ("op", "prompt", "steps"), ("op", "prompt", "steps"))
            item = {"op": op, "prompt": validate_local_policy_prompt(supplied["prompt"]),
                    "steps": _steps(supplied["steps"])}
        elif op == "gripper":
            _keys(supplied, ("op", "value", "steps"), ("op", "value", "steps"))
            item = {"op": op, "value": _number(supplied["value"], -1, 1), "steps": _steps(supplied["steps"])}
        elif op == "move_delta":
            _keys(supplied, ("op", "delta", "rotation", "steps", "gripper"), ("op", "delta", "rotation", "steps", "gripper"))
            item = {"op": op, "delta": _vector(supplied["delta"], .15),
                    "rotation": _vector(supplied["rotation"], .5), "steps": _steps(supplied["steps"]),
                    "gripper": _number(supplied["gripper"], -1, 1)}
        elif op == "move_bound_target":
            _keys(supplied, ("op", "target", "offset", "rotation", "steps", "gripper"),
                  ("op", "target", "offset", "steps", "gripper"))
            if not isinstance(supplied["target"], str) or supplied["target"] not in names:
                raise ValueError("Unknown current visual target")
            item = {"op": op, "target": supplied["target"], "offset": _vector(supplied["offset"], .15),
                    "rotation": _vector(supplied.get("rotation", [0, 0, 0]), .5),
                    "steps": _steps(supplied["steps"]), "gripper": _number(supplied["gripper"], -1, 1)}
        elif op == "pixel_target":
            _keys(supplied, ("op", "camera", "pixel", "offset", "steps", "gripper"),
                  ("op", "camera", "pixel", "offset", "steps", "gripper"))
            if moved or pixel_used:
                raise ValueError("Pixel target requires a fresh visual plan after motion")
            camera = supplied["camera"]
            if camera not in CAMERAS or camera not in observation:
                raise ValueError("Unknown current public camera")
            pixel = supplied["pixel"]
            height, width = observation[camera]["images"]["rgb"].shape[:2]
            if (not isinstance(pixel, list) or len(pixel) != 2 or any(type(x) is not int for x in pixel)
                    or not 0 <= pixel[0] < width or not 0 <= pixel[1] < height):
                raise ValueError("Pixel is outside the supplied current frame")
            item = {"op": op, "camera": camera, "pixel": pixel,
                    "offset": _vector(supplied["offset"], .15), "steps": _steps(supplied["steps"]),
                    "gripper": _number(supplied["gripper"], -1, 1)}
            pixel_used = True
        else:
            raise ValueError("Unknown public operation")
        total += item["steps"]
        moved = True
        validated.append(item)
    for index, operation in enumerate(validated):
        if operation["op"] == "vla_act" and any(later["op"] != "observe" for later in validated[index+1:]):
            raise ValueError("Local policy call must end the plan's action sequence")
    if total > remaining_steps:
        raise ValueError("Plan exceeds the remaining action budget")
    return {"schema": raw["schema"], "hypothesis": hypothesis, "operations": validated,
            "recipe": raw.get("recipe"), "next": next_action, "planned_steps": total,
            "visual_targets": targets}


def _remaining(controller):
    value = controller.get_budget()
    if isinstance(value, dict):
        value = value.get("remaining_steps")
    if type(value) is not int or value < 0:
        raise ValueError("Controller budget must be remaining steps only")
    return value


def _feedback(value):
    """Never forward an arbitrary controller result, exception or scoring flag."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("steps", "actions_executed"):
        if type(value.get(key)) is int and 0 <= value[key] <= 80:
            result[key] = value[key]
    for key in ("position_error", "rotation_error", "initial_position_error", "initial_rotation_error"):
        number = value.get(key)
        if type(number) in (int, float) and math.isfinite(number) and number >= 0:
            result[key] = number
    reduction = value.get("position_error_reduction")
    if type(reduction) in (int, float) and math.isfinite(reduction):
        result["position_error_reduction"] = reduction
    rotation_reduction = value.get("rotation_error_reduction")
    if type(rotation_reduction) in (int, float) and math.isfinite(rotation_reduction):
        result["rotation_error_reduction"] = rotation_reduction
    if value.get("motion_status") == "stalled":
        result["motion_status"] = "stalled"
        window_steps = value.get("stall_window_steps")
        if type(window_steps) is int and 1 <= window_steps <= 80:
            result["stall_window_steps"] = window_steps
        displacement = value.get("stall_window_displacement")
        if type(displacement) in (int, float) and math.isfinite(displacement) and displacement >= 0:
            result["stall_window_displacement"] = displacement
        window_reduction = value.get("stall_window_error_reduction")
        if type(window_reduction) in (int, float) and math.isfinite(window_reduction):
            result["stall_window_error_reduction"] = window_reduction
    if "gripper_position" in value:
        try:
            position = list(value["gripper_position"])
            if len(position) in (1, 2) and all(math.isfinite(float(x)) for x in position):
                result["gripper_position"] = [float(x) for x in position]
        except (TypeError, ValueError):
            pass
    return result


def _cavity_alignment_feedback(value):
    """Project only finite relative measurements; no coordinates or raw errors."""
    if not isinstance(value, dict):
        return {}
    height = value.get("interior_minus_entrance_height_m")
    horizontal = value.get("horizontal_distance_m")
    if (type(height) not in (int, float) or not math.isfinite(height)
            or type(horizontal) not in (int, float) or not math.isfinite(horizontal)
            or horizontal < 0):
        return {}
    return {"interior_minus_entrance_height_m": height,
            "horizontal_distance_m": horizontal,
            "maximum_absolute_height_difference_m": CAVITY_MAX_HEIGHT_DIFFERENCE_M,
            "minimum_horizontal_distance_m": CAVITY_MIN_HORIZONTAL_DISTANCE_M}


def _execute_plan(controller, plan, observation, entry, check_deadline, flush, *,
                  support_surface_guard=False):
    definitions = plan.get("visual_targets", [])
    bound_moves = 0
    if plan.get("recipe") in PHASE_RECIPES:
        entry["interaction_assessment"] = {
            "status": "unverified_action_attempt",
            "message": "Closing or finishing this recipe does not establish retention, contact, placement, or task completion. Use a subsequent planning observation when available.",
        }
    try:
        if definitions:
            try:
                controller._begin_visual_plan(definitions, observation)
            except ValueError:
                entry["binding_status"] = "current_public_depth_unavailable"
                flush()
                return True
        if support_surface_guard:
            from .public_support_geometry import assess_placement_support
            assessment = assess_placement_support(plan, observation)
            if assessment is not None:
                entry["support_surface_assessment"] = assessment
                if assessment["status"] == "near_vertical":
                    entry["status"] = "public_replan_needed"
                    entry["binding_status"] = "public_placement_support_reselection_needed"
                    entry["validation_reason"] = {
                        "code": "placement_support_near_vertical",
                        "correction": "The current public metric-depth neighborhood around this placement-support pixel consistently fits a nearly vertical surface at two scales. This recipe expects a supporting surface, not a side wall. No plan actions were executed. Reinspect current views and choose a currently visible supporting surface, or another appropriate bounded plan, in the next existing revision. Unknown geometry is not rejected or certified. This evidence does not establish an object's identity, clearance, contact, retention or task completion; no target is automatically changed.",
                    }
                    flush()
                    return True
        if plan.get("recipe") == "bound_cavity_place":
            try:
                controller._validate_cavity_route(plan["operations"][0]["offset"],
                                                  plan["operations"][1]["offset"])
            except CavityRouteAlignmentError as exc:
                entry["binding_status"] = "public_cavity_route_unavailable"
                entry["binding_feedback"] = _cavity_alignment_feedback({
                    "interior_minus_entrance_height_m": exc.height_difference_m,
                    "horizontal_distance_m": exc.horizontal_distance_m})
                entry["validation_reason"] = {
                    "code": "cavity_route_alignment",
                    "correction": "The signed height difference is interior minus entrance after applying THIS plan's requested offsets to its current public RGB-D anchors; positive means the requested interior waypoint is higher. Use current views and these relative measurements to reconsider pixels or offsets for a nearly level lateral route (absolute height difference at most0.04m and horizontal distance at least0.001m). No targets are automatically changed. Do not use this recipe if the entrance or interior support is hidden. These measurements do not establish free space, contact, retention, or task completion.",
                }
                flush()
                return True
            except ValueError:
                entry["binding_status"] = "public_cavity_route_unavailable"
                entry["validation_reason"] = {
                    "code": "cavity_route_binding_invalid",
                    "correction": "The current public anchors or requested offsets could not form finite cavity waypoints. Reconsider currently visible pixels and bounded offsets; no alignment measurement is available. No route or retention is confirmed.",
                }
                flush()
                return True
        for operation in plan["operations"]:
            check_deadline()
            item = {"operation": operation, "status": "pending"}
            entry["execution"].append(item)
            flush()
            try:
                if operation["op"] == "observe":
                    public_observation(controller.observe())
                    feedback = {}
                else:
                    if _remaining(controller) < operation["steps"]:
                        item["status"] = "insufficient_action_budget"
                        flush()
                        return True
                    kwargs = {k: v for k, v in operation.items() if k != "op"}
                    if (plan.get("recipe") in BOUND_RECIPES
                            and operation["op"] == "move_bound_target"):
                        if bound_moves == 0 or (plan.get("recipe") in {"bound_transfer", "bound_cavity_place"} and bound_moves == 2):
                            kwargs["_stop_on_stall"] = True
                        bound_moves += 1
                    if operation["op"] == "vla_act":
                        before = _remaining(controller)
                        item["status"] = "executing"
                        try:
                            controller.vla_act(**kwargs)
                        finally:
                            item["feedback"] = {
                                "local_policy_prompt": operation["prompt"],
                                "requested_steps": operation["steps"],
                                "actions_executed": before - _remaining(controller),
                            }
                            if item["status"] == "executing":
                                item["status"] = "interrupted"
                            flush()
                        item["status"] = "completed"
                        item["reobserve_reason"] = {
                            "code": "local_policy_observation_required",
                            "message": "A bounded local-policy attempt ended. Inspect fresh public views in the next available original planning revision before further actions; this is not evidence of contact, retention, or task completion.",
                        }
                        entry["status"] = "public_reobserve_needed"
                        flush()
                        return True
                    feedback = _feedback(getattr(controller, operation["op"])(**kwargs))
                if (getattr(controller, "_relative_orientation_feedback", False)
                        and operation["op"] in {"move_delta", "move_bound_target"}
                        and any(v != 0 for v in operation.get("rotation", [0, 0, 0]))):
                    angle_error = feedback.get("rotation_error")
                    if angle_error is None or angle_error > .12:
                        item.update(status="public_replan_needed", feedback=feedback,
                                    replan_reason={
                                        "code": "relative_rotation_feedback_unavailable" if angle_error is None
                                                else "relative_rotation_unreached",
                                        "rotation_error_limit_rad": .12,
                                        "message": "Public end-effector orientation did not confirm the requested bounded relative rotation. Remaining plan actions were skipped with the commanded gripper unchanged; use fresh views and public pose feedback to replan. This does not identify contact, retention, or task completion.",
                                    })
                        entry["status"] = "public_replan_needed"
                        flush()
                        return True
                if feedback.get("motion_status") == "stalled":
                    item.update(status="public_replan_needed", feedback=feedback,
                                replan_reason={
                                    "code": "bound_approach_stalled",
                                    "message": "Public end-effector position made no appreciable progress during approach. Preserve the gripper state and use fresh views to reassess the route or orientation; this is not a contact or grasp judgement.",
                                })
                    entry["status"] = "public_replan_needed"
                    flush()
                    return True
                if (operation["op"] == "move_bound_target"
                        or (plan.get("recipe") in LEGACY_POSITION_GUARDED_RECIPES
                            and operation["op"] == "pixel_target")):
                    error = feedback.get("position_error")
                    if error is None or error > BOUND_RECIPE_POSITION_ERROR_LIMIT_M:
                        item.update(status="public_replan_needed", feedback=feedback,
                                    replan_reason={
                                        "code": "bound_position_feedback_unavailable" if error is None
                                                else "bound_waypoint_unreached",
                                        "message": BOUND_RECIPE_REPLAN_MESSAGE,
                                        "position_error_limit_m": BOUND_RECIPE_POSITION_ERROR_LIMIT_M,
                                    })
                        entry["status"] = "public_replan_needed"
                        flush()
                        return True
                item.update(status="completed", feedback=feedback)
            except ValueError as exc:
                item.update(status="public_primitive_rejected", error_type=type(exc).__name__)
                flush()
                return True
            except Exception as exc:
                item.update(status="controller_unavailable", error_type=type(exc).__name__)
                flush()
                raise
            flush()
        return False
    finally:
        if definitions:
            controller._end_visual_plan()


def _bounded_policy_continuation(controller, steps, entry, reason, check_deadline, flush):
    """Public scheduling override; native stop remains private BaseException flow."""
    check_deadline()
    before = _remaining(controller)
    requested = min(steps, before)
    override = {"reason": reason, "requested_steps": requested, "executed_steps": 0,
                "status": "pending",
                "message": "A bounded policy continuation obtains current public observations within the existing action budget. This scheduling rule is not a task-completion judgement."}
    entry["continuation_override"] = override
    item = {"operation": {"op": "run_policy", "steps": requested},
            "origin": "runtime_finish_or_zero_action_continuation", "status": "pending"}
    entry["execution"].append(item)
    flush()
    try:
        controller.run_policy(requested)
        check_deadline()
        item["status"] = "completed"
        override["status"] = "completed"
    except Exception as error:
        item.update(status="controller_unavailable", error_type=type(error).__name__)
        override["status"] = "controller_unavailable"
        raise
    finally:
        executed = before - _remaining(controller)
        override["executed_steps"] = executed
        item["feedback"] = {"actions_executed": executed}
        if override["status"] == "pending":
            override["status"] = "interrupted"
            item["status"] = "interrupted"
        flush()


def recover(controller, config, output, planner=None):
    """Execute up to three visual revisions, returning only a public receipt.

Inject ``planner(request)`` for CPU checks. The planner receives fresh allowlisted
observations, frozen common recipes/lessons, remaining steps and public motion
receipts. The policy identity, task/init IDs and evaluator are never forwarded.
"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    extra_view = config.get('recovery_extra_view', False)
    if type(extra_view) is not bool:
        raise ValueError('recovery_extra_view must be Boolean')
    orientation_feedback = config.get("public_relative_orientation_feedback", False)
    if type(orientation_feedback) is not bool:
        raise ValueError("public_relative_orientation_feedback must be Boolean")
    if orientation_feedback:
        controller._set_relative_orientation_feedback(True)
    support_surface_guard = config.get("public_support_surface_guard", False)
    if type(support_surface_guard) is not bool:
        raise ValueError("public_support_surface_guard must be Boolean")
    limits = config.get("recovery", {})
    continuation_steps = config.get("finish_policy_continuation_steps", 0)
    if type(continuation_steps) is not int or not 0 <= continuation_steps <= 80:
        raise ValueError("finish_policy_continuation_steps must be integer 0..80")
    if continuation_steps:
        recovery_cap = config.get("recovery_steps", 400)
        if type(recovery_cap) is not int or not 0 <= recovery_cap <= 400:
            raise ValueError("Policy continuation requires the original recovery cap <=400")
    rounds = config.get("repair_rounds", limits.get("max_rounds", config.get("max_rounds", 3)))
    if type(rounds) is not int or not 1 <= rounds <= 3:
        raise ValueError("Recovery permits one to three visual revisions")
    deadline_seconds = limits.get("wall_timeout_seconds", 1800)
    if type(deadline_seconds) not in (int, float) or not 0 < deadline_seconds <= 7200:
        raise ValueError("Invalid recovery wall timeout")
    skills = json.loads((ROOT / "skills/catalog.json").read_text())
    lessons = json.loads((ROOT / "memory/general/public_lessons.json").read_text())
    from .experience import retrieve_public_experiences
    deadline = time.monotonic() + deadline_seconds
    inherited_deadline = config.get("case_deadline_monotonic")
    if inherited_deadline is not None:
        if type(inherited_deadline) not in (int, float) or not math.isfinite(inherited_deadline):
            raise ValueError("Invalid inherited case deadline")
        deadline = min(deadline, inherited_deadline)
    def account_queue_wait(seconds):
        nonlocal deadline
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid measured coding admission wait")
        trusted_account = getattr(controller, "_account_queue_wait", None)
        if trusted_account is not None:
            trusted_account(seconds)
        deadline += seconds
    if planner is None:
        settings = dict(config.get("coding_agent", {}))
        settings["case_deadline_monotonic"] = deadline
        planner = CodingPlanner(settings, output / "coding", on_queue_wait=account_queue_wait)
    elif isinstance(planner, CodingPlanner):
        planner._on_queue_wait = account_queue_wait
    receipt = {"schema": "libero.public_recovery.v2", "status": "started", "revisions": [],
               "memory_promoted": False, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "frozen_context_sha256": hashlib.sha256(json.dumps([skills, lessons], sort_keys=True).encode()).hexdigest()}
    def flush():
        temporary = output / "public_execution.json.tmp"
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "public_execution.json")
    flush()
    def check_deadline():
        if time.monotonic() >= deadline:
            receipt["status"] = "actor_wall_timeout"
            flush()
            raise TimeoutError("Recovery reached the fixed case wall deadline")
    history = []
    for revision in range(1, rounds + 1):
        remaining = _remaining(controller)
        if remaining == 0:
            receipt["status"] = "action_budget_exhausted"
            break
        check_deadline()
        observation = public_observation(controller._observe_for_planner(extra_view=True)
                                         if extra_view else controller.observe())
        experiences = retrieve_public_experiences(observation["instruction"], observation,
            previous_public_execution=copy.deepcopy(history), top_k=3,
            remaining_revisions=rounds - revision + 1, remaining_steps=remaining)
        request = {"observation": observation, "remaining_steps": remaining,
                   "remaining_revisions": rounds - revision + 1,
                   "skills": copy.deepcopy(skills), "general_lessons": copy.deepcopy(lessons),
                   "reusable_experiences": copy.deepcopy(experiences),
                   "previous_public_execution": copy.deepcopy(history)}
        entry = {"revision": revision, "remaining_revisions": rounds - revision + 1,
                 "status": "planning", "execution": []}
        receipt["revisions"].append(entry)
        flush()
        try:
            raw = planner(request)
        except Exception as exc:
            entry.update(status="planner_unavailable", error_type=type(exc).__name__)
            receipt["status"] = "planner_unavailable"
            flush()
            raise
        check_deadline()
        try:
            plan = validate_plan(raw, observation, remaining_steps=remaining)
        except (ValueError, TypeError, KeyError) as exc:
            reason = _validation_feedback(exc)
            entry.update(status="invalid_plan", error_type=type(exc).__name__, validation_reason=reason)
            history.append({"status": "invalid_plan", "validation_reason": copy.deepcopy(reason)})
            flush()
            continue
        entry.update(status="executing", plan=plan,
                     plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest())
        flush()
        try:
            execution_failed = _execute_plan(controller, plan, observation, entry, check_deadline, flush,
                                             support_surface_guard=support_surface_guard)
        except Exception:
            receipt["status"] = "controller_unavailable"
            flush()
            raise
        if entry["status"] == "executing":
            entry["status"] = "public_replan_needed" if execution_failed else "plan_completed"
        continuation_ran = False
        if (continuation_steps and not execution_failed and _remaining(controller) > 0
                and (plan["next"] == "finish" or _remaining(controller) == remaining)):
            reason = "actor_finish_is_advisory" if plan["next"] == "finish" else "zero_action_plan_has_no_new_observation"
            try:
                _bounded_policy_continuation(controller, continuation_steps, entry, reason, check_deadline, flush)
            except Exception:
                receipt["status"] = "controller_unavailable"
                flush()
                raise
            continuation_ran = True
        history_item = {"plan": plan, "status": entry["status"], "execution": copy.deepcopy(entry["execution"]),
                        "binding_status": entry.get("binding_status")}
        if "validation_reason" in entry:
            history_item["validation_reason"] = copy.deepcopy(entry["validation_reason"])
        if "binding_feedback" in entry:
            history_item["binding_feedback"] = _cavity_alignment_feedback(entry["binding_feedback"])
        if "support_surface_assessment" in entry:
            history_item["support_surface_assessment"] = copy.deepcopy(entry["support_surface_assessment"])
        if "interaction_assessment" in entry:
            history_item["interaction_assessment"] = copy.deepcopy(entry["interaction_assessment"])
        if "continuation_override" in entry:
            history_item["continuation_override"] = copy.deepcopy(entry["continuation_override"])
        history.append(history_item)
        if continuation_ran:
            continue
        if continuation_steps and _remaining(controller) == 0:
            receipt["status"] = "action_budget_exhausted"
            break
        if plan["next"] == "finish" and not execution_failed:
            receipt["status"] = "actor_finished"
            break
    else:
        receipt["status"] = "revision_budget_exhausted"
    receipt["remaining_steps"] = _remaining(controller)
    flush()
    return receipt
