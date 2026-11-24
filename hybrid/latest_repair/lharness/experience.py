"""Small retrieval layer for hand-curated public recovery experience cards.

Only fixed, reviewed summaries from ``memory/experiences/public_index.json``
are returned to the planner. Runtime instructions and receipts can affect
retrieval, but their free-form text, coordinates, identifiers, and scores are
never copied into an experience card.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "memory" / "experiences" / "public_index.json"
PUBLIC_CAMERAS = ("agentview", "robot0_eye_in_hand", "sideview")
PUBLIC_PROPRIO = (
    "robot_cartesian_pos", "robot_eef_quat", "robot_gripper_pos", "robot_joint_pos",
)
PUBLIC_OPERATIONS = {
    "observe", "pixel_target", "move_delta", "gripper", "run_policy", "vla_act",
    "bound_transfer", "bound_drawer_pull", "bound_surface_stroke",
}
PUBLIC_OPERATION_STATUSES = {
    "completed", "public_primitive_rejected", "insufficient_action_budget",
}
PUBLIC_VALIDATION_CODES = {"pixel_target_after_steps"}
PUBLIC_REPLAN_CODES = {
    "bound_waypoint_unreached",
    "bound_position_feedback_unavailable",
    "bound_approach_stalled",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_FINAL_REVISION_CARD_ID = "final_revision_action_closure"
_RETURN_FIELDS = ("id", "title", "when_public", "action_template", "recipe_ids", "limits")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


def _matches_keyword(keyword: str, instruction: str, instruction_tokens: set[str]) -> bool:
    keyword = keyword.casefold().strip()
    if not keyword:
        return False
    if _CJK_RE.search(keyword):
        return keyword in instruction.casefold()
    words = _tokens(keyword)
    return bool(words) and words <= instruction_tokens


def _explicit_surface_closing(instruction: str) -> bool:
    """Public language coverage only; never infer the current task phase."""
    if not re.search(r"\b(?:microwave|drawer|cabinet|door|oven)\b", instruction):
        return False
    for match in re.finditer(r"\b(?:close|shut)\s+(?:(?:the|a|an)\s+)?(?:it|microwave|drawer|cabinet|door|oven)\b", instruction):
        before = instruction[max(0, match.start()-24):match.start()]
        if not re.search(r"(?:do\s+not|don't|don’t|never|not)\s*$", before):
            return True
    return False


def _local_contact_instruction(instruction: str) -> bool:
    """Offer a conditional option, not a prediction of the current phase."""
    return bool(re.search(
        r"\b(?:grasp|grab|pick|put|place|insert|push|pull|press|open|close|shut)\b",
        instruction,
    ))


def _compound_transfer_instruction(instruction: str) -> bool:
    """Recognize two explicit transfer clauses, not arbitrary 'and' text."""
    transfer = r"\b(?:put|place|set|insert|transfer|move)\b"
    return bool(re.search(transfer + r".+\b(?:and|then)\b.+" + transfer, instruction))


def _has_rgbd(public_obs: object) -> bool:
    return bool(_rgbd_cameras(public_obs))


def _rgbd_cameras(public_obs: object) -> set[str]:
    if not isinstance(public_obs, dict):
        return set()
    cameras = set()
    for camera in PUBLIC_CAMERAS:
        entry = public_obs.get(camera)
        images = entry.get("images") if isinstance(entry, dict) else None
        if (isinstance(images, dict) and images.get("rgb") is not None
                and images.get("depth") is not None):
            cameras.add(camera)
    return cameras


def _has_alternate_rgbd(public_obs: object) -> bool:
    """Require two configured public RGB-D views; never inspect image content."""
    return len(_rgbd_cameras(public_obs)) >= 2


def _has_proprio(public_obs: object) -> bool:
    return isinstance(public_obs, dict) and any(key in public_obs for key in PUBLIC_PROPRIO)


def _commanded_position_not_reached(item: dict) -> bool:
    """Finite public positional progress only; no inference of action intent."""
    operation = item.get("operation")
    if (not isinstance(operation, dict)
            or operation.get("op") not in {"move_delta", "pixel_target", "move_bound_target"}
            or item.get("status") not in {"completed", "public_replan_needed"}):
        return False
    feedback = item.get("feedback")
    if not isinstance(feedback, dict):
        return False
    values = [feedback.get(key) for key in
              ("initial_position_error", "position_error", "position_error_reduction")]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
        return False
    initial, remaining, reduction = values
    return initial >= .04 and remaining >= .04 and reduction <= .002


def _history_flags(previous_public_execution: object) -> set[str]:
    """Read known public operation/status enums and finite positional progress."""
    if isinstance(previous_public_execution, dict):
        revisions = previous_public_execution.get("revisions", [])
    elif isinstance(previous_public_execution, list):
        revisions = previous_public_execution
    else:
        return set()
    if not isinstance(revisions, list):
        return set()

    flags: set[str] = set()
    for revision in revisions:
        if not isinstance(revision, dict):
            continue
        reason = revision.get("validation_reason")
        if isinstance(reason, dict) and reason.get("code") in PUBLIC_VALIDATION_CODES:
            flags.add("stale_pixel_rejected")
        execution = revision.get("execution", [])
        if not isinstance(execution, list):
            continue
        plan = revision.get("plan")
        recipe = plan.get("recipe") if isinstance(plan, dict) else None
        if (revision.get("status") == "plan_completed"
                and recipe in {"bound_transfer", "bound_drawer_pull", "bound_surface_stroke"}
                and execution
                and all(isinstance(item, dict) and item.get("status") == "completed"
                        for item in execution)):
            flags.add(recipe + "_completed")
        for item in execution:
            if not isinstance(item, dict):
                continue
            operation = item.get("operation")
            if not isinstance(operation, dict):
                continue
            name, status = operation.get("op"), item.get("status")
            if name == "vla_act" and status in {"completed", "public_reobserve_needed"}:
                flags.add("local_policy_attempt_completed")
            if _commanded_position_not_reached(item):
                flags.add("commanded_position_not_reached")
            if status == "insufficient_action_budget":
                flags.add("action_budget_limited")
            if name in PUBLIC_OPERATIONS and status == "completed":
                if name in {"bound_transfer", "bound_drawer_pull", "bound_surface_stroke"}:
                    flags.add(name + "_completed")
                if name == "gripper" and operation.get("value") == 1:
                    flags.add("gripper_close_completed")
            replan_reason = item.get("replan_reason")
            if (name in {"move_delta", "move_bound_target"} and status == "public_replan_needed"
                    and isinstance(replan_reason, dict)
                    and replan_reason.get("code") in {"relative_rotation_unreached", "relative_rotation_feedback_unavailable"}):
                flags.add("relative_rotation_unreached")
            if (name == "move_bound_target" and status == "public_replan_needed"
                    and isinstance(replan_reason, dict)
                    and replan_reason.get("code") in PUBLIC_REPLAN_CODES):
                flags.add("bound_waypoint_unreached")
            if name not in PUBLIC_OPERATIONS or status not in PUBLIC_OPERATION_STATUSES:
                continue
            if name == "pixel_target" and status == "public_primitive_rejected":
                flags.add("stale_pixel_rejected")
    return flags


def _load_cards() -> list[dict]:
    try:
        value = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(value, dict) or value.get("schema") != "libero.public_experiences.v2":
        return []
    cards = value.get("experiences")
    return [card for card in cards if isinstance(card, dict)] if isinstance(cards, list) else []


def _public_card(card: dict) -> dict:
    """Copy only fixed public fields; do not echo retrieval metadata or inputs."""
    result = {}
    for key in _RETURN_FIELDS:
        value = card.get(key)
        if key in {"recipe_ids", "limits"}:
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                result[key] = list(value)
        elif isinstance(value, str):
            result[key] = value
    return result


def retrieve_public_experiences(
    instruction,
    public_obs,
    previous_public_execution=None,
    top_k=3,
    remaining_revisions=None,
    remaining_steps=None,
):
    """Retrieve a few fixed public action templates using public-only signals.

    Matching uses instruction keywords, the presence of configured RGB-D and
    proprioceptive fields, allowlisted operation/status enums, and three finite
    public position-progress values from prior execution. Other fields are ignored.
    """
    if type(top_k) is not int or top_k <= 0:
        return []
    if (remaining_revisions is not None
            and (type(remaining_revisions) is not int or remaining_revisions <= 0)):
        return []
    if (remaining_steps is not None
            and (type(remaining_steps) is not int or remaining_steps <= 0)):
        return []
    if not isinstance(instruction, str):
        instruction = ""
    instruction = instruction.casefold()
    instruction_tokens = _tokens(instruction)
    has_rgbd = _has_rgbd(public_obs)
    has_proprio = _has_proprio(public_obs)
    history_flags = _history_flags(previous_public_execution)
    final_revision = remaining_revisions == 1
    closing_requested = _explicit_surface_closing(instruction) and has_rgbd and has_proprio
    contact_option = (_local_contact_instruction(instruction)
                      or "local_policy_attempt_completed" in history_flags)
    compound_transfer = _compound_transfer_instruction(instruction)

    ranked = []
    cards = _load_cards()
    final_card = next(
        (_public_card(card) for card in cards
         if card.get("id") == _FINAL_REVISION_CARD_ID),
        None,
    ) if final_revision else None
    for order, card in enumerate(cards):
        if card.get("id") == _FINAL_REVISION_CARD_ID:
            continue
        keywords = card.get("keywords", [])
        intents = card.get("intents", [])
        if not isinstance(keywords, list) or not isinstance(intents, list):
            continue
        keyword_hits = sum(
            1 for keyword in keywords
            if isinstance(keyword, str)
            and _matches_keyword(keyword, instruction, instruction_tokens)
        )
        intent_tokens = card.get("intent_tokens", [])
        if not isinstance(intent_tokens, dict):
            intent_tokens = {}
        intent_hits = 0
        for intent in intents:
            terms = intent_tokens.get(intent, [])
            if isinstance(terms, list) and any(
                isinstance(term, str)
                and _matches_keyword(term, instruction, instruction_tokens)
                for term in terms
            ):
                intent_hits += 1

        score = 2 * keyword_hits + 3 * intent_hits
        if card.get("id") == "bounded_policy_continuation" and contact_option:
            score += 9
        if card.get("id") == "ordered_multi_object_transfer" and compound_transfer:
            score += 8
        if card.get("id") == "visible_retention_gate_before_place" and intent_hits:
            score += 2
        requires = card.get("requires", [])
        if not isinstance(requires, list):
            requires = []
        if "rgbd" in requires and not has_rgbd:
            score -= 100
        if "alternate_rgbd" in requires and not _has_alternate_rgbd(public_obs):
            score -= 100
        if "proprio" in requires and not has_proprio:
            score -= 100
        if ("stale_pixel_rejected" in history_flags
                and card.get("history_trigger") == "stale_pixel_rejected"):
            score += 10
        if ("bound_waypoint_unreached" in history_flags
                and card.get("id") == "visible_relative_clearance"):
            score += 12
        if ("commanded_position_not_reached" in history_flags
                and card.get("id") == "commanded_position_not_reached"):
            score += 14
        if ("relative_rotation_unreached" in history_flags
                and card.get("id") == "current_image_grasp_axis"):
            score += 12
        triggers = card.get("history_triggers", [])
        if isinstance(triggers, str):
            triggers = [triggers]
        if (score > 0 and isinstance(triggers, list)
                and any(isinstance(flag, str) and flag in history_flags for flag in triggers)):
            score += 6
        if score > 0:
            ranked.append((score, order, _public_card(card)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    if not ranked:
        fallback = next(
            (card for card in cards if card.get("id") == "observe_before_motion"),
            None,
        )
        ranked = [(0, len(cards), _public_card(fallback))] if fallback is not None else []

    results = [item[2] for item in ranked]
    if closing_requested:
        closing_card = next((_public_card(card) for card in cards
                             if card.get("id") == "visible_surface_contact_stroke"), None)
        if closing_card is not None:
            results = [closing_card] + [card for card in results
                                      if card.get("id") != closing_card["id"]]
    if final_card is not None:
        results = [final_card] + [
            card for card in results if card.get("id") != _FINAL_REVISION_CARD_ID
        ]
    return results[:top_k]
