"""Small retrieval layer for hand-curated public recovery experience cards.

Only fixed, reviewed summaries from ``memory/experiences/public_index.json``
are returned to the planner. Runtime instructions and receipts can affect
retrieval, but their free-form text, coordinates, identifiers, and scores are
never copied into an experience card.
"""
from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "memory" / "experiences" / "public_index.json"
PUBLIC_CAMERAS = ("agentview", "robot0_eye_in_hand", "sideview")
PUBLIC_PROPRIO = (
    "robot_cartesian_pos", "robot_eef_quat", "robot_gripper_pos", "robot_joint_pos",
)
PUBLIC_OPERATIONS = {"observe", "pixel_target", "move_delta", "gripper", "run_policy"}
PUBLIC_OPERATION_STATUSES = {
    "completed", "public_primitive_rejected", "insufficient_action_budget",
}
PUBLIC_VALIDATION_CODES = {"pixel_target_after_steps"}
PUBLIC_REPLAN_CODES = {
    "bound_waypoint_unreached",
    "bound_position_feedback_unavailable",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
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


def _has_rgbd(public_obs: object) -> bool:
    if not isinstance(public_obs, dict):
        return False
    for camera in PUBLIC_CAMERAS:
        entry = public_obs.get(camera)
        images = entry.get("images") if isinstance(entry, dict) else None
        if (isinstance(images, dict) and images.get("rgb") is not None
                and images.get("depth") is not None):
            return True
    return False


def _has_proprio(public_obs: object) -> bool:
    return isinstance(public_obs, dict) and any(key in public_obs for key in PUBLIC_PROPRIO)


def _history_flags(previous_public_execution: object) -> set[str]:
    """Read only known public operation/status enums from prior receipts."""
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
        for item in execution:
            if not isinstance(item, dict):
                continue
            operation = item.get("operation")
            if not isinstance(operation, dict):
                continue
            name, status = operation.get("op"), item.get("status")
            replan_reason = item.get("replan_reason")
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
):
    """Retrieve a few fixed public action templates using public-only signals.

    Matching uses instruction keywords, the presence of configured RGB-D and
    proprioceptive fields, and allowlisted operation/status enums from prior
    public execution. Other observation/history fields are ignored.
    """
    if type(top_k) is not int or top_k <= 0:
        return []
    if not isinstance(instruction, str):
        instruction = ""
    instruction = instruction.casefold()
    instruction_tokens = _tokens(instruction)
    has_rgbd = _has_rgbd(public_obs)
    has_proprio = _has_proprio(public_obs)
    history_flags = _history_flags(previous_public_execution)

    ranked = []
    for order, card in enumerate(_load_cards()):
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
        if card.get("id") == "visible_retention_gate_before_place" and intent_hits:
            score += 2
        requires = card.get("requires", [])
        if not isinstance(requires, list):
            requires = []
        if "rgbd" in requires and not has_rgbd:
            score -= 100
        if "proprio" in requires and not has_proprio:
            score -= 100
        if ("stale_pixel_rejected" in history_flags
                and card.get("history_trigger") == "stale_pixel_rejected"):
            score += 10
        if ("bound_waypoint_unreached" in history_flags
                and card.get("id") == "visible_relative_clearance"):
            score += 12
        if score > 0:
            ranked.append((score, order, _public_card(card)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    if not ranked:
        fallback = next(
            (card for card in _load_cards() if card.get("id") == "observe_before_motion"),
            None,
        )
        return [_public_card(fallback)] if fallback is not None else []
    return [item[2] for item in ranked[:top_k]]
