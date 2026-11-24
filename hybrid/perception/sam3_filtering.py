"""Pure mask-selection/serialization helpers for the optional SAM3 candidate gate."""
from __future__ import annotations

import math
from typing import Callable


def validate_min_score(min_score: float | None) -> None:
    if min_score is not None and not math.isfinite(min_score):
        raise ValueError("min_score must be finite or None")


def build_mask_results(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    text_prompt: str,
    min_score: float | None,
    encode_mask: Callable,
    prepare_mask: Callable = lambda mask: mask > 0,
    mask_shape: Callable = lambda mask: mask.shape,
    box_values: Callable = lambda box: box.tolist(),
) -> list[dict]:
    """Filter finite scores before mask conversion/encoding; retain all matches."""
    validate_min_score(min_score)
    results = []
    for i, raw_score in enumerate(scores):
        score = float(raw_score)
        if min_score is not None and (not math.isfinite(score) or score < min_score):
            continue
        mask = prepare_mask(masks[i])
        results.append({
            "mask_base64": encode_mask(mask),
            "shape": list(mask_shape(mask)),
            "box": list(box_values(boxes[i])),
            "score": score,
            "label": text_prompt,
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results
