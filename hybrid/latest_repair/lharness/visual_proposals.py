"""Bounded, optional SAM proposals from the current public RGB; no native state access."""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit

import numpy as np

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_PROPOSALS = 64
DISCLAIMER = ("Unverified visual proposals only. Scores do not establish object identity, "
              "drawer level, contact, grasp, or success. Select the instructed drawer from "
              "the current original views; a candidate may be wrong or incomplete.")


def proposal_settings(value):
    if value is None:
        return None
    allowed = {"enabled", "endpoint", "timeout_seconds", "max_candidates", "min_score",
               "admission_root", "admission_slots", "admission_wait_seconds", "tabletop_enabled"}
    if not isinstance(value, dict) or set(value) - allowed or type(value.get("enabled")) is not bool:
        raise ValueError("Invalid public visual proposal configuration")
    if "tabletop_enabled" in value and type(value["tabletop_enabled"]) is not bool:
        raise ValueError("Tabletop proposals require an explicit Boolean")
    if value["enabled"] is False:
        return None
    result = dict(enabled=True, timeout_seconds=5., max_candidates=3, min_score=.3)
    result.update(value)
    result.setdefault("tabletop_enabled", False)
    _loopback_endpoint(result.get("endpoint"))
    if (type(result["timeout_seconds"]) not in (int, float)
            or not math.isfinite(result["timeout_seconds"])
            or not 0 < result["timeout_seconds"] <= 10):
        raise ValueError("SAM timeout must be finite and at most 10 seconds")
    if type(result["max_candidates"]) is not int or not 1 <= result["max_candidates"] <= 3:
        raise ValueError("At most three visual proposals per camera are allowed")
    if type(result["min_score"]) not in (int, float) or result["min_score"] != .3:
        raise ValueError("The proposal-only score filter must remain 0.3")
    if "admission_root" in result:
        root = result["admission_root"]
        if not isinstance(root, str) or not root or not Path(root).is_absolute():
            raise ValueError("SAM admission requires an absolute shared root")
        slots = result.setdefault("admission_slots", 1)
        if type(slots) is not int or slots != 1:
            raise ValueError("Shared SAM admission requires exactly one slot")
        wait = result.setdefault("admission_wait_seconds", 60)
        if type(wait) not in (int, float) or not math.isfinite(wait) or not 0 < wait <= 60:
            raise ValueError("SAM admission wait must be finite and at most60 seconds")
    elif "admission_slots" in result or "admission_wait_seconds" in result:
        raise ValueError("SAM admission limits require a shared root")
    return result


def is_drawer_instruction(instruction):
    return re.search(r"\bdrawers?\b", instruction, flags=re.IGNORECASE) is not None


def _loopback_endpoint(endpoint):
    """Parse the configured SAM service URL; only a local http://host:port/segment is accepted."""
    address = urlsplit(endpoint if isinstance(endpoint, str) else "")
    if (address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost", "::1"}
            or not address.port or address.path != "/segment" or address.query or address.fragment):
        raise ValueError("Set public_visual_proposals.endpoint to the local SAM service, "
                         "http://127.0.0.1:PORT/segment")
    return address.hostname, address.port


def request_proposals(png, timeout_seconds, deadline, *, endpoint, text_prompt="drawer"):
    """One POST to the local SAM service; no proxy, redirects, tools or retries."""
    if text_prompt not in {"drawer", "tabletop"}:
        raise ValueError("Only fixed public drawer/tabletop proposal queries are allowed")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Active case deadline exhausted before visual proposal")
    budget = min(timeout_seconds, remaining)
    end = time.monotonic() + budget
    host, port = _loopback_endpoint(endpoint)
    connection = http.client.HTTPConnection(host, port, timeout=budget)
    response_socket = [None]
    def expire():
        sock = response_socket[0] or connection.sock
        if sock is not None:
            try:
                sock.shutdown(2)
            except OSError:
                pass
        connection.close()
    timer = threading.Timer(budget, expire)
    timer.daemon = True
    timer.start()
    try:
        payload = json.dumps({"image_base64": base64.b64encode(png).decode("ascii"),
                              "text_prompt": text_prompt, "min_score": .3}).encode()
        connection.request("POST", "/segment", payload, {"Content-Type": "application/json"})
        response = connection.getresponse()
        response_socket[0] = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if response.status != 200:
            raise ValueError("SAM response status was not 200")
        chunks, size = [], 0
        while True:
            if time.monotonic() >= end:
                raise TimeoutError("Visual proposal request exceeded its wall-time bound")
            chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("SAM response exceeds its byte bound")
        if time.monotonic() >= end:
            raise TimeoutError("Visual proposal request exceeded its wall-time bound")
        return json.loads(b"".join(chunks))
    finally:
        timer.cancel()
        connection.close()


def candidate_masks(response, shape, limit=3, min_score=.3):
    """Strictly decode bounded raw uint8 masks, then preserve score ordering."""
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise ValueError("Invalid SAM response schema")
    rows = response["results"]
    if len(rows) > MAX_PROPOSALS:
        raise ValueError("Too many SAM masks")
    size = int(np.prod(shape))
    good = []
    for item in rows:
        if not isinstance(item, dict) or item.get("shape") != list(shape):
            raise ValueError("SAM mask shape differs from its current RGB")
        score = item.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid SAM proposal score")
        encoded = item.get("mask_base64")
        if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
            raise ValueError("Invalid SAM mask byte count")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != size:
            raise ValueError("Invalid SAM mask byte count")
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("SAM mask must be binary uint8")
        if score < min_score or not mask.any():
            continue
        y, x = np.nonzero(mask)
        good.append({"mask": mask, "score": float(score),
                     "bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]})
    return sorted(good, key=lambda item: item["score"], reverse=True)[:limit]


def panel_crop(rgb, bounds):
    x0, y0, x1, y1 = bounds
    padding = max(8, int(np.ceil(.15 * max(x1 - x0, y1 - y0))))
    left, top = max(0, x0 - padding), max(0, y0 - padding)
    right, bottom = min(rgb.shape[1], x1 + padding), min(rgb.shape[0], y1 + padding)
    return rgb[top:bottom, left:right].copy(), [left, top, right, bottom]


def full_pixel(pixel, roi):
    """Display coordinate illustration; controller still requires original pixels."""
    if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
        return None
    try:
        x, y = float(pixel[0]), float(pixel[1])
    except (TypeError, ValueError):
        return None
    left, top, right, bottom = roi
    if not np.isfinite([x, y]).all() or not (0 <= x < right-left and 0 <= y < bottom-top):
        return None
    return [int(x) + left, int(y) + top]


def proposal_crop(rgb, response, settings):
    candidates = candidate_masks(response, rgb.shape[:2], settings["max_candidates"], settings["min_score"])
    metadata = {"status": "available" if candidates else "empty", "query": "drawer",
                "interpretation": DISCLAIMER, "candidates": [],
                "bbox_convention": "original camera pixels [left,top,right,bottom), never world coordinates"}
    for rank, item in enumerate(candidates, 1):
        metadata["candidates"].append({"rank": rank, "bbox_xyxy": item["bbox_xyxy"],
                                       "score": item["score"]})
    if not candidates:
        return metadata, None
    boxes = np.asarray([item["bbox_xyxy"] for item in candidates])
    union = [int(boxes[:, 0].min()), int(boxes[:, 1].min()),
             int(boxes[:, 2].max()), int(boxes[:, 3].max())]
    crop, roi = panel_crop(rgb, union)
    metadata["crop"] = {"roi_xyxy": roi, "width": int(crop.shape[1]), "height": int(crop.shape[0]),
                        "resized": False, "original_pixel_offset_xy": roi[:2],
                        "mapping": "original_x=crop_x+left; original_y=crop_y+top; use ORIGINAL pixels in actions"}
    return metadata, crop


TABLETOP_DISCLAIMER = (
    "Unverified current-RGB tabletop mask suggestions only. Scores and distance to "
    "a predicted mask boundary do not prove a support surface, free space, object "
    "identity, contact, grasp, retention, placement or success. Check current RGB-D "
    "and other views; discard suggestions inconsistent with the instructed relation. "
    "The actor may choose another original-image pixel using the predicted outline, "
    "but must satisfy the instructed relative placement; maximum interior distance "
    "is not the task target. These points do not define the task-language left/right reference frame.")
TABLETOP_MIN_BOUNDARY_DISTANCE_PX = 3.


def is_tabletop_instruction(instruction):
    """Public-language routing, not a task-ID or benchmark-instance lookup."""
    if not isinstance(instruction, str) or is_drawer_instruction(instruction):
        return False
    text = instruction.lower()
    if not re.search(r"\b(?:put|place|move|set)\b", text):
        return False
    if re.search(r"\b(?:on|onto|on top of)\s+(?:the |a )?(?:tabletop|table)\b", text):
        return True
    return re.search(r"\b(?:put|place|move|set)\b[^.;!?]*\b(?:to the (?:left|right) of|"
                     r"in front of|behind|next to|beside)\s+(?:the |a )?plate\b", text) is not None


def tabletop_proposal_crop(rgb, response, settings):
    """At most three interior pixels from THIS predicted binary mask, no depth/truth."""
    import cv2
    masks = candidate_masks(response, rgb.shape[:2], settings["max_candidates"], settings["min_score"])
    metadata = {"status": "empty", "query": "tabletop", "interpretation": TABLETOP_DISCLAIMER,
                "candidates": [], "pixel_convention": "original camera integer [x,y], top-left origin",
                "distance_convention": "Euclidean image pixels to nearest predicted-mask exterior; image edge is exterior",
                "minimum_predicted_boundary_distance_px": TABLETOP_MIN_BOUNDARY_DISTANCE_PX}
    pools = []
    yy, xx = np.indices(rgb.shape[:2])
    for rank, item in enumerate(masks, 1):
        distance = cv2.distanceTransform(np.pad(item["mask"], 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        available = distance.copy()
        points = []
        for _ in range(3):
            y, x = np.unravel_index(int(available.argmax()), available.shape)
            margin = float(available[y, x])
            if margin < TABLETOP_MIN_BOUNDARY_DISTANCE_PX:
                break
            points.append({"pixel": [int(x), int(y)], "score": item["score"],
                "mask_rank": rank, "distance_to_predicted_boundary_px": float(distance[y, x]),
                "bbox_xyxy": item["bbox_xyxy"]})
            separation = max(8., .5 * margin)
            available[(xx-x)**2 + (yy-y)**2 < separation**2] = 0
        pools.append(points)
    for depth in range(3):
        for points in pools:
            if depth < len(points) and len(metadata["candidates"]) < 3:
                metadata["candidates"].append(points[depth])
    if not metadata["candidates"]:
        return metadata, None
    metadata["status"] = "available"
    boxes = np.array([p["bbox_xyxy"] for p in metadata["candidates"]])
    annotated = rgb.copy()
    for item in masks:
        contours, _ = cv2.findContours(item["mask"].copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 255, 255), 1)
    for rank, point in enumerate(metadata["candidates"], 1):
        pixel = tuple(point["pixel"])
        cv2.drawMarker(annotated, pixel, (255, 0, 255), cv2.MARKER_CROSS, 7, 1)
        cv2.putText(annotated, str(rank), (pixel[0]+4, max(9,pixel[1]-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 0, 255), 1)
    crop, roi = panel_crop(annotated, [int(boxes[:,0].min()),int(boxes[:,1].min()),
                               int(boxes[:,2].max()),int(boxes[:,3].max())])
    metadata["crop"] = {"roi_xyxy": roi, "width": int(crop.shape[1]), "height": int(crop.shape[0]),
        "resized": False, "original_pixel_offset_xy": roi[:2],
        "mapping": "original_x=crop_x+left; original_y=crop_y+top; suggestions already use ORIGINAL pixels",
        "overlay": "CYAN predicted mask outlines, MAGENTA numbered candidate crosses; original RGB is unchanged",
        "selection": "May select other original pixels using the predicted outline and current RGB-D; maximum margin does not define the task-correct location"}
    return metadata, crop


def _current_proposals(rgb, png_path, settings, deadline, prompt):
    start = time.monotonic()
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= start:
        raise TimeoutError("Active case deadline exhausted before visual proposal")
    png = png_path.read_bytes()
    receipt = {"input_png_sha256": hashlib.sha256(png).hexdigest(), "query": prompt,
               "attempts": 0, "timeout_seconds": settings["timeout_seconds"],
               "queue_seconds": 0., "service_seconds": 0.}
    queued_at = None
    admitted = False
    def query():
        receipt["attempts"] = 1
        service_start = time.monotonic()
        try:
            if prompt == "drawer":
                return request_proposals(png, settings["timeout_seconds"], deadline,
                                         endpoint=settings["endpoint"])
            return request_proposals(png, settings["timeout_seconds"], deadline,
                                     endpoint=settings["endpoint"], text_prompt="tabletop")
        finally:
            receipt["service_seconds"] = time.monotonic() - service_start
    def case_timeout(cause=None):
        receipt.update(status="case_deadline_exhausted", elapsed_seconds=time.monotonic()-start)
        error = TimeoutError("Active case deadline exhausted during visual proposal")
        error.proposal_receipt = dict(receipt)
        raise error from cause
    try:
        if settings.get("admission_root"):
            from .coding import coding_slot
            queued_at = time.monotonic()
            admission_deadline = min(deadline, queued_at + settings["admission_wait_seconds"])
            receipt.update(admission_slots=1, admission_wait_limit_seconds=settings["admission_wait_seconds"])
            with coding_slot(settings["admission_root"], admission_deadline, slots=1):
                receipt["queue_seconds"] = time.monotonic() - queued_at
                admitted = True
                if time.monotonic() >= deadline:
                    case_timeout()
                response = query()
        else:
            response = query()
        builder = proposal_crop if prompt == "drawer" else tabletop_proposal_crop
        metadata, crop = builder(rgb, response, settings)
        receipt["response_sha256"] = hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()
    except Exception as exc:
        if queued_at is not None and not admitted:
            receipt["queue_seconds"] = time.monotonic() - queued_at
        if time.monotonic() >= deadline:
            case_timeout(exc)
        metadata, crop = {"status": "unavailable", "query": prompt, "candidates": [],
                          "interpretation": DISCLAIMER if prompt == "drawer" else TABLETOP_DISCLAIMER}, None
        receipt["error_type"] = type(exc).__name__
    if time.monotonic() >= deadline:
        case_timeout()
    receipt.update(status=metadata["status"], elapsed_seconds=time.monotonic()-start,
                   public_metadata=metadata)
    return metadata, crop, receipt


def current_drawer_proposals(rgb, png_path, settings, deadline):
    return _current_proposals(rgb, png_path, settings, deadline, "drawer")


def current_tabletop_proposals(rgb, png_path, settings, deadline):
    return _current_proposals(rgb, png_path, settings, deadline, "tabletop")
