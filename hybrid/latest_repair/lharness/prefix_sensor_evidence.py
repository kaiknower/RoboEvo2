"""Public sensor evidence for sparse wrist RGB rounding at a saved handoff.

This optional rule supplements, never silently rewrites, the strict RGB test.
Original depth is a quantized preview, not retained raw metric depth. No native
object state, predicates, contact flags or instance identifiers are consulted.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = ("agentview", "robot0_eye_in_hand", "sideview")
ROBOT_SIZES = {"robot_cartesian_pos": 3, "robot_eef_quat": 4,
               "robot_gripper_pos": 2, "robot_joint_pos": 7}
WRIST_CHANGED_CHANNEL_LIMIT = 64


def original_public_context(source):
    contexts, hashes = [], {}
    for path in sorted((Path(source)/"coding/request_01/transport").glob("api/messages.json")):
        for message in json.loads(path.read_text()):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            texts = [content] if isinstance(content, str) else [p.get("text", "") for p in
                (content if isinstance(content, list) else []) if isinstance(p, dict)
                and p.get("type") in {"text", "input_text"}]
            for text in texts:
                try:
                    value = json.loads(text)
                except (TypeError, ValueError):
                    continue
                if (isinstance(value, dict) and isinstance(value.get("instruction"), str)
                        and isinstance(value.get("robot"), dict) and isinstance(value.get("cameras"), dict)):
                    contexts.append({k: value[k] for k in ("instruction", "robot", "cameras")})
                    hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not contexts or any(value != contexts[0] for value in contexts[1:]):
        raise ValueError("Original public sensor context missing or ambiguous")
    return contexts[0], hashes


def depth_preview(depth):
    """Match the recorded per-frame normalized depth preview exactly."""
    depth = np.asarray(depth, np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError("Finite positive metric depth is required")
    near, far = float(depth.min()), float(depth.max())
    image = (1 + 254*np.clip((depth-near)/max(far-near, 1e-6), 0, 1)).astype(np.uint8)
    return np.repeat(image[:, :, None], 3, axis=2), {"near": near, "far": far, "black": "invalid"}


def confirm_wrist_roundoff(obs, source, cameras):
    import cv2
    report = dict(matched=False, rule="public_sensor_confirmed_wrist_roundoff_v1",
        wrist_changed_channel_limit=WRIST_CHANGED_CHANNEL_LIMIT, rgb_max_abs_error_limit=1,
        scope="All 16 public robot components exact; three normalized depth previews and near/far exact; "
              "external RGB views exact; at most 64 wrist channels differ by one uint8 level.")
    try:
        if tuple(cameras) != CAMERAS:
            raise ValueError("The three configured cameras in native order are required")
        context, hashes = original_public_context(source)
        report["source_sha256"] = hashes
        if obs["instruction"] != context["instruction"]:
            raise ValueError("Instruction differs")
        report["robot_exact"] = {}
        for name, size in ROBOT_SIZES.items():
            current, old = np.asarray(obs[name]), np.asarray(context["robot"][name])
            same = (current.shape == old.shape == (size,) and np.isfinite(current).all()
                    and np.isfinite(old).all() and np.array_equal(current, old))
            report["robot_exact"][name] = bool(same)
            if not same:
                raise ValueError("Public robot observations differ")
        report["cameras"] = {}
        for camera in CAMERAS:
            images, metadata = obs[camera]["images"], context["cameras"][camera]
            current = np.asarray(images["rgb"])
            if current.dtype != np.uint8 or current.shape != (256, 256, 3):
                raise ValueError("Expected native uint8 RGB dimensions")
            saved = {}
            for kind in ("rgb", "depth"):
                path = Path(source)/"coding/request_01"/f"{camera}_{kind}.png"
                array = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if array is None or array.shape != current.shape:
                    raise ValueError("Original camera artifact missing or incompatible")
                saved[kind] = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
            difference = np.abs(current.astype(np.int16)-saved["rgb"].astype(np.int16))
            changed, maximum = int(np.count_nonzero(difference)), int(difference.max())
            preview, bounds = depth_preview(images["depth"])
            if preview.shape != current.shape:
                raise ValueError("RGB and depth dimensions differ")
            rgb_ok = (maximum <= 1 and changed <= WRIST_CHANGED_CHANNEL_LIMIT
                      if camera == "robot0_eye_in_hand" else changed == 0)
            depth_ok = np.array_equal(preview, saved["depth"])
            bounds_ok = metadata.get("depth_visualization_m") == bounds
            report["cameras"][camera] = dict(changed_channels=changed, max_abs_error=maximum,
                rgb_accepted=bool(rgb_ok), depth_preview_exact=bool(depth_ok), depth_bounds_exact=bool(bounds_ok))
            if not (rgb_ok and depth_ok and bounds_ok):
                raise ValueError("Public camera evidence differs beyond sparse wrist rounding")
        report["matched"] = True
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        report["reason"] = str(exc)
    return report


def save_public_snapshot(path, obs):
    arrays = {"instruction": np.asarray(obs["instruction"])}
    arrays.update({key: np.asarray(obs[key]) for key in ROBOT_SIZES})
    for camera in CAMERAS:
        for kind in ("rgb", "depth"):
            arrays[camera+"_"+kind] = np.asarray(obs[camera]["images"][kind])
    calibration_present = [(camera, key) for camera in CAMERAS for key in ("intrinsics", "extrinsics")
                           if key in obs[camera]]
    if calibration_present and len(calibration_present) != 2*len(CAMERAS):
        raise ValueError("Partial public camera calibration snapshot")
    for camera, key in calibration_present:
        arrays[camera+"_"+key] = np.asarray(obs[camera][key])
    np.savez_compressed(path, **arrays)


def load_public_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        expected = {"instruction", *ROBOT_SIZES, *(c+"_"+k for c in CAMERAS for k in ("rgb", "depth"))}
        calibration = {c+"_"+k for c in CAMERAS for k in ("intrinsics", "extrinsics")}
        if set(data.files) not in (expected, expected | calibration):
            raise ValueError("Unexpected public snapshot fields")
        obs = {key: data[key].copy() for key in ROBOT_SIZES}
        obs["instruction"] = str(data["instruction"].item())
        for camera in CAMERAS:
            obs[camera] = {"images": {kind: data[camera+"_"+kind].copy() for kind in ("rgb", "depth")}}
            if calibration <= set(data.files):
                for kind, shape in (("intrinsics", (3, 3)), ("extrinsics", (4, 4))):
                    array = data[camera+"_"+kind]
                    if array.shape != shape or not np.isfinite(array).all():
                        raise ValueError("Invalid public camera calibration snapshot")
                    obs[camera][kind] = array.copy()
    return obs
