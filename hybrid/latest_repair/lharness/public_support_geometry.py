"""Conservative local surface evidence from the current public RGB-D only.

This is a negative geometric check for explicitly requested placement supports.
It never certifies a support, clearance, object identity, contact or retention.
Missing, discontinuous, non-planar or unstable patches remain uncertain.
"""
from __future__ import annotations

import numpy as np


RADII = (4, 6)
MIN_VALID_FRACTION = .95
MAX_ADJACENT_DEPTH_JUMP_M = .015
MIN_MINOR_SPREAD_M = .005
MAX_PLANE_RMS_M = .002
MAX_PLANE_RESIDUAL_M = .006
MAX_NORMAL_TO_MINOR_SPREAD_RATIO = .30
MAX_ABS_NORMAL_DOT_WORLD_UP = .15
MIN_CROSS_SCALE_NORMAL_ALIGNMENT = .985
PLACEMENT_TARGETS = {
    "bound_place": "anchor", "bound_transfer": "support",
    "bound_cavity_place": "interior",
}


def _uncertain(reason, scales=()):
    return {"status": "uncertain", "reason": reason, "scales": list(scales)}


def assess_support_surface(view, pixel):
    """Return finite scalar evidence, never point coordinates or simulator state."""
    try:
        depth = np.asarray(view["images"]["depth"], dtype=float)
        k = np.asarray(view["intrinsics"], dtype=float)
        t = np.asarray(view["extrinsics"], dtype=float)
        if (depth.ndim != 2 or k.shape != (3, 3) or t.shape != (4, 4)
                or not np.isfinite(k).all() or not np.isfinite(t).all()
                or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-5)
                or len(pixel) != 2 or any(type(v) is not int for v in pixel)):
            return _uncertain("invalid_public_geometry")
        x, y = pixel
        height, width = depth.shape
        if not (0 <= x < width and 0 <= y < height):
            return _uncertain("invalid_public_geometry")
        if not np.isfinite(depth[y, x]) or not 0 < depth[y, x] < 10:
            return _uncertain("center_depth_unavailable")
        scales, normals = [], []
        for radius in RADII:
            if x-radius < 0 or y-radius < 0 or x+radius >= width or y+radius >= height:
                return _uncertain("incomplete_neighborhood", scales)
            yy, xx = np.mgrid[y-radius:y+radius+1, x-radius:x+radius+1]
            z = depth[yy, xx]
            valid = np.isfinite(z) & (z > 0) & (z < 10)
            fraction = float(valid.mean())
            if fraction < MIN_VALID_FRACTION:
                return _uncertain("insufficient_valid_depth", scales)
            jumps = []
            for axis in (0, 1):
                adjacent = (valid[1:, :] & valid[:-1, :]) if axis == 0 else (valid[:, 1:] & valid[:, :-1])
                jumps.extend(np.abs(np.diff(z, axis=axis))[adjacent].tolist())
            if not jumps or max(jumps) > MAX_ADJACENT_DEPTH_JUMP_M:
                return _uncertain("depth_discontinuity", scales)
            rays = np.linalg.solve(k, np.stack([xx[valid], yy[valid], np.ones(valid.sum())]))
            points = (t[:3, :3] @ (rays*z[valid]) + t[:3, 3:4]).T
            centered = points - points.mean(axis=0)
            _, singular, axes = np.linalg.svd(centered, full_matrices=False)
            normal = axes[-1]
            spread = singular / np.sqrt(len(points))
            residual = np.abs(centered @ normal)
            values = [fraction, float(max(jumps)), float(spread[1]),
                      float(spread[2]), float(residual.max()), float(abs(normal[2]))]
            if not np.isfinite(values).all():
                return _uncertain("invalid_public_geometry", scales)
            scales.append(dict(zip(("valid_fraction", "max_adjacent_depth_jump_m",
                                    "minor_spread_m", "plane_rms_m", "plane_max_residual_m",
                                    "abs_normal_dot_world_up"), values), radius_px=radius))
            if spread[1] < MIN_MINOR_SPREAD_M:
                return _uncertain("insufficient_metric_extent", scales)
            if (spread[2] > MAX_PLANE_RMS_M or residual.max() > MAX_PLANE_RESIDUAL_M
                    or spread[2]/spread[1] > MAX_NORMAL_TO_MINOR_SPREAD_RATIO):
                return _uncertain("nonplanar_neighborhood", scales)
            normals.append(normal)
        alignment = float(abs(np.dot(normals[0], normals[1])))
        if alignment < MIN_CROSS_SCALE_NORMAL_ALIGNMENT:
            return _uncertain("cross_scale_normal_disagreement", scales)
        vertical = all(abs(normal[2]) <= MAX_ABS_NORMAL_DOT_WORLD_UP for normal in normals)
        return {"status": "near_vertical" if vertical else "not_near_vertical",
                "reason": "stable_local_plane", "scales": scales,
                "cross_scale_normal_alignment": alignment,
                "near_vertical_max_abs_normal_dot_world_up": MAX_ABS_NORMAL_DOT_WORLD_UP,
                "interpretation": "Local surface orientation only; no support, clearance, contact, retention or completion is certified."}
    except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return _uncertain("invalid_public_geometry")


def assess_placement_support(plan, observation):
    """Recipe semantics identify supports; raw moves/grasp/stroke are excluded."""
    name = PLACEMENT_TARGETS.get(plan.get("recipe"))
    if name is None:
        return None
    definition = next((v for v in plan.get("visual_targets", []) if v.get("name") == name), None)
    if definition is None:
        return _uncertain("support_definition_unavailable")
    result = assess_support_surface(observation.get(definition["camera"], {}), definition["pixel"])
    return {"target_role": "placement_support", "target_name": name, **result}
