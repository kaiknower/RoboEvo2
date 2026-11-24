#!/usr/bin/env python3
"""Fold a trained checkpoint's LoRA adapters into its base weights.

The SFT v8 checkpoint (``pi05_libero_v8_lora``) is a full fine-tune: its freeze
filter is the default ``nnx.Nothing()``, so base weights and LoRA factors were
trained jointly. CPD and EWC take their frozen reference from "zero every
LoRA leaf", which on this checkpoint would give neither the trained policy nor
the pretrained base. Folding makes the adapter-off model equal the trained
policy exactly:

    W' = W + a @ b            (scaling = alpha/rank = 1.0 for every site here)
    then drop a, b            (CheckpointWeightLoader re-adds fresh LoRA leaves
                               through its missing_regex=".*lora.*" path)

A CPD run warm-started from the folded checkpoint therefore starts at the SFT v8
policy and uses it as the reference.

Usage:
  python scripts/merge_lora_into_base.py \
      --src /path/to/pi05_libero_v8_lora/params \
      --out /path/to/pi05_libero_v8_merged/params
"""

import argparse
import datetime
import json
import pathlib

import flax.traverse_util as traverse_util
import numpy as np
import orbax.checkpoint as ocp


def base_path_for(lora_a_path: str) -> str:
    """'<stem>lora_a' -> the base tensor it additively modifies (same map as the Fisher)."""
    stem = lora_a_path[: -len("lora_a")]
    return stem + "w" if stem.endswith("/") else stem.rstrip("_")


def fold_lora_site(w, a, b, path: str) -> tuple[np.ndarray, float]:
    """Fold one LoRA site and verify dense-plus-adapter algebra in float32."""
    w32 = np.asarray(w, dtype=np.float32)
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    dw = np.matmul(a32, b32)
    if dw.shape != w32.shape:
        raise ValueError(f"shape mismatch on {path}: dW {dw.shape} vs W {w32.shape}")

    merged = w32 + dw
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, *w32.shape[:-2], w32.shape[-2])).astype(np.float32)
    y_reference = np.einsum("n...i,...ij->n...j", x, w32) + np.einsum(
        "n...r,...rj->n...j",
        np.einsum("n...i,...ir->n...r", x, a32),
        b32,
    )
    y_merged = np.einsum("n...i,...ij->n...j", x, merged)
    max_error = float(np.max(np.abs(y_reference - y_merged)))
    if max_error > 1e-3:
        raise ValueError(f"fold algebra mismatch on {path}: max_abs_error={max_error:.6g}")
    return merged, max_error


def wrap_checkpoint_tree(params: dict) -> dict:
    """Add the top-level params item expected by openpi restore exactly once."""
    return params if "params" in params else {"params": params}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="…/<step>/params of the trained checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import openpi.models.model as _model

    params = _model.restore_params(args.src, restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params, sep="/")
    print(f"[merge] loaded {len(flat)} tensors from {args.src}")

    a_paths = sorted(k for k in flat if k.endswith("lora_a"))
    if not a_paths:
        raise SystemExit("no LoRA tensors in this checkpoint -- nothing to fold")

    merged = 0
    max_rel = 0.0
    max_algebra_error = 0.0
    for a_path in a_paths:
        b_path = a_path[:-1] + "b"
        w_path = base_path_for(a_path)
        if w_path not in flat:
            raise SystemExit(f"no base tensor for {a_path} (guessed {w_path})")
        a = np.asarray(flat[a_path], np.float32)
        b = np.asarray(flat[b_path], np.float32)
        w_original = np.asarray(flat[w_path])
        w = np.asarray(w_original, np.float32)
        folded, algebra_error = fold_lora_site(w, a, b, w_path)
        dw = folded - w
        rel = float(np.abs(dw).mean() / max(np.abs(w).mean(), 1e-12))
        max_rel = max(max_rel, rel)
        max_algebra_error = max(max_algebra_error, algebra_error)
        flat[w_path] = folded.astype(w_original.dtype)
        merged += 1
        print(f"  {w_path:52} |dW|/|W| = {rel:.4f}  algebra_err={algebra_error:.3e}")

    for a_path in a_paths:
        flat.pop(a_path)
        flat.pop(a_path[:-1] + "b")

    out_params = traverse_util.unflatten_dict(flat, sep="/")
    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(out, wrap_checkpoint_tree(out_params))

    persisted = _model.restore_params(out, restore_type=np.ndarray)
    persisted_flat = traverse_util.flatten_dict(persisted, sep="/")
    persisted_lora = [key for key in persisted_flat if "lora" in key.lower()]
    if persisted_lora:
        raise SystemExit(f"persisted checkpoint still contains LoRA leaves: {persisted_lora[:3]}")

    manifest = {
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": str(pathlib.Path(args.src).resolve()),
        "output": str(out),
        "folded_sites": merged,
        "max_algebra_error_float32": max_algebra_error,
        "max_mean_abs_relative_update": max_rel,
        "persisted_tensor_count": len(persisted_flat),
        "persisted_lora_leaves": 0,
    }
    manifest_path = out.parent / "MERGE_MANIFEST.json"
    manifest_tmp = out.parent / ".MERGE_MANIFEST.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_tmp.replace(manifest_path)
    print(f"[merge] folded {merged} LoRA sites (max |dW|/|W| = {max_rel:.4f}), "
          f"max algebra error = {max_algebra_error:.3e}, dropped {2 * merged} adapter tensors")
    print(f"[merge] wrote {out}")
    print(f"[merge] wrote {manifest_path}")


if __name__ == "__main__":
    main()
