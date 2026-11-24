#!/usr/bin/env python3
"""Merge raw Fisher marginal shards into the factorized EWC NPZ format."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


METADATA = {"__batch_indices", "__total_size"}
EXPECTED_SITES = 10


def load_shard(path, batches):
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        sites = sorted(key.removesuffix("::u_total") for key in keys if key.endswith("::u_total"))
        expected = METADATA | {f"{site}::{kind}_total" for site in sites for kind in ("u", "v")}
        if (len(archive.files) != len(keys) or keys != expected or len(sites) != EXPECTED_SITES
                or any(not site for site in sites)):
            raise ValueError(f"{path} must contain complete marginal pairs for ten LoRA sites and both metadata fields")
        indices = archive["__batch_indices"]
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not indices.size
                or (indices < 0).any() or (indices >= batches).any()
                or np.unique(indices).size != indices.size):
            raise ValueError(f"Invalid or duplicate batch indices in {path}")
        size = archive["__total_size"]
        if size.ndim != 0 or size.dtype.kind not in "iu" or int(size) <= 0:
            raise ValueError(f"Invalid dense parameter count in {path}")
        total_size = int(size)
        marginals = {}
        inferred_size = 0
        for site in sites:
            u = archive[f"{site}::u_total"]
            v = archive[f"{site}::v_total"]
            if (u.ndim < 1 or v.ndim < 1 or u.shape[:-1] != v.shape[:-1] or not u.size or not v.size):
                raise ValueError(f"Unmatched or empty marginal dimensions at {path}: {site}")
            for marginal in (u, v):
                if (marginal.dtype.kind != "f" or marginal.dtype.itemsize != 8
                        or not np.isfinite(marginal).all() or (marginal < 0).any()):
                    raise ValueError(f"Invalid float64 Fisher marginal values at {path}: {site}")
            inferred_size += math.prod(u.shape[:-1]) * u.shape[-1] * v.shape[-1]
            marginals[site] = (u, v)
        if inferred_size != total_size:
            raise ValueError(f"Dense parameter count in {path} is {total_size}, expected {inferred_size}")
    return indices, total_size, marginals


def merge_shards(shards, batches):
    if batches < 1 or not shards:
        raise ValueError("A positive batch count and at least one shard are required")
    seen = np.zeros(batches, dtype=bool)
    totals = None
    total_size = None
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for path in sorted(shards):
            indices, size, marginals = load_shard(path, batches)
            if seen[indices].any():
                duplicates = indices[seen[indices]].tolist()
                raise ValueError(f"Overlapping Fisher batches in {path}: {duplicates}")
            seen[indices] = True
            if totals is None:
                totals, total_size = marginals, size
                continue
            if size != total_size or marginals.keys() != totals.keys():
                raise ValueError(f"Fisher shard site or dense parameter count mismatch: {path}")
            for site, (u_total, v_total) in totals.items():
                u, v = marginals[site]
                if u.shape != u_total.shape or v.shape != v_total.shape:
                    raise ValueError(f"Fisher marginal shape differs at {path}: {site}")
                u_total += u
                v_total += v
        if not seen.all():
            missing = np.flatnonzero(~seen)
            raise ValueError(f"Missing {missing.size} Fisher batches: {missing[:20].tolist()}")
        global_mean = sum(float(u.sum()) for u, _ in totals.values()) / total_size / batches
        if not math.isfinite(global_mean) or global_mean <= 0:
            raise ValueError(f"Expected a positive finite global Fisher mean, got {global_mean}")
        output = {}
        for site, (u_total, v_total) in totals.items():
            u = u_total / batches
            v = v_total / batches
            mass = np.maximum(u.sum(axis=-1), 1e-30)
            output[f"{site}::u"] = np.asarray(u / global_mean, dtype=np.float32)
            output[f"{site}::v"] = np.asarray(v, dtype=np.float32)
            output[f"{site}::mass"] = np.asarray(mass, dtype=np.float32)
    return output, global_mean, total_size


def publish(output, arrays):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--shard", type=Path, required=True, action="append")
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    shards = [path.expanduser().resolve() for path in args.shard]
    if len(set(shards)) != len(shards):
        parser.error("Each shard path must be distinct")
    if os.path.lexists(output):
        parser.error(f"Output already exists at {output}")
    arrays, global_mean, total_size = merge_shards(shards, args.batches)
    publish(output, arrays)
    print(json.dumps(dict(output=str(output), shards=len(shards), batches=args.batches,
                          sites=EXPECTED_SITES, total_size=total_size, global_mean=global_mean)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
