#!/usr/bin/env python3
"""Estimate independent complete Fisher batches on one GPU per worker."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from run import parser


def main():
    cli = parser()
    cli.description = __doc__
    args = cli.parse_args()
    if args.command != "fisher" or args.recipe != "pi05_libero_v8_ewc_lora":
        cli.error("Expected the rank-factorized LIBERO Fisher recipe")
    here = Path(__file__).resolve().parent
    worker = [sys.executable, "-u", str(here / "checked_fisher.py"), *sys.argv[1:]]
    if args.check_only:
        subprocess.run(worker, check=True)
        return
    if args.out.suffix != ".npz" or os.path.lexists(args.out):
        cli.error(f"Use a fresh Fisher .npz output path: {args.out}")
    steps = args.steps or 500
    devices = [device.strip() for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if not all(devices) or not 2 <= len(set(devices)) == len(devices) <= steps:
        cli.error("Set CUDA_VISIBLE_DEVICES to distinct GPUs, at least two and at most the batch count")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="independent-", dir=args.out.parent))
    workers = []
    try:
        for index, gpu in enumerate(devices):
            shard = attempt / f"shard-{index}.npz"
            log = attempt / f"shard-{index}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                   "G2E_FISHER_SHARD": f"{index}/{len(devices)}"}
            env.pop("G2E_FISHER_PROBE_BATCH", None)
            with log.open("xb") as stream:
                process = subprocess.Popen(
                    [*worker, "--out", str(shard)],
                    env=env, stdout=stream, stderr=subprocess.STDOUT,
                )
            workers.append((process, log, shard))
            print(f"Fisher shard {index} on GPU {gpu}, PID {process.pid}, log {log}", flush=True)
        failure = []
        for process, log, _ in workers:
            code = process.wait()
            if code:
                failure.append(f"{log}: exit {code}")
        if failure:
            raise RuntimeError("Fisher shard failures: " + ", ".join(failure))
        merge = [sys.executable, "-u", str(here / "merge_fisher_shards.py"),
                 "--output", str(args.out), "--batches", str(steps)]
        for _, _, shard in workers:
            merge.extend(("--shard", str(shard)))
        subprocess.run(merge, check=True)
    finally:
        for process, _, _ in workers:
            if process.poll() is None:
                process.terminate()
        for process, _, _ in workers:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
