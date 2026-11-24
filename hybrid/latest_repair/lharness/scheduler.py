"""Conservative, CPU-only-by-default launcher for the LIBERO-PRO campaign."""
from __future__ import annotations

import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from .campaign import GROUPS, PROTOCOL
from .worker_environment import worker_environment
CONDITIONS = ("spatial_position", "spatial_task", "object_position", "object_task",
              "goal_position", "goal_task", "long_position", "long_task")
POLL_SECONDS = 30
IDLE_POLLS = 3


def parse_gpu_csv(text: str) -> dict[str, dict[str, float]]:
    """Parse nvidia-smi index, utilization %, and used memory MiB CSV."""
    rows: dict[str, dict[str, float]] = {}
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(row) != 3:
            raise ValueError(f"expected 3 CSV fields, got {len(row)}")
        idx, util, memory = (part.strip() for part in row)
        rows[idx] = {"utilization": float(util.rstrip("% ")), "memory_mib": float(memory.split()[0])}
    return rows


def gpu_snapshot() -> tuple[dict[str, dict[str, float]], dict[str, list[int]]] | None:
    """Return GPU metrics and active compute PIDs; None means status is unknown."""
    try:
        metrics = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        parsed: dict[str, dict[str, float]] = {}
        uuid_to_index: dict[str, str] = {}
        for row in csv.reader(metrics.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 5:
                raise ValueError("unexpected GPU metrics CSV")
            idx, uuid, util, memory, total = (part.strip() for part in row)
            parsed[idx] = {"utilization": float(util.rstrip("% ")), "memory_mib": float(memory.split()[0]),
                           "memory_total_mib": float(total.split()[0])}
            uuid_to_index[uuid] = idx
        pids_by_gpu: dict[str, list[int]] = {}
        for row in csv.reader(apps.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 2:
                raise ValueError("unexpected compute-app CSV")
            uuid, pid_text = (part.strip() for part in row)
            if uuid not in uuid_to_index or pid_text in ("N/A", "[Not Found]"):
                raise ValueError("unknown compute-app identity")
            pid = int(pid_text)
            if pid <= 0:
                raise ValueError("invalid compute-app PID")
            pids_by_gpu.setdefault(uuid_to_index[uuid], []).append(pid)
        return parsed, pids_by_gpu
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _is_complete(output: Path, group: str, condition: str) -> bool:
    marker = output / group / condition / "summary.json"
    try:
        data = json.loads(marker.read_text())
        return (data.get("status") == "complete" and data.get("episodes") == 100
                and data.get("group") == group and data.get("condition") == condition
                and data.get("unknown") == 0
                and data.get("protocol") == PROTOCOL)
    except (OSError, json.JSONDecodeError):
        return False


def _build_waves(jobs: list[dict[str, Any]], pending: list[int]) -> list[list[int]]:
    waves: list[list[int]] = []
    for idx in pending:
        gpu = jobs[idx]["gpu_id"]
        wave = next((candidate for candidate in waves
                     if len(candidate) < 8 and all(jobs[j]["gpu_id"] != gpu for j in candidate)), None)
        if wave is None:
            wave = []
            waves.append(wave)
        wave.append(idx)
    return waves


def _idle_service_policy(config: dict[str, Any]) -> tuple[dict[int, int], float]:
    """Read the explicit, process-incarnation-specific exception to strict idle admission."""
    entries = config.get("idle_service_pids", [])
    if not isinstance(entries, list):
        raise ValueError("idle_service_pids must be a list of {pid, start_ticks} objects")
    allowed: dict[int, int] = {}
    for entry in entries:
        if (not isinstance(entry, dict)
                or type(entry.get("pid")) is not int or entry["pid"] <= 0
                or type(entry.get("start_ticks")) is not int or entry["start_ticks"] <= 0
                or entry["pid"] in allowed):
            raise ValueError("idle_service_pids requires unique positive integer pid and start_ticks pairs")
        allowed[entry["pid"]] = entry["start_ticks"]
    minimum = config.get("minimum_free_memory_mib", 24000)
    if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
            or not math.isfinite(minimum) or minimum <= 0):
        raise ValueError("minimum_free_memory_mib must be a finite positive number")
    return allowed, float(minimum)


def _process_start_ticks(pid: int) -> int | None:
    """Read Linux /proc field 22; command names may contain spaces and parentheses."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _gpus_idle(gpu_ids: list[str], metrics: dict[str, dict[str, float]],
               pids_by_gpu: dict[str, list[int]], idle_service_pids: dict[int, int] | None = None,
               minimum_free_memory_mib: float = 24000) -> bool:
    """Keep strict admission unless an explicit list of existing idle services is supplied."""
    allowed = idle_service_pids or {}
    if not gpu_ids:
        return False
    for gpu in gpu_ids:
        metric = metrics.get(gpu)
        if metric is None:
            return False
        utilization = metric.get("utilization", math.nan)
        used = metric.get("memory_mib", math.nan)
        if not (math.isfinite(utilization) and 0 <= utilization < 5
                and math.isfinite(used) and used >= 0):
            return False
        pids = pids_by_gpu.get(gpu, [])
        if not allowed:
            if pids or used >= 1024:
                return False
        else:
            total = metric.get("memory_total_mib", math.nan)
            if not math.isfinite(total) or total - used < minimum_free_memory_mib:
                return False
            for pid in pids:
                if pid not in allowed or _process_start_ticks(pid) != allowed[pid]:
                    return False
    return True


def _wait_idle(gpu_ids: list[str], status: dict[str, Any], status_path: Path,
               idle_service_pids: dict[int, int] | None = None,
               minimum_free_memory_mib: float = 24000) -> None:
    consecutive = 0
    while consecutive < IDLE_POLLS:
        snap = gpu_snapshot()
        if snap is None:
            consecutive = 0
            status["gpu_state"] = "unknown"
            status["idle_polls"] = 0
        else:
            metrics, pids_by_gpu = snap
            free = _gpus_idle(gpu_ids, metrics, pids_by_gpu, idle_service_pids, minimum_free_memory_mib)
            consecutive = consecutive + 1 if free else 0
            status["gpu_state"] = "free" if free else "busy"
            status["idle_polls"] = consecutive
            status["gpu_metrics"] = {g: metrics.get(g) for g in gpu_ids}
        _atomic_json(status_path, status)
        if consecutive < IDLE_POLLS:
            time.sleep(POLL_SECONDS)


def run(config_path: str | Path, output: str | Path | None = None, execute: bool = False,
        wait_for_idle: bool = False, once: bool = False, resume: bool = False,
        adaptive_shards: bool = False, shard_count: int = 5,
        admission_seconds: float | None = None) -> dict[str, Any]:
    """Prepare the queue or run one worker per GPU, refilling each GPU independently."""
    if adaptive_shards:
        from .capacity_scheduler import run_capacity
        return run_capacity(config_path, output, execute=execute, resume=resume,
                            wait_for_idle=wait_for_idle, once=once, shard_count=shard_count,
                            admission_seconds=admission_seconds)
    if admission_seconds is not None:
        raise ValueError('admission_seconds requires adaptive_shards')
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    runtime_python = str(config["runtime_python"])
    root = Path(__file__).resolve().parents[1]
    output = Path(output or config["output_root"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    gpu_ids = [str(g) for g in config.get("gpu_ids", list(range(8)))]
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        raise ValueError("gpu_ids must contain exactly eight unique GPU identifiers")
    jobs = [{"group": group, "condition": condition, "gpu_id": gpu_ids[i % 8],
             "command": [runtime_python, str(root / "run.py"), "worker", "--config", str(config_path),
                         "--group", group, "--condition", condition, "--output", str(output)]}
            for i, (group, condition) in enumerate((g, c) for g in GROUPS for c in CONDITIONS)]
    if resume:
        for job in jobs:
            job["command"].append("--resume")
    status_path = output / ("campaign_status.json" if execute else "queue_preview.json")
    status: dict[str, Any] = {"status": "prepared", "execute": bool(execute), "resume": resume,
                              "output": str(output), "jobs": []}
    for job in jobs:
        job_status = {k: job[k] for k in ("group", "condition", "gpu_id")}
        if _is_complete(output, job["group"], job["condition"]):
            job_status["status"] = "complete_skipped"
        else:
            job_status["status"] = "queued" if not execute else "pending"
            job_status["command"] = job["command"]
        status["jobs"].append(job_status)
    if not execute:
        _atomic_json(status_path, status)
        return status

    idle_service_pids, minimum_free_memory_mib = _idle_service_policy(config)
    lock_path = output / "campaign.lock"
    lock_file = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"campaign already locked: {lock_path}") from exc
        if resume and status_path.exists():
            history = output / "scheduler_history"
            history.mkdir(exist_ok=True)
            shutil.copy2(status_path, history / f"campaign_status-{time.time_ns()}.json")
        _atomic_json(status_path, status)
        for js in status["jobs"]:
            if js["status"] != "pending":
                continue
            attempt_dir = output / "attempts" / js["group"] / js["condition"]
            prior = sorted(attempt_dir.glob("attempt-*.json")) if attempt_dir.exists() else []
            if prior and not resume:
                js["status"] = "prior_attempt_unknown"
                continue
            js["attempt"] = max([int(p.stem.split("-")[-1]) for p in prior] or [0]) + 1
            if prior:
                history = attempt_dir / "retry_history" / f"{time.time_ns()}-{os.getpid()}"
                history.mkdir(parents=True, exist_ok=False)
                for path in prior + ([attempt_dir / "worker.log"] if (attempt_dir / "worker.log").exists() else []):
                    shutil.move(str(path), str(history / path.name))
                _atomic_json(history / "manifest.json", {"reason": "explicit_infrastructure_resume",
                    "previous_attempts": [p.name for p in prior], "next_attempt": js["attempt"]})
        _atomic_json(status_path, status)
        pending = [i for i, j in enumerate(status["jobs"]) if j["status"] == "pending"]
        if once and pending:
            pending = _build_waves(status["jobs"], pending)[0]
        children: dict[str, tuple[int, Any]] = {}
        idle_polls = {g: 0 for g in gpu_ids}
        next_check = {g: 0.0 for g in gpu_ids}
        while pending or children:
            for gpu, (i, proc) in list(children.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                js = status["jobs"][i]
                js["status"] = "complete" if _is_complete(output, js["group"], js["condition"]) else "failed_unknown"
                js["returncode"] = rc
                attempt = output / "attempts" / js["group"] / js["condition"] / f"attempt-{js['attempt']}.json"
                _atomic_json(attempt, {"status": js["status"], "returncode": rc, "pid": js["pid"], "command": js["command"]})
                del children[gpu]
                idle_polls[gpu], next_check[gpu] = 0, 0.0
            available = []
            seen = set(children)
            for i in pending:
                js = status["jobs"][i]
                gpu = js["gpu_id"]
                if gpu not in seen and time.monotonic() >= next_check[gpu]:
                    available.append(i)
                    seen.add(gpu)
            snap = gpu_snapshot() if available else None
            if available and not wait_for_idle and (snap is None or any(
                    not _gpus_idle([status["jobs"][i]["gpu_id"]], *snap, idle_service_pids, minimum_free_memory_mib)
                    for i in available)):
                status["gpu_state"] = "unknown" if snap is None else "busy"
                for i in pending:
                    status["jobs"][i]["status"] = "gpu_unknown" if snap is None else "gpu_busy"
                pending, available = [], []
            for i in available:
                js = status["jobs"][i]
                gpu = js["gpu_id"]
                idle = snap is not None and _gpus_idle([gpu], *snap, idle_service_pids, minimum_free_memory_mib)
                status["gpu_state"] = "unknown" if snap is None else ("free" if idle else "busy")
                idle_polls[gpu] = idle_polls[gpu] + 1 if idle else 0
                status["idle_polls_by_gpu"] = dict(idle_polls)
                next_check[gpu] = time.monotonic() + POLL_SECONDS
                if not idle or (wait_for_idle and idle_polls[gpu] < IDLE_POLLS):
                    if not wait_for_idle:
                        js["status"] = "gpu_unknown" if snap is None else "gpu_busy"
                        pending.remove(i)
                    continue
                worker_lock = output / js["group"] / js["condition"] / "worker.lock"
                if worker_lock.exists():
                    with worker_lock.open("a") as worker_guard:
                        try:
                            fcntl.flock(worker_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            js["status"] = "worker_already_running"
                            pending.remove(i)
                            continue
                pending.remove(i)
                attempt_dir = output / "attempts" / js["group"] / js["condition"]
                attempt_dir.mkdir(parents=True, exist_ok=True)
                attempt = attempt_dir / f"attempt-{js['attempt']}.json"
                _atomic_json(attempt, {"status": "launching", "command": js["command"]})
                child_env = worker_environment(config, CUDA_VISIBLE_DEVICES=js['gpu_id'],
                    MUJOCO_EGL_DEVICE_ID=js['gpu_id'], MUJOCO_GL='egl',
                    PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1')
                with (attempt_dir / "worker.log").open("ab") as log:
                    proc = subprocess.Popen(js["command"], env=child_env, stdout=log, stderr=subprocess.STDOUT)
                js["status"] = "running"
                js["pid"] = proc.pid
                status["status"] = "running"
                _atomic_json(attempt, {"status": "running", "pid": proc.pid, "command": js["command"]})
                children[gpu] = (i, proc)
                _atomic_json(status_path, status)
            status["status"] = "running" if children else "waiting_for_idle"
            _atomic_json(status_path, status)
            if pending or children:
                time.sleep(1 if children else POLL_SECONDS)
        status["status"] = "finished" if all(j["status"] in ("complete", "complete_skipped") for j in status["jobs"]) else "needs_attention"
        _atomic_json(status_path, status)
        return status
    except KeyboardInterrupt:
        status["status"] = "interrupted_unknown"
        _atomic_json(status_path, status)
        raise
    finally:
        lock_file.close()
