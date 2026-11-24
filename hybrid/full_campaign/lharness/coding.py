"""Public image planning through the coding agent; inert on import.

The model is called only for an explicit recovery request, through
``lharness/model_api.py``, which you implement for your own API. No generated
Python is evaluated.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import struct
import time
import uuid
import zlib

import numpy as np

CAMERAS = ("agentview", "robot0_eye_in_hand", "sideview")
ROBOT_FIELDS = {
    "robot_cartesian_pos": 3, "robot_eef_quat": 4,
    "robot_gripper_pos": 2, "robot_joint_pos": 7,
}


def _process_start_ticks(pid):
    """Linux process identity: a reused PID cannot keep an abandoned ticket live."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return None
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _admission_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Shared coding admission reached the case deadline")


@contextmanager
def _admission_mutex(handle, deadline):
    while True:
        _admission_deadline(deadline)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(min(.01, max(0., deadline - time.monotonic())))
    try:
        _admission_deadline(deadline)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)


def _live_admission_tickets(queue, deadline):
    """Called only under the queue mutex; remove queue metadata, never processes."""
    live = []
    for path in sorted(queue.glob("[0-9]" * 20 + ".json")):
        _admission_deadline(deadline)
        try:
            if path.stat().st_size > 2048:
                raise ValueError("Oversized admission ticket")
            item = json.loads(path.read_text())
            alive = (type(item.get("pid")) is int and item["pid"] > 0
                     and type(item.get("start_ticks")) is int
                     and item["start_ticks"] == _process_start_ticks(item["pid"])
                     and type(item.get("deadline_monotonic")) in (int, float)
                     and np.isfinite(item["deadline_monotonic"])
                     and item["deadline_monotonic"] > time.monotonic()
                     and item.get("sequence") == int(path.stem))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, TypeError, AttributeError):
            alive = False
        if alive:
            live.append(path)
        else:
            path.unlink(missing_ok=True)
    return live


@contextmanager
def coding_slot(root, deadline, slots=2):
    """FIFO cross-process admission with a configured count of flock leases.

All campaign workers use the same explicit root. A persisted sequence counter
under a short mutex orders tickets; no arrival uses wall-clock ordering. Only
the live queue head may claim a lease. PID/start-ticks and fixed deadlines reap
abandoned waiters. The OS releases leases after crashes, and the ticket is
removed even on cancellation.
"""
    if type(slots) is not int or not 1 <= slots <= 8:
        raise ValueError("Shared coding admission limit must be an integer from 1 to 8")
    if type(deadline) not in (int, float) or not np.isfinite(deadline):
        raise ValueError("A finite monotonic admission deadline is required")
    _admission_deadline(deadline)
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    queue = directory / "tickets"
    queue.mkdir(exist_ok=True)
    handles = []
    mutex = None
    ticket = None
    counter_temp = None
    held = None
    try:
        handles = [(directory / ("slot_%d.lock" % index)).open("a+") for index in range(slots)]
        mutex = (directory / "queue.lock").open("a+")
        pid = os.getpid()
        start_ticks = _process_start_ticks(pid)
        if start_ticks is None:
            raise RuntimeError("Cannot identify admission ticket owner")
        with _admission_mutex(mutex, deadline):
            _live_admission_tickets(queue, deadline)
            counter = directory / "sequence.json"
            sequence = json.loads(counter.read_text()) if counter.exists() else 0
            if type(sequence) is not int or not 0 <= sequence < 10**20 - 1:
                raise ValueError("Invalid admission sequence counter")
            sequence += 1
            counter_temp = directory / ("sequence." + uuid.uuid4().hex + ".tmp")
            counter_temp.write_text(json.dumps(sequence) + "\n")
            counter_temp.replace(counter)
            counter_temp = None
            ticket = queue / (f"{sequence:020d}.json")
            with ticket.open("x") as stream:
                json.dump({"sequence": sequence, "pid": pid, "start_ticks": start_ticks,
                           "deadline_monotonic": deadline}, stream)
        while held is None:
            with _admission_mutex(mutex, deadline):
                live = _live_admission_tickets(queue, deadline)
                if live and live[0] == ticket:
                    for handle in handles:
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            held = handle
                            ticket.unlink(missing_ok=True)
                            break
                        except BlockingIOError:
                            pass
            if held is None:
                time.sleep(min(.01, max(0., deadline - time.monotonic())))
        _admission_deadline(deadline)
        yield
    finally:
        if ticket is not None:
            ticket.unlink(missing_ok=True)
        if counter_temp is not None:
            counter_temp.unlink(missing_ok=True)
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
        for handle in handles:
            handle.close()
        if mutex is not None:
            mutex.close()


def public_observation(observation):
    """Rebuild the public schema, dropping every unrecognized field recursively."""
    if not isinstance(observation, dict) or not isinstance(observation.get("instruction"), str):
        raise ValueError("Public observation requires an instruction")
    instruction = observation["instruction"]
    if not 0 < len(instruction) <= 8192:
        raise ValueError("Invalid instruction length")
    result = {"instruction": instruction}
    for key, size in ROBOT_FIELDS.items():
        if key not in observation:
            continue
        values = np.asarray(observation[key], dtype=float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError("Invalid public proprioception")
        result[key] = values.tolist()
    for camera in CAMERAS:
        if camera not in observation:
            continue
        raw = observation[camera]
        rgb = np.asarray(raw["images"]["rgb"])
        depth = np.asarray(raw["images"]["depth"], dtype=np.float32)
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
                or not 1 <= rgb.shape[0] <= 4096 or not 1 <= rgb.shape[1] <= 4096
                or depth.shape != rgb.shape[:2]):
            raise ValueError("Invalid public RGB-D frame")
        entry = {"images": {"rgb": rgb.copy(), "depth": depth.copy()}}
        for name, shape in (("intrinsics", (3, 3)), ("extrinsics", (4, 4))):
            if name in raw:
                matrix = np.asarray(raw[name], dtype=float)
                if matrix.shape != shape or not np.isfinite(matrix).all():
                    raise ValueError("Invalid public camera calibration")
                entry[name] = matrix.tolist()
        result[camera] = entry
    if not any(camera in result for camera in CAMERAS):
        raise ValueError("A public camera frame is required")
    return result


def _png(path, array):
    """Small dependency-free lossless RGB writer, used only for supplied frames."""
    array = np.ascontiguousarray(array, dtype=np.uint8)
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError("RGB image required")
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    scanlines = b"".join(b"\0" + row.tobytes() for row in array)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


SYSTEM_PROMPT = """You control a robot from the attached CURRENT RGB-D images and
robot proprioception. Return one JSON object, without Markdown or Python.
Use only the listed bounded operations or an approved parameterized recipe.
The instruction is the task. No benchmark identity, hidden object poses,
native predicates, rewards, simulator contacts, or per-instance lookup exists.
Select pixels in the ORIGINAL camera resolution; depth and calibration are
resolved by the controller. Never invent a world-coordinate target. Depth
visualizations are supplementary; RGB identifies objects and occlusion.
move_delta is a RELATIVE Cartesian displacement in METERS and RELATIVE
rotation in RADIANS, NOT a normalized control action. pixel_target.offset
is a small relative offset in meters from the CURRENT depth pixel, transformed
using camera-to-world calibration. A pixel_target MUST precede ANY operation
that advances simulation steps, including gripper, run_policy, move_delta, or
another pixel_target. Even opening the gripper counts as advancing steps.
An observe inside the same plan does NOT rebind its pixels to a fresh image.
To open while approaching, use gripper=-1 INSIDE pixel_target itself; do not
prepend a gripper operation. Use the approved grasp_observed_target recipe
when its visual preconditions hold and the fingers are already visibly open.
If opening must happen first, prefer bound_grasp: it binds the visible target
before opening and needs no extra planning call just to open. Only when choosing
the plain pixel_target interface without a bound visual target must a separate
opening plan end with next=reobserve before selecting a new pixel. At most one pixel_target is allowed per
plan. Read any validation_reason in previous_public_execution and correct that
specific grammar violation before proposing another plan.
For multi-phase manipulation, prefer bound_grasp or bound_place when their
visible preconditions hold. They bind a current RGB-D point BEFORE opening or
moving, then perform the bounded approach/contact/grip-or-release/retreat phases
within this one planning call. If public position feedback is missing or a
waypoint remains more than 0.04 m away, these recipes skip their remaining phases
and request fresh visual planning while preserving the current gripper command.
This is a gross motion check, not evidence of a grasp or placement.
Both bound recipes accept optional approach_steps, an integer 1..80 (default 40).
Pure translation still stops early on measured convergence. Public motion feedback
includes initial_position_error and position_error_reduction, the initial minus
final end-effector distance to that waypoint in meters; reduction may be negative.
When this distance is shrinking and fresh images show a clear route, bind a fresh
visual target and consider increasing approach_steps up to 80 within remaining_steps.
Without progress, do not merely add steps: first inspect the public views and
robot posture for an obstructed path or unsuitable approach/orientation.
All offsets are in WORLD-aligned meters relative
to that visually measured point; approach_offset is an independent offset from
the same point, not an increment added to contact_offset/offset. rotation is a
bounded relative robot rotation in radians. No world-coordinate target may be
provided. A bound point is not a tracked object pose: after manipulating or
occluding it, use fresh images in the next planning call to assess the result.
For a custom v2 plan, visual_targets is a list of {name,camera,pixel} definitions
from THIS image snapshot (at most four). move_bound_target takes target=name,
offset, optional rotation, steps and gripper. Definitions are resolved before
all actions and expire at the end of THIS plan. Never refer to a previous plan's
name without defining it again from the new images. Existing primitives and
bounded run_policy continuations may be combined with these moves.
reusable_experiences contains general public motion templates and cautions.
Use them only when current images justify their preconditions; they provide
neither remembered instance locations nor evidence that a grasp succeeded.
Use observe between uncertain phases, then stop the plan to request fresh
visual reasoning. Plan execution and observation do not themselves prove task
completion. Reuse approved recipes when applicable. Propose only a short
plan to improve the currently observed situation; keep necessary contact holds.
remaining_revisions includes THIS planning call. When it is 1, no later visual
planning call is available: next=reobserve cannot reserve another revision.
Use the remaining_steps budget for enough bounded operations to finish a
feasible interaction supported by the current images. For a visibly supported
placement, include the needed lowering and release, or use bound_place for the
complete approach/lower/release/retreat sequence. The place_held_object
recipe remains available when its direct-approach preconditions hold. Do not merely move
above a destination and defer the rest to an unavailable next call. Release is
not mandatory: if the current view does not justify it, preserve the appropriate
gripper state and choose a bounded action that the available evidence supports.
Schema: {"schema":"libero.public_plan.v2","hypothesis":"brief visual reason",
"operations":[{"op":"observe"}],"next":"reobserve"}.
Alternative: replace operations with "recipe":approved_id and "arguments":{}.
next is reobserve or finish; finish only stops this actor and is not a score.
Before choosing finish, check EVERY requested object and spatial relation against
the current views and state the visible evidence for each in the hypothesis.
Projected overlap does not prove that an object rests on another object, and
camera-image left/right does not by itself establish the task's spatial left/right.
If apparent completion is ambiguous or views disagree, use a justified existing
bounded action and fresh observation to clarify it within the remaining budget.
An observe alone changes no camera viewpoint. Do not infer completion or failure
from native results, hidden predicates, or the fact that recovery was invoked.
Each request is bounded: at most 12 operations, steps 1..80 each, delta/offset
components within +/-0.15 meters, rotation within +/-0.5 radians, gripper in
[-1,1] where -1 opens and +1 closes. Camera pixel coordinates are integer x,y.
"""


class CodingPlanner:
    def __init__(self, settings, output, backend_factory=None, on_queue_wait=None):
        self.settings = copy.deepcopy(settings)
        self.output = Path(output)
        self.backend_factory = backend_factory
        self._on_queue_wait = on_queue_wait
        self.queue_seconds = 0.0
        self.calls = 0

    def __call__(self, request):
        if self.settings.get("enabled") is False:
            raise RuntimeError("Coding transport is disabled")
        obs = public_observation(request["observation"])
        self.calls += 1
        directory = self.output / ("request_%02d" % self.calls)
        directory.mkdir(parents=True, exist_ok=False)
        content = []
        camera_metadata = {}
        for camera in CAMERAS:
            if camera not in obs:
                continue
            rgb = obs[camera]["images"]["rgb"]
            depth = obs[camera]["images"]["depth"]
            camera_metadata[camera] = {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])}
            rgb_path = directory / (camera + "_rgb.png")
            _png(rgb_path, rgb)
            content.extend([{"type": "text", "text": camera + " current RGB"},
                            {"type": "image_url", "image_url": {"url": str(rgb_path.absolute())}}])
            valid = np.isfinite(depth) & (depth > 0)
            normalized = np.zeros(depth.shape, dtype=np.uint8)
            if valid.any():
                lo, hi = float(np.min(depth[valid])), float(np.max(depth[valid]))
                normalized[valid] = (1 + 254 * np.clip((depth[valid] - lo) / max(hi-lo, 1e-6), 0, 1)).astype(np.uint8)
                camera_metadata[camera]["depth_visualization_m"] = {"near": lo, "far": hi, "black": "invalid"}
            depth_path = directory / (camera + "_depth.png")
            _png(depth_path, np.repeat(normalized[:, :, None], 3, axis=2))
            content.extend([{"type": "text", "text": camera + " current depth (near dark, far bright)"},
                            {"type": "image_url", "image_url": {"url": str(depth_path.absolute())}}])
        context = {
            "instruction": obs["instruction"], "cameras": camera_metadata,
            "robot": {key: obs[key] for key in ROBOT_FIELDS if key in obs},
            "remaining_steps": request["remaining_steps"],
            "remaining_revisions": request["remaining_revisions"],
            "skills": request["skills"], "general_lessons": request["general_lessons"],
            "reusable_experiences": request.get("reusable_experiences", []),
            "previous_public_execution": request.get("previous_public_execution", []),
        }
        content.insert(0, {"type": "text", "text": json.dumps(context, ensure_ascii=False, allow_nan=False)})
        factory = self.backend_factory
        if factory is None:
            from .model_api import CodingAgent
            factory = CodingAgent
        admission_root = self.settings.get("admission_root")
        if not admission_root:
            raise ValueError("A campaign-shared coding admission_root is required")
        deadline = self.settings.get("case_deadline_monotonic")
        if type(deadline) not in (int, float) or not np.isfinite(deadline):
            raise ValueError("A fixed case deadline is required for coding admission")
        if deadline <= time.monotonic():
            raise TimeoutError("Active case deadline exhausted before coding admission")
        concurrency = self.settings.get("max_inflight", self.settings.get("provider_max_inflight", 4))
        if concurrency != self.settings.get("provider_max_inflight", concurrency):
            raise ValueError("Outer coding admission and provider concurrency must match")
        queue_limit = self.settings.get("admission_wait_timeout_seconds", 7200)
        if type(queue_limit) not in (int, float) or not 0 < queue_limit <= 7200:
            raise ValueError("Coding queue wait must be bounded to at most 7200 seconds")
        queued_at = time.monotonic()
        admission = {"status": "waiting", "queue_wait_limit_seconds": queue_limit,
                     "queued_monotonic": queued_at, "max_inflight": concurrency}
        admission_path = directory / "admission.private.json"
        admission_path.write_text(json.dumps(admission, indent=2)+"\n")
        try:
            with coding_slot(admission_root, queued_at + queue_limit, slots=concurrency):
                admitted_at = time.monotonic()
                waited = admitted_at - queued_at
                self.queue_seconds += waited
                if self._on_queue_wait is not None:
                    self._on_queue_wait(waited)
                self.settings["case_deadline_monotonic"] = deadline + waited
                settings = dict(self.settings, io_dir=str(directory / "transport"))
                admission.update(status="admitted", queue_seconds=waited,
                                 admitted_monotonic=admitted_at)
                admission_path.write_text(json.dumps(admission, indent=2)+"\n")
                response = factory(settings).query([
                    {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content},
                ], json_mode=True)
        except BaseException:
            if admission["status"] == "waiting":
                admission.update(status="not_admitted", queue_seconds=time.monotonic()-queued_at)
                admission_path.write_text(json.dumps(admission, indent=2)+"\n")
            raise
        return response["content"]
