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
import stat
import struct
import time
import uuid
import zlib

import numpy as np

PREFIX_CAMERAS = ("agentview", "robot0_eye_in_hand", "sideview")
CAMERAS = PREFIX_CAMERAS + ("oppositeview",)
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


def _admission_control_path(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError("admission_control_file must be an absolute path")
    return Path(value)


def _controlled_admission_cap(path, slots):
    """Read one small owner-only data file; no cache, fallback, or code parsing."""
    if path is None:
        return slots
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                or not 0 < info.st_size <= 1024):
            raise ValueError("Admission control must be an owned0600 small regular file")
        raw = os.read(descriptor, 1025)
        if len(raw) > 1024:
            raise ValueError("Oversized admission control")
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate admission control field")
                result[key] = value
            return result
        control = json.loads(raw, object_pairs_hook=unique_object)
        if (not isinstance(control, dict) or set(control) != {"schema", "cap"}
                or control["schema"] != "libero.coding_admission.v1"
                or type(control["cap"]) is not int or not 1 <= control["cap"] <= 32):
            raise ValueError("Admission control requires exact schema and integer cap1..32")
        return min(slots, control["cap"])
    finally:
        os.close(descriptor)


@contextmanager
def coding_slot(root, deadline, slots=2, admission_control_file=None):
    """FIFO cross-process admission with a configured count of flock leases.

All campaign workers use the same explicit root. A persisted sequence counter
under a short mutex orders tickets; no arrival uses wall-clock ordering. Only
the live queue head may claim a lease. PID/start-ticks and fixed deadlines reap
abandoned waiters. The OS releases leases after crashes, and the ticket is
removed even on cancellation.
An optional owned0600 data file can lower or increase admission up to the
configured slots while existing waiters keep their tickets. It never revokes
held leases; corrupt/missing control fails closed for new admissions.
"""
    if type(slots) is not int or not 1 <= slots <= 32:
        raise ValueError("Shared coding admission limit must be an integer from 1 to 32")
    if type(deadline) not in (int, float) or not np.isfinite(deadline):
        raise ValueError("A finite monotonic admission deadline is required")
    _admission_deadline(deadline)
    control_path = _admission_control_path(admission_control_file)
    _controlled_admission_cap(control_path, slots)
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
        handles = [(directory / ("slot_%d.lock" % index)).open("a+")
                   for index in range(32 if control_path is not None else slots)]
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
                active_cap = _controlled_admission_cap(control_path, slots)
                live = _live_admission_tickets(queue, deadline)
                if live and live[0] == ticket:
                    candidates = handles[:active_cap]
                    if control_path is not None:
                        occupied = 0
                        for handle in handles:
                            try:
                                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                fcntl.flock(handle, fcntl.LOCK_UN)
                            except BlockingIOError:
                                occupied += 1
                        if occupied >= active_cap:
                            candidates = []
                    for handle in candidates:
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


def planner_depth_preview(depth, mapping="linear"):
    """Recovery-request display only; never mutate metric depth or prefix evidence.

    Linear encoding is the default. Opt-in log encoding uses
    the same valid pixels, bounds and resolution; it changes only PNG shades.
    """
    if mapping not in ("linear", "log"):
        raise ValueError("depth_preview_mapping must be linear or log")
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("A two-dimensional depth display is required")
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if not valid.any():
        return normalized, None
    lo, hi = float(np.min(depth[valid])), float(np.max(depth[valid]))
    metadata = {"near": lo, "far": hi, "black": "invalid"}
    if mapping == "linear":
        normalized[valid] = (1 + 254 * np.clip((depth[valid] - lo) / max(hi-lo, 1e-6), 0, 1)).astype(np.uint8)
    else:
        log_near, log_far = float(np.log(lo)), float(np.log(hi))
        span = max(log_far - log_near, 1e-6)
        levels = (np.log(np.asarray(depth[valid], dtype=float)) - log_near) / span
        normalized[valid] = (1 + 254 * np.clip(levels, 0, 1)).astype(np.uint8)
        metadata.update(
            mapping="natural_log_depth", log_near=log_near, log_span=span,
            encoding="uint8=floor(1+254*clip((ln(depth_m)-log_near)/log_span,0,1))",
            inverse_bin_lower_edge_m="exp(log_near+((uint8-1)/254)*log_span), for uint8>=1",
            pixel_registration="Unchanged resolution and pixel coordinates, paired with the current RGB and metric depth.",
            interpretation="Display shading only: darker means nearer, not contact, retention, or task completion.",
        )
    return normalized, metadata


def planner_camera_calibration(view):
    """Only serialize already-public calibration from this same RGB-D observation.

    No simulator access, resampling, stored camera assumptions, or object state.
    public_observation has validated matrix shapes and finite entries upstream.
    """
    present = [name in view for name in ("intrinsics", "extrinsics")]
    if any(present) and not all(present):
        raise ValueError("Planner requires paired public K and camera-to-world T")
    if not all(present):
        return {"calibration_available": False}
    k = np.asarray(view["intrinsics"], dtype=float)
    t = np.asarray(view["extrinsics"], dtype=float)
    if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(t).all():
        raise ValueError("Invalid public camera calibration")
    return {
        "calibration_available": True,
        "intrinsics": k.tolist(),
        "camera_to_world": t.tolist(),
        "pixel_convention": "top-left origin; u right, v down; optical-axis depth in metres",
        "camera_axes_world": {"right": t[:3, 0].tolist(),
                              "down": t[:3, 1].tolist(),
                              "forward": t[:3, 2].tolist()},
    }


SYSTEM_PROMPT = """You control a robot from the attached CURRENT RGB-D images and
robot proprioception. Return one JSON object, without Markdown or Python.
Use only the listed bounded operations or an approved parameterized recipe.
The instruction is the task. No benchmark identity, hidden object poses,
native predicates, rewards, simulator contacts, or per-instance lookup exists.
Select pixels in the ORIGINAL camera resolution; depth and calibration are
resolved by the controller. Never invent a world-coordinate target. Depth
visualizations are supplementary; RGB identifies objects and occlusion.
When camera metadata includes drawer_proposals, these are UNVERIFIED SAM masks,
not object identities or proof of the instructed drawer level. Compare all
original views with the instruction (top/middle/bottom, if specified), not just
the highest score. A crop shows the union of at most three candidates with
padding; it is not resized. Its origin is roi_xyxy[:2] in the ORIGINAL image.
Convert crop pixels by adding this origin before submitting an action pixel.
Candidate boxes are already in ORIGINAL camera pixels with exclusive right/bottom.
Never select a box center blindly: inspect the visible handle/contact surface,
occlusion and current RGB-D. Scores do not prove grasp, contact, or success.
Unavailable/empty proposals add no evidence; use the original views. These
proposals change no image axes or task-language spatial reference convention.
When camera metadata includes tabletop_proposals, up to three ORIGINAL-image
pixels lie inside an UNVERIFIED predicted tabletop mask with stated image-space
boundary margins. These are suggestions, not verified supports or mandatory targets.
Inspect current RGB-D and other views before choosing; reject points on the floor,
table edge, occluded regions, or inconsistent with the instructed relation. A mask
score or interior margin does not prove clearance, grasp, retention or completion.
The crop overlays CYAN predicted mask outlines and MAGENTA numbered suggestions.
You may choose another original-image pixel based on the predicted outline and
current RGB-D, but must satisfy the task-relative placement: the maximum interior
margin is not automatically the correct target. The original RGB is unchanged.
The crop uses the same original-pixel mapping as drawer crops. Empty/unavailable
proposals add no evidence; keep planning from the original images.
move_delta is a RELATIVE Cartesian displacement in METERS and RELATIVE
rotation in RADIANS, NOT a normalized control action. pixel_target.offset
is a small relative offset in meters from the CURRENT depth pixel, transformed
using camera-to-world calibration. A pixel_target MUST precede ANY operation
that advances simulation steps, including gripper, run_policy, vla_act, move_delta, or
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
Only bound_grasp and bound_place accept arguments.approach_steps, an integer
1..80 (default 40). bound_drawer_pull, bound_surface_stroke and bound_transfer
instead accept arguments.phase_steps.approach (integer 1..80, default 40);
bound_transfer uses arguments.phase_steps.carry for its support approach.
Never put approach_steps in these three recipes, even alongside
phase_steps, and never put phase_steps in bound_grasp or bound_place.
Use the exact argument names and phase keys listed for the selected recipe.
Pure translation still stops early on measured convergence. Public motion feedback
includes initial_position_error and position_error_reduction, the initial minus
final end-effector distance to that waypoint in meters; reduction may be negative.
When this distance is shrinking and fresh images show a clear route, bind a fresh
visual target and consider increasing its recipe-specific approach phase cap
up to 80 within remaining_steps, using the argument mapping above.
Without progress, do not merely add steps: first inspect the public views and
robot posture for an obstructed path or unsuitable approach/orientation.
An explicitly requested nonzero relative rotation is controlled using current
robot quaternion feedback toward a bounded relative goal. Public rotation_error,
initial_rotation_error and rotation_error_reduction are in radians. An unreached
rotation stops the remaining plan before subsequent close/release actions.
Reassess the visible route and wrist pose; never interpret this as grasp/contact.
A completed move_delta can still have nearly zero measured position progress;
do not assume it raised the gripper or created clearance. Read the public
position errors before repeating it or planning from its intended endpoint.
Camera metadata contains intrinsics K and camera_to_world T copied from the SAME
public RGB-D observation, when calibration_available is true. T maps OpenCV
camera axes (right, down, forward) to the robot's WORLD frame; its first three
rotation columns are also listed as camera_axes_world. These axes describe the
image, not an object's orientation, surface normal, drawer axis, or task left/right.
Convert a camera-frame relative vector to WORLD using T[:3,:3] only: never add
T's translation to a displacement. Image-right therefore need not be world +X,
and wrist-camera axes change with the current view. For a pure robot-relative
translation use move_delta: a pixel_target plus [0,0,dz] moves to that measured
point plus a height offset, NOT upward dz from the current end effector.
K maps camera coordinates
to pixels; metric depth is optical z, not distance along a unit ray. Pixel-derived
points remain bound by the controller, not supplied as absolute world targets.
If calibration is unavailable, do not assume a camera orientation. Infer a
bounded direction from current calibrated views and visible geometry; when
ambiguous, use fresh observation rather than inventing an axis. Calibration
identifies coordinates, not physical contact, retention, or successful closure.
move_delta.delta and recipe stroke vectors are WORLD-aligned displacements.
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
Division of work: use analytic primitives for visible target
staging, clearance, transport and release. For local contact, you may invoke
{"op":"vla_act","prompt":"grasp the visible white mug","steps":40}.
The prompt must be one short public instruction for the intended local attempt,
a single line of at most 400 characters. steps is 1..80 within remaining_steps.
This invokes the SAME frozen policy with only a temporary local instruction;
it does not change the episode task, camera inputs, policy weights or success rule.
Use current RGB-D to choose the intended object and favorable pre-contact pose.
Select the local contact phase from CURRENT RGB-D and public robot feedback,
not merely from the next verb in the task. A local prompt must not assert
"held" or "holding" unless recent public views credibly support that inference,
for example the intended object visibly moving with the fingers after a lift,
consistent with the measured end-effector motion and another available view.
Closed fingers, a close command, proximity, occlusion or a completed approach
alone do not establish retention. If the object remains visibly separate and
reachable, choose a pick/grasp attempt rather than a place/insert instruction
that presupposes it is carried. If visibility is insufficient, restage using
visible clearance or inspect another available view; do not invent retention
or issue a blind grasp. After that bounded attempt, reassess fresh views in
the next available original revision before choosing transport or release.
This is a public-evidence planning rule, not a holding detector, success signal,
or mandatory policy call; the same supported phase may use pure-code skills.
Any analytic staging in this plan must precede vla_act. After vla_act, only
observe is allowed, and the runtime returns to the next EXISTING planning
revision for fresh visual assessment. Use next=reobserve. No fallback policy
tail, extra planning call, contact-success flag, or task-success claim is granted.
A local call consumes its requested action budget; reaching the step limit
alone says nothing about grasp or placement. Re-ground and re-stage from the
new public views before any further contact attempt, within the same budget.
run_policy remains an optional continuation with the ORIGINAL full instruction.
Pure code, pure policy continuation, and staged code plus local policy are all
available; choose from the current evidence rather than forcing one mode.
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
next is reobserve or finish; finish is a stopping suggestion and is not a score.
If runtime policy continuation is enabled, a completed finish or zero-action plan
may be followed by at most 80 bounded policy steps within remaining_steps. This
scheduling rule does not judge task completion. If a planning revision remains,
use the resulting current images and public continuation receipt to replan.
A legal zero-action stopping suggestion uses operations:[{"op":"observe"}]
with next=finish; an empty operations list is invalid.
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
        self._depth_preview_mapping = self.settings.get("depth_preview_mapping", "linear")
        if self._depth_preview_mapping not in ("linear", "log"):
            raise ValueError("depth_preview_mapping must be linear or log")
        from .visual_proposals import proposal_settings
        self._visual_proposals = proposal_settings(self.settings.get("public_visual_proposals"))
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
        tabletop_camera = None
        if self._visual_proposals and self._visual_proposals.get("tabletop_enabled", False):
            from .visual_proposals import is_tabletop_instruction
            if is_tabletop_instruction(obs["instruction"]):
                tabletop_camera = "oppositeview" if "oppositeview" in obs else ("agentview" if "agentview" in obs else None)
        for camera in CAMERAS:
            if camera not in obs:
                continue
            rgb = obs[camera]["images"]["rgb"]
            depth = obs[camera]["images"]["depth"]
            camera_metadata[camera] = {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])}
            camera_metadata[camera].update(planner_camera_calibration(obs[camera]))
            if camera == 'oppositeview':
                camera_metadata[camera]['capture_context'] = (
                    'Fixed bench external camera, freshly rendered for this recovery revision with paired K/T. '
                    'Original views and robot proprioception retain their own cached sensor timing; '
                    'cross-view simultaneity, visibility of an interior, contact and retention are not asserted.')
            rgb_path = directory / (camera + "_rgb.png")
            _png(rgb_path, rgb)
            content.extend([{"type": "text", "text": camera + " current RGB"},
                            {"type": "image_url", "image_url": {"url": str(rgb_path.absolute())}}])
            normalized, depth_metadata = planner_depth_preview(depth, self._depth_preview_mapping)
            if depth_metadata is not None:
                camera_metadata[camera]["depth_visualization_m"] = depth_metadata
            depth_path = directory / (camera + "_depth.png")
            _png(depth_path, np.repeat(normalized[:, :, None], 3, axis=2))
            depth_label = (" current depth (near dark, far bright)" if self._depth_preview_mapping == "linear"
                           else " current depth (LOG display; near dark, far bright; mapping in camera metadata)")
            content.extend([{"type": "text", "text": camera + depth_label},
                            {"type": "image_url", "image_url": {"url": str(depth_path.absolute())}}])
            if self._visual_proposals and camera in PREFIX_CAMERAS:
                from .visual_proposals import is_drawer_instruction, current_drawer_proposals
                if is_drawer_instruction(obs["instruction"]):
                    receipt_path = directory / (camera + "_visual_proposals.actual.json")
                    try:
                        metadata, crop, receipt = current_drawer_proposals(
                            rgb, rgb_path, self._visual_proposals,
                            self.settings.get("case_deadline_monotonic"))
                    except TimeoutError as exc:
                        if hasattr(exc, "proposal_receipt"):
                            receipt_path.write_text(json.dumps(exc.proposal_receipt,
                                indent=2, allow_nan=False) + "\n")
                        raise
                    camera_metadata[camera]["drawer_proposals"] = metadata
                    if crop is not None:
                        crop_path = directory / (camera + "_drawer_proposals_crop.png")
                        _png(crop_path, crop)
                        content.extend([
                            {"type": "text", "text": camera + " UNVERIFIED drawer proposal union crop; "
                             "use camera metadata roi_xyxy origin to map crop pixels to ORIGINAL image pixels"},
                            {"type": "image_url", "image_url": {"url": str(crop_path.absolute())}}])
                    receipt_path.write_text(
                        json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            if camera == tabletop_camera:
                from .visual_proposals import current_tabletop_proposals
                receipt_path = directory / (camera + "_tabletop_proposals.actual.json")
                try:
                    metadata, crop, receipt = current_tabletop_proposals(
                        rgb, rgb_path, self._visual_proposals,
                        self.settings.get("case_deadline_monotonic"))
                except TimeoutError as exc:
                    if hasattr(exc, "proposal_receipt"):
                        receipt_path.write_text(json.dumps(exc.proposal_receipt, indent=2, allow_nan=False) + "\n")
                    raise
                camera_metadata[camera]["tabletop_proposals"] = metadata
                if crop is not None:
                    crop_path = directory / (camera + "_tabletop_proposals_crop.png")
                    _png(crop_path, crop)
                    content.extend([
                        {"type": "text", "text": camera + " UNVERIFIED tabletop mask union crop; candidate pixels are in the ORIGINAL image, not crop coordinates"},
                        {"type": "image_url", "image_url": {"url": str(crop_path.absolute())}}])
                receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
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
        control_file = self.settings.get("admission_control_file")
        _admission_control_path(control_file)
        queue_limit = self.settings.get("admission_wait_timeout_seconds", 7200)
        if type(queue_limit) not in (int, float) or not 0 < queue_limit <= 7200:
            raise ValueError("Coding queue wait must be bounded to at most 7200 seconds")
        queued_at = time.monotonic()
        admission = {"status": "waiting", "queue_wait_limit_seconds": queue_limit,
                     "queued_monotonic": queued_at, "max_inflight": concurrency}
        if control_file is not None:
            admission["admission_control_file"] = control_file
        admission_path = directory / "admission.private.json"
        admission_path.write_text(json.dumps(admission, indent=2)+"\n")
        try:
            with coding_slot(admission_root, queued_at + queue_limit, slots=concurrency,
                             admission_control_file=control_file):
                admitted_at = time.monotonic()
                waited = admitted_at - queued_at
                self.queue_seconds += waited
                if self._on_queue_wait is not None:
                    self._on_queue_wait(waited)
                self.settings["case_deadline_monotonic"] = deadline + waited
                settings = dict(self.settings, io_dir=str(directory / "transport"))
                settings.pop("admission_control_file", None)
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
