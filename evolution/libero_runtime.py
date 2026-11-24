"""Native LIBERO-PRO rollout, repair and development hooks for evolution.run.

Run with the companion LIBERO-PRO GPU Python and LIBERO_PRO_ROOT set:
python -m evolution.libero_runtime REQUEST.json RESPONSE.json

The policy is a converted PyTorch directory with the matching normalization
assets. Specify condition, task_id and init_id in the case's libero object.
The built-in planner calls your model through query_model() in
hybrid/full_campaign/lharness/model_api.py; see "Connect your model" in the
repository README. ICL
demonstrations can supply local RGB image paths in frames or this module's
trajectory manifest. A trusted --planner module:factory accepts
(harness_state, route, demo) and returns a callable taking public observations,
budget and previous attempts. It returns {"skill": name, "arguments": {...}}.
Each invocation runs in its own process and records into a new directory.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from hashlib import sha256
import importlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from hybrid.full_campaign.lharness.controller import EpisodeStop, PublicController
from hybrid.full_campaign.lharness.model_api import query_model
from hybrid.full_campaign.lharness.runtime_adapter import CAPS, DEFAULT_CAMERAS, NativePolicy, NativeSession
from .evidence import action_digest, normalized_jerk
from .initial.skills import PRIMITIVES, compile_skill, execute_skill
from .integrate import validate_guidance, validate_name, validate_skill


SPLITS = {"train": range(10), "development": range(20, 30)}
CAMERAS = tuple(DEFAULT_CAMERAS)


def _read(path):
    return json.loads(Path(path).read_text())


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False)


def _save(path, value):
    with path.open("x") as stream:
        stream.write(_json(value) + "\n")


def _array_hash(values):
    digest = sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(_json([array.dtype.str, list(array.shape)]).encode() + b"\x00")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state(obs):
    q = np.asarray(obs["robot_eef_quat"], dtype=float)
    w = np.clip(q[3], -1., 1.)
    den = np.sqrt(1. - w * w)
    angle = np.zeros(3) if math.isclose(den, 0.) else q[:3] * (2 * math.acos(w)) / den
    return np.concatenate((obs["robot_cartesian_pos"], angle, obs["robot_gripper_pos"])).astype("float32")


def _boundary(session, obs):
    native = session._env.get_sim_state()
    visible = []
    for name in CAMERAS:
        camera = obs[name]
        visible.extend((camera["images"]["rgb"], camera["images"]["depth"],
                        camera["intrinsics"], camera["extrinsics"]))
    return dict(start_state=_array_hash([native]), observation=_array_hash(
        [*visible, obs["robot_joint_pos"], obs["robot_cartesian_pos"], obs["robot_eef_quat"],
         obs["robot_gripper_pos"], np.frombuffer(obs["instruction"].encode(), dtype=np.uint8)]))


class SplitSession(NativeSession):
    """Use the Hybrid sensor calibration/reset/step with training-side IDs.

    The Hybrid evaluator only admits the held-out initial states 10-19. Only the
    split admission changes here; the reset and native verifier are inherited.
    """
    def __init__(self, condition, task, init, seed, split, runtime_dir, device):
        suite, axis = condition.split("_")
        if suite not in CAPS or axis not in {"position", "task"}:
            raise ValueError("Unknown LIBERO-PRO condition")
        if type(task) is not int or task not in range(10):
            raise ValueError("Expected one of the ten LIBERO task IDs")
        if type(init) is not int or init not in SPLITS[split]:
            raise ValueError("LIBERO initial state belongs to a different split")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Invalid reset seed")
        self.condition, self.task_id, self.init_id, self.seed = condition, task, init, seed
        self.runtime_dir, self.device, self.cameras = runtime_dir, device, CAMERAS
        self._env = None
        self._ever_success, self._ready = False, False
        self.sensor_time_calibration, self._calibration_samples = True, {}


class LocalController(PublicController):
    def vla_act(self, prompt, steps):
        class PromptedPolicy:
            def actions(_, obs):
                return original.actions({**obs, "instruction": prompt})
        original = self._policy
        self._policy = PromptedPolicy()
        try:
            return self.run_policy(steps)
        finally:
            self._policy = original


class Capture:
    """Write synchronized pre-action training inputs in lossless bounded chunks."""
    def __init__(self, path, clock, first_observation):
        self.path, self.clock = path, clock
        self.times = [clock()]
        self.positions = [np.array(first_observation["robot_cartesian_pos"])]
        self.instruction = first_observation["instruction"]
        self.buffer, self.chunks, self.actions = [], [], []

    def __call__(self, obs, action, stage):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.buffer.append(dict(image=obs["agentview"]["images"]["rgb"][:, ::-1].copy(),
            wrist_image=obs["robot0_eye_in_hand"]["images"]["rgb"][:, ::-1].copy(),
            state=_state(obs), action=self.actions[-1], stage=stage, time=self.times[-1]))
        if len(self.actions) > 1:
            self.positions.append(np.array(obs["robot_cartesian_pos"]))
        self.times.append(self.clock())
        if len(self.buffer) >= 16:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        name = f"steps-{len(self.chunks):05}.npz"
        with (self.path / name).open("xb") as stream:
            np.savez_compressed(stream, **{key: np.stack([row[key] for row in self.buffer])
                                for key in self.buffer[0]})
        self.chunks.append(name)
        self.buffer = []

    def seal(self, final, success):
        self.flush()
        if self.actions:
            self.positions.append(np.array(final["robot_cartesian_pos"]))
        actions = np.asarray(self.actions, dtype=np.float32).reshape(-1, 7)
        np.save(self.path / "actions.npy", actions, allow_pickle=False)
        np.savez(self.path / "motion.npz", time=self.times, xyz=self.positions)
        path = self.path / "trajectory.json"
        _save(path, dict(schema="roboevo.libero.trajectory.v1", instruction=self.instruction,
            steps=len(actions), success=success, chunks=self.chunks, actions="actions.npy",
            motion="motion.npz", image_orientation="policy", action_reference="executed"))
        return path


def _chunk(directory, actions, label, hz):
    if len(actions) < 10:
        return {}
    block = np.asarray(actions[:10], dtype=np.float32)
    path = directory / f"{label}.npy"
    np.save(path, block, allow_pickle=False)
    return dict(actions=str(path), action_sha256=action_digest(block), action_horizon=10,
                control_hz=hz, control_interface="libero_osc_pose_7d")


def _image_part(image):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB").save(buf, format="PNG")
    return {"type": "png", "data": buf.getvalue()}


def _demo_parts(demo):
    """Show only the demonstration’s public frames, robot states and executed actions."""
    from PIL import Image
    frames = demo.get("frames")
    if frames:
        if not isinstance(frames, list):
            raise ValueError("Expected a list of RGB image paths")
        parts = []
        for index in np.linspace(0, len(frames) - 1, min(len(frames), 5), dtype=int):
            with Image.open(frames[index]) as image:
                parts.append(_image_part(np.asarray(image.convert("RGB"))))
        return parts
    path = Path(demo["trajectory"])
    manifest = _read(path)
    if (manifest["schema"] != "roboevo.libero.trajectory.v1" or not manifest["chunks"]
            or manifest["success"] is not True):
        raise ValueError("Supply a verified native training demonstration or sampled RGB frames")
    parts = []
    chunks = manifest["chunks"]
    for index in np.linspace(0, len(chunks)-1, min(len(chunks), 5), dtype=int):
        with np.load(path.parent / chunks[index], allow_pickle=False) as data:
            parts.append({"type": "text", "text": _json({"demo_robot_state": data["state"][-1].tolist(),
                          "demo_executed_action": data["action"][-1].tolist()})})
            parts.append(_image_part(data["image"][-1, :, ::-1]))
    return parts


def _planner_state(state, route, *, allow_edits=False):
    catalog = {}
    for name, spec in state["skills"].items():
        validate_name(name)
        validate_skill(spec)
        if route != "code_policy" and any(step["op"] == "vla_act" for step in spec["procedure"]):
            continue
        catalog[name] = {k: spec[k] for k in ("scope", "completion", "use_when", "arguments", "procedure")}
    lessons = [{k: row[k] for k in ("condition", "skill", "instruction")}
               for row in state.get("memory", {}).get("general", []) if row["skill"] in catalog]
    for lesson in lessons:
        validate_guidance(lesson)
    rules = [{k: row[k] for k in ("condition", "skill")}
             for row in state["routing"].get("guidance", []) if row["skill"] in catalog]
    return {"allow_skill_edits": allow_edits, "skills": catalog, "memory": {"general": lessons},
            "routing": {"guidance": rules, "routes": {route: state["routing"]["routes"][route]}}}


def _branch_plan(proposed, state, route, budget, *, allow_edits):
    if not isinstance(proposed, dict) or not {"skill", "arguments"} <= proposed.keys() <= {
            "skill", "arguments", "definition", "guidance"}:
        raise ValueError("Expected a typed skill invocation")
    if not allow_edits and ("definition" in proposed or "guidance" in proposed):
        raise ValueError("Evaluate the frozen harness with existing skills")
    name = proposed["skill"]
    validate_name(name)
    candidate = state
    if "definition" in proposed:
        spec = proposed["definition"]
        if not isinstance(spec, dict) or set(spec) != {"scope", "use_when", "completion", "arguments", "procedure"}:
            raise ValueError("Unexpected skill definition fields")
        validate_skill(spec, stored=False)
        candidate = {**state, "skills": {**state["skills"], name: spec}}
    if "guidance" in proposed:
        lesson = proposed["guidance"]
        validate_guidance(lesson)
        if lesson["skill"] != name:
            raise ValueError("Routing recommendation must name the invoked skill")
    ops = compile_skill(candidate, route, name, proposed["arguments"])
    if sum(op.get("steps", 0) for op in ops) > budget:
        raise ValueError("Plan exceeds action budget")
    return candidate


def _prior_experiences(request):
    prior = []
    for row in request.get("prior_experiences", []):
        if "task" in row and row["task"] != request["case"]["task"]:
            continue
        route = row.get("route")
        if route not in {"code", "code_icl", "code_policy"}:
            raise ValueError("Invalid route experience")
        context = {"route": route}
        lesson = row.get("guidance")
        if lesson:
            validate_guidance(lesson)
            context["guidance"] = {k: lesson[k] for k in ("condition", "instruction", "skill")}
        if context not in prior:
            prior.append(context)
    return prior[-6:]


def _call_model(prompt, parts):
    """Send the prompt and parts through query_model() and parse its JSON reply."""
    with tempfile.TemporaryDirectory(prefix="roboevo2-planner-") as scratch:
        blocks = []
        for index, part in enumerate(parts):
            if part["type"] == "png":
                path = Path(scratch) / f"image_{index:02}.png"
                path.write_bytes(part["data"])
                blocks.append({"type": "image_url", "image_url": {"url": str(path)}})
            else:
                blocks.append(part)
        reply = query_model([{"role": "system", "content": prompt},
                             {"role": "user", "content": blocks}], {})
    return json.loads(reply)


def vlm_planner(harness, route, demo):
    """Bind grounded typed skill proposals to public RGB and proprioception."""
    prompt = ("Select one current skill and its grounded typed arguments or revise a typed skill. "
              "Return JSON with skill and arguments only when allow_skill_edits is false. When it is true, "
              "a skill revision can also include definition with scope, "
              "use_when, completion, arguments and procedure, as in the skill catalog. An optional "
              "guidance object contains observable condition, instruction and skill (the selected name). "
              "Reusable edits and guidance must be generic: no world coordinates, literal target pixels, "
              "asset/instance IDs, paths, hidden state or case-specific strategy. Bind camera, pixel and "
              "language to $named arguments at each invocation. Ground pixels in the current RGB view. "
              "Reobserve after motion. Use observed changes to refine your next call. "
              "Gripper closure alone does not verify a grasp. Never emit a task verdict.\n" +
              _json({"route": route, "primitives": {k: v for k, v in PRIMITIVES.items()
                     if route == "code_policy" or k != "vla_act"}, **harness}))
    demo_parts = _demo_parts(demo) if route == "code_icl" else []
    def propose(request):
        obs = request["observation"]
        parts = [{"type": "text", "text": _json({
            "instruction": obs["instruction"], "remaining_steps": request["remaining_steps"],
            "previous": request["previous"], "prior_experiences": request["prior_experiences"],
            "robot": {k: np.asarray(obs[k]).tolist() for k in (
                "robot_cartesian_pos", "robot_eef_quat", "robot_gripper_pos", "robot_joint_pos")}})}]
        failure = request.get("failure_evidence")
        if failure is not None:
            parts.append({"type": "text", "text": "End of the original failed attempt: " +
                          _json({"robot_state": failure["state"], "last_executed_action": failure["action"]})})
            parts.extend([_image_part(failure["image"]), _image_part(failure["wrist_image"])])
        if demo_parts:
            parts.append({"type": "text", "text": "Demonstration frames in temporal order:"})
            parts.extend(demo_parts)
        for name in CAMERAS:
            parts.extend([{"type": "text", "text": f"Current {name} camera:"},
                          _image_part(obs[name]["images"]["rgb"])])
        return _call_model(prompt, parts)
    return propose


def run_request(request, *, planner_factory=vlm_planner, device="cuda:0"):
    stage, case = request["stage"], request["case"]
    if stage not in {"rollout", "repair", "evaluate"}:
        raise ValueError("This adapter handles rollout, repair and evaluate")
    split = case["split"]
    if split not in SPLITS or (stage != "evaluate" and split != "train"):
        raise ValueError("Native stage/split mismatch")
    lib = case.get("libero", case)
    condition, task, init, seed = lib["condition"], lib["task_id"], lib["init_id"], case["seed"]
    cap = lib.get("policy_steps", CAPS[condition.split("_")[0]])
    if type(cap) is not int or not 1 <= cap <= 1000:
        raise ValueError("Expected bounded positive policy steps")
    if stage in {"evaluate", "rollout"} and request["mode"] not in {"standalone", "hybrid"}:
        raise ValueError("Unknown execution mode")
    if stage == "rollout" and request["mode"] != "standalone":
        raise ValueError("Training failures must come from standalone policy execution")
    if stage == "evaluate" and (request["mode"] == "hybrid") != bool(request["harness"]):
        raise ValueError("Hybrid evaluation requires the selected harness snapshot")
    hybrid = stage == "evaluate" and request["mode"] == "hybrid"
    repair = stage == "repair"
    route = request["route"] if repair else "code_policy" if hybrid else None
    if route not in {None, "code", "code_icl", "code_policy"}:
        raise ValueError("Unknown route")
    if stage == "repair":
        scale = lib.get("task_scale_m")
        window = lib.get("jerk_window", 7)
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("A fixed positive task_scale_m is required to compare native repairs")
        if type(window) is not int or window < 4:
            raise ValueError("A common cubic jerk fit requires at least four samples")
    demo = request.get("demo")
    if (route == "code_icl") != (demo is not None):
        raise ValueError("Code+ICL requires one training demonstration")
    if demo is not None and (demo["split"], demo["task"]) != ("train", case["task"]):
        raise ValueError("Mismatched training demonstration")
    model = request["policy"]
    if repair and ((model is not None) != (route == "code_policy")):
        raise ValueError("Only the Code+Policy branch gets the frozen policy")
    path = Path(request["workdir"]).resolve() / "episode"
    path.mkdir(parents=True, exist_ok=False)
    identity = sha256(_json([request["round"], stage, case["id"], model,
                             request.get("harness"), route]).encode()).hexdigest()
    fork = lib.get("takeover_step", 0)
    if type(fork) is not int or not 0 <= fork < cap:
        raise ValueError("Invalid takeover step")
    cfg = dict(case=case["conditions"], condition=condition, task=task, init=init, seed=seed,
               verifier="LIBERO-native-success-v1", cap=cap,
               task_scale_m=lib.get("task_scale_m"), jerk_window=lib.get("jerk_window", 7))
    def conditions(offset):
        return sha256(_json({**cfg, "remaining": cap-offset}).encode()).hexdigest()
    with ExitStack() as stack:
        session = SplitSession(condition, task, init, seed, split, path/"runtime", device)
        stack.callback(session.close)
        raw = session.reset()
        bound = _boundary(session, raw)
        if raw["instruction"] != case["goal"]:
            raise ValueError("Case goal must match the native BDDL instruction")
        hz = float(session._env.env.control_freq)
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("Invalid native control frequency")
        policy = None
        if model:
            policy = NativePolicy(model, condition, model, device, path/"model")
            stack.callback(policy.close)
            policy.reset(seed)
        if repair and fork:
            failed = request["failure"]
            original = Path(failed["trajectory"])
            files = _read(original)
            with (original.parent / files["actions"]).open("rb") as stream:
                prefix = np.load(stream, allow_pickle=False)
            if len(prefix) <= fork:
                raise ValueError("Failure does not have the requested recovery boundary")
            for act in prefix[:fork]:
                raw, success = session.step(act)
                if success:
                    raise ValueError("Failed prefix replay reached native success")
            bound = _boundary(session, raw)
        if repair:
            expected = request["failure"]
            anchors = [expected, *expected.get("anchors", [])]
            if not any(all(a.get(k) == v for k, v in bound.items()) and a.get("conditions") == conditions(fork)
                       for a in anchors):
                raise ValueError("Measured replay/reset does not match the source policy failure")
        capture = Capture(path, lambda: float(session._env.sim.data.time), raw) if stage != "evaluate" else None
        remaining = cap - fork if repair else lib.get("recovery_steps", 400)
        if type(remaining) is not int or not 1 <= remaining <= 1000:
            raise ValueError("Expected bounded positive recovery steps")
        ctrl = LocalController(session, policy, raw, cameras=CAMERAS, policy_cap=cap,
                               recovery_cap=remaining, recorder=capture)
        programs, history, anchor = [], [], None
        prior_experiences = _prior_experiences(request)
        failure_evidence = None
        if repair:
            original = Path(request["failure"]["trajectory"])
            chunks = _read(original)["chunks"]
            if chunks:
                with np.load(original.parent / chunks[-1], allow_pickle=False) as last:
                    failure_evidence = {"state": last["state"][-1].tolist(),
                                        "action": last["action"][-1].tolist(),
                                        "image": last["image"][-1, :, ::-1].copy(),
                                        "wrist_image": last["wrist_image"][-1, :, ::-1].copy()}
        max_plans = lib.get("max_plans", 3)
        if type(max_plans) is not int or not 1 <= max_plans <= 100:
            raise ValueError("Expected 1 to 100 bounded planning iterations")
        success = direct = False
        try:
            if not repair and not success:
                if fork and stage == "rollout":
                    ctrl.run_policy(fork)
                    anchor = _boundary(session, ctrl.observe())
                ctrl.run_policy(ctrl.get_budget()["remaining_steps"])
            if route and not success:
                ctrl.begin_recovery()
                state = _read(request["harness"])
                if not isinstance(state.get("skills"), dict) or not state["skills"]:
                    raise ValueError("Empty selected harness skill library")
                propose = planner_factory(_planner_state(state, route, allow_edits=repair), route, demo)
                for _ in range(max_plans):
                    public = ctrl.observe()
                    remaining_steps = ctrl.get_budget()["remaining_steps"]
                    if not remaining_steps:
                        break
                    proposed = propose(dict(observation=public, remaining_steps=remaining_steps, previous=history,
                                            failure_evidence=failure_evidence, prior_experiences=prior_experiences))
                    try:
                        candidate = _branch_plan(proposed, state, route, remaining_steps, allow_edits=repair)
                        for op in compile_skill(candidate, route, proposed["skill"], proposed["arguments"]):
                            if op["op"] != "pixel_target":
                                continue
                            x, y = op["pixel"]
                            depth = public[op["camera"]]["images"]["depth"]
                            if not (0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]
                                    and 0 < depth[y, x] < 10):
                                raise ValueError("Current depth cannot ground target")
                    except (KeyError, TypeError, ValueError) as exc:
                        history.append(dict(error=type(exc).__name__, reason="Invalid skill or parameters"))
                        continue
                    state = candidate
                    programs.append(proposed)
                    history.append(dict(program=proposed, end_effector_before=public["robot_cartesian_pos"].tolist()))
                    execute_skill(ctrl, state, route, proposed["skill"], proposed["arguments"])
                    if "definition" in proposed:
                        propose = planner_factory(_planner_state(state, route, allow_edits=repair), route, demo)
        except EpisodeStop as stop:
            if stop.reason != "native_success":
                raise
            success = True
            direct = not repair and ctrl.evaluator_record()["stage"] == "policy"
        if stage == "evaluate":
            return dict(success=success, policy_success=direct, case_id=case["id"],
                        policy=model, harness=request["harness"])
        trajectory = capture.seal(ctrl.observe(), success)
        offset = fork if repair else 0
        receipt = dict(id=identity, task=case["task"], goal=case["goal"], case_id=case["id"],
            split=split, success=success, policy_success=direct, trajectory=str(trajectory),
            steps=len(capture.actions), status="completed", **bound, conditions=conditions(offset),
            **_chunk(path, capture.actions, "target", hz))
        if anchor and len(capture.actions) > fork:
            receipt["anchors"] = [dict(**anchor, conditions=conditions(fork), verified=True,
                                       **_chunk(path, capture.actions[fork:], "anchor-target", hz))]
        if repair:
            receipt.update(failure_id=request["failure"]["id"], route=route)
            program = path / "program.json"
            _save(program, programs)
            receipt["program"] = str(program)
            if success:
                receipt["skill_edits"] = {p["skill"]: p["definition"] for p in programs if "definition" in p}
                lessons = [p["guidance"] for p in programs if "guidance" in p]
                if lessons:
                    receipt["guidance"] = lessons[-1]
                window = lib.get("jerk_window", 7)
                if len(capture.times) < window:
                    receipt.update(rank_eligible=False, rank_reason="insufficient_motion_samples")
                else:
                    receipt["jerk"] = normalized_jerk(capture.times, capture.positions,
                        task_scale=lib["task_scale_m"], grid_dt=1/hz, window=window)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument("--planner", help="Trusted module:factory producing a public-observation planner")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    planner = vlm_planner
    if args.planner:
        module, name = args.planner.split(":")
        planner = getattr(importlib.import_module(module), name)
    _save(args.response, run_request(_read(args.request), planner_factory=planner, device=args.device))


if __name__ == "__main__":
    main()
