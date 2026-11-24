"""Private LIBERO runtime and public-observation policy adapter.

Importing this module creates no simulator, model, CUDA context, or network request.
Use the configured LIBERO-PRO Python environment for actual inference.

Public cameras use upright, top-left-origin RGB and aligned optical-axis depth in
metres. ``intrinsics`` is OpenCV K and ``extrinsics`` maps OpenCV camera coordinates
(right, down, forward) to world. Robosuite's calibration helpers already use this
upright convention. For raw OpenGL image rows, K_raw = F_y @ K_public, where
F_y = [[1,0,0],[0,-1,H-1],[0,0,1]]; flipping the raw rows recovers K_public.
Do not flip K again. The policy horizontally flips the public RGB, matching
the LIBERO evaluator's raw_image[::-1, ::-1] exactly.
"""
from __future__ import annotations

from pathlib import Path
from functools import wraps
import hashlib
import os
import sys
from typing import Any, Sequence

import numpy as np

DEFAULT_CAMERAS = ("agentview", "robot0_eye_in_hand", "sideview")
SUITES = ("spatial", "object", "goal", "long")
CAPS = {"spatial": 220, "object": 280, "goal": 300, "long": 520}


def _condition_parts(condition: str) -> tuple[str, str]:
    suite, separator, axis = condition.partition("_")
    if not separator or suite not in SUITES or axis not in ("position", "task"):
        raise ValueError(f"Unknown LIBERO-PRO condition: {condition}")
    return suite, axis


def resolve_checkpoint(model_path: str | Path, condition: str,
                       baseline_path: str | Path) -> Path:
    """Resolve a direct export or PPO snapshot; object_task always selects SFT."""
    _condition_parts(condition)
    baseline = Path(baseline_path).expanduser().resolve()
    candidate = Path(model_path).expanduser().resolve()
    if condition == "object_task":
        candidate = baseline
    elif not (candidate / "model.safetensors").is_file():
        candidate = candidate / condition
    if not (candidate / "model.safetensors").is_file():
        raise FileNotFoundError(candidate / "model.safetensors")
    if not (candidate / "assets/roboevo2/libero_expert_v8/norm_stats.json").is_file():
        raise FileNotFoundError(candidate / "assets/roboevo2/libero_expert_v8/norm_stats.json")
    return candidate


def _setup_runtime(runtime_dir: str | Path) -> None:
    checkout = os.environ.get("LIBERO_PRO_ROOT")
    if checkout:
        source = str(Path(checkout).expanduser().resolve() / "src")
        if not (Path(source) / "libero_pro/runtime.py").is_file():
            raise FileNotFoundError("LIBERO_PRO_ROOT must point to the companion LIBERO-PRO checkout")
        if source not in sys.path:
            sys.path.insert(0, source)
    from libero_pro.runtime import setup_runtime
    setup_runtime(runtime_dir)


def _seed_episode(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)


def _create_native(condition: str, task_id: int, init_id: int,
                   cameras: Sequence[str]):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_pro.campaign import Condition
    from libero_pro.protocol import bddl_language
    from robosuite import macros

    if macros.IMAGE_CONVENTION != "opengl":
        raise RuntimeError("Expected robosuite IMAGE_CONVENTION='opengl'")
    suite = benchmark.get_benchmark(Condition.parse(condition).benchmark)()
    if suite.get_num_tasks() != 10:
        raise ValueError("Expected all ten tasks")
    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    if init_id >= len(states):
        raise ValueError("Requested held-out initial state is absent")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    instruction = bddl_language(bddl)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_names=list(cameras),
        camera_heights=256, camera_widths=256, camera_depths=True,
    )
    try:
        for camera in cameras:
            env.sim.model.camera_name2id(camera)
        env.env.modify_observable("robot0_joint_pos", "active", True)
    except BaseException:
        env.close()
        raise
    return env, states[init_id], instruction


def _camera_calibration(sim: Any, name: str, height: int, width: int):
    cam_id = sim.model.camera_name2id(name)
    fovy = sim.model.cam_fovy[cam_id]
    focal = 0.5 * height / np.tan(fovy * np.pi / 360)
    intrinsics = np.array([[focal, 0, width / 2],
                           [0, focal, height / 2], [0, 0, 1]])
    camera_to_world = np.zeros((4, 4))
    camera_to_world[:3, :3] = sim.data.cam_xmat[cam_id].reshape(3, 3)
    camera_to_world[:3, 3] = sim.data.cam_xpos[cam_id]
    camera_to_world[3, 3] = 1.0
    camera_to_world = camera_to_world @ np.diag([1.0, -1.0, -1.0, 1.0])
    return intrinsics, camera_to_world


def _metric_depth(sim: Any, depth: np.ndarray) -> np.ndarray:
    assert np.all(depth >= 0.0) and np.all(depth <= 1.0)
    extent = sim.model.stat.extent
    far = sim.model.vis.map.zfar * extent
    near = sim.model.vis.map.znear * extent
    metric = near / (1.0 - depth * (1.0 - near / far))
    return np.asarray(metric, dtype=np.float32)


def _public_observation(obs: dict[str, Any], instruction: str,
                        cameras: Sequence[str], sim: Any, *, camera_calibrations=None) -> dict[str, Any]:
    """Allowlist projection. No reward, predicate, object state, ID, or contact."""
    public = {
        "instruction": instruction,
        "robot_cartesian_pos": np.asarray(obs["robot0_eef_pos"]).copy(),
        "robot_eef_quat": np.asarray(obs["robot0_eef_quat"]).copy(),
        "robot_gripper_pos": np.asarray(obs["robot0_gripper_qpos"]).copy(),
        "robot_joint_pos": np.asarray(obs["robot0_joint_pos"]).copy(),
    }
    for name in cameras:
        rgb = np.asarray(obs[f"{name}_image"])
        depth = np.asarray(obs[f"{name}_depth"])
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError(f"Misaligned RGB-D for {name}")
        if rgb.dtype != np.uint8:
            raise TypeError(f"Expected uint8 RGB for {name}")
        if camera_calibrations is None:
            k, camera_to_world = _camera_calibration(sim, name, *rgb.shape[:2])
        else:
            k, camera_to_world = camera_calibrations[name]
        public[name] = {
            "images": {"rgb": rgb[::-1].copy(),
                       "depth": _metric_depth(sim, depth)[::-1].copy()},
            "intrinsics": k.copy(),
            "extrinsics": camera_to_world.copy(),
        }
    return public


class NativeSession:
    """Evaluator-owned simulator; never pass this object to the public actor.

    ``seed`` is the actual episode seed (42 + 1000*task_id + init_id), not a
    base seed. Construction is cheap; reset creates the simulator. The only
    success-bearing return is step's separate Boolean, for private evaluation.
    Ten settling actions are excluded from the native success latch. No action cap or automatic reset is imposed here:
    the evaluator owns prefix/recovery caps and continues the same state.
    """

    def __init__(self, condition: str, task_id: int, init_id: int, seed: int,
                 runtime_dir: str | Path, device: str = "cuda:0",
                 cameras: Sequence[str] = DEFAULT_CAMERAS, *, sensor_time_calibration: bool = False,
                 recovery_extra_view: bool = False):
        _condition_parts(condition)
        if type(task_id) is not int or not 0 <= task_id < 10:
            raise ValueError("task_id must be 0 through 9")
        if type(init_id) is not int or not 10 <= init_id < 20:
            raise ValueError("init_id must be a held-out state 10 through 19")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a nonnegative uint32 integer")
        if type(sensor_time_calibration) is not bool:
            raise ValueError("sensor_time_calibration must be Boolean")
        if type(recovery_extra_view) is not bool:
            raise ValueError("recovery_extra_view must be Boolean")
        cameras = tuple(cameras)
        if not cameras or len(set(cameras)) != len(cameras) or any(c not in DEFAULT_CAMERAS for c in cameras):
            raise ValueError("Use unique, preconfigured LIBERO camera names")
        self.condition, self.task_id, self.init_id, self.seed = condition, task_id, init_id, seed
        self.runtime_dir, self.device, self.cameras = Path(runtime_dir), device, cameras
        self._env = None
        self._ever_success = False
        self._ready = False
        self.sensor_time_calibration = sensor_time_calibration
        self._calibration_samples = {}
        self.recovery_extra_view = recovery_extra_view
        self._extra_view_captures = 0

    def reset(self) -> dict[str, Any]:
        if self._env is None:
            _setup_runtime(self.runtime_dir)
            self._env, self._init_state, self._instruction = _create_native(
                self.condition, self.task_id, self.init_id, self.cameras,
            )
            if self.recovery_extra_view:
                from .public_extra_view import append_fixed_camera
                native = self._env.env
                if getattr(native, '_xml_processor', None) is not None:
                    raise ValueError('Cannot replace an existing native XML processor')
                native.set_xml_processor(append_fixed_camera)
        _seed_episode(self.seed)
        self._env.seed(self.seed)
        self._env.reset()
        if self.recovery_extra_view:
            from .public_extra_view import NAME
            self._env.sim.model.camera_name2id(NAME)
        if self.sensor_time_calibration:
            self._install_camera_calibration_capture()
        obs = self._env.set_init_state(self._init_state)
        for _ in range(10):
            obs, _, _, _ = self._env.step([0, 0, 0, 0, 0, 0, -1])
        self._ever_success = False
        self._ready = True
        return self._public(obs)

    def capture_recovery_extra_view(self):
        """Trusted recovery/probe only; one fresh RGB-D render, no physics step.

        Original camera samples/timers and cached robot observations are not
        refreshed. Cross-camera simultaneity is not claimed.
        """
        if not self.recovery_extra_view or not self._ready:
            raise RuntimeError('Extra recovery view is not configured/ready')
        from .public_extra_view import NAME
        rgb, depth = self._env.sim.render(width=256, height=256,
                                         camera_name=NAME, depth=True)
        rgb, depth = np.asarray(rgb), np.asarray(depth)
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8 or depth.shape != (256, 256):
            raise ValueError('Invalid extra camera RGB-D render')
        k, pose = _camera_calibration(self._env.sim, NAME, 256, 256)
        view = dict(images=dict(rgb=rgb[::-1].copy(), depth=_metric_depth(self._env.sim, depth)[::-1].copy()),
                    intrinsics=k.copy(), extrinsics=pose.copy())
        self._extra_view_captures += 1
        directory = self.runtime_dir/'extra_recovery_views'
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/f'capture_{self._extra_view_captures:04d}.npz').open('xb') as stream:
            np.savez_compressed(stream, rgb=view['images']['rgb'], depth=view['images']['depth'],
                                intrinsics=k, extrinsics=pose)
        return view

    def _public(self, obs: dict[str, Any]) -> dict[str, Any]:
        if getattr(self, "sensor_time_calibration", False):
            calibrations = self._camera_calibrations_for(obs)
            return _public_observation(obs, self._instruction, self.cameras, self._env.sim,
                                       camera_calibrations=calibrations)
        return _public_observation(obs, self._instruction, self.cameras, self._env.sim)

    def _install_camera_calibration_capture(self) -> None:
        """Attach K/T to existing camera samples without adding sensor updates.

        The native setter validates a sensor by invoking it with {}. During
        installation alone, return the already cached image for that validation
        so it cannot add a render. A later genuine call with {} still renders.
        No force-update, forward, step, sensor timer or original cache mutation
        is introduced here. Only the original RGB sensor writes its depth cache.
        """
        from robosuite.utils.observables import NO_CORRUPTION, NO_FILTER, NO_DELAY

        native = self._env.env
        self._calibration_samples = {}
        installed = []

        def prepare(camera):
            observable = native._observables[camera + "_image"]
            depth_observable = native._observables[camera + "_depth"]
            for sensor in (observable, depth_observable):
                if (sensor._corrupter is not NO_CORRUPTION or sensor._filter is not NO_FILTER
                        or sensor._delayer is not NO_DELAY):
                    raise ValueError("Sensor-time calibration requires unmodified RGB-D sensors")
            if observable._sampling_timestep != depth_observable._sampling_timestep:
                raise ValueError("Sensor-time calibration requires paired RGB-D sampling")
            cached = np.asarray(observable.obs).copy()
            if cached.dtype != np.uint8 or cached.ndim != 3 or cached.shape[-1] != 3:
                raise ValueError("Cached camera RGB is unavailable for sensor installation")
            previous = observable._sensor
            original = getattr(previous, "_sensor_time_calibration_original", previous)
            if not callable(original) or not hasattr(original, "__modality__"):
                raise ValueError("Unexpected native camera sensor")
            installing = True
            count = 0

            @wraps(original)
            def wrapped(cache):
                nonlocal count
                if installing:
                    return cached.copy()
                rgb = original(cache)
                image = np.asarray(rgb)
                k, pose = _camera_calibration(self._env.sim, camera, *image.shape[:2])
                depth = np.asarray(cache[camera + "_depth"])
                if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
                        or depth.shape not in (image.shape[:2], (*image.shape[:2], 1))):
                    raise ValueError("Native camera sensor returned an invalid RGB-D pair")
                count += 1
                self._calibration_samples[camera] = dict(
                    sample_count=count, rgb=image.copy(), depth=depth.copy(),
                    intrinsics=np.asarray(k).copy(), extrinsics=np.asarray(pose).copy())
                return rgb

            wrapped._sensor_time_calibration_original = original
            try:
                native.modify_observable(camera + "_image", "sensor", wrapped)
            except BaseException:
                observable._sensor = previous
                raise
            finally:
                installing = False
            return observable, previous

        try:
            for camera in self.cameras:
                installed.append(prepare(camera))
        except BaseException:
            for observable, previous in installed:
                observable._sensor = previous
            self._calibration_samples = {}
            raise

    def _camera_calibrations_for(self, obs):
        calibrations = {}
        for camera in self.cameras:
            sample = self._calibration_samples.get(camera)
            if sample is None:
                raise ValueError("Missing sensor-time camera calibration")
            for suffix, field in (("_image", "rgb"), ("_depth", "depth")):
                value, paired = np.asarray(obs[camera + suffix]), sample[field]
                if value.dtype != paired.dtype or not np.array_equal(value, paired):
                    raise ValueError("Cached RGB-D does not match sensor-time camera calibration")
            calibrations[camera] = sample["intrinsics"], sample["extrinsics"]
        return calibrations

    def camera_calibration_evidence(self):
        """Private diagnostic of public camera data, without rendering or sampling."""
        if not self.sensor_time_calibration:
            raise ValueError("Sensor-time calibration is not enabled")
        result = dict(schema="libero.sensor_time_calibration.v1", cameras={},
            source="K/T copied immediately after each original native camera RGB-D render",
            robot_state_source="Unmodified native cached proprio; no simultaneous-proprio claim",
            additional_render_calls=0, additional_observable_updates=0)
        for camera in self.cameras:
            sample = self._calibration_samples.get(camera)
            if sample is None:
                raise ValueError("Missing sensor-time camera calibration")
            k, pose = _camera_calibration(self._env.sim, camera, *sample["rgb"].shape[:2])
            result["cameras"][camera] = dict(
                sampled_sensor_calls=sample["sample_count"],
                raw_rgb_shape=list(sample["rgb"].shape), raw_depth_shape=list(sample["depth"].shape),
                raw_rgb_sha256=hashlib.sha256(sample["rgb"].tobytes()).hexdigest(),
                raw_depth_sha256=hashlib.sha256(sample["depth"].tobytes()).hexdigest(),
                sample_intrinsics=sample["intrinsics"].tolist(), sample_extrinsics=sample["extrinsics"].tolist(),
                current_intrinsics=k.tolist(), current_extrinsics=pose.tolist(),
                intrinsics_max_abs_difference=float(np.max(np.abs(k-sample["intrinsics"]))),
                extrinsics_max_abs_difference=float(np.max(np.abs(pose-sample["extrinsics"]))))
        return result

    def step(self, action: Sequence[float]) -> tuple[dict[str, Any], bool]:
        if not self._ready:
            raise RuntimeError("Call reset before step")
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected one finite seven-dimensional LIBERO action")
        obs, _reward, _done, _info = self._env.step(action)
        current_success = bool(self._env.check_success())
        self._ever_success = self._ever_success or current_success
        return self._public(obs), bool(self._ever_success)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._ready = False


def policy_batch(raw_public: dict[str, Any], device: str = "cuda:0") -> dict[str, Any]:
    """Policy input derived exclusively from public fields."""
    import torch
    from rlinf.envs.libero.utils import quat2axisangle

    state = np.concatenate([
        raw_public["robot_cartesian_pos"],
        quat2axisangle(np.asarray(raw_public["robot_eef_quat"]).copy()),
        raw_public["robot_gripper_pos"],
    ]).astype(np.float32)
    if state.shape != (8,):
        raise ValueError("Policy state must contain xyz, axis-angle, two gripper joints")
    batch = {
        "main_images": torch.from_numpy(raw_public["agentview"]["images"]["rgb"][:, ::-1].copy())[None],
        "wrist_images": torch.from_numpy(raw_public["robot0_eye_in_hand"]["images"]["rgb"][:, ::-1].copy())[None],
        "states": torch.from_numpy(state)[None],
        "task_descriptions": [raw_public["instruction"]],
        "extra_view_images": None,
    }
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _load_policy(model_path: Path, condition: str, baseline_path: Path,
                 device: str, runtime_dir: Path):
    _setup_runtime(runtime_dir)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from libero_pro.training import build_config
    from rlinf.models.embodiment.openpi import get_model

    cfg = build_config(condition, baseline_path)
    config_path = runtime_dir.resolve() / "native_policy.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config_path)
    with initialize_config_dir(config_dir=str(config_path.parent), version_base="1.1"):
        resolved = compose(config_name="native_policy")
    resolved.actor.model.model_path = str(model_path)
    resolved.actor.model.openpi_data.norm_stats_path = str(
        model_path / "assets/roboevo2/libero_expert_v8/norm_stats.json"
    )
    if (resolved.actor.model.num_steps != 3 or resolved.actor.model.num_action_chunks != 5
            or not resolved.actor.model.add_value_head):
        raise ValueError("Unexpected native policy inference configuration")
    return get_model(resolved.actor.model).to(device).eval()


class NativePolicy:
    """Load once per condition, then reset(seed) at each episode before actions.

    The native loader performs checked full actor loading and sets up quantile
    unnormalization. Returned actions are already physical deltas; no clipping,
    delta conversion, gripper remapping, or second unnormalization is applied.
    ODE inference still samples its initial Gaussian noise, so seeding matters.
    """

    def __init__(self, model_path: str | Path, condition: str,
                 baseline_path: str | Path, device: str, runtime_dir: str | Path):
        self.model_path = resolve_checkpoint(model_path, condition, baseline_path)
        self.device = device
        self._model = _load_policy(self.model_path, condition, Path(baseline_path).resolve(),
                                   device, Path(runtime_dir))

    def reset(self, seed: int) -> None:
        _seed_episode(seed)

    def actions(self, raw_public: dict[str, Any]) -> np.ndarray:
        import torch
        batch = policy_batch(raw_public, self.device)
        with torch.inference_mode():
            actions, _ = self._model.predict_action_batch(batch, mode="eval", compute_values=False)
        result = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if result.ndim != 3 or result.shape[0] != 1 or result.shape[1] < 5 or result.shape[2] < 7:
            raise ValueError(f"Unexpected policy action shape: {result.shape}")
        result = result[0, :5, :7].copy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite policy actions")
        return result

    def close(self) -> None:
        self._model = None
