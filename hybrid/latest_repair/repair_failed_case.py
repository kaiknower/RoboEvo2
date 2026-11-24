"""Targeted repair from a saved, observation-checked policy failure prefix.

Baseline files are read-only. Replayed-prefix successes are never attributed to
the repair harness. Diagnostic prefix-only runs are not benchmark outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


class PrefixMismatch(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def robot_state(obs):
    return np.r_[obs['robot_cartesian_pos'], obs['robot_eef_quat'], obs['robot_gripper_pos']]


class ReplayPrefixPolicy:
    def __init__(self, policy, actions, *, advance_model_rng=True):
        self.policy, self.recorded = policy, np.asarray(actions)
        self.advance_model_rng = advance_model_rng
        self.cursor = 0

    def reset(self, seed):
        self.cursor = 0
        self.policy.reset(seed)

    def actions(self, obs):
        if self.cursor >= len(self.recorded):
            return self.policy.actions(obs)
        if self.advance_model_rng:
            self.policy.actions(obs)
        chunk = self.recorded[self.cursor:self.cursor+5].copy()
        self.cursor += len(chunk)
        return chunk


class PrefixConsistency:
    def __init__(self, recorder, reference_states, prefix_steps, *, tolerance=1e-6):
        self.recorder = recorder
        self.reference = np.asarray(reference_states)
        self.prefix_steps = prefix_steps
        self.tolerance = tolerance
        self.seen = 0
        self.max_state_error = 0.0

    def __call__(self, obs, action, stage):
        self.recorder(obs, action, stage)
        if stage != 'policy':
            return
        if self.seen >= self.prefix_steps:
            raise PrefixMismatch('Policy replay exceeds its saved prefix')
        error = float(np.max(np.abs(robot_state(obs)-self.reference[self.seen])))
        self.max_state_error = max(self.max_state_error, error)
        self.seen += 1
        if not np.isfinite(error) or error > self.tolerance:
            raise PrefixMismatch('Robot observations do not reproduce the saved prefix')


def check_handoff(obs, source, checker, cameras, *, rgb_mae_limit=0.0,
                  rgb_max_abs_error=0, rgb_changed_fraction_limit=0.0,
                  sensor_confirmed_wrist_roundoff=False):
    import cv2
    if isinstance(rgb_mae_limit, bool) or rgb_mae_limit != 0.0:
        raise ValueError('Use explicit tiny-render max-error/fraction limits instead of relaxing RGB MAE')
    if type(rgb_max_abs_error) is not int or not 0 <= rgb_max_abs_error <= 1:
        raise ValueError('RGB max absolute error limit must be integer 0 or 1')
    if (isinstance(rgb_changed_fraction_limit, bool)
            or not isinstance(rgb_changed_fraction_limit, (int, float))
            or not np.isfinite(rgb_changed_fraction_limit)
            or not 0 <= rgb_changed_fraction_limit <= 0.0001):
        raise ValueError('RGB changed-channel fraction limit must be finite and between 0 and 0.0001')
    if checker.seen != checker.prefix_steps:
        raise PrefixMismatch('Incomplete policy prefix')
    images = {}
    for camera in cameras:
        path = Path(source)/'coding/request_01'/f'{camera}_rgb.png'
        expected = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if expected is None:
            raise PrefixMismatch('Original handoff camera image is unavailable')
        expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)
        current = np.asarray(obs[camera]['images']['rgb'])
        if expected.shape != current.shape:
            raise PrefixMismatch('Handoff image dimensions differ')
        if current.dtype != np.uint8:
            raise PrefixMismatch('Handoff RGB must be uint8')
        error = np.abs(expected.astype(np.int16)-current.astype(np.int16))
        nonzero = int(np.count_nonzero(error))
        fraction = nonzero/error.size
        maximum = int(error.max())
        exact = nonzero == 0
        accepted = maximum <= rgb_max_abs_error and fraction <= rgb_changed_fraction_limit
        images[camera] = dict(mean_abs_error=float(error.mean()), max_abs_error=maximum,
            total_channels=int(error.size), nonzero_channels=nonzero, changed_fraction=fraction,
            strictly_equal=exact, matched=accepted,
            match_type='exact' if exact else ('within_render_tolerance' if accepted else 'mismatch'))
    if type(sensor_confirmed_wrist_roundoff) is not bool:
        raise ValueError('Sensor-confirmed wrist rounding must be explicitly Boolean')
    matched = bool(images) and all(v['matched'] for v in images.values())
    exact = bool(images) and all(v['strictly_equal'] for v in images.values())
    result = dict(matched=matched, strictly_equal=exact, strict_rgb_matched=matched,
                match_type='exact' if exact else ('within_render_tolerance' if matched else 'mismatch'),
                checked_prefix_actions=checker.seen,
                max_robot_state_abs_error=checker.max_state_error,
                robot_state_tolerance=checker.tolerance,
                rgb_max_abs_error_limit=rgb_max_abs_error,
                rgb_changed_fraction_limit=float(rgb_changed_fraction_limit),
                cameras=images,
                method='Saved action replay and public robot/RGB consistency; no object-state truth used')
    if not matched and sensor_confirmed_wrist_roundoff:
        from lharness.prefix_sensor_evidence import confirm_wrist_roundoff
        evidence = confirm_wrist_roundoff(obs, source, cameras)
        result['sensor_evidence'] = evidence
        if evidence['matched']:
            result.update(matched=True, match_type='sensor_confirmed_wrist_roundoff',
                          method='Identical saved actions and public robot trace; joint, RGB and quantized depth evidence at handoff')
    return result


def load_source(target, config):
    source = Path(target['source_case'])
    for name, field in [('outcome.json', 'source_outcome_sha256'),
                        ('actions_proprioception.npz', 'source_actions_sha256')]:
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != target[field]:
            raise ValueError('Frozen baseline artifact changed')
    original = json.loads((source/'outcome.json').read_text())
    if original != target['source_outcome'] or original['native_success'] is not False or original['policy_success'] is not False:
        raise ValueError('Target must be an unchanged completed native failure')
    for name in ('group', 'condition', 'task_id', 'init_id', 'seed'):
        if original[name] != target[name]:
            raise ValueError('Target identity differs from baseline')
    count = original['policy_steps']
    if count != config['policy_caps'][target['condition'].split('_')[0]]:
        raise ValueError('Baseline policy prefix did not reach the frozen action cap')
    with np.load(source/'actions_proprioception.npz', allow_pickle=False) as data:
        if data['stages'][:count].tolist() != ['policy']*count:
            raise ValueError('Invalid saved policy prefix stages')
        actions, states = data['actions'][:count].copy(), data['states'][:count].copy()
    if actions.shape != (count, 7) or states.shape != (count, 9):
        raise ValueError('Invalid saved trajectory shape')
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError('Invalid saved trajectory values')
    return original, actions, states


class SeedOnlyPolicy:
    def reset(self, seed):
        from lharness.runtime_adapter import _seed_episode
        _seed_episode(seed)

    def actions(self, obs):
        raise RuntimeError('Prefix-only diagnostics cannot invoke the policy')

    def close(self):
        pass


def run_case(config_path, manifest_path, target_index, output, gpu_id, *, prefix_only=False):
    config = json.loads(Path(config_path).read_text())
    targets = json.loads(Path(manifest_path).read_text())['targets']
    if not 0 <= target_index < len(targets):
        raise ValueError('Unknown repair target')
    target = targets[target_index]
    original, actions, states = load_source(target, config)
    output = Path(output).resolve()
    if output.is_relative_to(Path(config['reference_output']).resolve()):
        raise ValueError('The frozen baseline is read-only')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu_id), MUJOCO_EGL_DEVICE_ID=str(gpu_id),
                      MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1')
    proxy = config.get('coding_agent', {}).get('proxy_url')
    if proxy:
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            os.environ[name] = proxy
        os.environ['NO_PROXY'] = os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
    from lharness.scheduler import _process_start_ticks
    from lharness.runtime_adapter import NativeSession, NativePolicy
    from lharness.evaluation import evaluate_episode, CompactRecorder
    from lharness.campaign import model_path
    from lharness.repair import recover
    identity = dict(pid=os.getpid(), start_ticks=_process_start_ticks(os.getpid()), gpu_id=str(gpu_id))
    atomic_json(output/'worker_identity.json', identity)
    atomic_json(output/'reference.private.json', dict(target=target, prefix_only=prefix_only,
        config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(), started_at_unix=time.time()))
    recorder = CompactRecorder(output, config)
    consistency = PrefixConsistency(recorder, states, len(actions))
    policy = session = None
    prefix = dict(status='replaying', matched=False)
    row = dict(status='unknown', native_success=None)
    try:
        policy = SeedOnlyPolicy() if prefix_only else NativePolicy(
            str(model_path(config, target['group'], target['condition'])), target['condition'],
            config['baseline_path'], 'cuda:0', output/'policy_runtime')
        atomic_json(output/'model_ready.json', dict(identity, ready=True, prefix_only=prefix_only))
        session = NativeSession(target['condition'], target['task_id'], target['init_id'], target['seed'],
                                output/'runtime', device='cuda:0', cameras=config['cameras'],
                                sensor_time_calibration=config.get('sensor_time_calibration', False),
                                recovery_extra_view=config.get('recovery_extra_view', False))
        replay = ReplayPrefixPolicy(policy, actions, advance_model_rng=not prefix_only)

        def repair_from_checked_prefix(controller, runtime_config, case_output):
            nonlocal prefix
            observation = controller.observe()
            if config.get('sensor_time_calibration', False):
                atomic_json(output/'sensor_time_calibration.private.json', session.camera_calibration_evidence())
            extra = {}
            if config.get('prefix_sensor_confirmed_wrist_roundoff', False):
                extra['sensor_confirmed_wrist_roundoff'] = config['prefix_sensor_confirmed_wrist_roundoff']
            if config.get('prefix_sensor_confirmed_wrist_roundoff', False) or config.get('sensor_time_calibration', False):
                from lharness.prefix_sensor_evidence import save_public_snapshot
                save_public_snapshot(output/'prefix_public_snapshot.npz', observation)
            prefix = check_handoff(observation, target['source_case'], consistency, config['cameras'],
                rgb_max_abs_error=config.get('prefix_rgb_max_abs_error', 0),
                rgb_changed_fraction_limit=config.get('prefix_rgb_changed_fraction_limit', 0.0), **extra)
            prefix['status'] = 'matched' if prefix['matched'] else 'handoff_mismatch'
            atomic_json(output/'prefix_reproduction.private.json', prefix)
            if not prefix['matched']:
                raise PrefixMismatch('Public images do not reproduce the saved handoff')
            if not prefix_only:
                return recover(controller, runtime_config, case_output)

        case = {k: target[k] for k in ('group', 'condition', 'task_id', 'init_id', 'seed')}
        episode_config = dict(config, policy_tail_after_harness=False) if prefix_only else config
        row = evaluate_episode(session, replay, episode_config, case, output,
                               repair_fn=repair_from_checked_prefix, recorder=consistency)
        if row.get('policy_success') is True:
            prefix.update(status='native_success_before_repair', matched=False)
        if prefix_only:
            row.update(status='prefix_validated' if prefix['matched'] else 'prefix_unreproduced',
                       native_success=None, harness_called=False)
        row.update(evaluation_mode='prefix_validation_only' if prefix_only else 'targeted_native_failure_repair',
                   prefix_reproduced=prefix['matched'],
                   repair_eligible=not prefix_only and prefix['matched'], harness_version=config['harness_version'],
                   prefix_validation_mode=prefix.get('match_type', 'not_reached'))
    except Exception as exc:
        import traceback
        (output/'worker_error.private.txt').write_text(traceback.format_exc())
        row.update(status='unknown', native_success=None, error_type=type(exc).__name__)
    finally:
        for resource in (recorder, session, policy):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    row.update(status='unknown', native_success=None, cleanup_error_type=type(exc).__name__)
        prefix.update(checked_prefix_actions=consistency.seen, max_robot_state_abs_error=consistency.max_state_error)
        atomic_json(output/'prefix_reproduction.private.json', prefix)
        atomic_json(output/'outcome.json', row)
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--target-index', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-id', required=True)
    p.add_argument('--prefix-only', action='store_true')
    args = p.parse_args()
    result = run_case(args.config, args.manifest, args.target_index, args.output, args.gpu_id,
                      prefix_only=args.prefix_only)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['status'] in {'completed', 'prefix_validated'} else 1)
