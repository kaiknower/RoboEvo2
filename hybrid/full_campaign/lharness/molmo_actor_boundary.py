"""Public actor API/data projection, not a sandbox for malicious Python.

Native judging/recording stays outside this mapping. Only configured public
cameras and explicit robot/task/time fields cross it. Bound callables are
captured from the RATs functions map.
"""
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
import inspect
import numbers
import numpy as np


class ActorBoundaryError(ValueError):
    pass


ORACLES = frozenset({
    'get_native_success', 'judge_success', 'case_status', 'case_begin', 'case_end',
    'get_reward', 'compute_reward', 'task_completed', 'get_task_info',
    'get_object_pose', 'get_object_position', 'describe_object_relation',
    'get_contacts', 'get_contact', 'is_holding', 'is_grasped', 'get_state_trace',
    'get_sim_state', 'get_joint_state', 'get_target_joint',
})
ROBOT_FIELDS = frozenset({
    'robot_joint_pos', 'robot_joint_vel', 'robot_cartesian_pos',
    'robot_base_pose', 'robot_gripper_pos', 'gripper_width', 'gripper_position',
})
TIME_FIELDS = frozenset({'time', 'timestamp', 'sim_time', 'step_count'})
TASK_FIELDS = frozenset({'instruction', 'task_id', 'task_type'})
VISUAL_FIELDS = frozenset({'verified', 'confidence', 'actual', 'reasoning',
                           'answer', 'holding', 'grasped', 'attached'})
RETURN_KINDS = {
    'get_observation': 'observation', 'run_policy': 'policy',
    'vlm_verify': 'visual', 'verify_object_identity': 'visual',
    'get_execution_budget': 'budget',
    'point_prompt_molmo': 'point', 'get_segmentation_masks': 'perception',
    'segment_sam3_text_prompt': 'perception',
    'segment_sam3_point_prompt': 'perception',
    'segment_sam3_point_prompt_roi': 'perception',
    'plan_grasp': 'perception', 'plan_grasp_from_point_clouds': 'perception',
    'subsample_point_cloud': 'numeric', 'filter_noise': 'numeric',
    'goto_home_joint_position': 'control',
    'point_prompt_sam3': 'perception', 'get_sam3_masks': 'perception',
    'get_grasp_poses': 'perception', 'get_grasps': 'perception',
    'mask_to_world_points': 'numeric', 'pixel_to_world_point': 'numeric',
    'depth_to_point_cloud': 'numeric', 'transform_points': 'numeric',
    'get_oriented_bounding_box_from_3d_points': 'perception',
    'rotation_matrix_to_quaternion': 'numeric', 'decompose_transform': 'numeric',
    'interpolate_segment': 'numeric', 'normalize_vector': 'numeric',
    'select_top_down_grasp': 'perception', 'select_horizontal_grasp': 'perception',
    'select_grasp_along_direction': 'perception',
    'solve_ik': 'numeric', 'forward_kinematics': 'numeric',
    'goto_pose': 'control', 'move_to_joints': 'control',
    'open_gripper': 'control', 'close_gripper': 'control',
    'set_arm_speed': 'control', 'set_gripper': 'control',
}
PERCEPTION_FIELDS = frozenset({
    'masks', 'mask', 'box', 'boxes', 'scores', 'score', 'points', 'point', 'pixel', 'pixels',
    'positions', 'position', 'quaternion', 'quaternions', 'grasps', 'grasp_poses',
    'transforms', 'R', 'center', 'extent', 'rotation', 'translation',
})
POLICY_FIELDS = frozenset({'steps', 'policy_steps', 'actions_executed',
    'elapsed_seconds', 'execution_seconds', 'segment_steps', 'remaining_steps', 'remaining_seconds'})
CONTROL_FIELDS = POLICY_FIELDS | frozenset({'success','reached','target_reached',
    'position_error','orientation_error','joint_error','tcp_position','joint_positions',
    'gripper_position','gripper_width','executed_steps','max_joint_step_rad',
    'move_max_steps','arm_speed'})
BUDGET_FIELDS = frozenset({'steps', 'max_steps', 'remaining_steps', 'elapsed_seconds',
    'remaining_seconds', 'execution_remaining_seconds', 'execution_elapsed_seconds'})


def _numeric(value):
    """Copy numeric leaves only; reject object arrays and custom containers."""
    if isinstance(value, np.generic):
        if value.dtype.kind not in 'buif':
            raise ActorBoundaryError('Non-numeric scalar cannot cross actor boundary')
        value = value.item()
    if value is None or isinstance(value, (bool, numbers.Real)):
        return value.item() if isinstance(value, np.generic) else value
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'buif':
            raise ActorBoundaryError('Non-numeric array cannot cross actor boundary')
        return value.copy()
    if type(value) in (tuple, list):
        return type(value)(_numeric(x) for x in value)
    raise ActorBoundaryError('Only numeric public values may cross this field')


def _fields(raw, allowed, *, strings=False):
    if not isinstance(raw, Mapping):
        return {}
    out = {}
    for k in allowed:
        if k not in raw:
            continue
        v = raw[k]
        if strings and isinstance(v, str):
            out[k] = v
        else:
            try:
                out[k] = _numeric(v)
            except ActorBoundaryError:
                pass
    return out


def _perception(value):
    """Production SAM list-of-records and CGN numeric tuples, schema restricted."""
    if isinstance(value, Mapping):
        return _fields(value, PERCEPTION_FIELDS)
    if type(value) in (list, tuple):
        return type(value)(_perception(x) for x in value)
    return _numeric(value)


def _same_callable(left, right):
    return left is right or (getattr(left, '__self__', None) is not None
        and getattr(left, '__self__', None) is getattr(right, '__self__', None)
        and getattr(left, '__func__', None) is getattr(right, '__func__', None))


@dataclass(frozen=True)
class Binding:
    """Explicit trusted-runtime attestation for an injected orchestration API."""
    function: object
    return_kind: str
    provenance: str


class ActorBoundary:
    def __init__(self, configured_cameras):
        self.cameras = tuple(configured_cameras)
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ActorBoundaryError('Explicit unique public camera names required')
        if any(not isinstance(x, str) or x in ROBOT_FIELDS | TIME_FIELDS | TASK_FIELDS
               or x in {'object_pose','object_pos','target_joint','holding','grasped','world_state'}
               or any(word in x.lower() for word in ('sim_state', 'contact', 'attachment', 'native_success'))
               for x in self.cameras):
            raise ActorBoundaryError('Invalid public camera name')

    def project_observation(self, raw):
        if not isinstance(raw, Mapping):
            raise ActorBoundaryError('Observation must be a mapping')
        out = _fields(raw, ROBOT_FIELDS | TIME_FIELDS)
        out.update(_fields(raw, TASK_FIELDS, strings=True))
        task = _fields(raw.get('task_descriptor'), TASK_FIELDS, strings=True)
        if task:
            out['task_descriptor'] = task
        for name in self.cameras:
            if name not in raw:
                continue
            view = raw[name]
            if isinstance(view, Mapping):
                camera = _fields(view, {'intrinsics', 'pose_mat', 'extrinsics'})
                images = _fields(view.get('images'), {'rgb', 'depth'})
                if images:
                    camera['images'] = images
                camera.update(_fields(view, {'rgb', 'depth'}))
                if camera:
                    out[name] = camera
            else:
                try:
                    out[name] = _numeric(view)
                except ActorBoundaryError:
                    pass
        out['_actor_evidence'] = {'source': 'public_observation_projection_v1',
                                  'native_judge_exposed': False}
        return out

    def project_policy_result(self, raw):
        out = _fields(raw, POLICY_FIELDS)
        if isinstance(raw, Mapping) and raw.get('status') in {
                'completed', 'stopped', 'budget_exhausted', 'cancelled'}:
            out['status'] = raw['status']
        out['_actor_evidence'] = {'source': 'policy_execution_public_feedback',
                                  'native_judge_exposed': False}
        return out

    def project_return(self, kind, value, *, public_keys=()):
        if kind == 'observation':
            return self.project_observation(value)
        if kind == 'policy':
            return self.project_policy_result(value)
        if kind == 'point':
            if not isinstance(value, Mapping):
                raise ActorBoundaryError('Molmo point result must be a mapping')
            return {key: _numeric(value[key]) for key in public_keys if key in value}
        if kind == 'visual':
            if not isinstance(value, Mapping):
                raise ActorBoundaryError('VLM verdict must be a structured mapping')
            out = _fields(value, VISUAL_FIELDS, strings=True)
            if public_keys:
                answers = value.get('predicates')
                out['predicates'] = {key: _fields(answers[key], VISUAL_FIELDS, strings=True)
                    for key in public_keys if isinstance(answers, Mapping) and key in answers}
                out.update(_fields(value, {'complete'}))
            out['_actor_evidence'] = {'source': 'vlm_visual_inference',
                'native_judge_exposed': False, 'simulation_contact_truth': False}
            return out
        if kind == 'budget':
            return _fields(value, BUDGET_FIELDS)
        if kind == 'perception':
            return _perception(value)
        if kind in {'numeric', 'control'}:
            if isinstance(value, Mapping):
                fields = CONTROL_FIELDS if kind == 'control' else PERCEPTION_FIELDS
                return _fields(value, fields)
            return _numeric(value)
        raise ActorBoundaryError('Unknown public return contract')

    def bind_rats_api(self, api, *, trusted_extra=None):
        """Wrap real api.functions() values with output contracts.

        Extra closures (e.g. run_policy supplied by CaseController) must be
        explicitly supplied as Binding objects by the trusted runtime. The
        returned map exposes no oracle capabilities. Unknown APIs are absent.
        This does not stop malicious Python from introspecting closures.
        """
        actual = api.functions()
        if not isinstance(actual, Mapping):
            raise ActorBoundaryError('RATs functions() must return a mapping')
        bindings = {}
        for name, function in actual.items():
            if name in ORACLES or name not in RETURN_KINDS:
                continue
            method = getattr(api, name, None)
            try:
                underlying = inspect.unwrap(function)
                bound_method = inspect.unwrap(method) if callable(method) else None
            except ValueError as exc:
                raise ActorBoundaryError('Cyclic API wrapper') from exc
            if not callable(function) or not _same_callable(underlying, bound_method):
                raise ActorBoundaryError('Functions mapping is not its bound API method: '+name)
            bindings[name] = Binding(function, RETURN_KINDS[name], 'rats_bound_public_api')
        for name, binding in (trusted_extra or {}).items():
            if (name in ORACLES or name not in RETURN_KINDS or not isinstance(binding, Binding)
                    or binding.return_kind != RETURN_KINDS[name]
                    or not callable(binding.function) or not binding.provenance):
                raise ActorBoundaryError('Unattested or forbidden extra actor API: '+name)
            bindings[name] = binding
        wrapped = {}
        for name, binding in bindings.items():
            def invoke(*args, _binding=binding, _name=name, **kwargs):
                public_keys = ()
                if _name == 'point_prompt_molmo':
                    prompt = kwargs.get('text_prompt', args[1] if len(args) > 1 else None)
                    if not isinstance(prompt, str):
                        raise ActorBoundaryError('Explicit public Molmo text prompt required')
                    public_keys = (prompt,)
                elif _name == 'vlm_verify' and kwargs.get('questions') is not None:
                    questions = kwargs['questions']
                    if not isinstance(questions, dict) or any(not isinstance(k, str) for k in questions):
                        raise ActorBoundaryError('Public predicate request IDs required')
                    public_keys = tuple(questions)
                return self.project_return(_binding.return_kind,
                    _binding.function(*args, **kwargs), public_keys=public_keys)
            wrapped[name] = invoke
        return MappingProxyType(wrapped)
