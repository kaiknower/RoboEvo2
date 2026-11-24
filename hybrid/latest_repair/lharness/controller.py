"""Public observation/control facade. Native verdicts never return to the actor.

The actor submits a bounded JSON primitive plan, not unrestricted Python with
access to this controller's attributes. Simulator handles stay in this process.
"""
from __future__ import annotations
import copy
import time
import numpy as np
from .molmo_actor_boundary import ActorBoundary


APPROACH_STALL_WINDOW_STEPS = 20
APPROACH_STALL_MIN_ERROR_M = .04
APPROACH_STALL_MAX_DISPLACEMENT_M = .002
APPROACH_STALL_MAX_ERROR_REDUCTION_M = .001
CAVITY_MAX_HEIGHT_DIFFERENCE_M = .04
CAVITY_MIN_HORIZONTAL_DISTANCE_M = .001


class CavityRouteAlignmentError(ValueError):
    """Measured public waypoint mismatch; never an arbitrary exception payload."""
    def __init__(self, height_difference_m, horizontal_distance_m):
        super().__init__('Cavity route must be lateral at a nearly shared height')
        self.height_difference_m = height_difference_m
        self.horizontal_distance_m = horizontal_distance_m


class EpisodeStop(BaseException):
    """Private evaluator control flow, outside actor feedback and diagnostics."""
    def __init__(self, reason):
        self.reason = reason


def validate_local_policy_prompt(prompt):
    """Bounded public language only; never evaluated as code or a stop predicate."""
    if (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 400
            or any(ord(character) < 32 or ord(character) == 127 for character in prompt)):
        raise ValueError('Local policy prompt must be nonempty single-line text of at most400 characters')
    return prompt


class PublicController:
    def __init__(self, session, policy, raw, *, cameras, policy_cap, recovery_cap,
                 wall_seconds=1800, recorder=None):
        self._session, self._policy, self._raw = session, policy, raw
        self._boundary = ActorBoundary(cameras)
        self._recorder = recorder
        self._policy_cap, self._recovery_cap = policy_cap, recovery_cap
        self._stage = 'policy'
        self._counts = {'policy': 0, 'recovery': 0}
        self._native_success = False
        self._started = time.monotonic()
        self._deadline = self._started + wall_seconds
        self._control_seconds = 0.0
        self._queue_seconds = 0.0
        self._feedback = []
        self._visual_targets = {}
        self._relative_orientation_feedback = False
        self._planner_extra_view = None
        self._planner_extra_action_count = None

    def _set_relative_orientation_feedback(self, enabled):
        """Trusted recovery configuration; never an actor primitive."""
        if type(enabled) is not bool:
            raise ValueError("Relative orientation feedback must be Boolean")
        self._relative_orientation_feedback = enabled

    def observe(self):
        out = self._boundary.project_observation(self._raw)
        q = np.asarray(self._raw['robot_eef_quat'], dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError('Invalid robot quaternion')
        out['robot_eef_quat'] = q.copy()
        out.pop('task_id', None)
        out.pop('task_type', None)
        out.pop('task_descriptor', None)
        return out

    def get_budget(self):
        return {'remaining_steps': max(0, self._cap() - self._counts[self._stage])}

    def _observe_for_planner(self, *, extra_view=False):
        """Recovery-only camera capability; policy/recorder observe stays original."""
        if type(extra_view) is not bool:
            raise ValueError('Extra-view choice must be Boolean')
        public = self.observe()
        if not extra_view:
            return public
        if self._stage != 'recovery':
            raise ValueError('Extra camera is available only after policy prefix')
        self._check()
        from .public_extra_view import NAME
        from .coding import public_observation
        view = self._session.capture_recovery_extra_view()
        projected = public_observation({**public, NAME: view})
        self._planner_extra_view = projected[NAME]
        self._planner_extra_action_count = sum(self._counts.values())
        return projected

    def _account_queue_wait(self, seconds):
        """Trusted runtime only: credit measured FIFO wait, never actor/API work."""
        if type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid measured admission wait')
        self._queue_seconds += seconds
        self._deadline += seconds

    def _cap(self):
        return self._policy_cap if self._stage == 'policy' else self._recovery_cap

    def _check(self):
        if self._native_success:
            raise EpisodeStop('native_success')
        if time.monotonic() >= self._deadline:
            raise TimeoutError('Case wall deadline; no native failure inferred')
        if self._counts[self._stage] >= self._cap():
            raise EpisodeStop('action_budget')

    def _step(self, action):
        self._check()
        a = np.asarray(action, dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError('Finite 7D robot action required')
        a = np.clip(a, -1, 1)
        before = self.observe()
        start = time.monotonic()
        raw, success = self._session.step(a)
        self._control_seconds += time.monotonic() - start
        if type(success) is not bool:
            raise TypeError('Native evaluator must return a Boolean')
        self._raw = raw
        self._counts[self._stage] += 1
        self._native_success |= success
        if self._recorder:
            self._recorder(before, a, self._stage)
        if success:
            raise EpisodeStop('native_success')

    def run_policy(self, steps=5):
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise ValueError('Bounded integer policy steps required')
        target = min(steps, self.get_budget()['remaining_steps'])
        done = 0
        while done < target:
            self._check()
            actions = np.asarray(self._policy.actions(self.observe()))
            if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
                raise ValueError('Policy must provide an Nx7 action chunk')
            for a in actions[:min(5, target - done)]:
                self._step(a)
                done += 1
        return {'actions_executed': done}

    def vla_act(self, prompt, steps):
        """Recovery-only local instruction, same frozen policy and five-action cadence."""
        if self._stage != 'recovery':
            raise ValueError('Local policy instruction is available only during recovery')
        prompt = validate_local_policy_prompt(prompt)
        if type(steps) is not int or not 1 <= steps <= 80:
            raise ValueError('Local policy steps must be integer1..80')
        self._check()
        if steps > self.get_budget()['remaining_steps']:
            raise ValueError('Local policy steps exceed the remaining action budget')
        done = 0
        while done < steps:
            self._check()
            local_observation = copy.deepcopy(self.observe())
            local_observation['instruction'] = prompt
            actions = np.asarray(self._policy.actions(local_observation))
            if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
                raise ValueError('Policy must provide an Nx7 action chunk')
            for action in actions[:min(5, steps-done)]:
                self._step(action)
                done += 1
        return {'local_policy_prompt': prompt, 'requested_steps': steps, 'actions_executed': done}

    def begin_recovery(self):
        if self._native_success:
            raise EpisodeStop('native_success')
        self._stage = 'recovery'

    def _move_to(self, target, rotation, steps, gripper, *, stop_when_reached=False,
                 stop_on_stall=False):
        target, rotation = np.asarray(target, float), np.asarray(rotation, float)
        if target.shape != (3,) or rotation.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(rotation).all():
            raise ValueError('Invalid Cartesian target')
        if type(steps) is not int or not 1 <= steps <= 80 or not -1 <= gripper <= 1:
            raise ValueError('Invalid public motion bound')
        done = 0
        settled = 0
        initial_position = np.asarray(self.observe()['robot_cartesian_pos'], float).copy()
        initial_error = float(np.linalg.norm(target - initial_position))
        translation_only = np.linalg.norm(rotation) <= 1e-12
        orientation_goal = None
        if self._relative_orientation_feedback and not translation_only:
            from .public_orientation import RelativeOrientationGoal
            orientation_goal = RelativeOrientationGoal(self.observe()["robot_eef_quat"], rotation)
        positions = [initial_position]
        stall_feedback = {}
        for _ in range(steps):
            self._check()
            error = target - np.asarray(self.observe()['robot_cartesian_pos'])
            rotation_action = (orientation_goal.action(self.observe()["robot_eef_quat"])
                               if orientation_goal is not None else
                               np.clip(rotation / (max(steps, 1) * .5), -1, 1))
            action = np.r_[np.clip(error / .05, -1, 1), rotation_action, gripper]
            self._step(action)
            done += 1
            if stop_when_reached and translation_only:
                remaining_error = np.linalg.norm(target - np.asarray(self.observe()['robot_cartesian_pos']))
                settled = settled + 1 if remaining_error <= .003 else 0
                if settled >= 3:
                    break
            if stop_on_stall and translation_only:
                positions.append(np.asarray(self.observe()['robot_cartesian_pos'], float).copy())
                positions = positions[-(APPROACH_STALL_WINDOW_STEPS + 1):]
                if len(positions) == APPROACH_STALL_WINDOW_STEPS + 1:
                    window = np.asarray(positions)
                    errors = np.linalg.norm(target - window, axis=1)
                    displacement = float(np.linalg.norm(window - window[0], axis=1).max())
                    reduction = float(errors[0] - errors[-1])
                    if (float(errors.min()) > APPROACH_STALL_MIN_ERROR_M
                            and displacement <= APPROACH_STALL_MAX_DISPLACEMENT_M
                            and reduction <= APPROACH_STALL_MAX_ERROR_REDUCTION_M):
                        stall_feedback = {
                            'motion_status': 'stalled',
                            'stall_window_steps': APPROACH_STALL_WINDOW_STEPS,
                            'stall_window_displacement': displacement,
                            'stall_window_error_reduction': reduction,
                        }
                        break
        error = float(np.linalg.norm(target - self.observe()['robot_cartesian_pos']))
        return {'actions_executed': done, 'position_error': error,
                'initial_position_error': initial_error, 'position_error_reduction': initial_error-error,
                **(orientation_goal.feedback(self.observe()["robot_eef_quat"])
                   if orientation_goal is not None else {}), **stall_feedback}

    def move_delta(self, delta, rotation=None, steps=20, gripper=1):
        delta = np.asarray(delta, float)
        rotation = np.asarray(rotation if rotation is not None else [0, 0, 0], float)
        if delta.shape != (3,) or np.any(np.abs(delta) > .15) or np.any(np.abs(rotation) > .5):
            raise ValueError('Relative motion exceeds public primitive bound')
        target = self.observe()['robot_cartesian_pos'] + delta
        return self._move_to(target, rotation, steps, gripper)

    def gripper(self, value, steps=10):
        if type(steps) is not int or not 1 <= steps <= 80 or not -1 <= value <= 1:
            raise ValueError('Invalid gripper command')
        for _ in range(steps):
            self._step([0, 0, 0, 0, 0, 0, value])
        return {'actions_executed': steps, 'gripper_position': self.observe()['robot_gripper_pos']}

    def pixel_target(self, camera, pixel, offset=None, steps=20, gripper=-1):
        view = self.observe().get(camera)
        if camera == 'oppositeview':
            if (self._stage != 'recovery' or self._planner_extra_view is None
                    or self._planner_extra_action_count != sum(self._counts.values())):
                raise ValueError('Extra camera target requires this unchanged planner observation')
            view = self._planner_extra_view
        if not isinstance(view, dict):
            raise ValueError('Camera is not configured')
        depth = np.asarray(view['images']['depth'])
        if len(pixel) != 2 or any(type(v) not in (int, float) for v in pixel):
            raise ValueError('Pixel must be [x,y]')
        x, y = map(lambda v: int(round(v)), pixel)
        if not 0 <= x < depth.shape[1] or not 0 <= y < depth.shape[0]:
            raise ValueError('Pixel outside camera')
        z = float(depth[y, x])
        if not np.isfinite(z) or not 0 < z < 10:
            raise ValueError('No usable metric depth at predicted pixel; observe another view')
        offset = np.asarray(offset if offset is not None else [0, 0, 0], float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or np.any(np.abs(offset) > .15):
            raise ValueError('Invalid local approach offset')
        xyz = np.linalg.solve(np.asarray(view['intrinsics']), [x, y, 1]) * z
        target = (np.asarray(view['extrinsics']) @ np.r_[xyz, 1])[:3] + offset
        return self._move_to(target, [0, 0, 0], steps, gripper)

    def _begin_visual_plan(self, definitions, observation):
        """Trusted dispatcher binds only the exact public snapshot shown this round.

        The actor receives names, never these internal metric coordinates. A
        bound point is a fixed visual waypoint, not an object-state tracker.
        """
        self._visual_targets.clear()
        try:
            for definition in definitions:
                view = observation[definition['camera']]
                x, y = definition['pixel']
                depth = np.asarray(view['images']['depth'])
                z = float(depth[y, x])
                if not np.isfinite(z) or not 0 < z < 10:
                    raise ValueError('No usable current metric depth for visual target')
                intrinsics, extrinsics = np.asarray(view['intrinsics'], float), np.asarray(view['extrinsics'], float)
                if intrinsics.shape != (3, 3) or extrinsics.shape != (4, 4):
                    raise ValueError('Invalid public camera calibration')
                if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
                    raise ValueError('Invalid public camera calibration')
                xyz = np.linalg.solve(intrinsics, [x, y, 1]) * z
                target = (extrinsics @ np.r_[xyz, 1])[:3]
                if not np.isfinite(target).all():
                    raise ValueError('Invalid public visual target')
                self._visual_targets[definition['name']] = target.copy()
        except (KeyError, IndexError, np.linalg.LinAlgError, ValueError) as exc:
            self._visual_targets.clear()
            raise ValueError('Current public RGB-D cannot bind this visual target') from exc

    def _validate_cavity_route(self, entrance_offset, interior_offset):
        """Trusted recipe preflight using only this plan's public RGB-D anchors.

        This checks route topology, not visibility, collision, retention or a
        task predicate. Pixels must be selected from actually visible surfaces.
        """
        points = []
        for name, offset in (("entrance", entrance_offset), ("interior", interior_offset)):
            offset = np.asarray(offset, float)
            if (name not in self._visual_targets or offset.shape != (3,)
                    or not np.isfinite(offset).all() or np.any(np.abs(offset) > .15)):
                raise ValueError('Invalid current public cavity route')
            point = self._visual_targets[name] + offset
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('Invalid current public cavity route')
            points.append(point)
        displacement = points[1] - points[0]
        height_difference = float(displacement[2])
        horizontal_distance = float(np.linalg.norm(displacement[:2]))
        if not np.isfinite([height_difference, horizontal_distance]).all():
            raise ValueError('Invalid current public cavity route')
        if (abs(height_difference) > CAVITY_MAX_HEIGHT_DIFFERENCE_M
                or horizontal_distance < CAVITY_MIN_HORIZONTAL_DISTANCE_M):
            raise CavityRouteAlignmentError(height_difference, horizontal_distance)

    def _end_visual_plan(self):
        self._visual_targets.clear()
        self._planner_extra_view = None
        self._planner_extra_action_count = None

    def move_bound_target(self, target, offset, steps, gripper, rotation=None, *, _stop_on_stall=False):
        if target not in self._visual_targets:
            raise ValueError('Visual target is not bound in this planning round')
        offset = np.asarray(offset, float)
        rotation = np.asarray(rotation if rotation is not None else [0, 0, 0], float)
        if (offset.shape != (3,) or rotation.shape != (3,) or not np.isfinite(offset).all()
                or not np.isfinite(rotation).all() or np.any(np.abs(offset) > .15)
                or np.any(np.abs(rotation) > .5)):
            raise ValueError('Bound visual motion exceeds the public primitive bound')
        return self._move_to(self._visual_targets[target] + offset, rotation, steps, gripper,
                             stop_when_reached=True, stop_on_stall=_stop_on_stall)

    def evaluator_record(self):
        """Private: this function is never in the actor's dispatch table."""
        return dict(native_success=self._native_success, stage=self._stage,
                    policy_steps=self._counts['policy'], recovery_steps=self._counts['recovery'],
                    control_seconds=self._control_seconds,
                    queue_seconds=self._queue_seconds,
                    active_wall_seconds=max(0., time.monotonic() - self._started - self._queue_seconds),
                    wall_seconds=time.monotonic() - self._started)
