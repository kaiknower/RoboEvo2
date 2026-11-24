"""Finite relative orientation feedback using only public xyzw proprioception.

Robosuite OSC composes a world
delta on the left. A constant body-to-tool rotation cancels in the error.
"""
import math
import numpy as np


def quaternion_matrix_xyzw(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('Finite public xyzw quaternion required')
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError('Public quaternion has zero norm')
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rotation_matrix(vector):
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.any(np.abs(v) > .5):
        raise ValueError('Relative rotation exceeds existing public bound')
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)


def rotation_vector(matrix):
    r = np.asarray(matrix, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError('Finite rotation matrix required')
    angle = math.acos(float(np.clip((np.trace(r)-1)/2, -1, 1)))
    if angle < 1e-8:
        return np.zeros(3)
    if abs(angle-math.pi) < 1e-5:
        _, vectors = np.linalg.eigh((r+r.T+2*np.eye(3))/4)
        return vectors[:, -1]*angle
    return np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])*angle/(2*math.sin(angle))


class RelativeOrientationGoal:
    def __init__(self, public_quaternion, relative_rotation):
        self.target = rotation_matrix(relative_rotation) @ quaternion_matrix_xyzw(public_quaternion)
        self.initial_error = float(np.linalg.norm(relative_rotation))

    def error(self, public_quaternion):
        return rotation_vector(self.target @ quaternion_matrix_xyzw(public_quaternion).T)

    def action(self, public_quaternion):
        return np.clip(2.8*self.error(public_quaternion), -.5, .5)

    def feedback(self, public_quaternion):
        error = float(np.linalg.norm(self.error(public_quaternion)))
        return dict(rotation_error=error, initial_rotation_error=self.initial_error,
                    rotation_error_reduction=self.initial_error-error)
