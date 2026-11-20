"""Geometric Jacobian for the LeRobot SO-101 gripper."""

from __future__ import annotations

import numpy as np

from .forward_kinematics import exp_twist_revolute
from .so101_model import AXIS_POINTS, HOME_POSE, N_JOINTS, SCREW_AXES


def geometric_jacobian(q: np.ndarray) -> np.ndarray:
    """Return the 6-by-5 gripper Jacobian in the base frame.

    Joint angles are in radians. The first three rows map joint rates
    to gripper linear velocity, and the last three map to angular velocity.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != N_JOINTS or not np.all(np.isfinite(q)):
        raise ValueError(f"Expected {N_JOINTS} finite joint angles in radians.")

    T = np.eye(4)
    axes = np.empty((N_JOINTS, 3))
    axis_points = np.empty((N_JOINTS, 3))

    for i, (S, theta) in enumerate(zip(SCREW_AXES, q)):
        # The preceding joints carry this joint's axis into the base frame.
        R = T[:3, :3]
        axes[i] = R @ S[:3]
        axis_points[i] = R @ AXIS_POINTS[i] + T[:3, 3]
        T = T @ exp_twist_revolute(S, theta)

    p_gripper = (T @ HOME_POSE)[:3, 3]
    # Each revolute joint moves the gripper tangentially about its axis.
    J_pos = np.cross(axes, p_gripper - axis_points).T
    J_rot = axes.T
    return np.vstack((J_pos, J_rot))