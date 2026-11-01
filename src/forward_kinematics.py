"""Forward kinematics of the LeRobot SO-101 using the Product of Exponentials formula.

    T_0e(q) = exp([S1] q1) exp([S2] q2) ... exp([S5] q5) M

with screw axes S_i and home pose M defined in ``so101_model.py`` (space form).

Interface (as suggested in the project brief)::

    T = forward_kinematics(q)                    # 4x4 homogeneous transform {0} -> {e}
    position, rotation = forward_kinematics_pose(q)
"""
from __future__ import annotations

import numpy as np

from .so101_model import HOME_POSE, JOINT_LIMITS, N_JOINTS, SCREW_AXES


def skew(w: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix [w]x such that [w]x a = w x a."""
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_twist_revolute(S: np.ndarray, theta: float) -> np.ndarray:
    """Matrix exponential exp([S] theta) for a revolute screw axis S = (w, v), |w| = 1.

    Closed form (Rodrigues):
        R = I + sin(t) [w] + (1 - cos(t)) [w]^2
        p = (I t + (1 - cos t) [w] + (t - sin t) [w]^2) v
    """
    w, v = S[:3], S[3:]
    W = skew(w)
    W2 = W @ W
    s, c = np.sin(theta), np.cos(theta)
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + s * W + (1.0 - c) * W2
    T[:3, 3] = (np.eye(3) * theta + (1.0 - c) * W + (theta - s) * W2) @ v
    return T


def _check_q(q) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} joint angles (arm joints only), got {q.shape[0]}")
    return q


def forward_kinematics(q) -> np.ndarray:
    """End-effector pose T_0e (4x4) for the 5 arm joint angles q [rad]."""
    q = _check_q(q)
    T = np.eye(4)
    for S, theta in zip(SCREW_AXES, q):
        T = T @ exp_twist_revolute(S, theta)
    return T @ HOME_POSE


def forward_kinematics_pose(q) -> tuple[np.ndarray, np.ndarray]:
    """Return (position (3,), rotation matrix R in SO(3) (3x3)) of the end-effector."""
    T = forward_kinematics(q)
    return T[:3, 3].copy(), T[:3, :3].copy()


def within_joint_limits(q, margin: float = 0.0) -> bool:
    """True if every joint angle lies inside its limits shrunk by ``margin`` [rad]."""
    q = _check_q(q)
    return bool(np.all(q >= JOINT_LIMITS[:, 0] + margin) and np.all(q <= JOINT_LIMITS[:, 1] - margin))


def joint_frames(q) -> list[np.ndarray]:
    """Frames for plotting: {0}, then one frame per joint located at the point p_i on its
    axis (carried by joints 1..i), then the end-effector frame {e}. Returns 7 4x4 transforms."""
    from .so101_model import AXIS_POINTS

    q = _check_q(q)
    frames = [np.eye(4)]
    T = np.eye(4)
    for i, (S, theta) in enumerate(zip(SCREW_AXES, q)):
        T = T @ exp_twist_revolute(S, theta)
        Ti = T.copy()
        Ti[:3, 3] = (T @ np.r_[AXIS_POINTS[i], 1.0])[:3]
        frames.append(Ti)
    frames.append(T @ HOME_POSE)
    return frames
