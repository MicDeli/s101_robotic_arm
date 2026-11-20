"""Damped least squares inverse kinematics for the LeRobot SO-101."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .forward_kinematics import forward_kinematics
from .jacobian import geometric_jacobian
from .so101_model import JOINT_LIMITS, N_JOINTS


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error: float
    orientation_error: float | None


def inverse_kinematics(
    target: np.ndarray,
    q0: np.ndarray,
    *,
    position_tol: float = 1e-4,
    orientation_tol: float = 1e-3,
    damping: float = 1e-2,
    orientation_weight: float = 0.2,
    max_step: float = 0.2,
    max_iterations: int = 200,
) -> IKResult:
    """Solve for a base-frame position (3,) or homogeneous pose (4, 4).

    Joint angles, angular errors, and joint steps are in radians.
    Positions and position errors are in metres.
    """
    target = np.asarray(target, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    if q0.shape != (N_JOINTS,) or not np.all(np.isfinite(q0)):
        raise ValueError(f"Expected {N_JOINTS} finite initial joint angles.")

    if not np.all(np.isfinite(target)):
        raise ValueError("The target must contain finite values.")

    R_target = None
    if target.shape == (3,):
        p_target = target
    elif target.shape == (4, 4):
        p_target = target[:3, 3]
        R_target = target[:3, :3]
        if (
            not np.allclose(target[3], [0, 0, 0, 1])
            or not np.allclose(R_target.T @ R_target, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R_target), 1.0, atol=1e-6)
        ):
            raise ValueError("Expected a rigid homogeneous transform.")
    else:
        raise ValueError("Expected a position of shape (3,) or a pose of shape (4, 4).")

    settings = np.array(
        [position_tol, orientation_tol, damping, orientation_weight, max_step]
    )
    if not np.all(np.isfinite(settings)) or np.any(settings <= 0):
        raise ValueError("Tolerances, damping, weight, and max_step must be positive.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    q_min, q_max = JOINT_LIMITS.T
    q = np.clip(q0, q_min, q_max)

    for iteration in range(max_iterations + 1):
        T = forward_kinematics(q)
        e_pos = p_target - T[:3, 3]
        position_error = float(np.linalg.norm(e_pos))

        orientation_error = None
        e = e_pos
        if R_target is not None:
            R_error = R_target @ T[:3, :3].T
            e_rot = Rotation.from_matrix(R_error).as_rotvec()
            orientation_error = float(np.linalg.norm(e_rot))
            e = np.concatenate((e_pos, orientation_weight * e_rot))

        converged = position_error <= position_tol and (
            orientation_error is None or orientation_error <= orientation_tol
        )
        if converged or iteration == max_iterations:
            return IKResult(q, converged, iteration, position_error, orientation_error)

        J = geometric_jacobian(q)
        if R_target is None:
            J = J[:3]
        else:
            # Scale the angular rows and angular error together.
            J[3:] *= orientation_weight

        # Damping limits joint updates along poorly conditioned directions.
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(J.shape[0]), e
        )
        step_norm = np.linalg.norm(dq)
        if step_norm > max_step:
            dq *= max_step / step_norm

        q = np.clip(q + dq, q_min, q_max)