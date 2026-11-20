"""Test the SO-101 geometric Jacobian and inverse kinematics."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.forward_kinematics import forward_kinematics, within_joint_limits
from src.inverse_kinematics import inverse_kinematics
from src.jacobian import geometric_jacobian


def test_geometric_jacobian_matches_finite_differences():
    q = np.array([0.2, -0.4, 0.6, -0.3, 0.5])
    h = 1e-6
    J_fd = np.empty((6, q.size))

    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = h
        T_plus = forward_kinematics(q + dq)
        T_minus = forward_kinematics(q - dq)

        J_fd[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * h)

        # This rotation order expresses the angular change in the base frame.
        R_delta = T_plus[:3, :3] @ T_minus[:3, :3].T
        J_fd[3:, i] = Rotation.from_matrix(R_delta).as_rotvec() / (2 * h)

    np.testing.assert_allclose(geometric_jacobian(q), J_fd, atol=1e-8, rtol=0)


@pytest.mark.parametrize("mode", ["position", "pose"])
def test_ik_reaches_target(mode):
    q_goal = np.array([0.2, -0.4, 0.6, -0.3, 0.5])
    q0 = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
    T_target = forward_kinematics(q_goal)
    target = T_target[:3, 3] if mode == "position" else T_target

    result = inverse_kinematics(target, q0)

    assert result.converged
    assert within_joint_limits(result.q)

    # Check the achieved Cartesian target since multiple joint solutions can exist.
    T_achieved = forward_kinematics(result.q)
    e_pos = np.linalg.norm(T_target[:3, 3] - T_achieved[:3, 3])
    assert e_pos <= 1e-4

    if mode == "pose":
        R_error = T_target[:3, :3] @ T_achieved[:3, :3].T
        assert Rotation.from_matrix(R_error).magnitude() <= 1e-3
    else:
        assert result.orientation_error is None


def test_ik_reports_unreachable_target():
    p_target = np.array([2.0, 0.0, 0.0])
    result = inverse_kinematics(p_target, np.zeros(5), max_iterations=25)

    assert not result.converged
    assert result.iterations == 25
    assert within_joint_limits(result.q)
    assert result.position_error > 1e-4