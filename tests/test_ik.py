"""Test the SO-101 geometric Jacobian and inverse kinematics."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.forward_kinematics import forward_kinematics, within_joint_limits
from src.inverse_kinematics import inverse_kinematics
from src.jacobian import geometric_jacobian

# Five reachable Cartesian targets, each defined by a joint configuration that
# reaches it (FK maps the configuration to the Cartesian target).
REACHABLE_GOALS = {
    "A": [0.2, -0.4, 0.6, -0.3, 0.5],
    "B": [-0.4, -0.5, 0.8, 0.2, -0.5],
    "C": [0.6, 0.2, -0.7, 0.4, 0.8],
    "D": [-0.7, 0.3, 0.5, -0.6, -0.8],
    "E": [0.1, -0.8, 1.0, 0.1, 0.3],
}
Q0_OFFSET = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
POSITION_TOL = 1e-4
ORIENTATION_TOL = 1e-3

# The unreachable target is outside the robot's workspace
UNREACHABLE_TARGET = np.array([0.5, 0.0, 0.0])


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
@pytest.mark.parametrize("goal", list(REACHABLE_GOALS), ids=list(REACHABLE_GOALS))
def test_ik_reaches_target(goal, mode):
    q_goal = np.array(REACHABLE_GOALS[goal])
    T_target = forward_kinematics(q_goal)
    target = T_target[:3, 3] if mode == "position" else T_target

    result = inverse_kinematics(target, q_goal + Q0_OFFSET)

    assert result.converged
    assert within_joint_limits(result.q)

    # Multiple joint solutions can exist, so compare the achieved Cartesian pose.
    T_achieved = forward_kinematics(result.q)
    e_pos = np.linalg.norm(T_target[:3, 3] - T_achieved[:3, 3])
    assert e_pos <= POSITION_TOL

    if mode == "pose":
        R_error = T_target[:3, :3] @ T_achieved[:3, :3].T
        assert Rotation.from_matrix(R_error).magnitude() <= ORIENTATION_TOL
    else:
        assert result.orientation_error is None


def test_ik_reports_unreachable_target():
    result = inverse_kinematics(UNREACHABLE_TARGET, np.zeros(5), max_iterations=25)

    assert not result.converged
    assert result.iterations == 25
    assert within_joint_limits(result.q)
    assert result.position_error > POSITION_TOL


def test_ik_non_converging_with_decreased_iteration_limit():
    """Taking the C target with max_iterations=3."""
    p_target = forward_kinematics(np.array(REACHABLE_GOALS["C"]))[:3, 3]
    result = inverse_kinematics(p_target, np.zeros(5), max_iterations=3)

    assert not result.converged
    assert result.iterations == 3
    assert result.position_error > POSITION_TOL
