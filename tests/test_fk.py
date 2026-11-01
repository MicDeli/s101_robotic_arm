"""Unit tests for the SO-101 forward kinematics (run with ``pytest``)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forward_kinematics import (  # noqa: E402
    exp_twist_revolute,
    forward_kinematics,
    forward_kinematics_pose,
    within_joint_limits,
)
from src.so101_model import HOME_POSE, JOINT_LIMITS, SCREW_AXES  # noqa: E402

RNG = np.random.default_rng(1)


def random_q(n: int = 1) -> np.ndarray:
    return RNG.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(n, 5))


def test_zero_configuration_is_home_pose():
    assert np.allclose(forward_kinematics(np.zeros(5)), HOME_POSE, atol=1e-12)


def test_rotation_is_in_so3_and_transform_is_homogeneous():
    for q in random_q(200):
        T = forward_kinematics(q)
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
        assert np.allclose(T[3], [0, 0, 0, 1])


def test_screw_axes_are_unit_and_orthogonal_form():
    w, v = SCREW_AXES[:, :3], SCREW_AXES[:, 3:]
    assert np.allclose(np.linalg.norm(w, axis=1), 1.0)
    assert np.allclose(np.einsum("ij,ij->i", w, v), 0.0, atol=1e-12)  # pitch 0 (revolute)


def test_exponential_matches_scipy_expm():
    from scipy.linalg import expm

    for S in SCREW_AXES:
        for theta in (-2.0, 0.3, 1.7):
            se3 = np.zeros((4, 4))
            W = np.array([[0, -S[2], S[1]], [S[2], 0, -S[0]], [-S[1], S[0], 0]])
            se3[:3, :3], se3[:3, 3] = W, S[3:]
            assert np.allclose(exp_twist_revolute(S, theta), expm(se3 * theta), atol=1e-12)


def test_pan_rotates_tool_about_vertical_axis():
    """Pan only: tool keeps its height and its distance to the pan axis."""
    q0 = np.zeros(5)
    p0 = forward_kinematics(q0)[:3, 3]
    pan_axis_xy = np.array([0.0388353, 0.0])
    for pan in (-1.5, 0.4, 1.9):
        p = forward_kinematics(q0 + np.array([pan, 0, 0, 0, 0]))[:3, 3]
        assert np.isclose(p[2], p0[2], atol=1e-12)
        assert np.isclose(np.linalg.norm(p[:2] - pan_axis_xy), np.linalg.norm(p0[:2] - pan_axis_xy), atol=1e-12)


def test_wrist_roll_moves_tool_on_circle_about_roll_axis():
    """The tool point (jaw midpoint) is ~7.9 mm off the roll axis, so roll sweeps it on a
    circle about that axis: distance to the axis line and position along it are invariant."""
    from src.so101_model import AXES, AXIS_POINTS

    w, a = AXES[4], AXIS_POINTS[4]
    radii, along = [], []
    for roll in (-2.0, 0.0, 1.0, 2.5):
        r = forward_kinematics(np.array([0, 0, 0, 0, roll]))[:3, 3] - a
        along.append(r @ w)
        radii.append(np.linalg.norm(r - (r @ w) * w))
    assert np.allclose(radii, radii[0], atol=1e-12)
    assert np.allclose(along, along[0], atol=1e-12)
    assert abs(radii[0] - 0.0079) < 3e-4


def test_planar_motion_when_pan_and_roll_are_zero():
    """With pan = roll = 0 joints 2-4 are parallel y-axes: the tool stays in the x-z plane."""
    y_home = forward_kinematics(np.zeros(5))[1, 3]
    for q in random_q(50):
        q[[0, 4]] = 0.0
        assert abs(forward_kinematics(q)[1, 3] - y_home) < 1e-12


def test_shoulder_lift_preserves_distance_to_shoulder_axis():
    """Rotating about the shoulder_lift axis cannot change the distance from a point on
    that axis to the tool point."""
    from src.so101_model import AXIS_POINTS

    q = random_q()[0]
    ref = None
    for lift in np.linspace(-1.7, 1.7, 9):
        q2 = q.copy()
        q2[1] = lift
        d = np.linalg.norm(forward_kinematics(q2)[:3, 3] - np.r_[0, 0, 0] - _after_pan(q2, AXIS_POINTS[1]))
        ref = d if ref is None else ref
        assert np.isclose(d, ref, atol=1e-12)


def _after_pan(q, point):
    """Position of a point fixed on the shoulder link (on the lift axis) after the pan rotation."""
    from src.forward_kinematics import exp_twist_revolute

    return (exp_twist_revolute(SCREW_AXES[0], q[0]) @ np.r_[point, 1.0])[:3]


def test_pose_interface_matches_matrix():
    q = random_q()[0]
    T = forward_kinematics(q)
    p, R = forward_kinematics_pose(q)
    assert np.allclose(p, T[:3, 3]) and np.allclose(R, T[:3, :3])


def test_bad_input_length_raises():
    with pytest.raises(ValueError):
        forward_kinematics(np.zeros(6))


def test_joint_limit_helper():
    assert within_joint_limits(np.zeros(5))
    assert not within_joint_limits(np.array([2.5, 0, 0, 0, 0]))


def test_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    from simulation.validate_fk_mujoco import MujocoFK, compare

    sim = MujocoFK()
    for q in random_q(100):
        e_p, e_r = compare(sim, q)
        assert e_p < 1e-5 and e_r < 1e-5
