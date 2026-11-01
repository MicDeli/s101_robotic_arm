"""Kinematic model constants of the LeRobot SO-101 follower arm (Product of Exponentials).

Conventions
-----------
* Base frame {0}: origin at the base_link origin (centre of the base plate on the
  table), x pointing forward (towards the arm), y left, z up. Units: metres, radians.
* Joint angle q = 0 is the "new calibration" zero pose (upper arm pointing up, forearm
  pointing forward, gripper axis along +x). Positive rotation follows the right-hand
  rule about the axis w_i given below (the same sign convention as the official URDF
  and MuJoCo model, so joint values can be exchanged with MuJoCo without conversion).
* End-effector frame {e}: ``gripper_frame_link`` of the official URDF, i.e. the tool
  point between the gripper jaws. z_e points along the gripper (approach direction).
* Only the five arm joints position/orient the end-effector. The gripper jaw joint
  (6th servo) does not move {e} and is not part of this chain.
"""
from __future__ import annotations

import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
N_JOINTS = len(JOINT_NAMES)

# Joint position limits [rad] (lower, upper).
JOINT_LIMITS = np.array(
    [
        (-1.91986, 1.91986),   # shoulder_pan
        (-1.74533, 1.74533),   # shoulder_lift
        (-1.69000, 1.69000),   # elbow_flex
        (-1.65806, 1.65806),   # wrist_flex
        (-2.74385, 2.84121),   # wrist_roll
    ]
)

# Unit rotation axes w_i expressed in {0} at q = 0.
AXES = np.array(
    [
        (0.0, 0.0, -1.0),  # shoulder_pan   : vertical, positive = clockwise seen from above
        (0.0, 1.0, 0.0),   # shoulder_lift  : horizontal, positive tilts the arm forward/down
        (0.0, 1.0, 0.0),   # elbow_flex     : parallel to shoulder_lift
        (0.0, 1.0, 0.0),   # wrist_flex     : parallel to shoulder_lift
        (-1.0, 0.0, 0.0),  # wrist_roll     : along the forearm/gripper direction
    ]
)

# A point p_i on each joint axis, in {0} at q = 0 [m].
AXIS_POINTS = np.array(
    [
        (0.0388353, 0.0, 0.0624),                  # shoulder_pan
        (0.0692345, -0.0182778090, 0.1166),        # shoulder_lift
        (0.0972345, -0.0182778090, 0.22917),       # elbow_flex
        (0.2321345, -0.0182778090, 0.23437),       # wrist_flex
        (0.2932345, -0.0001778090, 0.23437),       # wrist_roll
    ]
)

# Screw axes in the space (base) frame. Shape (5, 6).
SCREW_AXES = np.hstack([AXES, -np.cross(AXES, AXIS_POINTS)])

# Home pose M = T_0e(q = 0).
# The rotation is a fixed right-angle rotation plus a 2.79 deg tilt (ALPHA) caused by

ALPHA = 0.0486795  # [rad]
_sa, _ca = np.sin(ALPHA), np.cos(ALPHA)
HOME_POSE = np.array(
    [
        [0.0, 0.0, 1.0, 0.3913619],
        [_sa, _ca, 0.0, -0.0000112554],
        [-_ca, _sa, 0.0, 0.2264687446],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Nominal link dimensions
NOMINAL_LINKS = {
    "shoulder_pan axis -> shoulder_lift axis (dx, dy, dz)": tuple(
        float(v) for v in (AXIS_POINTS[1] - AXIS_POINTS[0])
    ),
    "shoulder_lift axis -> elbow_flex axis (upper arm)": float(
        np.linalg.norm(AXIS_POINTS[2] - AXIS_POINTS[1])
    ),
    "elbow_flex axis -> wrist_flex axis (forearm)": float(
        np.linalg.norm(AXIS_POINTS[3] - AXIS_POINTS[2])
    ),
    "wrist_flex axis -> wrist_roll axis point": float(
        np.linalg.norm(AXIS_POINTS[4] - AXIS_POINTS[3])
    ),
    "wrist_roll axis -> tool point (distance along the gripper)": float(
        np.linalg.norm(HOME_POSE[:3, 3] - AXIS_POINTS[4])
    ),
}
