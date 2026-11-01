"""Derive the Product-of-Exponentials (PoE) parameters of the SO-101.

composes the fixed joint-to-joint transforms at the zero configuration q = 0 and prints

    * the point p_i on each joint axis and the unit axis w_i, in the base frame {0}
    * the screw axes S_i = (w_i, v_i) with v_i = -w_i x p_i
    * the home pose M of the end-effector frame (``gripper_frame_link``)

The URDF stores angles rounded to 5 digits (e.g. 1.5708 instead of pi/2). Angles
within 1e-4 rad of a multiple of pi/2 are snapped to that multiple so that the
derived axes are exactly axis-aligned. The one genuine non-right-angle value
(the 0.0486795 rad wrist-roll mounting offset) is kept as is.

Usage:  python tools/derive_poe_from_urdf.py [path/to/so101_new_calib.urdf]
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_JOINT = "gripper_frame_joint"  # fixed joint to the end-effector (tool) frame
SNAP_TOL = 1e-4


def snap(angle: float) -> float:
    k = round(angle / (np.pi / 2))
    return k * np.pi / 2 if abs(angle - k * np.pi / 2) < SNAP_TOL else angle


def joint_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = np.array([float(v) for v in origin.get("xyz").split()])
    rpy = [snap(float(v)) for v in origin.get("rpy").split()]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()  # URDF: R = Rz Ry Rx
    T[:3, 3] = xyz
    return T


def clean(a: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    a = np.array(a, dtype=float)
    a[np.abs(a) < tol] = 0.0
    return a


def derive(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    T = np.eye(4)
    axes, points, limits = [], [], []
    for name in CHAIN:
        T = T @ joint_transform(joints[name])
        axes.append(clean(T[:3, :3] @ np.array([0.0, 0.0, 1.0])))  # all joint axes are local z
        points.append(clean(T[:3, 3]))
        lim = joints[name].find("limit")
        limits.append((float(lim.get("lower")), float(lim.get("upper"))))
    M = T @ joint_transform(joints[EE_JOINT])
    S = np.array([np.r_[w, -np.cross(w, p)] for w, p in zip(axes, points)])
    return np.array(axes), np.array(points), S, M, limits


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "models/so101/so101_new_calib.urdf"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    w, p, S, M, lim = derive(path)
    np.set_printoptions(precision=7, suppress=True)
    print("joint | axis w (base frame, q=0) | point p on axis [m] | limits [rad]")
    for n, wi, pi_, li in zip(CHAIN, w, p, lim):
        print(f"{n:13s} {wi}  {pi_}  {li}")
    print("\nScrew axes S_i = (w, v), one row per joint:")
    print(S)
    print("\nHome pose M of gripper_frame_link in base frame:")
    print(M)
    alpha = np.arcsin(M[1, 0])
    print(f"\nwrist mounting tilt alpha = asin(M[1,0]) = {alpha:.7f} rad = {np.degrees(alpha):.4f} deg")
