"""Validate the PoE forward kinematics against the official SO-101 MuJoCo model.

For named and random joint configurations (inside the joint limits) the end-effector
pose predicted by ``forward_kinematics`` is compared with the pose of the
``gripperframe`` site computed by MuJoCo. The MuJoCo world frame coincides with the
base frame {0} (the base body sits at the origin with identity orientation).

Usage:  python simulation/validate_fk_mujoco.py [--n 1000] [--seed 0]
Outputs: results/data/fk_validation.csv, results/data/fk_validation_summary.txt,
         results/figures/fk_validation_errors.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forward_kinematics import forward_kinematics  # noqa: E402
from src.so101_model import JOINT_LIMITS, JOINT_NAMES  # noqa: E402

SCENE = ROOT / "models/so101/scene.xml"
SITE = "gripperframe"

NAMED_CONFIGS = {
    "zero (L-pose)": [0, 0, 0, 0, 0],
    "pan +45 deg": [np.pi / 4, 0, 0, 0, 0],
    "shoulder_lift +45 deg": [0, np.pi / 4, 0, 0, 0],
    "elbow_flex +60 deg": [0, 0, np.pi / 3, 0, 0],
    "wrist_flex -45 deg": [0, 0, 0, -np.pi / 4, 0],
    "wrist_roll +90 deg": [0, 0, 0, 0, np.pi / 2],
    "folded (rest-like)": [0.0, -1.7, 1.6, 1.0, 0.0],
    "mixed A": [0.5, -0.4, 0.9, 0.3, -1.2],
    "mixed B": [-1.2, 0.8, -0.7, -1.0, 2.0],
}


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle [rad] of a rotation matrix."""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


class MujocoFK:
    def __init__(self, scene_path: Path = SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, SITE)
        self.qadr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES
        ]

    def pose(self, q) -> np.ndarray:
        self.data.qpos[:] = 0.0
        for adr, qi in zip(self.qadr, q):
            self.data.qpos[adr] = qi
        mujoco.mj_kinematics(self.model, self.data)
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site_id]
        return T


# The MuJoCo site "gripperframe" and the URDF frame "gripper_frame_link" share the same
# origin but differ by a fixed rotation of -90 deg about the tool y-axis:
#     R_site = R_urdf @ R_SITE_OFFSET.
# Our model uses the URDF frame (the one LeRobot's own kinematics targets). The offset
# was verified to be constant over 500 random configurations (deviation ~1e-14).
R_SITE_OFFSET = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # Ry(-90 deg)


def compare(sim: MujocoFK, q, compensate_frame_offset: bool = True) -> tuple[float, float]:
    """Return (position error [m], orientation error [rad]) between FK and MuJoCo."""
    T_fk, T_mj = forward_kinematics(q), sim.pose(q)
    R_fk = T_fk[:3, :3] @ R_SITE_OFFSET if compensate_frame_offset else T_fk[:3, :3]
    return (
        float(np.linalg.norm(T_fk[:3, 3] - T_mj[:3, 3])),
        rotation_angle(R_fk.T @ T_mj[:3, :3]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of random configurations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sim = MujocoFK()
    rng = np.random.default_rng(args.seed)
    out_dir = ROOT / "results"
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    raw = np.array([compare(sim, q, compensate_frame_offset=False)[1] for q in
                    rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(100, 5))])
    lines = [
        f"Raw orientation difference to MuJoCo site (no offset compensation): "
        f"{raw.min():.6f} .. {raw.max():.6f} rad (pi/2 = {np.pi / 2:.6f}) -> constant frame offset",
        "Errors below are after compensating the fixed Ry(-90 deg) site offset.",
        "",
        "Named configurations",
        "-" * 72,
    ]
    lines.append(f"{'configuration':24s} {'pos err [m]':>14s} {'rot err [rad]':>14s}   FK position [m]")
    for name, q in NAMED_CONFIGS.items():
        q = np.array(q, dtype=float)
        e_p, e_r = compare(sim, q)
        p = forward_kinematics(q)[:3, 3]
        lines.append(f"{name:24s} {e_p:14.3e} {e_r:14.3e}   [{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}]")

    Q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(args.n, 5))
    errs = np.array([compare(sim, q) for q in Q])
    np.savetxt(
        out_dir / "data/fk_validation.csv",
        np.hstack([Q, errs]),
        delimiter=",",
        header=",".join(JOINT_NAMES) + ",pos_err_m,rot_err_rad",
        comments="",
    )

    lines += [
        "",
        f"Random configurations inside joint limits: n = {args.n}, seed = {args.seed}",
        "-" * 72,
        f"position error  [m]  : max {errs[:, 0].max():.3e}   mean {errs[:, 0].mean():.3e}   rms {np.sqrt((errs[:, 0] ** 2).mean()):.3e}",
        f"rotation error [rad] : max {errs[:, 1].max():.3e}   mean {errs[:, 1].mean():.3e}   rms {np.sqrt((errs[:, 1] ** 2).mean()):.3e}",
    ]
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "data/fk_validation_summary.txt").write_text(summary + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    ax[0].hist(errs[:, 0] * 1e9, bins=30, color="#2a6f97")
    ax[0].set_xlabel("position error [nm]")
    ax[0].set_ylabel("configurations")
    ax[1].hist(errs[:, 1] * 1e9, bins=30, color="#bc6c25")
    ax[1].set_xlabel("orientation error [nrad]")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"FK vs MuJoCo, {args.n} random configurations", fontsize=10)
    fig.savefig(out_dir / "figures/fk_validation_errors.png", dpi=200)


if __name__ == "__main__":
    main()
