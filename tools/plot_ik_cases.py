"""Draw one kinematic diagram per inverse-kinematics validation case.

Each named reachable target is drawn in position and pose mode, plus the
unreachable target, in the same style as ``so101_kinematic_diagram.png``.
The solid skeleton is the IK solution, the dashed grey skeleton is the start
configuration q0, the star is the target position and the dashed triads are the
target orientation (pose mode only).

Output: results/figures/ik/ik_case_<name>_<mode>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forward_kinematics import forward_kinematics
from src.inverse_kinematics import inverse_kinematics
from tests.test_ik import (
    ORIENTATION_TOL,
    POSITION_TOL,
    Q0_OFFSET,
    REACHABLE_GOALS,
    UNREACHABLE_TARGET,
)
from tools.plot_kinematics import ACCENT, INK, MUTED, draw, skeleton

TARGET_COLOR = "#c0392b"
START_COLOR = "#b0b7c3"
AXIS_COLORS = ["#c0392b", "#27ae60", "#2980b9"]


def rotation_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def draw_start(ax, q0, dims):
    _, pts = skeleton(q0)
    ix, iy = dims
    ax.plot(pts[:, ix], pts[:, iy], "--", color=START_COLOR, lw=1.3, zorder=1)
    ax.plot(pts[1:6, ix], pts[1:6, iy], "o", ms=4, mfc="white",
            mec=START_COLOR, mew=1.0, zorder=1)


def draw_target(ax, dims, p, R=None):
    ix, iy = dims
    ax.plot(p[ix], p[iy], "*", ms=15, color=TARGET_COLOR, mec="white",
            mew=0.7, zorder=7)
    if R is None:
        return
    scale = 0.05
    for col, color in zip(range(3), AXIS_COLORS):
        d = R[:, col] * scale
        ax.annotate(
            "",
            xy=(p[ix] + d[ix], p[iy] + d[iy]),
            xytext=(p[ix], p[iy]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                            linestyle="--"),
            zorder=6,
        )


def fit_limits(ax, dims, points):
    vals = np.vstack(points)
    lo, hi = vals[:, dims].min(axis=0), vals[:, dims].max(axis=0)
    pad = 0.06
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)


def make_case(name, mode, target, p_target, R_target, q0):
    result = inverse_kinematics(
        target, q0, position_tol=POSITION_TOL, orientation_tol=ORIENTATION_TOL
    )
    T_sol = forward_kinematics(result.q)
    e_pos = float(np.linalg.norm(p_target - T_sol[:3, 3]))
    e_rot = (
        rotation_angle(R_target.T @ T_sol[:3, :3]) if R_target is not None else None
    )

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, dims, subtitle in (
        (axs[0], (0, 2), "(a) side view (x right, z up)"),
        (axs[1], (0, 1), "(b) top view (x right, y up)"),
    ):
        draw(ax, result.q, dims, subtitle, label_joints=False)
        draw_start(ax, q0, dims)
        draw_target(ax, dims, p_target, R_target)
        _, pts = skeleton(result.q)
        _, pts0 = skeleton(q0)
        fit_limits(ax, dims, [pts, pts0, p_target[None, :]])

    handles = [
        Line2D([], [], color=INK, lw=2, marker="o", mfc="white",
               label="IK solution"),
        Line2D([], [], color=START_COLOR, lw=1.3, ls="--", label="start q0"),
        Line2D([], [], color="none", marker="*", mfc=TARGET_COLOR, mec="white",
               ms=12, label="target position"),
    ]
    if R_target is not None:
        handles.append(
            Line2D([], [], color=MUTED, lw=1.2, ls="--",
                   label="target orientation")
        )
    axs[0].legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)

    status = "converged" if result.converged else "not converged"
    rot = "free" if e_rot is None else f"{e_rot:.2e} rad"
    fig.suptitle(
        f"IK case {name} ({mode}): {status}, {result.iterations} iterations, "
        f"|dp| = {e_pos:.2e} m, |dR| = {rot}",
        fontsize=10,
        color=INK,
    )
    fig.text(
        0.01, -0.02,
        "Solid black: IK solution.  Dashed grey: start configuration q0.  "
        "Star: target position.  Dashed triads: target orientation (x red, "
        "y green, z blue).",
        fontsize=7.5, color=MUTED,
    )

    out = ROOT / f"results/figures/ik/ik_case_{name}_{mode}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out, e_pos, e_rot, result


def main() -> None:
    cases = []
    for name, angles in REACHABLE_GOALS.items():
        q_goal = np.array(angles, dtype=float)
        T_target = forward_kinematics(q_goal)
        q0 = q_goal + Q0_OFFSET
        cases.append((name, "position", T_target[:3, 3], T_target[:3, 3], None, q0))
        cases.append((name, "pose", T_target, T_target[:3, 3], T_target[:3, :3], q0))
    cases.append(
        ("U", "position", UNREACHABLE_TARGET, UNREACHABLE_TARGET, None, np.zeros(5))
    )

    for name, mode, target, p_target, R_target, q0 in cases:
        out, e_pos, e_rot, result = make_case(
            name, mode, target, p_target, R_target, q0
        )
        rot = "free" if e_rot is None else f"{e_rot:.3e} rad"
        print(
            f"{name} {mode:8s} -> {out.name:26s} "
            f"iters {result.iterations:3d}  pos {e_pos:.3e} m  rot {rot}"
        )


if __name__ == "__main__":
    main()
