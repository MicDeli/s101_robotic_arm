"""Draw the SO-101 kinematic skeleton (joint axes, links, base and tool frames).

Output: results/figures/so101_kinematic_diagram.png
  (a) side view (x-z) at q = 0 with joint axes labelled
  (b) top view (x-y) and side view of a bent configuration, tool frame triad shown
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forward_kinematics import forward_kinematics, joint_frames  # noqa: E402
from src.so101_model import AXES, JOINT_NAMES  # noqa: E402

INK, MUTED, ACCENT = "#1f2933", "#7b8794", "#2a6f97"
LABELS = ["1 pan", "2 lift", "3 elbow", "4 wrist flex", "5 wrist roll"]
OFFSETS = [(8, -3), (8, -3), (-12, 9), (-28, 10), (-14, -17)]  # label offsets (points) for panel (a)


def skeleton(q):
    fr = joint_frames(q)
    pts = np.array([f[:3, 3] for f in fr])  # base origin, 5 axis points, tool point
    return fr, pts


def draw(ax, q, dims, title, label_joints=True, triad=True, axes_arrows=True):
    ix, iy = dims
    fr, pts = skeleton(q)
    # link chain: base origin -> pan axis point -> ... -> tool
    ax.plot(pts[:, ix], pts[:, iy], "-", color=INK, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(pts[1:6, ix], pts[1:6, iy], "o", ms=7, mfc="white", mec=INK, mew=1.5, zorder=3)
    ax.plot(*pts[6, [ix, iy]], "o", ms=7, color=ACCENT, zorder=4)
    ax.plot([0], [0], "s", ms=7, color=MUTED, zorder=3)
    if label_joints:
        for k in range(5):
            ax.annotate(LABELS[k], pts[k + 1, [ix, iy]], textcoords="offset points",
                        xytext=OFFSETS[k], fontsize=8, color=INK, ha="left")
    # frame triads
    def triad_at(T, scale=0.045):
        o = T[:3, 3]
        for col, c in zip(range(3), ["#c0392b", "#27ae60", "#2980b9"]):
            d = T[:3, col] * scale
            ax.annotate("", xy=(o[ix] + d[ix], o[iy] + d[iy]), xytext=(o[ix], o[iy]),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4), zorder=5)
    triad_at(np.eye(4))
    ax.text(0.003, -0.022, "{0}", fontsize=8, color=INK)
    if triad:
        triad_at(fr[-1])
        ax.annotate("{e}", pts[6, [ix, iy]], textcoords="offset points", xytext=(8, -12), fontsize=8, color=ACCENT)
    ax.set_aspect("equal")
    ax.set_xlabel(["x [m]", "y [m]", "z [m]"][ix], fontsize=8)
    ax.set_ylabel(["x [m]", "y [m]", "z [m]"][iy], fontsize=8)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(color="#e4e7eb", lw=0.6)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    q0 = np.zeros(5)
    q1 = np.array([0.5, -0.4, 0.9, 0.3, -1.2])
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    draw(axs[0], q0, (0, 2), "(a) side view, q = 0 (x right, z up)")
    axs[0].set_xlim(-0.06, 0.46)
    axs[0].set_ylim(-0.05, 0.30)
    draw(axs[1], q1, (0, 2), "(b) side view, q = (0.5, -0.4, 0.9, 0.3, -1.2) rad", label_joints=False)
    axs[1].set_xlim(-0.06, 0.46)
    axs[1].set_ylim(-0.05, 0.30)
    draw(axs[2], q1, (0, 1), "(c) top view, same configuration (x right, y up)", label_joints=False)
    axs[2].set_xlim(-0.06, 0.40)
    axs[2].set_ylim(-0.15, 0.15)
    fig.text(0.01, -0.02,
             "Axes triads: x red, y green, z blue.  Joint 1 axis is vertical (-z), joints 2-4 are parallel to +y, "
             "joint 5 is along -x at q = 0.", fontsize=7.5, color=MUTED)
    out = ROOT / "results/figures/so101_kinematic_diagram.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
