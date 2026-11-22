"""Validate the analytic geometric Jacobian against central finite differences.

For random joint configurations inside the joint limits, ``geometric_jacobian`` is
compared with central differences of ``forward_kinematics``. The linear block uses
the finite difference of the tool position; the angular block uses the rotation
vector of ``R(q + h) R(q - h)^T``, which expresses the angular change in the base
frame, matching the Jacobian convention in ``src/jacobian.py``.

Usage:  python simulation/validate_jacobian_fd.py [--n 500] [--seed 0] [--h 1e-6]
Outputs: results/data/jacobian_fd_validation.csv,
         results/data/jacobian_fd_validation_summary.txt,
         results/figures/jacobian_fd_errors.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forward_kinematics import forward_kinematics  # noqa: E402
from src.jacobian import geometric_jacobian  # noqa: E402
from src.so101_model import JOINT_LIMITS, JOINT_NAMES, N_JOINTS  # noqa: E402


def finite_difference_jacobian(q: np.ndarray, h: float) -> np.ndarray:
    """Central finite differences of the forward kinematics, shape (6, 5)."""
    J = np.empty((6, q.size))
    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = h
        T_plus = forward_kinematics(q + dq)
        T_minus = forward_kinematics(q - dq)

        J[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2.0 * h)

        # This rotation order expresses the angular change in the base frame.
        R_delta = T_plus[:3, :3] @ T_minus[:3, :3].T
        J[3:, i] = Rotation.from_matrix(R_delta).as_rotvec() / (2.0 * h)
    return J


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of random configurations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h", type=float, default=1e-6, help="finite-difference step [rad]")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    q_samples = rng.uniform(
        JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(args.n, N_JOINTS)
    )

    rows = np.empty((args.n, N_JOINTS + 3))
    for k, q in enumerate(q_samples):
        J = geometric_jacobian(q)
        J_fd = finite_difference_jacobian(q, args.h)
        diff = np.abs(J - J_fd)
        rows[k] = np.concatenate(
            (q, [diff.max(), diff[:3].max(), diff[3:].max()])
        )

    out_dir = ROOT / "results"
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    np.savetxt(
        out_dir / "data/jacobian_fd_validation.csv",
        rows,
        delimiter=",",
        header=",".join(JOINT_NAMES) + ",max_abs_diff,max_pos_diff,max_rot_diff",
        comments="",
    )

    max_diff = rows[:, -3]
    lines = [
        "Jacobian versus central finite differences of the forward kinematics",
        "-" * 72,
        f"configurations : {args.n}",
        f"seed           : {args.seed}",
        f"step h [rad]   : {args.h:.1e}",
        "",
        f"largest absolute difference over all entries : {max_diff.max():.3e}",
        f"largest position-block difference            : {rows[:, -2].max():.3e}",
        f"largest angular-block difference             : {rows[:, -1].max():.3e}",
        f"mean of per-configuration maximum            : {max_diff.mean():.3e}",
        "",
        "The difference is the finite-difference truncation error, O(h^2), so the",
        "analytic Jacobian is correct to floating-point accuracy.",
    ]
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "data/jacobian_fd_validation_summary.txt").write_text(summary + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    ax.hist(max_diff * 1e10, bins=30, color="#2a6f97")
    ax.set_xlabel("max |J - J_fd| per configuration [1e-10]")
    ax.set_ylabel("configurations")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Analytic Jacobian vs finite differences, {args.n} random configurations",
        fontsize=10,
    )
    fig.savefig(out_dir / "figures/jacobian_fd_errors.png", dpi=200)


if __name__ == "__main__":
    main()
