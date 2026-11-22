"""Validate the SO-101 inverse kinematics against the official SO-101 MuJoCo model.

For the named reachable targets from ``tests/test_ik.py`` (A-E, in position and
pose mode) and the unreachable target, the IK is solved in the base frame and
the achieved pose is read from the independent MuJoCo model. Writes the same
kind of outputs as ``validate_fk_mujoco.py``.

Usage:  python simulation/validate_ik_mujoco.py
Outputs: results/data/ik_validation.csv,
         results/data/ik_validation_summary.txt,
         results/figures/ik_validation_errors.png,
         results/figures/ik_convergence.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.validate_fk_mujoco import (
    MujocoFK,
    R_SITE_OFFSET,
    rotation_angle,
)
from src.forward_kinematics import within_joint_limits
from src.inverse_kinematics import inverse_kinematics
from tests.test_ik import (
    ORIENTATION_TOL,
    POSITION_TOL,
    Q0_OFFSET,
    REACHABLE_GOALS,
    UNREACHABLE_TARGET,
)


def gripper_pose(sim: MujocoFK, q: np.ndarray) -> np.ndarray:
    """Read the simulated pose in the same tool frame used by Phase 1 FK."""
    T = sim.pose(q)
    T[:3, :3] = T[:3, :3] @ R_SITE_OFFSET.T
    return T


def check_case(sim: MujocoFK, name: str, mode: str, target: np.ndarray,
               q0: np.ndarray, p_target: np.ndarray,
               R_target: np.ndarray) -> dict:
    """Solve one IK case and compare the achieved MuJoCo pose with the target."""
    result = inverse_kinematics(
        target, q0, position_tol=POSITION_TOL, orientation_tol=ORIENTATION_TOL
    )
    T_achieved = gripper_pose(sim, result.q)
    e_pos = float(np.linalg.norm(p_target - T_achieved[:3, 3]))
    e_rot = None
    if mode == "pose":
        e_rot = float(rotation_angle(R_target.T @ T_achieved[:3, :3]))
    limits_ok = within_joint_limits(result.q)
    passed = (
        result.converged
        and limits_ok
        and e_pos <= POSITION_TOL
        and (e_rot is None or e_rot <= ORIENTATION_TOL)
    )
    return {
        "case": name,
        "mode": mode,
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "pos_err": e_pos,
        "rot_err": e_rot,
        "limits_ok": bool(limits_ok),
        "passed": bool(passed),
    }


def format_row(row: dict) -> str:
    status = "PASS" if row["passed"] else "FAIL"
    rot = "free" if row["rot_err"] is None else f"{row['rot_err']:.3e} rad"
    return (
        f"{row['case']:8s} {row['mode']:8s} {status:6s} "
        f"{row['iterations']:5d} {row['pos_err']:12.3e} m {rot:>16s}"
    )


def check_unreachable(sim: MujocoFK) -> dict:
    result = inverse_kinematics(UNREACHABLE_TARGET, np.zeros(5))
    p_achieved = gripper_pose(sim, result.q)[:3, 3]
    e_pos = float(np.linalg.norm(UNREACHABLE_TARGET - p_achieved))
    limits_ok = within_joint_limits(result.q)
    passed = (not result.converged) and limits_ok and e_pos > POSITION_TOL
    return {
        "case": "U",
        "mode": "position",
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "pos_err": e_pos,
        "rot_err": None,
        "limits_ok": bool(limits_ok),
        "passed": bool(passed),
    }


def write_csv(rows: list[dict]) -> None:
    lines = ["case,mode,converged,iterations,pos_err_m,rot_err_rad,limits_ok,passed"]
    for row in rows:
        rot = "" if row["rot_err"] is None else f"{row['rot_err']:.6e}"
        lines.append(
            f"{row['case']},{row['mode']},{row['converged']},{row['iterations']},"
            f"{row['pos_err']:.6e},{rot},{row['limits_ok']},{row['passed']}"
        )
    out = ROOT / "results/data/ik_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def save_figures(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{r['case']}\n{'pos' if r['mode'] == 'position' else 'pose'}"
        for r in rows
    ]
    errors = np.array([max(r["pos_err"], 1e-12) for r in rows])
    iters = np.array([r["iterations"] for r in rows])
    colors = ["#2a6f97" if r["passed"] else "#c0392b" for r in rows]
    x = np.arange(len(rows))

    fig_dir = ROOT / "results/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    ax.bar(x, errors, color=colors)
    ax.set_yscale("log")
    ax.axhline(POSITION_TOL, color="#7b8794", ls="--", lw=1, label="position_tol")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("final position error [m]")
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("IK vs MuJoCo, named reachable targets + unreachable", fontsize=10)
    fig.savefig(fig_dir / "ik_validation_errors.png", dpi=200)

    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    ax.bar(x, iters, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("iterations")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("IK iterations (blue = passed, red = failed)", fontsize=10)
    fig.savefig(fig_dir / "ik_convergence.png", dpi=200)


def main() -> None:
    sim = MujocoFK()
    rows: list[dict] = []
    lines = [
        "Named reachable targets (from tests/test_ik.py)",
        "-" * 72,
        f"{'target':8s} {'mode':8s} {'status':6s} {'iters':>5s} "
        f"{'pos err [m]':>12s} {'rot err':>16s}",
    ]

    for name, angles in REACHABLE_GOALS.items():
        q_goal = np.array(angles, dtype=float)
        T_target = gripper_pose(sim, q_goal)
        p_target, R_target = T_target[:3, 3], T_target[:3, :3]
        q0 = q_goal + Q0_OFFSET
        for mode, target in (("position", p_target), ("pose", T_target)):
            row = check_case(sim, name, mode, target, q0, p_target, R_target)
            rows.append(row)
            lines.append(format_row(row))

    unreachable = check_unreachable(sim)
    rows.append(unreachable)
    lines.append("")
    lines.append("Unreachable target")
    lines.append("-" * 72)
    lines.append(
        f"{'U':8s} {unreachable['mode']:8s} "
        f"{'PASS' if unreachable['passed'] else 'FAIL':6s} "
        f"{unreachable['iterations']:5d} {unreachable['pos_err']:12.3e} m "
        f"{'(not converged)' if not unreachable['converged'] else '(converged)'}"
    )

    all_ok = all(r["passed"] for r in rows)
    lines += ["", "IK validation:", "PASS" if all_ok else "FAIL"]

    summary = "\n".join(lines)
    print(summary)

    out = ROOT / "results/data/ik_validation_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary + "\n")
    write_csv(rows)
    save_figures(rows)

    if not all_ok:
        raise SystemExit("IK validation failed.")


if __name__ == "__main__":
    main()
