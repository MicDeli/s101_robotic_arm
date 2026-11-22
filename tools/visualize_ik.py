"""View inverse-kinematics solutions in the MuJoCo viewer.

By default this uses the same target set as ``tests/test_ik.py``: the five
reachable configurations A-E plus the unreachable point U.
All reachable targets are drawn at once and labelled, and the arm moves
continuously from one solution to the next without resetting. The unreachable
target is then attempted and shown to fail.

The targets are joint configurations mapped through forward kinematics, so the
reachable poses are guaranteed reachable; the IK solver only receives the
Cartesian pose.

Examples
--------
    python tools/visualize_ik.py                        # validation targets A-E + unreachable
    python tools/visualize_ik.py --mode position        # only position targets, orientation is free
    python tools/visualize_ik.py --no-viewer
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.validate_fk_mujoco import MujocoFK  # noqa: E402
from src.forward_kinematics import forward_kinematics  # noqa: E402
from src.inverse_kinematics import inverse_kinematics  # noqa: E402
from tests.test_ik import REACHABLE_GOALS, UNREACHABLE_TARGET  # noqa: E402

TARGET_COLORS = [
    [0.95, 0.25, 0.20, 1.0],
    [0.20, 0.75, 0.30, 1.0],
    [0.20, 0.45, 0.95, 1.0],
    [0.95, 0.65, 0.10, 1.0],
    [0.75, 0.30, 0.85, 1.0],
]
UNREACHABLE_COLOR = [0.85, 0.10, 0.10, 1.0]


@dataclass
class Case:
    name: str
    q0: np.ndarray
    q_sol: np.ndarray
    p_target: np.ndarray
    R_target: np.ndarray
    converged: bool
    e_pos: float
    e_rot: float | None


def solve_case(name: str, q_goal: np.ndarray, mode: str, q0: np.ndarray) -> Case:
    T_target = forward_kinematics(q_goal)
    p_target, R_target = T_target[:3, 3], T_target[:3, :3]
    target = p_target if mode == "position" else T_target
    result = inverse_kinematics(target, q0)
    T_achieved = forward_kinematics(result.q)
    e_pos = float(np.linalg.norm(p_target - T_achieved[:3, 3]))
    e_rot = None
    if mode == "pose":
        e_rot = float(Rotation.from_matrix(
            R_target @ T_achieved[:3, :3].T).magnitude())
    return Case(name, q0, result.q, p_target, R_target,
                result.converged, e_pos, e_rot)


def solve_unreachable(q0: np.ndarray) -> Case:
    result = inverse_kinematics(UNREACHABLE_TARGET, q0)
    T_achieved = forward_kinematics(result.q)
    e_pos = float(np.linalg.norm(UNREACHABLE_TARGET - T_achieved[:3, 3]))
    return Case("U", q0, result.q, UNREACHABLE_TARGET, np.eye(3),
                result.converged, e_pos, None)


def set_config(sim: MujocoFK, q: np.ndarray) -> None:
    import mujoco

    sim.data.qpos[:] = 0.0
    for adr, qi in zip(sim.qadr, q):
        sim.data.qpos[adr] = qi
    mujoco.mj_forward(sim.model, sim.data)


def draw_targets(viewer, targets) -> None:
    """Draw all targets; each is (label, p, R, rgba)."""
    import mujoco

    scn = viewer.user_scn
    scn.ngeom = 0

    def add_geom(geom_type, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        mujoco.mjv_initGeom(g, geom_type, size, pos, np.eye(3).ravel(), rgba)
        return g

    for label, p, R, rgba in targets:
        add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.009, 0.0, 0.0]),
                 p, rgba)

        axis_colors = ([1.0, 0.2, 0.2, 1.0], [0.2, 1.0, 0.2, 1.0],
                       [0.2, 0.4, 1.0, 1.0])
        for col, axis_rgba in zip(range(3), axis_colors):
            g = add_geom(mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), p, axis_rgba)
            if g is None:
                break
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_ARROW, 0.003, p, p + R[:, col] * 0.04
            )
            g.rgba[:] = axis_rgba

        g = add_geom(mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                     p + np.array([0.0, 0.0, 0.03]), rgba)
        if g is not None:
            g.label = label


def print_table(cases: list[Case]) -> None:
    print(f"{'target':>6} {'converged':>10} {'pos err [m]':>12} {'rot err [rad]':>14}")
    print("-" * 48)
    for c in cases:
        rot = "free" if c.e_rot is None else f"{c.e_rot:.3e}"
        print(f"{c.name:>6} {str(c.converged):>10} {c.e_pos:>12.3e} {rot:>14}")


def show(sim: MujocoFK, cases: list[Case], unreachable: Case,
         duration: float) -> None:
    import mujoco.viewer

    reachable = [c for c in cases if c is not unreachable]
    targets = [
        (c.name, c.p_target, c.R_target, TARGET_COLORS[k % len(TARGET_COLORS)])
        for k, c in enumerate(reachable)
    ]
    targets.append(("U", unreachable.p_target, unreachable.R_target,
                    UNREACHABLE_COLOR))

    centre = np.mean([c.p_target for c in reachable], axis=0)
    radius = max(float(np.linalg.norm(c.p_target - centre)) for c in reachable)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = centre
        viewer.cam.distance = max(0.7, 2.4 * radius + 0.2)
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        q = np.zeros(5)
        set_config(sim, q)
        draw_targets(viewer, targets)
        viewer.sync()

        # Move continuously through the reachable solutions, then the attempt
        # at the unreachable target; the arm never resets to q0.
        for case in reachable + [unreachable]:
            q_start = q.copy()
            frames = max(2, int(duration * 60))
            for alpha in np.linspace(0.0, 1.0, frames):
                if not viewer.is_running():
                    return
                q = (1.0 - alpha) * q_start + alpha * case.q_sol
                set_config(sim, q)
                draw_targets(viewer, targets)
                viewer.sync()
                time.sleep(1.0 / 60.0)

        while viewer.is_running():
            draw_targets(viewer, targets)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q0", type=float, nargs=5, default=None,
                    help="initial joint guess (default: zeros)")
    ap.add_argument("--mode", choices=("position", "pose"), default="pose")
    ap.add_argument("--duration", type=float, default=0.9,
                    help="animation duration per target in seconds")
    ap.add_argument("--no-viewer", action="store_true",
                    help="solve and print the results without opening the viewer")
    args = ap.parse_args()

    q0 = np.zeros(5) if args.q0 is None else np.asarray(args.q0, dtype=float)

    reachable = [
        solve_case(name, np.array(angles, dtype=float), args.mode, q0)
        for name, angles in REACHABLE_GOALS.items()
    ]
    unreachable = solve_unreachable(q0)
    print_table(reachable + [unreachable])
    print("(U is the unreachable target; it is shown in red and cannot converge)")

    if not args.no_viewer:
        show(MujocoFK(), reachable, unreachable, args.duration)


if __name__ == "__main__":
    main()
