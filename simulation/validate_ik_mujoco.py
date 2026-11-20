"""Validate SO-101 position and pose IK with MuJoCo kinematics."""

import numpy as np
from scipy.spatial.transform import Rotation

from simulation.validate_fk_mujoco import MujocoFK, R_SITE_OFFSET
from src.forward_kinematics import within_joint_limits
from src.inverse_kinematics import inverse_kinematics


POSITION_TOL = 1e-4  # metres
ORIENTATION_TOL = 1e-3  # radians

TARGET_CONFIGS = {
    "A": [0.2, -0.4, 0.6, -0.3, 0.5],
    "B": [-0.4, -0.5, 0.8, 0.2, -0.5],
    "C": [0.6, 0.2, -0.7, 0.4, 0.8],
    "D": [-0.7, 0.3, 0.5, -0.6, -0.8],
    "E": [0.1, -0.8, 1.0, 0.1, 0.3],
}


def gripper_pose(sim: MujocoFK, q: np.ndarray) -> np.ndarray:
    """Read the simulated pose in the same tool frame used by Phase 1 FK."""
    T = sim.pose(q)
    T[:3, :3] = T[:3, :3] @ R_SITE_OFFSET.T
    return T


def main() -> None:
    sim = MujocoFK()
    failed = []

    for name, angles in TARGET_CONFIGS.items():
        q_goal = np.array(angles)
        T_target = gripper_pose(sim, q_goal)
        p_target = T_target[:3, 3]
        q0 = q_goal + np.array([-0.1, 0.2, -0.3, 0.2, -0.3])

        print(f"\nTarget {name} [m]: {np.array2string(p_target, precision=4)}")
        for mode, target in (("position", p_target), ("pose", T_target)):
            result = inverse_kinematics(
                target,
                q0,
                position_tol=POSITION_TOL,
                orientation_tol=ORIENTATION_TOL,
            )
            T_achieved = gripper_pose(sim, result.q)
            e_pos = np.linalg.norm(p_target - T_achieved[:3, 3])

            e_rot = None
            if mode == "pose":
                R_error = T_target[:3, :3] @ T_achieved[:3, :3].T
                e_rot = Rotation.from_matrix(R_error).magnitude()

            limits_ok = within_joint_limits(result.q)
            passed = (
                result.converged
                and limits_ok
                and e_pos <= POSITION_TOL
                and (e_rot is None or e_rot <= ORIENTATION_TOL)
            )
            status = "PASS" if passed else "FAIL"
            rotation_text = "free" if e_rot is None else f"{e_rot:.3e} rad"
            print(
                f"  {mode:8s} {status}  iterations={result.iterations:3d}  "
                f"position={e_pos:.3e} m  orientation={rotation_text}  "
                f"limits_ok={limits_ok}"
            )
            if not passed:
                failed.append(f"{name}/{mode}")

    # Two metres along the base x-axis is outside the SO-101 workspace.
    p_unreachable = np.array([2.0, 0.0, 0.0])
    result = inverse_kinematics(p_unreachable, np.zeros(5))
    p_achieved = gripper_pose(sim, result.q)[:3, 3]
    e_pos = np.linalg.norm(p_unreachable - p_achieved)
    limits_ok = within_joint_limits(result.q)
    passed = not result.converged and limits_ok and e_pos > POSITION_TOL
    status = "PASS" if passed else "FAIL"
    print(
        f"\nUnreachable {status}  converged={result.converged}  "
        f"iterations={result.iterations}  position={e_pos:.3e} m  "
        f"limits_ok={limits_ok}"
    )
    if not passed:
        failed.append("unreachable")

    if failed:
        raise SystemExit(f"Validation failed: {', '.join(failed)}")
    print("\nMuJoCo validation passed for all 11 cases.")


if __name__ == "__main__":
    main()