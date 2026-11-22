# LeRobot SO-101 – IK-based Cartesian Control

**Team members:**
  Palaniappan Apurva,
  Adrah Michael,
  Karmacharya Harjit,


Course project for *Introduction to Robotic Manipulation*: kinematic modelling, inverse kinematics and Cartesian control of a LeRobot SO-101 arm, validated in MuJoCo and demonstrated on hardware.

**Status:**  
Phase 1 (kinematic model + forward kinematics + validation) complete.  
Phase 2 (Manipulator Jacobian + DLS inverse kinematics + MuJoCo validation) complete.  
Phase 3 (trajectory, control, hardware demo) pending.

## Setup

```bash
# Python 3.12 (e.g. via Miniforge)
conda create -n lerobot_ik python=3.12 -y && conda activate lerobot_ik
pip install -r requirements.txt
```

The official SO-101 model (URDF, MuJoCo XML, meshes) is included in `models/so101/`
(source: TheRobotStudio/SO-ARM100, `Simulation/SO101`).

## Run

```bash
pytest tests                                   # unit tests (FK, Jacobian, IK)
python simulation/validate_fk_mujoco.py        # FK vs MuJoCo, writes results/
python simulation/validate_jacobian_fd.py      # Jacobian vs finite differences, writes results/
python simulation/validate_ik_mujoco.py        # DLS IK vs MuJoCo, writes results/
python tools/plot_kinematics.py                # kinematic diagram figure
python tools/plot_ik_cases.py                  # one diagram per IK validation case
python tools/visualize_ik.py                   # interactive MuJoCo IK viewer
python tools/derive_poe_from_urdf.py           # re-derive PoE parameters from the URDF
```

Final demonstration (Phase 3, not yet implemented): `simulation/mujoco_demo.py`, `hardware/lerobot_demo.py`.

## Layout

| Path | Content |
|---|---|
| `src/so101_model.py` | PoE parameters: screw axes, home pose, joint limits |
| `src/forward_kinematics.py` | `forward_kinematics(q)`, `forward_kinematics_pose(q)`, `joint_frames(q)` |
| `src/jacobian.py` | `geometric_jacobian(q)` (6×5 point Jacobian `[p_dot; omega]`) |
| `src/inverse_kinematics.py` | `inverse_kinematics(target, q0, ...)` (DLS), `IKResult` |
| `simulation/validate_fk_mujoco.py` | FK validation against the MuJoCo model |
| `simulation/validate_jacobian_fd.py` | Analytic Jacobian versus central finite differences over random configurations |
| `simulation/validate_ik_mujoco.py` | IK validation against MuJoCo: named reachable targets (position and pose) and an unreachable target |
| `tests/` | FK, Jacobian and IK unit tests (`test_fk.py`, `test_ik.py`) |
| `tools/plot_kinematics.py` | Kinematic diagram (`so101_kinematic_diagram.png`) |
| `tools/plot_ik_cases.py` | One kinematic diagram per IK validation case (`ik_case_*.png`) |
| `tools/visualize_ik.py` | Interactive MuJoCo viewer for the IK targets |
| `tools/derive_poe_from_urdf.py` | Re-derive the PoE parameters from the URDF |
| `docs/Technical_report.md` | Frames, PoE derivation, validation (report section 1) |
| `results/` | Figures and validation data |
| `models/so101/` | Official URDF, MuJoCo XML and meshes |

## Conventions

Metres and radians. Base frame {0}: x forward, y left, z up. End-effector frame = URDF `gripper_frame_link`. Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll. See `docs/Technical_report.md`.

IK accepts either a 3D position `(3,)` (orientation free) or a homogeneous pose `(4, 4)` (full 6D, solved in the least-squares sense since the arm has 5 DoF). The solver is damped least squares (`damping = 1e-2`, `orientation_weight = 0.2`) with joint-limit clamping and a `max_step` cap; convergence is checked against `position_tol = 1e-4` m and `orientation_tol = 1e-3` rad.
