# LeRobot SO-101 – IK-based Cartesian Control

**Team members:**
  Palaniappan Apurva,
  Adrah Michael,
  Karmacharya Harjit,


Course project for *Introduction to Robotic Manipulation*: kinematic modelling, inverse kinematics and Cartesian control of a LeRobot SO-101 arm, validated in MuJoCo and demonstrated on hardware.

**Status:** Phase 1 (kinematic model + forward kinematics + validation) complete. Phases 2–3 (IK, trajectory, control, hardware demo) pending.

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
pytest tests                                   # unit tests (FK)
python simulation/validate_fk_mujoco.py        # FK vs MuJoCo, writes results/
python tools/plot_kinematics.py                # kinematic diagram figure
python tools/derive_poe_from_urdf.py           # re-derive PoE parameters from the URDF
```

Final demonstration (Phase 3, not yet implemented): `simulation/mujoco_demo.py`, `hardware/lerobot_demo.py`.

## Layout

| Path | Content |
|---|---|
| `src/so101_model.py` | PoE parameters: screw axes, home pose, joint limits |
| `src/forward_kinematics.py` | `forward_kinematics(q)`, `forward_kinematics_pose(q)` |
| `simulation/validate_fk_mujoco.py` | FK validation against the MuJoCo model |
| `tests/test_fk.py` | FK unit tests |
| `tools/` | Derivation of the model from the URDF, figure generation |
| `docs/phase1_kinematic_model.md` | Frames, PoE derivation, validation (report section 1) |
| `results/` | Figures and validation data |
| `models/so101/` | Official URDF, MuJoCo XML and meshes |

## Conventions

Metres and radians. Base frame {0}: x forward, y left, z up. End-effector frame = URDF `gripper_frame_link`. Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll. See `docs/phase1_kinematic_model.md`.
