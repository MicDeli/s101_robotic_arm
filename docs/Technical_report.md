# Technical Report – Section 1: Kinematic Model of the LeRobot SO-101

Nov 1, 2026&#32;

## 1.1 Robot and coordinate frames

The SO-101 follower arm has five revolute joints that position and orient the tool, plus a gripper-jaw servo that does not move the tool frame. The kinematic chain therefore has n = 5 joints and the tool pose is an element of SE(3). All numbers below come from the official CAD-derived model of the SO-ARM100 repository (`so101_new_calib` URDF and MuJoCo XML), and all lengths are in metres and angles in radians unless a figure says otherwise.

&#91;image: SO-101 with labelled parts, joints and key dimensions\]

*Figure 1 – The SO-101 rendered from the CAD meshes at q = 0: labelled parts, joint numbers and axes, and key dimensions in mm (side view projections; top view shows the lateral offsets).*

**Frames.** Three kinds of frames are used:

- **Base frame {0}**: origin at the origin of `base_link`, on the table plane at the centre of the base plate. x points forward (towards the arm), y to the left, z up (right-handed).
- **End-effector frame {e}**: the URDF frame `gripper_frame_link`, located at the tool point between the two jaws. At q = 0 its z-axis points forward along the gripper (the approach direction), its y-axis points left, and its x-axis points down, tilted by 2.79° (see §1.2).
- **Joint axes**: each joint i rotates about a unit axis w\_i through a point p\_i, both expressed in {0} at the zero configuration. A positive joint angle is a right-hand rotation about w\_i, the same sign convention as the URDF and the MuJoCo model, so joint values can be exchanged with MuJoCo without conversion.

**Zero configuration.** q = 0 is the "new calibration" zero of the official model: upper arm vertical, forearm horizontal pointing forward, gripper pointing forward (the "L" pose).

**Joints.** Table 1 lists the five joints with their axes at q = 0 and their limits.

| i | Joint (motor ID) | Axis w\_i at q = 0 | Point p\_i on axis \[m\] | Limits \[rad\] | Limits \[deg\] |
| --- | --- | --- | --- | --- | --- |
| 1 | shoulder\_pan (1) | (0, 0, −1) | (0.0388, 0, 0.0624) | ±1.920 | ±110.0 |
| 2 | shoulder\_lift (2) | (0, 1, 0) | (0.0692, −0.0183, 0.1166) | ±1.745 | ±100.0 |
| 3 | elbow\_flex (3) | (0, 1, 0) | (0.0972, −0.0183, 0.2292) | ±1.690 | ±96.8 |
| 4 | wrist\_flex (4) | (0, 1, 0) | (0.2321, −0.0183, 0.2344) | ±1.658 | ±95.0 |
| 5 | wrist\_roll (5) | (−1, 0, 0) | (0.2932, −0.0002, 0.2344) | −2.744 … 2.841 | −157.2 … 162.8 |

*Table 1 – Joint axes and limits of the SO-101 in the base frame at q = 0 (source: official URDF).*

**Out-of-plane offset.** The shoulder-lift, elbow and wrist-flex axes are parallel and lie in a vertical plane 18.3 mm to the −y side of the pan axis, because the shoulder-lift motor is mounted on the side of the shoulder bracket. The wrist link returns the roll axis to y = −0.2 mm, i.e. back to the centreline. This offset is invisible in a side view, but it is part of the geometry and the reason a simple planar two-link model is not sufficient for this arm.

&#91;image: Kinematic skeleton of the SO-101\]

*Figure 2 – Kinematic skeleton: (a) side view at q = 0; (b) side view and (c) top view at q = (0.5, −0.4, 0.9, 0.3, −1.2) rad. Frame triads: x red, y green, z blue.*

## 1.2 Product-of-Exponentials representation

The arm is modelled with the Product-of-Exponentials (PoE) formulation in the space (base) frame, not with Denavit–Hartenberg (DH) parameters. PoE needs only the axis of every joint and the home pose of the end-effector, and all of these can be read directly from the CAD-derived model. A DH table would need extra offset parameters and a fixed tool transform to capture the 18.3 mm out-of-plane offset and the 2.79° wrist tilt, and its frame assignment would not match the URDF frames. The screw axes also give the space Jacobian needed for inverse kinematics in the next phase without further derivation.

**Screw axes.** Each revolute joint i is described by the screw axis S\_i = (w\_i, v\_i), where w\_i is the unit rotation axis and v\_i = −w\_i × p\_i is the linear part for a point p\_i on the axis, all in {0} at q = 0:

| i | Joint | w\_i | v\_i \[m\] |
| --- | --- | --- | --- |
| 1 | shoulder\_pan | (0, 0, −1) | (0, 0.0388353, 0) |
| 2 | shoulder\_lift | (0, 1, 0) | (−0.1166, 0, 0.0692345) |
| 3 | elbow\_flex | (0, 1, 0) | (−0.22917, 0, 0.0972345) |
| 4 | wrist\_flex | (0, 1, 0) | (−0.23437, 0, 0.2321345) |
| 5 | wrist\_roll | (−1, 0, 0) | (0, −0.23437, −0.0001778) |

*Table 2 – Screw axes S\_i = (w\_i, v\_i) of the SO-101 in the base frame.*

**Home pose.** The pose of {e} in {0} at q = 0 is

```latex
M = \begin{bmatrix} 0 & 0 & 1 & 0.39136 \\ \sin\alpha & \cos\alpha & 0 & -0.00001 \\ -\cos\alpha & \sin\alpha & 0 & 0.22647 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \qquad \alpha = 0.0486795\ \text{rad} = 2.79^\circ
```

The tilt α is the CAD mounting offset of the wrist-roll servo. It is a real feature of the model and is kept, not rounded away: ignoring it would shift the orientation of {e} by 2.79°.

**How the parameters were obtained.** `tools/derive_poe_from_urdf.py` composes the fixed joint-to-joint transforms of the URDF at q = 0, takes the z-axis of each joint frame as w\_i and its origin as p\_i, and appends the fixed `gripper_frame_joint` transform to obtain M. The URDF stores angles with five digits (1.5708 instead of π/2), so any angle within 10⁻⁴ rad of a multiple of π/2 is snapped to it. Without this the axes would be tilted by about 10⁻⁵ rad. The only angle that is not a right angle, the 0.0486795 rad wrist-roll offset, is kept unchanged.

## 1.3 Forward kinematics derivation

The end-effector pose for joint angles q = (q1, …, q5) is the product of five screw motions applied to the home pose:

```latex
T_{0e}(q) = e^{[S_1]q_1}\, e^{[S_2]q_2}\, e^{[S_3]q_3}\, e^{[S_4]q_4}\, e^{[S_5]q_5}\, M
```

**Why this product.** With every joint at zero the tool pose is M. Rotating the last joint (wrist roll) by q5 moves the tool rigidly about the fixed axis S\_5, giving e^{\[S5\]q5} M. Rotating joint 4 by q4 then carries the tool and axis 5 together about S\_4, which is still the axis as it was at q = 0 because it is expressed in the fixed space frame. Repeating this from the distal joint to the proximal one yields the product above, with joint 1 acting last on everything (Lynch and Park, *Modern Robotics*, ch. 4).

**Matrix exponential of a revolute joint.** For a screw axis with |w| = 1 and angle θ, Rodrigues' formula gives the closed form that is implemented, with \[w\] the skew-symmetric matrix of w:

```latex
e^{[S]\theta} = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}, \quad R = I + \sin\theta\,[w] + (1-\cos\theta)\,[w]^2, \quad p = \bigl(I\theta + (1-\cos\theta)[w] + (\theta - \sin\theta)[w]^2\bigr)\, v
```

The tool position is p\_e(q) = T\_{0e}\[0:3, 3\] and its orientation is R\_e(q) = T\_{0e}\[0:3, 0:3\] ∈ SO(3).

**Planar reduction as an analytic check.** With pan = roll = 0, joints 2–4 are parallel y-axes and the tool stays in the x–z plane at constant y. Nesting the three rotations about the points P\_1, P\_2, P\_3 on the axes gives, with R\_y(θ) the rotation about +y and the link vectors expressed at q = 0:

```latex
p_{xz} = P_1 + R_y(q_2)\Bigl[\ell_1 + R_y(q_3)\bigl[\ell_2 + R_y(q_4)\,\ell_3\bigr]\Bigr], \qquad \ell_1 = \begin{bmatrix}0.0280\\0.1126\end{bmatrix},\ \ell_2 = \begin{bmatrix}0.1349\\0.0052\end{bmatrix},\ \ell_3 = \begin{bmatrix}0.1592\\-0.0079\end{bmatrix}\ \text{m}
```

The tool pitch is the sum q2 + q3 + q4 applied to the home orientation. This closed form agrees with the PoE product to 1.7·10⁻¹⁶ m over 200 random planar configurations, which checks the screw axes of joints 2–4 and the home pose independently of the exponential code.

**Worked example.** For q = (0.5, −0.4, 0.9, 0.3, −1.2) rad the product gives

```latex
p_e = \begin{bmatrix}0.2549\\-0.1098\\0.0550\end{bmatrix}\ \text{m}, \qquad R_e = \begin{bmatrix}0.2555 & 0.7489 & 0.6114\\0.9413 & -0.0484 & -0.3340\\-0.2206 & 0.6609 & -0.7174\end{bmatrix}
```

MuJoCo returns the same position to within 4.5·10⁻⁹ m. A sign check: a pure pan of +45° moves the tool from (0.3914, 0, 0.2265) to (0.2881, −0.2493, 0.2265), i.e. clockwise seen from above, as expected for the axis w\_1 = (0, 0, −1).

**Link to the Jacobian.** The same screw axes give the space Jacobian column by column, J\_i(q) = Ad(e^{\[S1\]q1} ⋯ e^{\[S(i−1)\]q(i−1)}) S\_i, which is what the inverse-kinematics phase will use.

## 1.4 Implementation

The forward kinematics is implemented in Python (NumPy) as a small package; the model constants are kept separate from the algorithm so that a calibrated parameter set could replace them without touching the code.

| File | Content |
| --- | --- |
| `src/so101_model.py` | Joint names and limits, w\_i, p\_i, screw axes S\_i, home pose M |
| `src/forward_kinematics.py` | `forward_kinematics(q)` returns the 4×4 pose; `forward_kinematics_pose(q)` returns (position, rotation); `within_joint_limits(q)` |
| `tools/derive_poe_from_urdf.py` | Reproduces all model constants from the official URDF |
| `tests/test_fk.py` | 12 unit tests, including a comparison with MuJoCo |
| `simulation/validate_fk_mujoco.py` | FK versus MuJoCo over named and random configurations |

*Table 3 – Files of the Phase 1 implementation.*

The interface takes the five arm joint angles in radians; a vector of any other length raises a `ValueError`, so the gripper angle cannot be passed by mistake.

```python
from src.forward_kinematics import forward_kinematics_pose

position, rotation = forward_kinematics_pose([0.5, -0.4, 0.9, 0.3, -1.2])  # rad
# position -> [ 0.2549, -0.1098, 0.0550 ] m,  rotation -> 3x3 matrix in SO(3)
```

The matrix exponential is evaluated with the closed-form Rodrigues expression of §1.3, not with a general-purpose `expm`; the unit tests confirm that both agree to 10⁻¹².

## 1.5 Validation

The forward kinematics agrees with the official MuJoCo model to 15 nm in position and 6.7·10⁻⁸ rad in orientation over 1000 random configurations, and the CAD link dimensions agree with measurements on the physical arm to within 4.6 mm. Three independent checks support this.

**Unit tests.** The 12 tests in `tests/test_fk.py` all pass:

| Check | What it verifies |
| --- | --- |
| Zero pose | FK(0) returns M exactly |
| SO(3) and homogeneity | R Rᵀ = I, det R = 1, last row (0, 0, 0, 1) for 200 random configurations |
| Screw axes | Unit-length w\_i and zero pitch (w\_i · v\_i = 0) |
| Exponential | Closed form equals `scipy.linalg.expm` to 10⁻¹² |
| Pure pan | Tool height and distance to the pan axis stay constant |
| Pure roll | Tool moves on a circle of radius 7.9 mm about the roll axis |
| Planar motion | With pan = roll = 0 the tool stays in the x–z plane |
| Shoulder lift | Distance from the lift axis to the tool is independent of q2 |
| Interface | Pose and matrix outputs are consistent; a wrong-length input raises |
| Joint limits | Helper accepts q = 0 and rejects a pan of 2.5 rad |
| MuJoCo match | 100 random configurations within 10⁻⁵ m and 10⁻⁵ rad |

*Table 4 – Unit tests of the forward kinematics.*

**Comparison with MuJoCo.** `simulation/validate_fk_mujoco.py` sets the joint angles in the official MuJoCo model, runs its kinematics and reads the pose of the `gripperframe` site. MuJoCo’s world frame coincides with {0}. Nine named configurations (zero pose, single-joint moves, a folded pose, mixed poses) and 1000 random configurations drawn uniformly inside the joint limits (seed 0) were compared:

| Metric (1000 random configurations) | Max | Mean | RMS |
| --- | --- | --- | --- |
| Position error \[m\] | 1.5·10⁻⁸ | 7.9·10⁻⁹ | 9.1·10⁻⁹ |
| Orientation error \[rad\] | 6.7·10⁻⁸ | 3.5·10⁻⁸ | 3.8·10⁻⁸ |

*Table 5 – FK versus MuJoCo. The nine named configurations are within 1.0·10⁻⁸ m and 4.2·10⁻⁸ rad.*

&#91;image: Histograms of position and orientation error\]

*Figure 3 – Distribution of the FK error against MuJoCo (position in nm, orientation in nrad).*

The residual is at the level of floating-point and URDF angle rounding, so the implementation, the axis signs and the frame definitions are correct. One convention difference had to be handled: MuJoCo’s `gripperframe` site and the URDF `gripper_frame_link` share the same origin but differ by a fixed rotation of −90° about the tool y-axis. This offset is constant to 10⁻¹⁴ over 500 random configurations. The model uses the URDF frame, which is the one LeRobot’s own kinematics targets, and the validation script compensates the offset and also prints the raw 90° difference.

**Comparison with the physical arm.** Because MuJoCo and the model both derive from the same CAD numbers, the comparison above does not show that the real arm matches the CAD. The link dimensions were therefore measured on the arm and compared with the model:

| Dimension | CAD \[mm\] | Measured \[mm\] | Difference \[mm\] |
| --- | --- | --- | --- |
| Pan axis → lift axis, forward | 30.4 | 35 | +4.6 |
| Pan axis → lift axis, up | 54.2 | 55 | +0.8 |
| Upper arm (lift → elbow axis) | 116.0 | 114.0 | −2.0 |
| Forearm (elbow → wrist-flex axis) | 135.0 | 139.0 | +4.0 |
| Wrist (wrist-flex → roll axis) | 63.7 | 64 | +0.3 |
| Roll axis → tool point | 98.4 | 99 | +0.6 |

*Table 6 – Link dimensions: CAD model versus measurement on the arm.*

All differences are within 5 mm. The 4.6 mm difference of the first row is consistent with a straight-line horizontal measurement, which includes the 18.3 mm sideways offset: √(30.4² + 18.3²) = 35.5 mm. The remaining differences of −2.0 and +4.0 mm are of the size expected from reading the centres of servo horns on printed parts. The CAD values were kept as the nominal model, since a hand measurement is not more accurate than the CAD.

## 1.6 Limitations and discussion

The model is a rigid-body kinematic model of the CAD geometry, accurate to a few millimetres on the real arm. Its main limitations are:

- **Same-source validation.** The MuJoCo comparison shares its geometry with the model, so it validates the implementation but not the hardware. The physical check in §1.5 covers link lengths only; a tool-position check at known table positions remains to be done in the hardware phase.
- **Not modelled.** Servo backlash, gear compliance, printed-part tolerances and calibration offsets are ignored. An angular error of 0.5° at the shoulder-lift joint moves the tool by about 3 mm at the 0.34 m distance between that axis and the tool, which is the same size as the dimension differences in Table 6.
- **Joint units and limits.** The LeRobot software reports joint values in its own units and calibration. A conversion to radians with the correct sign and zero offset is needed before commanding the arm, and the limits in Table 1 come from the simulation model and must be checked against the real arm.
- **Five degrees of freedom.** Five joints cannot reach an arbitrary pose in SE(3): the reachable tool poses form a five-dimensional set, so the inverse-kinematics phase must treat position and orientation targets accordingly. The gripper jaw is not part of the chain and does not change the tool frame.
- **Tool frame.** {e} is the URDF `gripper_frame_link`, placed at the middle of the jaws and 7.9 mm off the roll axis. A different tool point would only change M, not the screw axes.

### References

1. K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*, Cambridge University Press, 2017, ch. 3–4 (screw axes, product of exponentials).
2. TheRobotStudio, [SO-ARM100 repository, `Simulation/SO101`](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101): `so101_new_calib.urdf`, `so101_new_calib.xml` and meshes.
3. E. Todorov, T. Erez and Y. Tassa, “MuJoCo: A physics engine for model-based control,” *IEEE/RSJ IROS*, 2012.
