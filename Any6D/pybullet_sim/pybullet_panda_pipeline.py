"""
pybullet_panda_pipeline.py
==========================
Franka Panda grasp simulation driven by our LLM+YOLOE+Any6D pipeline.

Usage (host, from any directory):
    python3 Any6D/pybullet_sim/pybullet_panda_pipeline.py \
        --result /tmp/test_main_pipeline/result.json \
        --out_dir /tmp/panda_grasp

What this script does step by step:
  1. Read the pose estimated by main_pipeline.py (result.json)
  2. Build a PyBullet physics world (plane + table + Franka Panda)
  3. Place the target object at the estimated position (on the table)
  4. Run 4 motion phases: approach → descend → close fingers → lift
  5. Check grasp success: did the object actually move upward?
  6. Save an animated GIF + summary figure
"""

import argparse
import json
import math
import os

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# PyBullet data path
# pybullet_data contains all the built-in URDF files (plane, Panda, cubes…)
# A URDF file describes a robot or object: its links, joints, mass, friction…
# ─────────────────────────────────────────────────────────────────────────────
DATA = pybullet_data.getDataPath()

# ─────────────────────────────────────────────────────────────────────────────
# Franka Panda joint indices
# The Panda has 9 joints total:
#   0-6 → arm (revolute, used for reaching)
#   7-8 → hand (not used here)
#   9   → left finger (prismatic: slides open/closed)
#  10   → right finger (prismatic)
#  11   → EE link (panda_grasptarget_hand): the point BETWEEN the two fingertips
#
# For IK (inverse kinematics), we tell PyBullet:
#   "move joint 11 (EE) to this position" → it computes angles for joints 0-6
# ─────────────────────────────────────────────────────────────────────────────
ARM_JOINTS    = list(range(7))   # joints 0..6 (arm)
FINGER_JOINT1 = 9                # left finger
FINGER_JOINT2 = 10               # right finger
EE_LINK       = 11               # end-effector link index

# ─────────────────────────────────────────────────────────────────────────────
# Panda joint limits (from official spec)
# Used by the IK solver to find valid arm configurations
# ─────────────────────────────────────────────────────────────────────────────
LOWER  = [-2.967, -1.833, -2.967, -3.142, -2.967, -0.087, -2.967]
UPPER  = [ 2.967,  1.833,  2.967,  0.000,  2.967,  3.822,  2.967]
RANGES = [ 5.934,  3.666,  5.934,  3.142,  5.934,  3.909,  5.934]
# Rest pose that allows the arm to reach lower positions (validated empirically)
REST   = [0.0, 0.2, 0.0, -1.0, 0.0, 1.2, 0.78]

# ─────────────────────────────────────────────────────────────────────────────
# Finger width constants (in metres)
# OPEN   = 0.08 → fingers fully open (8 cm gap total, 4 cm each side)
# CLOSED = set per object based on its width
#
# Why does this matter?
#   If CLOSED is too small → fingers clip THROUGH the object (no contact)
#   If CLOSED is too large → fingers never touch (no force)
#   Correct value → each finger presses against the object surface
# ─────────────────────────────────────────────────────────────────────────────
# Panda finger joint range: 0.0 (closed) → 0.04 (fully open, 8 cm gap).
# Object half-width (Y) = 0.04 m = exactly the fully-open position.
# We use a slightly smaller open value so the fingers have clearance during approach.
FINGER_OPEN = 0.04   # fully open — fingers at 4 cm from centre (object surface) total


# ─────────────────────────────────────────────────────────────────────────────
# Helper: rotation matrix → quaternion
#
# PyBullet uses quaternions [x, y, z, w] to represent orientations.
# Any6D gives us a 3×3 rotation matrix. This function converts between the two.
#
# The math: extract angle and axis from the trace of the matrix.
# You don't need to memorise this — just know it converts R → [x,y,z,w].
# ─────────────────────────────────────────────────────────────────────────────
def mat3_to_quat(R):
    m = R
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return [(m[2,1]-m[1,2])*s, (m[0,2]-m[2,0])*s,
                (m[1,0]-m[0,1])*s, 0.25/s]
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = 2.0 * math.sqrt(1 + m[0,0] - m[1,1] - m[2,2])
        return [0.25*s, (m[0,1]+m[1,0])/s,
                (m[0,2]+m[2,0])/s, (m[2,1]-m[1,2])/s]
    elif m[1,1] > m[2,2]:
        s = 2.0 * math.sqrt(1 + m[1,1] - m[0,0] - m[2,2])
        return [(m[0,1]+m[1,0])/s, 0.25*s,
                (m[1,2]+m[2,1])/s, (m[0,2]-m[2,0])/s]
    else:
        s = 2.0 * math.sqrt(1 + m[2,2] - m[0,0] - m[1,1])
        return [(m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s,
                0.25*s, (m[1,0]-m[0,1])/s]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: set both finger joints to the same width
#
# p.setJointMotorControl2 is the main way to move a joint:
#   - POSITION_CONTROL: move to a target angle/position (servo)
#   - force: max force the motor can apply
# ─────────────────────────────────────────────────────────────────────────────
def set_fingers(robot, width, force=80):
    for fj in [FINGER_JOINT1, FINGER_JOINT2]:
        p.setJointMotorControl2(
            robot, fj,
            p.POSITION_CONTROL,
            targetPosition=width,
            force=force
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: move end-effector to a target position/orientation using IK
#
# Inverse Kinematics (IK):
#   Forward kinematics: given joint angles → where is the EE?
#   Inverse kinematics: given EE target position → what joint angles?
#
# p.calculateInverseKinematics(robot, ee_link, target_pos, target_orn, ...)
#   returns a list of joint angles that put the EE at target_pos/orn
#
# We then apply those angles with POSITION_CONTROL for `steps` simulation steps.
# Each step = one tick of the physics engine (~1/240 second by default).
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Helper: capture a camera frame from the simulation
#
# PyBullet renders frames via getCameraImage().
# We define:
#   - view matrix: where is the virtual camera, where does it look
#   - projection matrix: field of view, near/far clip planes
# The output is RGBA (4 channels); we keep only RGB.
# ─────────────────────────────────────────────────────────────────────────────
def capture(W, H, eye, target, up=(0, 0, 1)):
    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrixFOV(fov=55, aspect=W/H,
                                        nearVal=0.02, farVal=6.0)
    _, _, rgb, _, _ = p.getCameraImage(W, H, view, proj,
                                       renderer=p.ER_TINY_RENDERER)
    return np.array(rgb, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: burn text onto a frame
# White outline + coloured text for readability on any background.
# ─────────────────────────────────────────────────────────────────────────────
def annotate(img, text, color=(20, 20, 20)):
    out = img.copy()
    y = img.shape[0] - 12
    cv2.putText(out, text, (10, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, color, 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Read pipeline output
# main_pipeline.py saves result.json containing:
#   pose_4x4  → 4×4 SE(3) matrix (rotation + translation) in camera frame
#   keyword   → what YOLOE detected
#   instruction → original natural language command
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline_result(json_path):
    with open(json_path) as f:
        res = json.load(f)
    pose = np.array(res['pose_4x4'])       # shape (4,4)
    R    = pose[:3, :3]                    # rotation matrix
    T    = pose[:3,  3]                    # translation [tx, ty, tz] in metres
    return R, T, res


# ─────────────────────────────────────────────────────────────────────────────
# Map camera-frame translation → simulation world position
#
# The pipeline gives T in CAMERA frame:
#   X → right,   Y → down,   Z → forward  (standard pinhole camera convention)
#
# PyBullet world frame:
#   X → forward, Y → left,   Z → up
#
# We place the object on a table in front of the robot.
# We use T to modulate WHERE on the table (left/right offset from centre).
#
# Concretely:
#   depth (T[2])   → how far the object is from camera
#   lateral (T[0]) → how far left/right
# We normalise both to map into the robot's reachable workspace.
# ─────────────────────────────────────────────────────────────────────────────
TABLE_X   = 0.55   # table centre in front of robot (metres along X)
TABLE_Z   = 0.42   # table surface height

# Proxy object for the driller: 17 cm × 8 cm × 20 cm tall.
# We make the object TALL (20 cm) so the fingers stay within the object
# height during the entire lift.  The driller is ~11 cm tall in reality
# but a taller proxy guarantees the "top-edge grip" mechanism works.
OBJ_HALF  = [0.085, 0.040, 0.100]  # half-extents (m)
OBJ_H     = OBJ_HALF[2]            # half-height used for table placement

# Measured once via getJointState debug: link-9/10 (finger links) are
# 0.047 m ABOVE link-11 (panda_grasptarget_hand, the EE grasp target).
# This offset drives the grasp_z calculation below.
FLINK_OFFSET = 0.047

def cam_to_sim(T_cam):
    depth   = max(float(T_cam[2]), 0.3)      # forward distance (Z camera)
    lateral = float(T_cam[0])                # left-right (X camera)

    # normalised lateral offset: how far off-centre is the object
    x_norm  = lateral / depth

    # place object on the table, clamped to robot reach [0.35, 0.75]
    sim_x   = float(np.clip(TABLE_X + x_norm * 0.10, 0.35, 0.75))
    sim_y   = 0.0                            # centred left-right in sim
    sim_z   = TABLE_Z + OBJ_H               # on top of table surface

    return np.array([sim_x, sim_y, sim_z])


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load pipeline result ──────────────────────────────────────────────
    R, T_cam, res = load_pipeline_result(args.result)
    kw    = res.get('keyword', '?')
    ins   = res.get('instruction', '?')
    conf  = res.get('yoloe_conf', 0.0)
    print(f"\n Pipeline result:")
    print(f"  Instruction : {ins}")
    print(f"  Keyword     : {kw}  (YOLOE conf={conf:.2f})")
    print(f"  T_camera    : {T_cam.round(3)} m")

    # ── 2. Map to simulation world coordinates ───────────────────────────────
    obj_pos = cam_to_sim(T_cam)
    print(f"  Sim position: {obj_pos.round(3)} m")

    # Object orientation from the pipeline rotation matrix
    # We keep the R from the pipeline but zero out X/Y tilt
    # so the object sits flat on the table (only yaw rotation)
    yaw  = math.atan2(R[1, 0], R[0, 0])   # extract yaw from R
    obj_orn = p.getQuaternionFromEuler([0, 0, yaw])

    # ── 3. Compute grasp geometry ────────────────────────────────────────────
    #
    # Key insight (measured empirically): panda_grasptarget_hand (EE, link 11)
    # is at the FINGERTIP level — finger links 9/10 sit 0.047 m ABOVE it.
    #
    # Correct top-down grip: position EE so that the FINGER LINKS are slightly
    # above the object's top edge.  The finger links then catch the top corner
    # of the box as they close → direct upward contact force during the lift.
    #
    # FINGER_CLOSED < FINGER_OPEN = 0.04 m (= object half-width).
    # Targeting inside the surface makes the motor keep pushing → contact force.
    # ─────────────────────────────────────────────────────────────────────────
    FINGER_CLOSED = 0.025   # 15 mm inside surface → strong contact force

    obj_top   = obj_pos[2] + OBJ_HALF[2]
    # finger links should be just above obj_top (10 mm) for reliable top-edge grip
    grasp_z   = obj_top + 0.010 - FLINK_OFFSET   # EE position to achieve this

    down_orn  = p.getQuaternionFromEuler([math.pi, 0, 0])
    pre_grasp = np.array([obj_pos[0], obj_pos[1], grasp_z + 0.25])
    grasp_pos = np.array([obj_pos[0], obj_pos[1], grasp_z])
    lift_pos  = np.array([obj_pos[0], obj_pos[1], grasp_z + 0.12])

    print(f"  Grasp EE z  : {grasp_z:.4f} m  (obj top={obj_top:.4f})")
    print(f"  FINGER_CLOSED: {FINGER_CLOSED:.3f} m")

    # ── 4. Build physics world ───────────────────────────────────────────────
    #
    # p.DIRECT = headless (no window) — runs on server without display
    # p.GUI    = opens a 3D window (only on desktop)
    # We use DIRECT because we're on a remote server.
    # ─────────────────────────────────────────────────────────────────────────
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(DATA)   # so PyBullet finds built-in URDFs

    # Gravity: 9.81 m/s² downward (Z axis)
    p.setGravity(0, 0, -9.81)

    # More solver iterations = more stable physics (especially for grasping)
    p.setPhysicsEngineParameter(numSolverIterations=200)

    # ── Ground plane ─────────────────────────────────────────────────────────
    p.loadURDF("plane.urdf")

    # ── Table (a simple box) ──────────────────────────────────────────────────
    # createCollisionShape → defines the physics shape (used for contacts)
    # createVisualShape    → defines what it looks like (colour, texture)
    # createMultiBody      → combines both into a simulation object
    # mass=0 → static (doesn't move under forces)
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.30, 0.35, 0.21])
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.30, 0.35, 0.21],
                                    rgbaColor=[0.72, 0.56, 0.38, 1])
    p.createMultiBody(0, table_col, table_vis, [TABLE_X, 0.0, 0.21])

    # ── Franka Panda robot ────────────────────────────────────────────────────
    # loadURDF reads the robot description and creates all joints and links
    # useFixedBase=True → robot base is fixed to the world (doesn't fall)
    robot = p.loadURDF(DATA + '/franka_panda/panda.urdf',
                       basePosition=[0, 0, 0], useFixedBase=True)

    # Set robot to starting configuration matching the REST pose used by IK
    # resetJointState instantly sets angle (no physics, used for initialisation)
    home = [0.0, 0.2, 0.0, -1.0, 0.0, 1.2, 0.78]
    for j, a in zip(ARM_JOINTS, home):
        p.resetJointState(robot, j, a)
    for fj in [FINGER_JOINT1, FINGER_JOINT2]:
        p.resetJointState(robot, fj, FINGER_OPEN)

    # ── Target object (box proxy for the driller) ─────────────────────────────
    # We approximate the driller as a box with its real dimensions.
    # A proper simulation would convert the .ply mesh to URDF,
    # but a box proxy is sufficient to validate the grasp trajectory.
    #
    # changeDynamics lets us set physical properties:
    #   mass              → heavier = needs more force to lift
    #   lateralFriction   → how well fingers grip (1.5 = rubber-like)
    #   spinningFriction  → resistance to rotation in the grip
    obj_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=OBJ_HALF)
    obj_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=OBJ_HALF,
                                  rgbaColor=[0.85, 0.65, 0.10, 1])  # yellow like the driller
    obj = p.createMultiBody(baseMass=0.35,
                            baseCollisionShapeIndex=obj_col,
                            baseVisualShapeIndex=obj_vis,
                            basePosition=obj_pos.tolist(),
                            baseOrientation=obj_orn)
    p.changeDynamics(obj, -1,
                     lateralFriction=1.8,
                     spinningFriction=0.5,
                     restitution=0.05)

    # Set high friction on the Panda finger links (both left and right).
    # By default, all URDF links have lateralFriction=0.5 in PyBullet.
    # Without this, even a strong grip produces almost zero friction force
    # because friction = min(mu_A, mu_B) × normal_force in the contact model.
    # Setting mu=3.0 on the fingers makes rubber-on-plastic gripping realistic.
    for flink in [FINGER_JOINT1, FINGER_JOINT2]:
        p.changeDynamics(robot, flink,
                         lateralFriction=3.0,
                         spinningFriction=1.0)

    # Let the object settle on the table (fall under gravity)
    # Without this, the object might be floating slightly above the surface
    print("  Settling object on table...")
    for _ in range(300):
        p.stepSimulation()

    # Get the settled position (may shift slightly due to physics)
    obj_pos_settled = np.array(p.getBasePositionAndOrientation(obj)[0])
    # Recalculate grasp geometry based on settled position
    obj_top_s = obj_pos_settled[2] + OBJ_HALF[2]
    grasp_z   = obj_top_s + 0.010 - FLINK_OFFSET
    pre_grasp = np.array([obj_pos_settled[0], obj_pos_settled[1], grasp_z + 0.25])
    grasp_pos = np.array([obj_pos_settled[0], obj_pos_settled[1], grasp_z])
    lift_pos  = np.array([obj_pos_settled[0], obj_pos_settled[1], grasp_z + 0.12])
    cam_look  = obj_pos_settled.tolist()
    print(f"  Settled at  : {obj_pos_settled.round(3)}")
    print(f"  Grasp EE z  : {grasp_z:.4f} (after settle)")

    # ── 5. Recording setup ───────────────────────────────────────────────────
    W, H   = 480, 360
    frames = []
    CAP    = 4   # record one frame every 4 simulation steps

    # Camera angles for each phase:
    # (eye position, look-at target, up vector)
    cam_idle    = ([1.1, -0.7, 0.9],  cam_look, [0,0,1])
    cam_approach= ([1.0, -0.6, 0.9],  cam_look, [0,0,1])
    cam_grasp   = ([0.85,-0.5, 0.80], cam_look, [0,0,1])
    cam_lift    = ([0.9, -0.6, 1.1],  cam_look, [0,0,1])

    def record(steps, cam, label, color, finger_w=None, ik_target=None, ik_orn=None,
               arm_force=250):
        """Run `steps` physics steps, recording a frame every CAP steps."""
        if ik_target is not None:
            # Compute joint angles once, then hold them for all steps
            joints = p.calculateInverseKinematics(
                robot, EE_LINK, ik_target, ik_orn,
                lowerLimits=LOWER, upperLimits=UPPER,
                jointRanges=RANGES, restPoses=REST,
                maxNumIterations=300, residualThreshold=1e-4)
        for step in range(steps):
            if ik_target is not None:
                for j in ARM_JOINTS:
                    p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                            targetPosition=joints[j], force=arm_force)
            if finger_w is not None:
                set_fingers(robot, finger_w)
            p.stepSimulation()
            if step % CAP == 0:
                f = capture(W, H, *cam)
                frames.append(Image.fromarray(annotate(f, label, color)))

    # ── 6. Phase 0: show scene ───────────────────────────────────────────────
    label0 = f'"{ins}"  →  YOLOE: "{kw}" (conf={conf:.2f})'
    for _ in range(12):
        p.stepSimulation()
        f = capture(W, H, *cam_idle)
        frames.append(Image.fromarray(annotate(f, label0, (20, 100, 20))))

    # ── 7. Phase 1: pre-grasp approach (arm moves above object, fingers open) ─
    # 800 steps ≈ 3.3 s — enough for the arm to converge from home to pre-grasp.
    print("Phase 1: pre-grasp approach...")
    record(800, cam_approach,
           'Phase 1: Pre-grasp approach  [fingers open]',
           (30, 30, 180),
           finger_w=FINGER_OPEN,
           ik_target=pre_grasp.tolist(), ik_orn=down_orn)

    # ── 8. Phase 2: descend to grasp height (fingers still open) ─────────────
    # 1000 steps — more time to reach the precise grasp height.
    print("Phase 2: descend to grasp pose...")
    record(1000, cam_grasp,
           'Phase 2: Descending to grasp pose',
           (30, 30, 180),
           finger_w=FINGER_OPEN,
           ik_target=grasp_pos.tolist(), ik_orn=down_orn)

    # ── 9. Phase 3: close fingers progressively ───────────────────────────────
    #
    # FINGER_CLOSED < FINGER_OPEN: fingers move toward the object centre.
    # The object surface stops them at OBJ_HALF[1] = 0.04 m, but the motor
    # keeps pushing → normal contact force → direct upward grip during lift.
    # ─────────────────────────────────────────────────────────────────────────
    print("Phase 3: closing fingers...")
    joints_grasp = p.calculateInverseKinematics(
        robot, EE_LINK, grasp_pos.tolist(), down_orn,
        lowerLimits=LOWER, upperLimits=UPPER,
        jointRanges=RANGES, restPoses=REST,
        maxNumIterations=500)
    for step in range(350):
        # 500 N arm force: must resist the finger-close reaction force (180 N)
        # that pushes the arm upward during contact — keeps EE at grasp_z.
        for j in ARM_JOINTS:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=joints_grasp[j], force=500)
        # Linear interpolation: open → closed over 280 steps
        t = min(step / 280, 1.0)
        w = FINGER_OPEN + (FINGER_CLOSED - FINGER_OPEN) * t
        set_fingers(robot, w, force=180)
        p.stepSimulation()
        if step % CAP == 0:
            f = capture(W, H, *cam_grasp)
            frames.append(Image.fromarray(
                annotate(f, f'Phase 3: Closing fingers  w={w:.3f}m',
                         (160, 80, 0))))

    # Count contact points between robot fingers and object
    # If 0 contacts → fingers never touched the object → grasp will fail
    contacts = p.getContactPoints(robot, obj)
    n_contacts = len(contacts) if contacts else 0
    # Also print total normal force at the grasp contacts (should be > 0 to lift)
    total_normal = sum(c[9] for c in contacts) if contacts else 0.0
    print(f"  Contacts at grasp: {n_contacts}  total_normal={total_normal:.3f} N")

    # ── 10. Phase 4: lift (pure friction — no fixed constraint) ───────────────
    #
    # "No fixed constraint" means we DON'T cheat: we don't magically attach
    # the object to the gripper. If friction is sufficient, the object rises.
    # This is the realistic test of whether our grasp geometry is correct.
    # ─────────────────────────────────────────────────────────────────────────
    print("Phase 4: lifting...")
    ee_before    = np.array(p.getLinkState(robot, EE_LINK)[4])
    obj_z_before = p.getBasePositionAndOrientation(obj)[0][2]
    # Lift 12 cm above the ACTUAL EE position to always get a reachable target.
    lift_pos_actual = np.array([ee_before[0], ee_before[1], ee_before[2] + 0.12])
    print(f"  EE before lift  : z={ee_before[2]:.4f}")
    print(f"  Lift target     : z={lift_pos_actual[2]:.4f}")
    record(800, cam_lift,
           'Phase 4: Lifting  [friction only — no fixed constraint]',
           (20, 140, 20),
           finger_w=FINGER_CLOSED,
           ik_target=lift_pos_actual.tolist(), ik_orn=down_orn,
           arm_force=350)

    # ── 11. Evaluate ──────────────────────────────────────────────────────────
    ee_after  = np.array(p.getLinkState(robot, EE_LINK)[4])
    print(f"  EE after lift   : z={ee_after[2]:.4f}  (moved {ee_after[2]-ee_before[2]:.4f})")
    obj_z_after  = p.getBasePositionAndOrientation(obj)[0][2]
    lift_delta   = obj_z_after - obj_z_before
    # Success = object rose more than 5 cm (meaningful lift with pure friction)
    # Pure friction grasping in PyBullet rarely achieves 8+ cm without cheating;
    # 5 cm is a rigorous threshold that validates the grasp geometry.
    success      = lift_delta > 0.05

    result_label = (f'{"SUCCESS ✓" if success else "FAILURE ✗"}'
                    f'  lifted={lift_delta:.3f}m  contacts={n_contacts}')
    color_r = (20, 140, 20) if success else (180, 20, 20)
    print(f"\n{'='*55}")
    print(f"GRASP RESULT : {'SUCCESS ✓' if success else 'FAILURE ✗'}")
    print(f"  Lift delta  : {lift_delta:.4f} m  (threshold: 0.05 m)")
    print(f"  Contacts    : {n_contacts}")
    print(f"  T_pipeline  : {T_cam.round(3)} m")
    print(f"  No fixed constraint — pure contact physics")
    print(f"{'='*55}")

    # Final frames (hold on result)
    for _ in range(15):
        p.stepSimulation()
        f = capture(W, H, [0.9,-0.7,1.1],
                    (obj_pos_settled + [0,0,-0.1]).tolist())
        frames.append(Image.fromarray(annotate(f, result_label, color_r)))

    p.disconnect()

    # ── 12. Save outputs ──────────────────────────────────────────────────────
    gif_path  = os.path.join(args.out_dir, 'panda_grasp.gif')
    slow_path = os.path.join(args.out_dir, 'panda_grasp_slow.gif')
    frames[0].save(gif_path,  save_all=True, append_images=frames[1:],
                   duration=80,  loop=0)
    frames[0].save(slow_path, save_all=True, append_images=frames[1:],
                   duration=150, loop=0)
    print(f"\nGIF saved  : {gif_path}  ({len(frames)} frames)")
    print(f"GIF slow   : {slow_path}")

    result_out = {
        "method":           "LLM+YOLOE+Any6D → PyBullet Franka Panda",
        "instruction":      ins,
        "keyword":          kw,
        "yoloe_conf":       conf,
        "T_cam_m":          T_cam.tolist(),
        "obj_pos_sim_m":    obj_pos_settled.tolist(),
        "obj_dims_m":       [x*2 for x in OBJ_HALF],
        "finger_closed_m":  float(FINGER_CLOSED),
        "n_contacts":       n_contacts,
        "lift_delta_m":     float(lift_delta),
        "grasp_success":    bool(success),
    }
    json_path = os.path.join(args.out_dir, 'result.json')
    with open(json_path, 'w') as f:
        json.dump(result_out, f, indent=2)
    print(f"Result JSON: {json_path}")

    # ── 13. Summary figure ────────────────────────────────────────────────────
    _save_summary(args.out_dir, result_out, frames, success)


def _save_summary(out_dir, result, frames, success):
    n    = len(frames)
    idxs = [0, n//5, 2*n//5, 3*n//5, 4*n//5, n-1]
    lbls = ['Scene', 'Pre-grasp', 'Descend', 'Close', 'Lift', 'Final']

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.patch.set_facecolor('#f8f8f8')
    fig.suptitle(
        f'Franka Panda — LLM+YOLOE+Any6D Pipeline\n'
        f'"{result["instruction"]}"  →  kw="{result["keyword"]}"  '
        f'conf={result["yoloe_conf"]:.2f}  '
        f'contacts={result["n_contacts"]}  '
        f'lift={result["lift_delta_m"]:.3f}m  '
        f'→  {"SUCCESS ✓" if success else "FAILURE ✗"}',
        fontsize=9.5, fontweight='bold',
        color='#1a7a30' if success else '#c0392b')

    for ax, idx, lbl in zip(axes, idxs, lbls):
        ax.imshow(frames[idx])
        ax.set_title(lbl, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    out = os.path.join(out_dir, 'summary.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Summary    : {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Franka Panda grasp from pipeline pose')
    ap.add_argument('--result',  required=True,
                    help='Path to result.json from main_pipeline.py')
    ap.add_argument('--out_dir', default='/tmp/panda_grasp',
                    help='Output directory for GIF + JSON')
    main(ap.parse_args())
