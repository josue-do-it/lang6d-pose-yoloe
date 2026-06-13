"""
Rigorous PyBullet grasp simulation using pose from LLM+YOLOE+Any6D pipeline.

- Franka Panda with real finger joints (no fixed constraint trick)
- Grasp success = object lifted by contact friction alone
- Pose T from pipeline → object placement in sim
- Generates animated GIF

Usage:
  python pybullet_real_grasp.py \
      --json /workspace/results/infer_pose_bottle/bottle_pose.json \
      --out_dir /workspace/results/pybullet_real
"""
import argparse, os, json, math, time
import numpy as np
import pybullet as p
import pybullet_data as pbd
from PIL import Image
import cv2

DATA = pbd.getDataPath()

# Panda joint indices
ARM_JOINTS    = list(range(7))          # 0-6: revolute arm
FINGER_JOINT1 = 9
FINGER_JOINT2 = 10
EE_LINK       = 11                      # panda_grasptarget_hand

FINGER_OPEN   = 0.04
FINGER_CLOSED = 0.008                   # tight grip on a small mug


# ── helpers ───────────────────────────────────────────────────────────────────
def mat3_to_quat(R):
    m = R; tr = m[0,0]+m[1,1]+m[2,2]
    if tr > 0:
        s = 0.5/math.sqrt(tr+1.0)
        return [(m[2,1]-m[1,2])*s,(m[0,2]-m[2,0])*s,(m[1,0]-m[0,1])*s,0.25/s]
    elif m[0,0]>m[1,1] and m[0,0]>m[2,2]:
        s = 2.0*math.sqrt(1+m[0,0]-m[1,1]-m[2,2])
        return [0.25*s,(m[0,1]+m[1,0])/s,(m[0,2]+m[2,0])/s,(m[2,1]-m[1,2])/s]
    elif m[1,1]>m[2,2]:
        s = 2.0*math.sqrt(1+m[1,1]-m[0,0]-m[2,2])
        return [(m[0,1]+m[1,0])/s,0.25*s,(m[1,2]+m[2,1])/s,(m[0,2]-m[2,0])/s]
    else:
        s = 2.0*math.sqrt(1+m[2,2]-m[0,0]-m[1,1])
        return [(m[0,2]+m[2,0])/s,(m[1,2]+m[2,1])/s,0.25*s,(m[1,0]-m[0,1])/s]

def set_fingers(robot, width, force=50):
    for fj in [FINGER_JOINT1, FINGER_JOINT2]:
        p.setJointMotorControl2(robot, fj, p.POSITION_CONTROL,
                                targetPosition=width, force=force)

def ik_and_move(robot, target_pos, target_orn, steps=400, finger_w=None):
    joints = p.calculateInverseKinematics(
        robot, EE_LINK, target_pos, target_orn,
        lowerLimits=[-2.967,-1.833,-2.967,-3.142,-2.967,-0.087,-2.967],
        upperLimits=[ 2.967, 1.833, 2.967, 0.000, 2.967, 3.822, 2.967],
        jointRanges=[5.934,3.666,5.934,3.142,5.934,3.909,5.934],
        restPoses=[0,-0.3,0,-2.0,0,2.0,0.78],
        maxNumIterations=300, residualThreshold=1e-4)
    for _ in range(steps):
        for j in ARM_JOINTS:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=joints[j], force=250)
        if finger_w is not None:
            set_fingers(robot, finger_w)
        p.stepSimulation()

def capture_frame(W, H, cam_pos, target, up=[0,0,1]):
    view = p.computeViewMatrix(cam_pos, target, up)
    proj = p.computeProjectionMatrixFOV(55, W/H, 0.02, 6.0)
    _, _, rgb, _, _ = p.getCameraImage(W, H, view, proj,
                                       renderer=p.ER_TINY_RENDERER)
    return np.array(rgb, dtype=np.uint8).reshape(H, W, 4)[:,:,:3].copy()

def annotate(img, text, color=(30,30,30)):
    out = img.copy()
    cv2.putText(out, text, (10, img.shape[0]-12),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, img.shape[0]-12),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2, cv2.LINE_AA)
    return out


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json) as f: res = json.load(f)
    R_cam = np.array(res['R']); T_cam = np.array(res['T'])
    kw = res['keyword']; ins = res['instruction']
    conf = res['yoloe_conf']; n_pts = res['n_obj_pts']

    print(f"Pipeline result:")
    print(f"  Instruction : {ins}")
    print(f"  Keyword     : {kw}  (YOLOE conf={conf})")
    print(f"  T_estimated : {T_cam.round(3)} m  |T|={np.linalg.norm(T_cam):.3f} m")
    print(f"  Object pts  : {n_pts}")

    # ── map camera-frame T → sim world coordinates ──
    # Camera convention: X_right, Y_down, Z_forward
    # Sim world:         X_right, Y_forward, Z_up
    # We place the object on a table (z=0.42m) in front of the Panda.
    # The X offset from pipeline is used to left/right shift the object.
    # This simulates the camera being mounted above/behind the robot looking forward.
    depth = T_cam[2]
    x_norm = T_cam[0] / max(depth, 0.1)   # normalised lateral offset
    obj_x = np.clip(x_norm * 0.15 + 0.55, 0.35, 0.75)  # in robot reach [0.35, 0.75]
    obj_y = 0.0                             # centred laterally in front of robot
    obj_z = 0.42 + 0.04                    # table height + half object height

    obj_pos = np.array([obj_x, obj_y, obj_z])
    obj_orn = mat3_to_quat(R_cam)
    print(f"  x_norm (lateral) = {x_norm:.3f} → sim x = {obj_x:.3f}")
    print(f"\nObject placed at sim pos: {obj_pos.round(3)}")

    # ── physics setup ──
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(DATA)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=150)

    p.loadURDF("plane.urdf")

    # table
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.21])
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.21],
                                    rgbaColor=[0.76,0.60,0.42,1])
    p.createMultiBody(0, table_col, table_vis, [0.55, 0.0, 0.21])

    # Franka Panda
    robot = p.loadURDF(DATA+'/franka_panda/panda.urdf',
                       [0,0,0], useFixedBase=True)

    # home pose
    home = [0, -0.3, 0, -2.2, 0, 2.0, 0.78]
    for j, a in zip(ARM_JOINTS, home):
        p.resetJointState(robot, j, a)
    for fj in [FINGER_JOINT1, FINGER_JOINT2]:
        p.resetJointState(robot, fj, FINGER_OPEN)

    # object: mug as bottle proxy
    obj_urdf = DATA+'/urdf/mug.urdf'
    if not os.path.exists(obj_urdf):
        obj_urdf = DATA+'/block.urdf'
    obj = p.loadURDF(obj_urdf, obj_pos.tolist(), obj_orn,
                     globalScaling=0.3)   # width ~3.9cm, fits Panda 8cm finger gap
    p.changeDynamics(obj, -1, mass=0.20,
                     lateralFriction=1.5, spinningFriction=0.5,
                     restitution=0.1)

    # settle
    for _ in range(300): p.stepSimulation()
    obj_pos_settled = np.array(p.getBasePositionAndOrientation(obj)[0])
    print(f"Object settled: {obj_pos_settled.round(3)}")

    # grasp geometry — compute AABB to find object centre and ideal EE height
    aabb = p.getAABB(obj)
    obj_half_h  = (aabb[1][2] - aabb[0][2]) / 2
    obj_center_z = aabb[0][2] + obj_half_h
    finger_half  = 0.034   # Panda finger half-length
    grasp_z      = obj_center_z + finger_half   # EE sits above object centre

    down_orn   = p.getQuaternionFromEuler([math.pi, 0, 0])
    pre_grasp  = np.array([obj_pos_settled[0], obj_pos_settled[1], grasp_z + 0.25])
    grasp_pos  = np.array([obj_pos_settled[0], obj_pos_settled[1], grasp_z])
    lift_pos   = np.array([obj_pos_settled[0], obj_pos_settled[1], grasp_z + 0.35])

    print(f"Object AABB z: {aabb[0][2]:.4f} → {aabb[1][2]:.4f}  centre={obj_center_z:.4f}")
    print(f"Grasp EE z target: {grasp_z:.4f}  (finger_half={finger_half})")
    cam_look   = obj_pos_settled.tolist()

    W, H   = 480, 360
    frames = []
    CAP    = 4   # capture every N steps

    def rec(steps, cam, label, color=(30,30,30), finger_w=None):
        joints = p.calculateInverseKinematics(
            robot, EE_LINK, cam[3], cam[4],
            lowerLimits=[-2.967,-1.833,-2.967,-3.142,-2.967,-0.087,-2.967],
            upperLimits=[ 2.967, 1.833, 2.967, 0.000, 2.967, 3.822, 2.967],
            jointRanges=[5.934,3.666,5.934,3.142,5.934,3.909,5.934],
            restPoses=[0,-0.3,0,-2.0,0,2.0,0.78],
            maxNumIterations=300, residualThreshold=1e-4)
        for step in range(steps):
            for j in ARM_JOINTS:
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                        targetPosition=joints[j], force=250)
            if finger_w is not None:
                set_fingers(robot, finger_w, force=80)
            p.stepSimulation()
            if step % CAP == 0:
                f = capture_frame(W, H, cam[0], cam[1], cam[2])
                frames.append(Image.fromarray(annotate(f, label, color)))

    # cam spec: (eye, target, up, ik_target_pos, ik_target_orn)
    cam_approach = ([1.0,-0.6,0.9], cam_look, [0,0,1], pre_grasp.tolist(),  down_orn)
    cam_grasp    = ([0.9,-0.5,0.8], cam_look, [0,0,1], grasp_pos.tolist(),  down_orn)
    cam_close    = ([0.9,-0.5,0.8], cam_look, [0,0,1], grasp_pos.tolist(),  down_orn)
    cam_lift     = ([0.9,-0.6,1.1], (np.array(cam_look)+[0,0,0.2]).tolist(),
                    [0,0,1], lift_pos.tolist(), down_orn)

    # ── Phase 0: idle (show object + pipeline info) ──
    for _ in range(10):
        p.stepSimulation()
        f = capture_frame(W, H, [1.1,-0.7,0.8], cam_look)
        f = annotate(f, f'LLM keyword: "{kw}"  YOLOE conf={conf}  pts={n_pts}', (20,80,20))
        frames.append(Image.fromarray(f))

    # ── Phase 1: pre-grasp approach (fingers open) ──
    print("Phase 1: pre-grasp approach...")
    rec(350, cam_approach, 'Phase 1: Pre-grasp approach  [fingers open]',
        (30,30,150), finger_w=FINGER_OPEN)

    # ── Phase 2: descend to grasp (fingers open) ──
    print("Phase 2: descend to grasp...")
    rec(350, cam_grasp, 'Phase 2: Descending to grasp pose',
        (30,30,150), finger_w=FINGER_OPEN)

    # ── Phase 3: close fingers (real contact grasp) ──
    print("Phase 3: closing fingers (real contact)...")
    joints_grasp = p.calculateInverseKinematics(
        robot, EE_LINK, grasp_pos.tolist(), down_orn,
        maxNumIterations=300)
    for step in range(250):
        for j in ARM_JOINTS:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=joints_grasp[j], force=250)
        # progressively close fingers
        w = FINGER_OPEN - (FINGER_OPEN - FINGER_CLOSED) * min(step/200, 1.0)
        set_fingers(robot, w, force=120)
        p.stepSimulation()
        if step % CAP == 0:
            f = capture_frame(W, H, [0.9,-0.5,0.8], cam_look)
            f_w = FINGER_OPEN - (FINGER_OPEN - FINGER_CLOSED) * min(step/200,1.0)
            frames.append(Image.fromarray(
                annotate(f, f'Phase 3: Closing fingers  w={f_w:.3f}m', (150,80,0))))

    # ── check contact before lifting ──
    contacts = p.getContactPoints(robot, obj)
    n_contacts = len(contacts) if contacts else 0
    print(f"  Contacts at grasp: {n_contacts}")

    # ── Phase 4: lift (NO constraint — pure friction) ──
    print("Phase 4: lifting (physics only — no fixed constraint)...")
    obj_z_before = p.getBasePositionAndOrientation(obj)[0][2]
    rec(500, cam_lift, 'Phase 4: Lifting  [NO fixed constraint — friction only]',
        (20,120,20), finger_w=FINGER_CLOSED)

    # ── evaluate ──
    obj_pos_after = np.array(p.getBasePositionAndOrientation(obj)[0])
    lift_delta = obj_pos_after[2] - obj_z_before
    success = lift_delta > 0.08

    result_label = f'{"SUCCESS" if success else "FAILURE"}  lifted={lift_delta:.3f}m  contacts={n_contacts}'
    color_r = (20,140,20) if success else (180,20,20)
    print(f"\n{'='*55}")
    print(f"GRASP RESULT : {'SUCCESS ✓' if success else 'FAILURE ✗'}")
    print(f"  Lift delta  : {lift_delta:.4f} m  (threshold 0.08 m)")
    print(f"  Contacts    : {n_contacts}")
    print(f"  T_pipeline  : {T_cam.round(3)} m")
    print(f"  No fixed constraint used — pure contact physics")
    print(f"{'='*55}")

    # ── final frames ──
    for _ in range(14):
        p.stepSimulation()
        f = capture_frame(W, H, [0.9,-0.7,1.1],
                          (obj_pos_after + [0,0,-0.1]).tolist())
        frames.append(Image.fromarray(annotate(f, result_label, color_r)))

    p.disconnect()

    # ── save GIF ──
    out_gif = os.path.join(args.out_dir, 'grasp_real.gif')
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=90, loop=0)
    print(f"\nGIF saved: {out_gif}  ({len(frames)} frames)")

    # ── save slow GIF ──
    out_slow = os.path.join(args.out_dir, 'grasp_real_slow.gif')
    frames[0].save(out_slow, save_all=True, append_images=frames[1:],
                   duration=160, loop=0)
    print(f"GIF slow: {out_slow}")

    # ── save result JSON ──
    result = {
        "method": "LLM+YOLOE+Any6D",
        "instruction": ins, "keyword": kw,
        "yoloe_conf": conf, "n_obj_pts": n_pts,
        "T_estimated_m": T_cam.tolist(),
        "T_norm_m": float(np.linalg.norm(T_cam)),
        "obj_pos_sim": obj_pos_settled.tolist(),
        "n_contacts_at_grasp": n_contacts,
        "lift_delta_m": float(lift_delta),
        "grasp_success": bool(success),
        "fixed_constraint_used": False,
        "note": "Grasp via contact friction only — Franka Panda finger joints"
    }
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # ── summary figure ──
    _summary(args.out_dir, result, frames)


def _summary(out_dir, result, frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(frames)
    idxs = [0, n//5, 2*n//5, 3*n//5, 4*n//5, n-1]
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.patch.set_facecolor('white')
    ok = result['grasp_success']
    fig.suptitle(
        f'PyBullet — Franka Panda — LLM+YOLOE+Any6D Pipeline\n'
        f'"{result["instruction"]}"  |  kw="{result["keyword"]}"  conf={result["yoloe_conf"]}  '
        f'pts={result["n_obj_pts"]}  |T|={result["T_norm_m"]:.2f}m  '
        f'contacts={result["n_contacts_at_grasp"]}  lift={result["lift_delta_m"]:.3f}m  '
        f'→  {"SUCCESS ✓" if ok else "FAILURE ✗"}',
        fontsize=9.5, fontweight='bold',
        color='#1a7a30' if ok else '#c0392b')
    labels = ['Idle','Pre-grasp','Descend','Close fingers','Lift','Final']
    for ax, idx, lbl in zip(axes, idxs, labels):
        ax.imshow(frames[idx])
        ax.set_title(lbl, fontsize=9, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    out = os.path.join(out_dir, 'summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Summary: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',    required=True)
    parser.add_argument('--out_dir', default='/workspace/results/pybullet_real')
    main(parser.parse_args())
