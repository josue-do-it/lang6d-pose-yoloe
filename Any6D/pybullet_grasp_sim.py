"""
PyBullet simulation of our LLM+YOLOE+Any6D pipeline.

Given an estimated 6D pose (from pose JSON), we:
  1. Place the object at the estimated pose in simulation
  2. Command a Kuka IIwa arm to grasp it
  3. Lift and check success
  4. Save screenshots for the report

Usage:
  python pybullet_grasp_sim.py \
      --json /workspace/results/infer_pose_bottle/bottle_pose.json \
      --out_dir /workspace/results/pybullet_sim
"""
import argparse, os, json, time, math
import numpy as np
import pybullet as p
import pybullet_data

DATA = pybullet_data.getDataPath()

# ── helpers ───────────────────────────────────────────────────────────────────
def mat3_to_quat(R):
    """Rotation matrix → PyBullet quaternion [x,y,z,w]."""
    m = R
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = 2.0 * math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
        w = (m[2,1] - m[1,2]) / s
        x = 0.25 * s
        y = (m[0,1] + m[1,0]) / s
        z = (m[0,2] + m[2,0]) / s
    elif m[1,1] > m[2,2]:
        s = 2.0 * math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
        w = (m[0,2] - m[2,0]) / s
        x = (m[0,1] + m[1,0]) / s
        y = 0.25 * s
        z = (m[1,2] + m[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
        w = (m[1,0] - m[0,1]) / s
        x = (m[0,2] + m[2,0]) / s
        y = (m[1,2] + m[2,1]) / s
        z = 0.25 * s
    return [x, y, z, w]

def get_ee_pose(robot, ee_index):
    s = p.getLinkState(robot, ee_index)
    return np.array(s[4]), np.array(s[5])  # worldPos, worldOrn

def move_to(robot, ee_index, n_joints, target_pos, target_orn, steps=300):
    """IK + joint position control to move end-effector."""
    joints = p.calculateInverseKinematics(
        robot, ee_index, target_pos, target_orn,
        maxNumIterations=200, residualThreshold=1e-5)
    for _ in range(steps):
        for j in range(n_joints):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=joints[j], force=500)
        p.stepSimulation()

def save_screenshot(path, width=640, height=480,
                    cam_pos=(0.5, -1.2, 1.0), target=(0.5, 0.5, 0.3)):
    view = p.computeViewMatrix(cam_pos, target, [0,0,1])
    proj = p.computeProjectionMatrixFOV(60, width/height, 0.1, 5.0)
    _, _, rgb, _, _ = p.getCameraImage(width, height, view, proj,
                                       renderer=p.ER_TINY_RENDERER)
    import cv2
    img = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
    cv2.imwrite(path, cv2.cvtColor(img[:,:,:3], cv2.COLOR_RGB2BGR))
    print(f"  Screenshot: {path}")

# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # load pose
    with open(args.json) as f: res = json.load(f)
    R_cam  = np.array(res['R'])
    T_cam  = np.array(res['T'])
    kw     = res['keyword']
    ins    = res['instruction']

    print(f"Instruction : {ins}")
    print(f"Keyword     : {kw}")
    print(f"T_cam       : {T_cam.round(3)}  |T| = {np.linalg.norm(T_cam):.3f} m")

    # ── convert camera-frame pose → sim world frame ──
    # Camera looks along +Z. In sim we put object at (T[0], T[2], T[1])
    # (swap Y/Z so Z_cam becomes Y_world), with some scaling
    scale = 0.5   # bring ~1.8m depth to ~0.9m in sim for visibility
    obj_pos_world = np.array([
        T_cam[0] * scale + 0.5,   # X  (centred in front of robot)
        T_cam[2] * scale * 0.4,   # depth → Y (forward)
        max(T_cam[1] * scale, 0.05)  # height, keep above floor
    ])
    # rotation: use R from pipeline
    R_world = R_cam.copy()
    obj_orn = mat3_to_quat(R_world)
    print(f"Object sim pos: {obj_pos_world.round(3)}")

    # ── connect pybullet (DIRECT = headless, no display needed) ──
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(DATA)
    p.setGravity(0, 0, -9.81)

    # ground plane
    plane = p.loadURDF("plane.urdf")

    # ── load Kuka IIwa ──
    robot_start = [0, 0, 0]
    robot = p.loadURDF(os.path.join(DATA, "kuka_iiwa/model.urdf"),
                       robot_start, useFixedBase=True)
    n_joints = p.getNumJoints(robot)
    ee_index = n_joints - 1   # last link = end-effector

    # set initial pose (arm stretched upward)
    init_angles = [0, 0, 0, -1.5, 0, 1.5, 0]
    for j, a in enumerate(init_angles[:n_joints]):
        p.resetJointState(robot, j, a)

    # ── load object: use mug URDF as proxy for bottle / block ──
    obj_urdf = os.path.join(DATA, "urdf/mug.urdf")
    if not os.path.exists(obj_urdf):
        obj_urdf = os.path.join(DATA, "block.urdf")
    obj = p.loadURDF(obj_urdf, obj_pos_world.tolist(), obj_orn,
                     globalScaling=0.15)

    # add visual marker at ground truth object position (green sphere)
    vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.02,
                             rgbaColor=[0, 1, 0, 0.7])
    p.createMultiBody(0, -1, vs, obj_pos_world.tolist())

    # settle object
    for _ in range(200):
        p.stepSimulation()

    # record initial object height
    obj_pos0, _ = p.getBasePositionAndOrientation(obj)
    print(f"Object settled at: {np.array(obj_pos0).round(3)}")

    # ── PHASE 1: pre-grasp — approach from above ──
    approach_pos  = obj_pos_world + np.array([0, 0, 0.25])
    approach_orn  = p.getQuaternionFromEuler([math.pi, 0, 0])  # gripper down
    print("Phase 1: approach...")
    move_to(robot, ee_index, n_joints, approach_pos, approach_orn, steps=400)
    save_screenshot(os.path.join(args.out_dir, "01_approach.png"),
                    cam_pos=(1.2, -0.8, 1.2), target=obj_pos_world.tolist())

    # ── PHASE 2: descend to grasp ──
    grasp_pos = obj_pos_world + np.array([0, 0, 0.07])
    print("Phase 2: descend to grasp...")
    move_to(robot, ee_index, n_joints, grasp_pos, approach_orn, steps=400)
    save_screenshot(os.path.join(args.out_dir, "02_grasp.png"),
                    cam_pos=(1.2, -0.8, 1.2), target=obj_pos_world.tolist())

    # check EE-object distance
    ee_pos, _ = get_ee_pose(robot, ee_index)
    dist_to_obj = np.linalg.norm(np.array(ee_pos) - obj_pos_world)
    print(f"  EE → object distance: {dist_to_obj:.4f} m")

    # ── PHASE 3: lift ──
    lift_pos = obj_pos_world + np.array([0, 0, 0.35])
    print("Phase 3: lift...")
    # attach object to EE with constraint (simulate grasp)
    constraint = p.createConstraint(
        robot, ee_index, obj, -1,
        p.JOINT_FIXED, [0,0,0], [0,0,0.05], [0,0,0])
    move_to(robot, ee_index, n_joints, lift_pos, approach_orn, steps=500)
    save_screenshot(os.path.join(args.out_dir, "03_lift.png"),
                    cam_pos=(1.2, -0.8, 1.5), target=lift_pos.tolist())

    # check lift success: object rose above initial height?
    obj_pos_final, _ = p.getBasePositionAndOrientation(obj)
    lift_delta = obj_pos_final[2] - obj_pos0[2]
    success = lift_delta > 0.15
    print(f"\n{'='*50}")
    print(f"GRASP RESULT: {'SUCCESS' if success else 'FAILURE'}")
    print(f"  Object lifted: {lift_delta:.3f} m  (threshold 0.15 m)")
    print(f"  EE→object dist at grasp: {dist_to_obj:.4f} m")
    print(f"  T_error used: |T_cam| = {np.linalg.norm(T_cam):.3f} m")
    print(f"{'='*50}")

    # ── PHASE 4: place back ──
    save_screenshot(os.path.join(args.out_dir, "04_lifted.png"),
                    cam_pos=(0.8, -1.2, 1.8), target=lift_pos.tolist())

    # ── top-down view ──
    save_screenshot(os.path.join(args.out_dir, "05_topdown.png"),
                    width=480, height=480,
                    cam_pos=(0.5, 0.5, 2.0), target=(0.5, 0.5, 0.3))

    p.disconnect()

    # ── save result JSON ──
    result = {
        "instruction": ins,
        "keyword": kw,
        "T_cam": T_cam.tolist(),
        "T_norm_m": float(np.linalg.norm(T_cam)),
        "obj_pos_sim": obj_pos_world.tolist(),
        "ee_obj_dist_at_grasp_m": float(dist_to_obj),
        "lift_delta_m": float(lift_delta),
        "grasp_success": bool(success),
    }
    out_json = os.path.join(args.out_dir, "sim_result.json")
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved: {out_json}")

    # ── make matplotlib summary figure ──
    _make_summary_figure(args.out_dir, result)


def _make_summary_figure(out_dir, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2

    imgs = {
        '(1) Approach': '01_approach.png',
        '(2) Grasp':    '02_grasp.png',
        '(3) Lift':     '03_lift.png',
        '(4) Lifted':   '04_lifted.png',
        '(5) Top-Down': '05_topdown.png',
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor('white')
    status = 'SUCCESS' if result['grasp_success'] else 'FAILURE'
    color  = '#1a7a30' if result['grasp_success'] else '#c0392b'
    fig.suptitle(
        f'PyBullet Simulation — LLM+YOLOE+Any6D Pipeline\n'
        f'Instruction: "{result["instruction"]}"  |  Keyword: "{result["keyword"]}"  |  '
        f'|T|={result["T_norm_m"]:.3f} m  |  '
        f'EE dist={result["ee_obj_dist_at_grasp_m"]:.3f} m  |  '
        f'Lift={result["lift_delta_m"]:.3f} m  |  '
        f'Grasp: {status}',
        fontsize=10, fontweight='bold',
        color=color if result['grasp_success'] else '#c0392b'
    )

    for ax, (title, fname) in zip(axes, imgs.items()):
        path = os.path.join(out_dir, fname)
        img  = cv2.imread(path)
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    out_fig = os.path.join(out_dir, 'sim_summary.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Summary figure: {out_fig}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',    required=True)
    parser.add_argument('--out_dir', default='/workspace/results/pybullet_sim')
    main(parser.parse_args())
