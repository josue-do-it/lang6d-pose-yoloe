"""
Re-runs the PyBullet grasp simulation capturing many frames → animated GIF.
"""
import argparse, os, json, math
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from PIL import Image

DATA = pybullet_data.getDataPath()

def mat3_to_quat(R):
    m = R
    trace = m[0,0]+m[1,1]+m[2,2]
    if trace > 0:
        s = 0.5/math.sqrt(trace+1.0)
        return [(m[2,1]-m[1,2])*s, (m[0,2]-m[2,0])*s, (m[1,0]-m[0,1])*s, 0.25/s]
    elif m[0,0]>m[1,1] and m[0,0]>m[2,2]:
        s = 2.0*math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])
        return [0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s, (m[2,1]-m[1,2])/s]
    elif m[1,1]>m[2,2]:
        s = 2.0*math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])
        return [(m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s, (m[0,2]-m[2,0])/s]
    else:
        s = 2.0*math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])
        return [(m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s, (m[1,0]-m[0,1])/s]

def capture(width, height, cam_pos, target):
    view = p.computeViewMatrix(cam_pos, target, [0,0,1])
    proj = p.computeProjectionMatrixFOV(55, width/height, 0.05, 6.0)
    _, _, rgb, _, _ = p.getCameraImage(width, height, view, proj,
                                       renderer=p.ER_TINY_RENDERER)
    img = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:,:,:3]
    return img  # RGB

def move_and_capture(robot, ee_idx, n_j, target_pos, target_orn,
                     steps, frames, capture_every,
                     cam_pos, look_at, W, H, label=''):
    joints = p.calculateInverseKinematics(
        robot, ee_idx, target_pos, target_orn,
        maxNumIterations=300, residualThreshold=1e-5)
    for step in range(steps):
        for j in range(n_j):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=joints[j], force=500)
        p.stepSimulation()
        if step % capture_every == 0:
            frame = capture(W, H, cam_pos, look_at).copy()
            if label:
                cv2.putText(frame, label, (10, H-15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (30,30,30), 2, cv2.LINE_AA)
            frames.append(Image.fromarray(frame))

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json) as f: res = json.load(f)
    R_cam = np.array(res['R']); T_cam = np.array(res['T'])
    kw = res['keyword']; ins = res['instruction']

    scale = 0.5
    obj_pos = np.array([
        T_cam[0]*scale + 0.5,
        T_cam[2]*scale*0.4,
        max(T_cam[1]*scale, 0.05)
    ])
    obj_orn = mat3_to_quat(R_cam)

    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(DATA)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    robot = p.loadURDF(os.path.join(DATA, "kuka_iiwa/model.urdf"),
                       [0,0,0], useFixedBase=True)
    n_j   = p.getNumJoints(robot)
    ee    = n_j - 1

    for j, a in enumerate([0,0,0,-1.5,0,1.5,0][:n_j]):
        p.resetJointState(robot, j, a)

    obj_urdf = os.path.join(DATA, "urdf/mug.urdf")
    if not os.path.exists(obj_urdf):
        obj_urdf = os.path.join(DATA, "block.urdf")
    obj = p.loadURDF(obj_urdf, obj_pos.tolist(), obj_orn, globalScaling=0.15)

    # green sphere marker at object centre
    vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.025, rgbaColor=[0,1,0,0.8])
    p.createMultiBody(0, -1, vs, obj_pos.tolist())

    for _ in range(200): p.stepSimulation()
    obj_pos0,_ = p.getBasePositionAndOrientation(obj)

    W, H = 480, 360
    CAP  = 3    # capture every N steps

    # camera orbit params for each phase
    phases = [
        # (steps, cam_x, cam_y, cam_z, label)
        (120, 1.2, -0.9, 1.2, 'Idle'),
        (300, 1.1, -0.9, 1.2, 'Approaching'),
        (300, 1.0, -0.8, 1.0, 'Descending to grasp'),
        (400, 0.9, -0.8, 1.3, 'Lifting'),
        (200, 0.7, -1.1, 1.6, 'Success!'),
    ]

    frames   = []
    down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    approach = obj_pos + np.array([0, 0, 0.28])
    grasp    = obj_pos + np.array([0, 0, 0.07])
    lifted   = obj_pos + np.array([0, 0, 0.42])

    # ── Phase 0: idle (just render) ──
    cx,cy,cz = phases[0][1:4]
    for _ in range(phases[0][0]):
        p.stepSimulation()
    for _ in range(8):
        frames.append(Image.fromarray(
            capture(W,H,[cx,cy,cz],obj_pos.tolist())))
    _add_label(frames, f'Instruction: "{ins[:45]}"', W, H, y=20)

    # ── Phase 1: approach ──
    move_and_capture(robot, ee, n_j, approach, down_orn,
                     phases[1][0], frames, CAP,
                     [phases[1][1],phases[1][2],phases[1][3]],
                     obj_pos.tolist(), W, H, 'Approaching object')

    # ── Phase 2: descend ──
    move_and_capture(robot, ee, n_j, grasp, down_orn,
                     phases[2][0], frames, CAP,
                     [phases[2][1],phases[2][2],phases[2][3]],
                     obj_pos.tolist(), W, H, 'Grasping')

    # ── Phase 3: attach + lift ──
    p.createConstraint(robot, ee, obj, -1, p.JOINT_FIXED,
                       [0,0,0], [0,0,0.05], [0,0,0])
    move_and_capture(robot, ee, n_j, lifted, down_orn,
                     phases[3][0], frames, CAP,
                     [phases[3][1],phases[3][2],phases[3][3]],
                     lifted.tolist(), W, H, 'Lifting')

    # ── Phase 4: hold and verify ──
    obj_posf,_ = p.getBasePositionAndOrientation(obj)
    delta = obj_posf[2] - obj_pos0[2]
    success = delta > 0.15
    label = f'GRASP {"SUCCESS" if success else "FAILURE"}  lift={delta:.2f}m'
    for _ in range(12):
        p.stepSimulation()
        frames.append(Image.fromarray(
            _annotate(capture(W,H,
                              [phases[4][1],phases[4][2],phases[4][3]],
                              lifted.tolist()),
                      label, W, H,
                      color=(30,160,30) if success else (200,30,30))))

    p.disconnect()

    # ── top-down bonus frames ──
    print(f"\nGrasp: {'SUCCESS' if success else 'FAILURE'}  lift={delta:.3f}m")
    print(f"Total frames captured: {len(frames)}")

    # ── save GIF ──
    out_gif = os.path.join(args.out_dir, 'grasp_sim.gif')
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=80,       # ms per frame
        loop=0,            # infinite loop
        optimize=False,
    )
    print(f"Saved GIF: {out_gif}  ({len(frames)} frames)")

    # also save a higher-quality version (slower playback)
    out_gif2 = os.path.join(args.out_dir, 'grasp_sim_slow.gif')
    frames[0].save(
        out_gif2,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
    )
    print(f"Saved GIF (slow): {out_gif2}")


def _annotate(img, text, W, H, color=(30,30,30)):
    out = np.ascontiguousarray(img.copy())
    cv2.putText(out, text, (10, H-15),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out

def _add_label(frames, text, W, H, y=20):
    for i in range(min(8, len(frames))):
        arr = np.ascontiguousarray(np.array(frames[-(i+1)]))
        cv2.putText(arr, text, (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (20,20,20), 1, cv2.LINE_AA)
        frames[-(i+1)] = Image.fromarray(arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',    required=True)
    parser.add_argument('--out_dir', default='/workspace/results/pybullet_sim')
    main(parser.parse_args())
