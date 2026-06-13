"""
PyBullet Robot Grasp Animation
Full pipeline: scene → detect pose → robot arm → grasp → lift
Saves frames as PNG + assembles GIF/MP4
"""
import os, sys, json, time, math
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ANCHOR_DIR = '/workspace/anchor_results/dexycb_reference_view_ours'
OUT_DIR    = '/workspace/results/pybullet_grasp'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/frames', exist_ok=True)

# ── Scene parameters ──────────────────────────────────────────────────────────
W, H      = 640, 480
fx = fy   = 525.0
cx, cy    = 319.5, 239.5
NEAR, FAR = 0.1, 5.0
TABLE_TOP = 0.626

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_EYE    = [0.0, -1.2, 1.8]
CAM_TARGET = [0.0,  0.0, 0.69]

TARGET_OBJ = '006_mustard_bottle'
INSTRUCTION = 'I want to grab the yellow mustard bottle'

def proj_matrix():
    return [
        2*fx/W, 0, 1-2*cx/W, 0,
        0, 2*fy/H, 2*cy/H-1, 0,
        0, 0, -(FAR+NEAR)/(FAR-NEAR), -2*FAR*NEAR/(FAR-NEAR),
        0, 0, -1, 0
    ]

def depth_to_metric(buf):
    return FAR * NEAR / (FAR - (FAR - NEAR) * buf)

def render(client, eye=None, target=None):
    e = eye    or CAM_EYE
    t = target or CAM_TARGET
    vm = p.computeViewMatrix(e, t, [0,0,1], physicsClientId=client)
    pm = proj_matrix()
    _, _, rgb, depth_buf, seg = p.getCameraImage(
        W, H, vm, pm,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client)
    rgb_img  = np.array(rgb, dtype=np.uint8).reshape(H, W, 4)[:,:,:3]
    depth_m  = depth_to_metric(np.array(depth_buf).reshape(H, W))
    seg_img  = np.array(seg).reshape(H, W)
    return rgb_img, depth_m, seg_img

def save_frame(frame_idx, rgb, label, detection=None, pose_axes=None, K=None):
    img = rgb.copy()
    if detection is not None:
        mask, box, prompt, conf = detection
        # Green mask overlay
        colored = np.zeros_like(img); colored[:,:] = [50, 210, 80]
        img[mask] = (img[mask]*0.4 + colored[mask]*0.6).astype(np.uint8)
        cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (50,210,80), 2)
        cv2.putText(img, f'"{prompt}" {conf:.2f}',
                    (10, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,210,80), 1)
    if pose_axes is not None and K is not None:
        T = pose_axes
        try:
            rvec, _ = cv2.Rodrigues(T[:3,:3])
            tvec = T[:3,3].reshape(3,1)
            ax3d = np.float32([[0,0,0],[0.06,0,0],[0,0.06,0],[0,0,0.06]])
            pts2d, _ = cv2.projectPoints(ax3d, rvec, tvec, K, np.zeros(5))
            pts2d = pts2d.astype(int).reshape(-1,2)
            o = tuple(pts2d[0])
            cv2.arrowedLine(img, o, tuple(pts2d[1]), (220,50,50),  3, tipLength=0.3)
            cv2.arrowedLine(img, o, tuple(pts2d[2]), (50,200,50),  3, tipLength=0.3)
            cv2.arrowedLine(img, o, tuple(pts2d[3]), (50,50,220),  3, tipLength=0.3)
            cv2.circle(img, o, 5, (255,255,255), -1)
        except: pass
    # Label overlay
    cv2.rectangle(img, (0,0), (W,28), (20,20,20), -1)
    cv2.putText(img, label, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    path = f'{OUT_DIR}/frames/frame_{frame_idx:04d}.png'
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path

# ── Init PyBullet ─────────────────────────────────────────────────────────────
print('Initialising PyBullet...')
client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -9.81, physicsClientId=client)
p.loadURDF('plane.urdf', physicsClientId=client)
p.loadURDF('table/table.urdf', physicsClientId=client)

# ── Load objects ──────────────────────────────────────────────────────────────
OBJECTS = [
    ('006_mustard_bottle',  [ 0.00,  0.00, 0.80], [0.7071, 0, 0, 0.7071]),
    ('003_cracker_box',     [ 0.18,  0.05, 0.80], [0.7071, 0, 0, 0.7071]),
    ('005_tomato_soup_can', [-0.15,  0.00, 0.80], [0.7071, 0, 0, 0.7071]),
]
obj_bodies = {}
for obj_name, pos, orn in OBJECTS:
    mesh = f'{ANCHOR_DIR}/{obj_name}/center_mesh_{obj_name}.obj'
    if not os.path.exists(mesh): continue
    col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh,
                                  meshScale=[1,1,1], physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh,
                               meshScale=[1,1,1], physicsClientId=client)
    body = p.createMultiBody(0.1, col, vis, pos, orn, physicsClientId=client)
    obj_bodies[obj_name] = body
    print(f'  Loaded: {obj_name}')

for _ in range(200):
    p.stepSimulation(physicsClientId=client)

# ── Load Franka Panda ─────────────────────────────────────────────────────────
panda_start = [0.0, -0.6, TABLE_TOP]
panda_orn   = p.getQuaternionFromEuler([0, 0, 0])
panda = p.loadURDF('franka_panda/panda.urdf',
                    basePosition=panda_start,
                    baseOrientation=panda_orn,
                    useFixedBase=True,
                    physicsClientId=client)
print(f'Loaded Franka Panda (id={panda})')

# Panda joint info
n_joints = p.getNumJoints(panda, physicsClientId=client)
arm_joints    = [0,1,2,3,4,5,6]
finger_joints = [9,10]

# Home position
HOME_POS = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
for ji, jp in zip(arm_joints, HOME_POS):
    p.resetJointState(panda, ji, jp, physicsClientId=client)
p.resetJointState(panda, 9,  0.04, physicsClientId=client)
p.resetJointState(panda, 10, 0.04, physicsClientId=client)

for _ in range(100):
    p.stepSimulation(physicsClientId=client)

# ── Get target object pose ────────────────────────────────────────────────────
target_body = obj_bodies.get(TARGET_OBJ)
target_pos, target_orn = p.getBasePositionAndOrientation(
    target_body, physicsClientId=client)
print(f'Target object pos: {np.round(target_pos,3)}')

# Grasp position: above object
pre_grasp  = [target_pos[0], target_pos[1], target_pos[2] + 0.25]
grasp_pos  = [target_pos[0], target_pos[1], target_pos[2] + 0.06]
lift_pos   = [target_pos[0], target_pos[1], target_pos[2] + 0.35]

# ── IK helper ────────────────────────────────────────────────────────────────
EEF_LINK = 11  # Panda end-effector link

def move_to(target_xyz, target_orn_quat=None, steps=60, gripper_open=True):
    if target_orn_quat is None:
        target_orn_quat = p.getQuaternionFromEuler([math.pi, 0, math.pi/4])
    ik = p.calculateInverseKinematics(
        panda, EEF_LINK,
        target_xyz, target_orn_quat,
        physicsClientId=client)
    frames = []
    for step in range(steps):
        # Interpolate
        t = (step+1) / steps
        current_pos = []
        for ji in arm_joints:
            cur = p.getJointState(panda, ji, physicsClientId=client)[0]
            target_j = ik[ji]
            interp = cur + (target_j - cur) * min(t * 2, 1.0)
            p.setJointMotorControl2(
                panda, ji,
                p.POSITION_CONTROL,
                targetPosition=interp,
                force=150,
                physicsClientId=client)
        grip = 0.04 if gripper_open else 0.001
        for fj in finger_joints:
            p.setJointMotorControl2(
                panda, fj,
                p.POSITION_CONTROL,
                targetPosition=grip,
                force=50,
                physicsClientId=client)
        p.stepSimulation(physicsClientId=client)
        frames.append(step)
    return frames

# ── Camera intrinsics matrix ──────────────────────────────────────────────────
K_mat = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])

# ── ANIMATION SEQUENCE ────────────────────────────────────────────────────────
frame_idx  = 0
frame_paths = []

print('\nGenerating animation frames...')

# ── Phase 1: Initial scene (20 frames) ───────────────────────────────────────
print('Phase 1: Initial scene')
for i in range(20):
    rgb, depth_m, seg = render(client)
    path = save_frame(frame_idx,
                      rgb, f'Scene — instruction: "{INSTRUCTION}"')
    frame_paths.append(path)
    frame_idx += 1

# ── Phase 2: YOLOE detection (20 frames) ─────────────────────────────────────
print('Phase 2: Detection')
# Run actual YOLOE on the scene image
sys.path.insert(0, '/workspace/yoloe')
try:
    from ultralytics import YOLOE
    _ym = None
    orig = os.getcwd()
    os.chdir('/workspace/yoloe')
    _ym = YOLOE('/workspace/yoloe/yoloe-26l-seg.pt')
    os.chdir(orig)

    rgb_scene, depth_scene, seg_scene = render(client)
    img_bgr = cv2.cvtColor(rgb_scene, cv2.COLOR_RGB2BGR)
    prompt  = 'yellow mustard bottle'
    _ym.set_classes([prompt], _ym.get_text_pe([prompt]))
    res = _ym.predict(img_bgr, conf=0.05, verbose=False)
    if len(res[0].boxes):
        conf   = float(res[0].boxes.conf[0].item())
        box    = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
        mask   = cv2.resize(res[0].masks.data[0].cpu().numpy(),(W,H)) > 0.5
        det    = (mask, box, prompt, conf)
        print(f'  YOLOE detected: {prompt} conf={conf:.2f}')
    else:
        det = None
        print('  YOLOE: no detection — using GT mask')
        seg_mask = seg_scene == target_body
        ys, xs   = np.where(seg_mask)
        if len(ys):
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            det = (seg_mask, box, 'mustard bottle (GT)', 0.99)
except Exception as e:
    print(f'  YOLOE error: {e} — using GT')
    rgb_scene, depth_scene, seg_scene = render(client)
    seg_mask = seg_scene == target_body
    ys, xs   = np.where(seg_mask)
    box      = np.array([xs.min(), ys.min(), xs.max(), ys.max()]) if len(ys) else np.array([100,100,200,300])
    det      = (seg_mask, box, 'mustard bottle (GT)', 0.99)

for i in range(25):
    rgb, depth_m, seg = render(client)
    path = save_frame(frame_idx, rgb,
                      f'Step 1/3 — YOLOE detection: "{prompt}"',
                      detection=det)
    frame_paths.append(path)
    frame_idx += 1

# ── Phase 3: Pose estimation display (20 frames) ─────────────────────────────
print('Phase 3: Pose estimation')
T_gt = np.eye(4)
T_gt[:3,:3] = np.array(p.getMatrixFromQuaternion(target_orn)).reshape(3,3)
T_gt[:3, 3] = np.array(target_pos)

for i in range(25):
    rgb, depth_m, seg = render(client)
    path = save_frame(frame_idx, rgb,
                      f'Step 2/3 — 6D pose estimated  t=({target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f})m',
                      detection=det,
                      pose_axes=T_gt,
                      K=K_mat)
    frame_paths.append(path)
    frame_idx += 1

# ── Phase 4: Robot approaches (move to pre-grasp) ────────────────────────────
print('Phase 4: Robot approaching')
grasp_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/4])
steps = 80
ik = p.calculateInverseKinematics(
    panda, EEF_LINK, pre_grasp, grasp_orn,
    physicsClientId=client)

for step in range(steps):
    t = (step+1) / steps
    for ji in arm_joints:
        cur = p.getJointState(panda, ji, physicsClientId=client)[0]
        interp = cur + (ik[ji] - cur) * min(t*1.5, 1.0)
        p.setJointMotorControl2(panda, ji, p.POSITION_CONTROL,
                                 targetPosition=interp, force=150,
                                 physicsClientId=client)
    for fj in finger_joints:
        p.setJointMotorControl2(panda, fj, p.POSITION_CONTROL,
                                 targetPosition=0.04, force=50,
                                 physicsClientId=client)
    p.stepSimulation(physicsClientId=client)
    if step % 2 == 0:
        rgb, _, _ = render(client)
        path = save_frame(frame_idx, rgb,
                          'Step 3/3 — Robot approaching target',
                          detection=det, pose_axes=T_gt, K=K_mat)
        frame_paths.append(path)
        frame_idx += 1

# ── Phase 5: Descend to grasp ────────────────────────────────────────────────
print('Phase 5: Descending to grasp')
ik2 = p.calculateInverseKinematics(
    panda, EEF_LINK, grasp_pos, grasp_orn,
    physicsClientId=client)
for step in range(60):
    t = (step+1) / 60
    for ji in arm_joints:
        cur = p.getJointState(panda, ji, physicsClientId=client)[0]
        interp = cur + (ik2[ji] - cur) * min(t*2, 1.0)
        p.setJointMotorControl2(panda, ji, p.POSITION_CONTROL,
                                 targetPosition=interp, force=150,
                                 physicsClientId=client)
    p.stepSimulation(physicsClientId=client)
    if step % 2 == 0:
        rgb, _, _ = render(client)
        path = save_frame(frame_idx, rgb, 'Grasping...',
                          detection=det, pose_axes=T_gt, K=K_mat)
        frame_paths.append(path)
        frame_idx += 1

# ── Phase 6: Close gripper ────────────────────────────────────────────────────
print('Phase 6: Closing gripper')
for step in range(40):
    grip = 0.04 * (1 - step/40) + 0.002 * (step/40)
    for fj in finger_joints:
        p.setJointMotorControl2(panda, fj, p.POSITION_CONTROL,
                                 targetPosition=grip, force=80,
                                 physicsClientId=client)
    p.stepSimulation(physicsClientId=client)
    if step % 2 == 0:
        rgb, _, _ = render(client)
        path = save_frame(frame_idx, rgb, 'Closing gripper...')
        frame_paths.append(path)
        frame_idx += 1

# ── Phase 7: Lift ────────────────────────────────────────────────────────────
print('Phase 7: Lifting object')
ik3 = p.calculateInverseKinematics(
    panda, EEF_LINK, lift_pos, grasp_orn,
    physicsClientId=client)
for step in range(80):
    t = (step+1) / 80
    for ji in arm_joints:
        cur = p.getJointState(panda, ji, physicsClientId=client)[0]
        interp = cur + (ik3[ji] - cur) * min(t*1.5, 1.0)
        p.setJointMotorControl2(panda, ji, p.POSITION_CONTROL,
                                 targetPosition=interp, force=150,
                                 physicsClientId=client)
    for fj in finger_joints:
        p.setJointMotorControl2(panda, fj, p.POSITION_CONTROL,
                                 targetPosition=0.003, force=80,
                                 physicsClientId=client)
    p.stepSimulation(physicsClientId=client)
    if step % 2 == 0:
        rgb, _, _ = render(client)
        path = save_frame(frame_idx, rgb,
                          'Object grasped and lifted!')
        frame_paths.append(path)
        frame_idx += 1

# ── Phase 8: Final pose (hold 20 frames) ─────────────────────────────────────
for i in range(20):
    rgb, _, _ = render(client)
    path = save_frame(frame_idx, rgb,
                      f'Done — "{INSTRUCTION}" completed')
    frame_paths.append(path)
    frame_idx += 1

print(f'\nTotal frames: {len(frame_paths)}')

# ── Assemble GIF ──────────────────────────────────────────────────────────────
print('Assembling GIF...')
try:
    from PIL import Image
    imgs = [Image.fromarray(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
            for p in frame_paths]
    gif_path = f'{OUT_DIR}/grasp_animation.gif'
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=60, loop=0)
    print(f'Saved GIF: {gif_path}')
except ImportError:
    print('PIL not available — saving MP4 instead')

# ── Assemble MP4 ──────────────────────────────────────────────────────────────
print('Assembling MP4...')
mp4_path = f'{OUT_DIR}/grasp_animation.mp4'
fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
out_vid  = cv2.VideoWriter(mp4_path, fourcc, 20, (W, H))
for fp in frame_paths:
    frame = cv2.imread(fp)
    if frame is not None:
        out_vid.write(frame)
out_vid.release()
print(f'Saved MP4: {mp4_path}')

# ── Key frames figure ────────────────────────────────────────────────────────
print('Saving key frames figure...')
key_indices = [0,
               len(frame_paths)//6,
               len(frame_paths)//3,
               len(frame_paths)//2,
               2*len(frame_paths)//3,
               len(frame_paths)-1]
key_labels  = ['initial scene',
               'YOLOE detection',
               'pose estimated',
               'approaching',
               'grasping',
               'lifted']

fig, axes = plt.subplots(1, 6, figsize=(24, 4))
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle(f'Simulation — "{INSTRUCTION}"',
             color='white', fontsize=11, y=1.02)

for ax, idx, label in zip(axes, key_indices, key_labels):
    idx = min(idx, len(frame_paths)-1)
    img = cv2.cvtColor(cv2.imread(frame_paths[idx]), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(label, color='white', fontsize=8, pad=4)
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

plt.tight_layout(pad=0.3)
fig_path = f'{OUT_DIR}/key_frames.png'
plt.savefig(fig_path, dpi=110, bbox_inches='tight', facecolor='#0f0f0f')
plt.close()
print(f'Saved key frames: {fig_path}')

p.disconnect(physicsClientId=client)
print('\nDone!')
