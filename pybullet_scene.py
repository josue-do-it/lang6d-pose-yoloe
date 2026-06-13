"""
PyBullet Scene — RGB-D capture with YCB objects
Generates: color.png, depth.png, T_gt per object
Then runs the full pipeline on the simulated image
"""
import os, sys, json, math
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR    = '/workspace/results/pybullet_scene'
YCB_MODELS = '/workspace/dataset/ho3d/YCB_Video_Models/models'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Camera parameters (Azure Kinect 640x480) ─────────────────────────────────
W, H  = 640, 480
fx = fy = 614.1
cx, cy  = 320.0, 240.0
NEAR, FAR = 0.01, 10.0

# ── Object list ──────────────────────────────────────────────────────────────
# Using YCB mesh files already on disk
OBJECTS = [
    ('006_mustard_bottle',  [0.0,  0.0,  0.65], [0, 0, 0, 1]),
    ('003_cracker_box',     [0.15, 0.05, 0.65], [0, 0, 0.3827, 0.9239]),
    ('005_tomato_soup_can', [-0.12, 0.0, 0.63], [0, 0, 0, 1]),
]

def get_projection_matrix(fx, fy, cx, cy, W, H, near, far):
    return [
        2*fx/W,   0,        1 - 2*cx/W,   0,
        0,        2*fy/H,   2*cy/H - 1,   0,
        0,        0,        -(far+near)/(far-near), -2*far*near/(far-near),
        0,        0,        -1,            0
    ]

def get_view_matrix(eye, target, up=[0, 0, 1]):
    return p.computeViewMatrix(eye, target, up)

def depth_buffer_to_metric(depth_buffer, near, far):
    return far * near / (far - (far - near) * depth_buffer)

def render_scene(physics_client, eye=[0, -0.8, 1.2], target=[0, 0, 0.6]):
    view_matrix = get_view_matrix(eye, target)
    proj_matrix = get_projection_matrix(fx, fy, cx, cy, W, H, NEAR, FAR)

    _, _, rgb, depth_buf, seg = p.getCameraImage(
        width=W, height=H,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=physics_client
    )

    rgb_img   = np.array(rgb,       dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
    depth_m   = depth_buffer_to_metric(np.array(depth_buf).reshape(H, W), NEAR, FAR)
    depth_mm  = (depth_m * 1000).astype(np.uint16)
    seg_img   = np.array(seg).reshape(H, W)

    return rgb_img, depth_mm, depth_m, seg_img

def get_object_pose(body_id, physics_client):
    pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
    R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = pos
    return T

# ── Main scene ───────────────────────────────────────────────────────────────
print('Starting PyBullet (offscreen)...')
client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -9.81, physicsClientId=client)

# Load plane
plane_id = p.loadURDF('plane.urdf', physicsClientId=client)

# Load table
table_id = p.loadURDF('table/table.urdf',
                       basePosition=[0, 0, 0],
                       baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=client)

# Load YCB objects
obj_ids   = []
obj_names = []
loaded    = []

for obj_name, pos, orn in OBJECTS:
    # Try OBJ mesh first
    mesh_path = f'{YCB_MODELS}/{obj_name}/textured_simple.obj'
    if not os.path.exists(mesh_path):
        mesh_path = f'{YCB_MODELS}/{obj_name}/textured.obj'
    if not os.path.exists(mesh_path):
        print(f'  SKIP {obj_name} — mesh not found')
        continue

    try:
        col_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[0.001, 0.001, 0.001],
            physicsClientId=client
        )
        vis_id = p.createVisualShape(
            p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[0.001, 0.001, 0.001],
            physicsClientId=client
        )
        body_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=client
        )
        obj_ids.append(body_id)
        obj_names.append(obj_name)
        loaded.append((obj_name, pos, body_id))
        print(f'  Loaded: {obj_name}')
    except Exception as e:
        print(f'  FAIL {obj_name}: {e}')

# Step simulation to settle objects
for _ in range(100):
    p.stepSimulation(physicsClientId=client)

# Render from two viewpoints
views = [
    {'name': 'front',  'eye': [0.0, -0.85, 1.15], 'target': [0.0, 0.0, 0.63]},
    {'name': 'angled', 'eye': [0.4, -0.75, 1.20], 'target': [0.0, 0.0, 0.63]},
]

scene_data = {}
for view in views:
    rgb, depth_mm, depth_m, seg = render_scene(
        client, eye=view['eye'], target=view['target'])

    color_path = f"{OUT_DIR}/color_{view['name']}.png"
    depth_path = f"{OUT_DIR}/depth_{view['name']}.png"
    cv2.imwrite(color_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_path, depth_mm)
    print(f"Saved: {color_path}")

    # Get GT poses for each object
    poses_gt = {}
    for obj_name, _, body_id in loaded:
        T = get_object_pose(body_id, client)
        poses_gt[obj_name] = T.tolist()

    scene_data[view['name']] = {
        'color_path': color_path,
        'depth_path': depth_path,
        'eye':        view['eye'],
        'target':     view['target'],
        'poses_gt':   poses_gt,
        'K': [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    }

with open(f'{OUT_DIR}/scene_data.json', 'w') as f:
    json.dump(scene_data, f, indent=2)
print(f'Saved: {OUT_DIR}/scene_data.json')

# ── Visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle('PyBullet simulated scene — YCB objects', color='white', fontsize=12)

for ri, view in enumerate(views):
    rgb   = cv2.cvtColor(cv2.imread(scene_data[view['name']]['color_path']), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(scene_data[view['name']]['depth_path'], cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth[depth < 1] = np.nan

    # RGB
    axes[ri][0].imshow(rgb)
    axes[ri][0].set_title(f"RGB — {view['name']} view", color='white', fontsize=9)
    axes[ri][0].axis('off')

    # Depth
    axes[ri][1].imshow(depth, cmap='jet', vmin=np.nanmin(depth), vmax=np.nanmax(depth))
    axes[ri][1].set_title('depth map', color='white', fontsize=9)
    axes[ri][1].axis('off')

    # Segmentation mask per object
    seg_vis = np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
    colors_seg = [(255, 100, 50), (50, 200, 100), (100, 100, 255)]
    for bi, (obj_name, _, body_id) in enumerate(loaded):
        _, _, _, _, seg_img = p.getCameraImage(
            width=W, height=H,
            viewMatrix=get_view_matrix(view['eye'], view['target']),
            projectionMatrix=get_projection_matrix(fx, fy, cx, cy, W, H, NEAR, FAR),
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=client
        )
        seg_arr = np.array(seg_img).reshape(H, W)
        mask = seg_arr == body_id
        seg_vis[mask] = colors_seg[bi % len(colors_seg)]

    overlay = (rgb * 0.5 + seg_vis * 0.5).astype(np.uint8)
    axes[ri][2].imshow(overlay)
    axes[ri][2].set_title('object segmentation (GT)', color='white', fontsize=9)
    axes[ri][2].axis('off')

    for ax in axes[ri]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

plt.tight_layout(pad=0.5)
viz_path = f'{OUT_DIR}/scene_visualization.png'
plt.savefig(viz_path, dpi=110, bbox_inches='tight', facecolor='#0f0f0f')
plt.close()
print(f'Saved: {viz_path}')

p.disconnect(physicsClientId=client)
print('Done.')
