"""
PyBullet Scene v2 — corrected camera and object positions
Object AABB z: 0.646-0.733 (placed at z=0.67)
Table top: z=0.626
Camera: eye=[0, -0.8, 1.2], target=[0, 0.05, 0.69]
"""
import os, sys, json, numpy as np
import pybullet as p
import pybullet_data
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR    = '/workspace/results/pybullet_scene'
ANCHOR_DIR = '/workspace/anchor_results/dexycb_reference_view_ours'
os.makedirs(OUT_DIR, exist_ok=True)

W, H   = 640, 480
fx = fy = 525.0
cx, cy = 319.5, 239.5
NEAR, FAR = 0.1, 5.0

OBJECTS = [
    ('006_mustard_bottle',   [ 0.00,  0.00, 0.67], [0, 0, 0, 1]),
    ('003_cracker_box',      [ 0.18,  0.05, 0.67], [0, 0, 0.3827, 0.9239]),
    ('005_tomato_soup_can',  [-0.15,  0.00, 0.67], [0, 0, 0, 1]),
]

VIEWS = [
    {'name': 'front',  'eye': [0.0,  -0.8, 1.2],  'target': [0.0,  0.05, 0.69]},
    {'name': 'angled', 'eye': [0.35, -0.7, 1.15],  'target': [0.05, 0.05, 0.69]},
]

def proj_matrix(fx, fy, cx, cy, W, H, near, far):
    return [
        2*fx/W, 0, 1-2*cx/W, 0,
        0, 2*fy/H, 2*cy/H-1, 0,
        0, 0, -(far+near)/(far-near), -2*far*near/(far-near),
        0, 0, -1, 0
    ]

def depth_to_metric(buf, near, far):
    return far * near / (far - (far - near) * buf)

print('Starting PyBullet...')
client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -9.81, physicsClientId=client)
p.loadURDF('plane.urdf', physicsClientId=client)
p.loadURDF('table/table.urdf', physicsClientId=client)

loaded = []
for obj_name, pos, orn in OBJECTS:
    mesh = f'{ANCHOR_DIR}/{obj_name}/center_mesh_{obj_name}.obj'
    if not os.path.exists(mesh):
        print(f'  SKIP {obj_name}')
        continue
    col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh,
                                  meshScale=[1,1,1], physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh,
                               meshScale=[1,1,1], physicsClientId=client)
    body = p.createMultiBody(0.1, col, vis, pos, orn, physicsClientId=client)
    for _ in range(200):
        p.stepSimulation(physicsClientId=client)
    final_pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=client)
    aabb = p.getAABB(body, physicsClientId=client)
    loaded.append((obj_name, body))
    print(f'  {obj_name}: z={final_pos[2]:.3f}  aabb_z=[{aabb[0][2]:.3f},{aabb[1][2]:.3f}]')

scene_data = {}
for view in VIEWS:
    vm = p.computeViewMatrix(view['eye'], view['target'], [0,0,1],
                              physicsClientId=client)
    pm = proj_matrix(fx, fy, cx, cy, W, H, NEAR, FAR)

    _, _, rgb, depth_buf, seg = p.getCameraImage(
        W, H, vm, pm,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client)

    rgb_img  = np.array(rgb,       dtype=np.uint8).reshape(H, W, 4)[:,:,:3]
    depth_m  = depth_to_metric(np.array(depth_buf).reshape(H,W), NEAR, FAR)
    depth_mm = (depth_m * 1000).astype(np.uint16)
    seg_img  = np.array(seg).reshape(H, W)

    print(f"  View {view['name']}: depth range [{depth_m.min():.2f},{depth_m.max():.2f}]m")
    print(f"  Unique seg IDs: {np.unique(seg_img)}")

    color_path = f"{OUT_DIR}/color_{view['name']}.png"
    depth_path = f"{OUT_DIR}/depth_{view['name']}.png"
    cv2.imwrite(color_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_path, depth_mm)

    poses_gt = {}
    for obj_name, body_id in loaded:
        pos2, orn2 = p.getBasePositionAndOrientation(body_id, physicsClientId=client)
        R = np.array(p.getMatrixFromQuaternion(orn2)).reshape(3,3)
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=pos2
        poses_gt[obj_name] = T.tolist()

    scene_data[view['name']] = {
        'color_path': color_path, 'depth_path': depth_path,
        'eye': view['eye'], 'target': view['target'],
        'poses_gt': poses_gt,
        'K': [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    }
    print(f"  Saved: {color_path}")

with open(f'{OUT_DIR}/scene_data.json','w') as f:
    json.dump(scene_data, f, indent=2)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle('PyBullet simulated scene — YCB objects', color='white', fontsize=12)

SEG_COLORS = {2: [255,100,50], 3: [50,200,100], 4: [100,150,255],
              5: [255,220,50], 6: [200,50,200]}

for ri, view in enumerate(VIEWS):
    rgb   = cv2.cvtColor(cv2.imread(scene_data[view['name']]['color_path']),
                         cv2.COLOR_BGR2RGB)
    depth = cv2.imread(scene_data[view['name']]['depth_path'],
                       cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = depth.copy(); d[d<100] = np.nan

    vm = p.computeViewMatrix(view['eye'], view['target'], [0,0,1],
                              physicsClientId=client)
    pm = proj_matrix(fx, fy, cx, cy, W, H, NEAR, FAR)
    _, _, _, _, seg_arr = p.getCameraImage(W, H, vm, pm,
        renderer=p.ER_TINY_RENDERER, physicsClientId=client)
    seg = np.array(seg_arr).reshape(H, W)

    seg_vis = np.zeros((H, W, 3), dtype=np.uint8)
    colors = [[255,80,50],[50,210,80],[80,120,255]]
    for ci, (_, body_id) in enumerate(loaded):
        mask = seg == body_id
        if mask.any():
            seg_vis[mask] = colors[ci % len(colors)]

    overlay = (rgb * 0.45 + seg_vis * 0.55).astype(np.uint8)

    axes[ri][0].imshow(rgb)
    axes[ri][0].set_title(f"RGB — {view['name']}", color='white', fontsize=9)
    axes[ri][0].axis('off')

    if not np.all(np.isnan(d)):
        axes[ri][1].imshow(d, cmap='plasma',
                           vmin=np.nanmin(d), vmax=np.nanmax(d))
    axes[ri][1].set_title('depth map', color='white', fontsize=9)
    axes[ri][1].axis('off')

    axes[ri][2].imshow(overlay)
    obj_names_short = [o.split('_',1)[1].replace('_',' ')
                       for o,_ in loaded]
    axes[ri][2].set_title('GT segmentation: ' + ' | '.join(obj_names_short),
                           color='white', fontsize=8)
    axes[ri][2].axis('off')

    for ax in axes[ri]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

plt.tight_layout(pad=0.5)
out = f'{OUT_DIR}/scene_visualization.png'
plt.savefig(out, dpi=110, bbox_inches='tight', facecolor='#0f0f0f')
plt.close()
print(f'Saved: {out}')

p.disconnect(physicsClientId=client)
print('Done.')
