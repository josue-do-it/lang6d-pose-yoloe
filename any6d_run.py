import sys
import os
import numpy as np
import cv2

# workspace FIRST — before foundationpose
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/foundationpose/mycpp/build')

import trimesh
import nvdiffrast.torch as dr
from estimater import Any6D
from pytorch_lightning import seed_everything

seed_everything(0)
glctx = dr.RasterizeCudaContext()

# Load inputs saved by YOLOE
ob_mask   = np.load('/tmp/yoloe_mask.npy')
scene_bgr = cv2.imread('/tmp/yoloe_scene_rgb.png')
scene_rgb = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB)
depth     = np.load('/tmp/yoloe_depth.npy')
K         = np.load('/tmp/yoloe_K.npy')

print('ob_mask pixels:', ob_mask.sum())
print('scene shape:', scene_rgb.shape)
print('depth range:', round(float(depth.min()), 2), '-', round(float(depth.max()), 2), 'm')

# Load mesh
mesh = trimesh.load('/workspace/demo_data/mustard.obj')
print('mesh vertices:', len(mesh.vertices))

# Initialize Any6D
os.makedirs('/tmp/any6d_results', exist_ok=True)
est = Any6D(
    symmetry_tfs=None,
    mesh=mesh,
    debug_dir='/tmp/any6d_results',
    debug=0
)

# Run pose estimation with YOLOE mask
pred_pose = est.register_any6d(
    K=K,
    rgb=scene_rgb,
    depth=depth,
    ob_mask=ob_mask,
    iteration=5,
    name='yoloe_any6d'
)

# Save results
np.save('/tmp/any6d_pose.npy', pred_pose)
np.savetxt('/tmp/any6d_pose.txt', pred_pose)

t = pred_pose[:3, 3]
print('POSE RESULT:')
print(pred_pose)
print('X:', round(t[0]*100, 1), 'cm')
print('Y:', round(t[1]*100, 1), 'cm')
print('Z:', round(t[2]*100, 1), 'cm')
