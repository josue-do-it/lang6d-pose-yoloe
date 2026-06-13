"""
Visualisation qualitative YCB-Video : pose estimée (bleu) + GT (vert) sur image réelle.
Grille 7×3 objets × 3 panels = 21 objets.

Run inside Docker:
    /opt/conda/envs/Any6D/bin/python3 /workspace/visualization/plot_qualitative_ycbv.py
"""
import os, sys, json, glob
import numpy as np
import cv2
import trimesh
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '/workspace')
from foundationpose.Utils import nvdiffrast_render, draw_xyz_axis
import nvdiffrast.torch as dr

YCBV_ROOT = "/dataset/ycbv"
POSES_DIR = "/workspace/results/ycbv_pipeline/stride5"
OUT_PATH  = "/workspace/results/qualitative_ycbv.png"
MM_TO_M   = 0.001
CROP_PAD  = 55

YCBV_NAMES = {
    1:"master chef can", 2:"cracker box",    3:"sugar box",
    4:"tomato soup can", 5:"mustard bottle", 6:"tuna fish can",
    7:"pudding box",     8:"gelatin box",    9:"potted meat can",
    10:"banana",         11:"pitcher base",  12:"bleach cleanser",
    13:"bowl",           14:"mug",           15:"power drill",
    16:"wood block",     17:"scissors",      18:"large marker",
    19:"large clamp",    20:"extra large clamp", 21:"foam brick",
}

glctx = dr.RasterizeCudaContext()
BLUE  = (50,  120, 220)
GREEN = (46,  160,  70)

# ── helpers ───────────────────────────────────────────────────────────────────

def get_K(scene_id):
    cam_path = f"{YCBV_ROOT}/test/{scene_id:06d}/scene_camera.json"
    with open(cam_path) as f: cam = json.load(f)
    km = list(cam.values())[0]['cam_K']
    return np.array([[km[0],km[1],km[2]],[km[3],km[4],km[5]],[km[6],km[7],km[8]]], dtype=np.float64)

def get_rgb(scene_id, im_id):
    path = f"{YCBV_ROOT}/test/{scene_id:06d}/rgb/{im_id:06d}.png"
    img  = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def get_gt_mask(scene_id, obj_id, im_id):
    """Find mask_visib for obj_id in the given frame."""
    with open(f"{YCBV_ROOT}/test/{scene_id:06d}/scene_gt.json") as f:
        gt = json.load(f)
    anns = gt.get(str(im_id), [])
    for idx, ann in enumerate(anns):
        if ann['obj_id'] == obj_id:
            path = f"{YCBV_ROOT}/test/{scene_id:06d}/mask_visib/{im_id:06d}_{idx:06d}.png"
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return (m > 0) if m is not None else None
    return None

def bbox_from_mask(mask, pad=CROP_PAD, H=None, W=None):
    H = H or mask.shape[0]; W = W or mask.shape[1]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0, 0, W, H
    side = max(xs.max()-xs.min(), ys.max()-ys.min())
    cx, cy = (xs.min()+xs.max())//2, (ys.min()+ys.max())//2
    half = side//2 + pad
    return max(0,cx-half), max(0,cy-half), min(W,cx+half), min(H,cy+half)

def make_pose(R, T):
    p = np.eye(4); p[:3,:3] = np.array(R); p[:3,3] = np.array(T)
    return p

def render_on_frame(rgb, mesh, pose_4x4, K, color_rgb, alpha=0.70):
    H, W = rgb.shape[:2]
    pose_t = torch.tensor(pose_4x4[None], dtype=torch.float32, device='cuda')
    ren_img, ren_depth, _ = nvdiffrast_render(
        K=K, H=H, W=W, mesh=mesh, ob_in_cams=pose_t,
        context='cuda', use_light=True, glctx=glctx, extra={})
    ren_np = (ren_img[0].detach().cpu().numpy() * 255).astype(np.uint8)
    dep_np = ren_depth[0].detach().cpu().numpy()
    mask   = dep_np > 0
    out = rgb.copy().astype(float)
    out[mask] = alpha * ren_np[mask].astype(float) + (1-alpha) * out[mask]
    out = out.clip(0,255).astype(np.uint8)
    contours, _ = cv2.findContours((mask.astype(np.uint8)*255),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.drawContours(out_bgr, contours, -1, color_rgb[::-1], 3)
    out_bgr = draw_xyz_axis(out_bgr, ob_in_cam=pose_4x4, scale=0.05,
                            K=K, thickness=3, transparency=0, is_input_rgb=False)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

def pick_frame(frames):
    """Pick a frame with visible (non-trivial) error: prefer 1–20° range."""
    cands = [fr for fr in frames if 1.0 < fr['R_error'] < 20]
    if cands:
        return min(cands, key=lambda fr: fr['R_error'])
    non_trivial = [fr for fr in frames if fr['R_error'] > 0.5]
    if non_trivial:
        return min(non_trivial, key=lambda fr: fr['R_error'])
    return frames[0]

# ── Grid 7×3 objects × 3 panels ──────────────────────────────────────────────
NGROUPS = 3
NROWS   = 7
NCOLS_PER = 3

fig, axes = plt.subplots(NROWS, NGROUPS * NCOLS_PER,
                         figsize=(NGROUPS * NCOLS_PER * 2.8, NROWS * 2.9))
fig.patch.set_facecolor('#1a1a1a')
for ax in axes.flat: ax.axis('off')

obj_ids = sorted(YCBV_NAMES.keys())   # 1..21

for idx, obj_id in enumerate(obj_ids):
    obj_name = YCBV_NAMES[obj_id]
    row = idx // NGROUPS
    grp = idx %  NGROUPS
    ax_rgb  = axes[row, grp*NCOLS_PER + 0]
    ax_pred = axes[row, grp*NCOLS_PER + 1]
    ax_gt   = axes[row, grp*NCOLS_PER + 2]

    poses_path = f"{POSES_DIR}/obj_{obj_id:06d}_poses.json"
    if not os.path.exists(poses_path):
        ax_rgb.set_title(f"{obj_name}\n(no data)", fontsize=6, color='gray')
        continue

    with open(poses_path) as f: d = json.load(f)
    frames = d['frames']
    best   = pick_frame(frames)

    scene_id  = best['scene_id']
    im_id     = best['im_id']
    r_err     = best['R_error']
    t_err     = best['T_error']
    pred_pose = make_pose(best['R_pred'], best['T_pred'])
    gt_pose   = make_pose(best['R_gt'],   best['T_gt'])

    K   = get_K(scene_id)
    rgb = get_rgb(scene_id, im_id)
    if rgb is None:
        ax_rgb.set_title(f"{obj_name}\n(img missing)", fontsize=6, color='red')
        continue

    # load mesh
    mesh = trimesh.load(f"{YCBV_ROOT}/models/obj_{obj_id:06d}.ply", force='mesh')
    mesh.apply_scale(MM_TO_M)

    # crop bbox from GT mask
    gt_mask = get_gt_mask(scene_id, obj_id, im_id)
    if gt_mask is not None:
        x1, y1, x2, y2 = bbox_from_mask(gt_mask, H=rgb.shape[0], W=rgb.shape[1])
    else:
        x1, y1, x2, y2 = 0, 0, rgb.shape[1], rgb.shape[0]

    vis_pred = render_on_frame(rgb, mesh, pred_pose, K, BLUE)
    vis_gt   = render_on_frame(rgb, mesh, gt_pose,   K, GREEN)

    ax_rgb.imshow(rgb[y1:y2, x1:x2])
    ax_pred.imshow(vis_pred[y1:y2, x1:x2])
    ax_gt.imshow(vis_gt[y1:y2, x1:x2])

    short = obj_name if len(obj_name) <= 14 else obj_name[:13]+'…'
    ax_rgb.set_title(short, fontsize=7, color='white', fontweight='bold', pad=2)

    color_err = '#00ff88' if r_err < 5 else '#ffcc00' if r_err < 20 else '#ff6655'
    ax_pred.set_xlabel(f"R={r_err:.1f}°  T={t_err:.1f}cm",
                       fontsize=6, color=color_err, labelpad=1)

    print(f"  {obj_name:22}: sc={scene_id} im={im_id:4d}  R={r_err:.1f}°  T={t_err:.2f}cm")

# ── legend & title ────────────────────────────────────────────────────────────
legend = [
    mpatches.Patch(facecolor=np.array(BLUE)/255,  label='Estimated pose  (LLM + YOLOE + Any6D)'),
    mpatches.Patch(facecolor=np.array(GREEN)/255, label='Ground truth pose'),
]
fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=9,
           facecolor='#2a2a2a', edgecolor='#555', labelcolor='white',
           framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

fig.suptitle("YCB-Video — Qualitative Pose Estimation Results  (MEAN ADD(S) = 96.3%)\n"
             "GT mesh rendered at estimated pose vs ground truth pose",
             fontsize=11, fontweight='bold', color='white', y=1.01)

plt.subplots_adjust(hspace=0.18, wspace=0.05, left=0.04, right=0.99, top=0.96, bottom=0.05)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"\nSaved → {OUT_PATH}")
