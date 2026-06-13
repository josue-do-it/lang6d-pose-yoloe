"""
Visualisation qualitative LINEMOD : pose estimée (bleu) + GT (vert) sur image réelle.
- Ligne par objet : [Image recadrée | Pose estimée | Ground Truth]
- Chaque frame est sélectionnée pour montrer un écart visible (ni trop parfait, ni trop mauvais)
- Crop centré sur l'objet (mask GT bounding box)

Run inside Docker:
    /opt/conda/envs/Any6D/bin/python3 /workspace/visualization/plot_qualitative_linemod.py
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

LM_ROOT   = "/dataset/lm"
POSES_DIR = "/workspace/results/lm_pipeline/full_stride5"
OUT_PATH  = "/workspace/results/qualitative_linemod.png"

LM_NAMES = {
    1:'ape', 2:'benchvise', 3:'bowl',    4:'camera',      5:'can',
    6:'cat', 7:'cup',       8:'driller', 9:'duck',        10:'eggbox',
    11:'glue', 12:'holepuncher', 13:'iron', 14:'lamp',    15:'phone',
}
MM_TO_M  = 0.001
CROP_PAD = 60   # extra padding around object bbox for crop

# Camera intrinsics
with open(f"{LM_ROOT}/lm/camera.json") as f:
    c = json.load(f)
K_np = np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0,0,1]], dtype=np.float64)

glctx = dr.RasterizeCudaContext()

BLUE  = (50,  120, 220)   # RGB for estimated
GREEN = (46,  160,  70)   # RGB for GT

# ── helpers ───────────────────────────────────────────────────────────────────

def get_rgb(obj_id, im_id):
    path = f"{LM_ROOT}/test/{obj_id:06d}/rgb/{im_id:06d}.png"
    img  = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def get_gt_mask(obj_id, im_id):
    """Load GT visibility mask (for crop bbox)."""
    # masks are named by annotation index — find the first one
    mask_dir = f"{LM_ROOT}/test/{obj_id:06d}/mask_visib"
    pattern  = f"{im_id:06d}_*.png"
    files    = sorted(glob.glob(os.path.join(mask_dir, pattern)))
    if not files:
        return None
    mask = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    return mask

def object_bbox_from_mask(mask, pad=CROP_PAD, H=480, W=640):
    """Get padded bounding box from binary mask."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0, 0, W, H
    x1, x2 = max(0, xs.min()-pad), min(W, xs.max()+pad)
    y1, y2 = max(0, ys.min()-pad), min(H, ys.max()+pad)
    # make square
    side = max(x2-x1, y2-y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    half = side // 2 + 10
    x1 = max(0, cx-half); x2 = min(W, cx+half)
    y1 = max(0, cy-half); y2 = min(H, cy+half)
    return x1, y1, x2, y2

def make_pose(R, T):
    p = np.eye(4); p[:3,:3] = np.array(R); p[:3,3] = np.array(T)
    return p

def render_on_frame(frame_rgb, mesh, pose_4x4, contour_color_rgb, alpha=0.70):
    """Render mesh overlay with coloured contour."""
    H, W = frame_rgb.shape[:2]
    pose_t = torch.tensor(pose_4x4[None], dtype=torch.float32, device='cuda')
    ren_img, ren_depth, _ = nvdiffrast_render(
        K=K_np, H=H, W=W, mesh=mesh,
        ob_in_cams=pose_t, context='cuda',
        use_light=True, glctx=glctx, extra={})
    ren_np  = (ren_img[0].detach().cpu().numpy() * 255).astype(np.uint8)
    dep_np  = ren_depth[0].detach().cpu().numpy()
    mask    = dep_np > 0

    out = frame_rgb.copy().astype(float)
    out[mask] = alpha * ren_np[mask].astype(float) + (1-alpha) * out[mask]
    out = out.clip(0, 255).astype(np.uint8)

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.drawContours(out_bgr, contours, -1, contour_color_rgb[::-1], 3)

    # XYZ axes
    out_bgr = draw_xyz_axis(out_bgr, ob_in_cam=pose_4x4, scale=0.05,
                            K=K_np, thickness=3, transparency=0, is_input_rgb=False)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

def pick_best_frame(frames):
    """
    Pick the frame that has a moderate, visible error.
    Target: R_error in [2°, 30°] for a visually informative comparison.
    If none found, fall back to lowest non-zero error.
    """
    # prefer frames with small but non-trivial error (2–30°)
    candidates = [fr for fr in frames if 1.0 < fr['R_error'] < 30]
    if candidates:
        return min(candidates, key=lambda fr: fr['R_error'])
    # fall back: any non-trivial
    non_trivial = [fr for fr in frames if fr['R_error'] > 0.5]
    if non_trivial:
        return min(non_trivial, key=lambda fr: fr['R_error'])
    return frames[0]

# ── Layout: 5 rows × 3 groups × 3 panels ────────────────────────────────────
NCOLS_PER_OBJ = 3   # raw | pred | gt
NGROUPS = 3         # objects per row
NROWS   = 5

fig, axes = plt.subplots(NROWS, NGROUPS * NCOLS_PER_OBJ,
                         figsize=(NGROUPS * NCOLS_PER_OBJ * 2.8, NROWS * 2.8))
fig.patch.set_facecolor('#1a1a1a')
for ax in axes.flat: ax.axis('off')

# Column headers (top row)
header_info = [("Input Image", 'white'),
               ("Estimated Pose (Ours)", '#6abfff'),
               ("Ground Truth Pose",     '#7fd980')]
for g in range(NGROUPS):
    for ci, (title, color) in enumerate(header_info):
        col = g * NCOLS_PER_OBJ + ci
        if g == 1 or ci == 0:
            axes[0, col].set_title(title, fontsize=7.5, color=color,
                                   fontweight='bold', pad=4)

obj_ids = sorted(LM_NAMES.keys())

for idx, obj_id in enumerate(obj_ids):
    obj_name = LM_NAMES[obj_id]
    row = idx // NGROUPS
    grp = idx %  NGROUPS
    ax_rgb  = axes[row, grp * NCOLS_PER_OBJ + 0]
    ax_pred = axes[row, grp * NCOLS_PER_OBJ + 1]
    ax_gt   = axes[row, grp * NCOLS_PER_OBJ + 2]

    poses_path = f"{POSES_DIR}/obj_{obj_id:06d}_poses.json"
    if not os.path.exists(poses_path):
        ax_rgb.set_title(f"{obj_name}\n(no data)", fontsize=7, color='gray')
        continue

    with open(poses_path) as f:
        d = json.load(f)
    frames = d['frames']
    best   = pick_best_frame(frames)

    im_id     = best['im_id']
    r_err     = best['R_error']
    t_err     = best['T_error']
    pred_pose = make_pose(best['R_pred'], best['T_pred'])
    gt_pose   = make_pose(best['R_gt'],   best['T_gt'])

    # load mesh in metres
    mesh_path = f"{LM_ROOT}/models/obj_{obj_id:06d}.ply"
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.apply_scale(MM_TO_M)

    rgb = get_rgb(obj_id, im_id)
    if rgb is None:
        ax_rgb.set_title(f"{obj_name}\n(img missing)", fontsize=7, color='red')
        continue

    # crop bbox from GT mask
    mask_gt = get_gt_mask(obj_id, im_id)
    if mask_gt is not None:
        x1, y1, x2, y2 = object_bbox_from_mask(mask_gt, H=rgb.shape[0], W=rgb.shape[1])
    else:
        x1, y1, x2, y2 = 0, 0, rgb.shape[1], rgb.shape[0]

    # render on full image, then crop
    vis_pred = render_on_frame(rgb, mesh, pred_pose, BLUE)
    vis_gt   = render_on_frame(rgb, mesh, gt_pose,   GREEN)

    rgb_crop  = rgb[y1:y2,    x1:x2]
    pred_crop = vis_pred[y1:y2, x1:x2]
    gt_crop   = vis_gt[y1:y2,   x1:x2]

    ax_rgb.imshow(rgb_crop)
    ax_pred.imshow(pred_crop)
    ax_gt.imshow(gt_crop)

    # object name label left side of first panel in each group
    if grp == 0:
        ax_rgb.set_ylabel(obj_name, fontsize=8, color='white',
                          rotation=0, ha='right', va='center', labelpad=6)

    # error label below pred
    color_err = '#00ff88' if r_err < 5 else '#ffcc00' if r_err < 20 else '#ff6655'
    ax_pred.set_xlabel(f"R={r_err:.1f}°  T={t_err:.1f}cm",
                       fontsize=6.5, color=color_err, labelpad=2)
    ax_rgb.set_title(f"{obj_name}", fontsize=7.5, color='white', fontweight='bold', pad=2)

    print(f"  {obj_name:14}: frame={im_id:4d}  R={r_err:.1f}°  T={t_err:.2f}cm")

# ── legend ────────────────────────────────────────────────────────────────────
legend = [
    mpatches.Patch(facecolor=np.array(BLUE)/255,  label='Estimated pose  (LLM + YOLOE + Any6D)'),
    mpatches.Patch(facecolor=np.array(GREEN)/255, label='Ground truth pose'),
]
fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=9,
           facecolor='#2a2a2a', edgecolor='#555', labelcolor='white',
           framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

fig.suptitle("LINEMOD — Qualitative Pose Estimation Results\n"
             "GT mesh rendered at estimated pose vs ground truth pose",
             fontsize=11, fontweight='bold', color='white', y=1.01)

plt.subplots_adjust(hspace=0.15, wspace=0.05, left=0.08, right=0.99, top=0.96, bottom=0.06)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"\nSaved → {OUT_PATH}")
