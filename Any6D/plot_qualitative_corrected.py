"""
Qualitative figure with CORRECTED poses (pred_q = relative pose correction applied).

Uses pred_q from AP13_poses.json (R_pred, T_pred after pose_aq @ gt_pose_a).
Renders the GT YCB CAD mesh at corrected pred_q and gt_pose_q using nvdiffrast.

Run inside Docker:
    /opt/conda/envs/Any6D/bin/python3 /workspace/plot_qualitative_corrected.py
"""
import os, sys, json, glob
import numpy as np
import cv2
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nvdiffrast.torch as dr

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/yoloe')

from foundationpose.Utils import nvdiffrast_render, make_mesh_tensors, draw_xyz_axis

# ── Paths ──────────────────────────────────────────────────────────────────────
POSES_DIR   = "/workspace/results/ho3d_pipeline/run_full"
HO3D_ROOT   = "/dataset/ho3d/HO3D_data/evaluation"
YCB_MODELS  = "/dataset/ho3d/models"
ANCHOR_DIR  = "/workspace/anchor_results/dexycb_reference_view_ours"
OUT_PATH    = "/workspace/results/qualitative_corrected.png"
FRAME_STRIDE = 10

# Best frames (by R_error on corrected pred_q)
ROWS = [
    ("AP13", "019_pitcher_base", "Pitcher Base",    "blue pouring container", 136),
    ("SM1",  "006_mustard_bottle","Mustard Bottle", "mustard bottle",          22),
    ("SB11", "021_bleach_cleanser","Bleach Cleanser","cleaning bottle",        134),
]

# ── Setup ──────────────────────────────────────────────────────────────────────
glctx = dr.RasterizeCudaContext()

def load_ho3d_frame(seq_id, eval_idx):
    rgb_dir = os.path.join(HO3D_ROOT, seq_id, "rgb")
    frames  = sorted(glob.glob(f"{rgb_dir}/*.jpg") + glob.glob(f"{rgb_dir}/*.png"),
                     key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    sampled = frames[::FRAME_STRIDE]
    img = cv2.cvtColor(cv2.imread(sampled[eval_idx]), cv2.COLOR_BGR2RGB)
    return img

def load_K(seq_id):
    import pickle, glob
    files = sorted(glob.glob(os.path.join(HO3D_ROOT, seq_id, "meta", "*.pkl")))
    with open(files[0], 'rb') as f: meta = pickle.load(f, encoding='latin1')
    return meta['camMat'].astype(np.float64)

def load_anchor_rgb(obj_folder):
    path = os.path.join(ANCHOR_DIR, obj_folder, f"{obj_folder}_img_00000.png")
    img  = cv2.imread(path)
    W    = img.shape[1] // 3
    return cv2.cvtColor(img[:, :W], cv2.COLOR_BGR2RGB)

def render_mesh_on_frame(frame_rgb, mesh_trimesh, pose_4x4, K, color_rgb, alpha=0.85):
    """Render mesh at given pose, blend coloured contour onto frame."""
    H, W = frame_rgb.shape[:2]
    K_t  = torch.tensor(K, dtype=torch.float32, device='cuda')
    pose_t = torch.tensor(pose_4x4[None], dtype=torch.float32, device='cuda')

    ren_img, ren_depth, _ = nvdiffrast_render(
        K=K, H=H, W=W, mesh=mesh_trimesh,
        ob_in_cams=pose_t, context='cuda',
        use_light=True, glctx=glctx, extra={})

    ren_np  = (ren_img[0].detach().cpu().numpy() * 255).astype(np.uint8)
    dep_np  = ren_depth[0].detach().cpu().numpy()
    mask    = (dep_np > 0)

    # coloured contour overlay
    out = frame_rgb.copy().astype(float)
    out[mask] = alpha * ren_np[mask].astype(float) + (1 - alpha) * out[mask]
    out = out.clip(0, 255).astype(np.uint8)

    # draw contour border
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.drawContours(out_bgr, contours, -1, color_rgb[::-1], 2)   # BGR

    # draw XYZ axis arrows (X=red, Y=green, Z=blue)
    out_bgr = draw_xyz_axis(out_bgr, ob_in_cam=pose_4x4, scale=0.05,
                            K=K, thickness=3, transparency=0, is_input_rgb=False)
    out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.2))
    fig.patch.set_facecolor('white')
    for ax in axes.flat: ax.axis('off')

    col_titles  = ["Anchor Image", "LLM + YOLOE + Any6D  (Ours)", "Ground Truth"]
    col_colors  = ["#333333", "#1a5276", "#1a7a30"]
    for ci, (t, c) in enumerate(zip(col_titles, col_colors)):
        axes[0, ci].set_title(t, fontsize=12, fontweight='bold', color=c, pad=10)

    BLUE  = (58,  130, 196)   # estimated
    GREEN = (46,  170,  80)   # GT

    for ri, (seq_id, obj_folder, display_name, keyword, fidx) in enumerate(ROWS):
        print(f"[{ri+1}/3] {seq_id}  frame={fidx}  ({display_name})")

        # ── load poses from JSON ───────────────────────────────────────────────
        with open(os.path.join(POSES_DIR, f"{seq_id}_poses.json")) as f:
            d = json.load(f)
        fr = d['frames'][fidx]
        R_pred = np.array(fr['R_pred']); T_pred = np.array(fr['T_pred'])
        R_gt   = np.array(fr['R_gt']);   T_gt   = np.array(fr['T_gt'])
        r_err  = fr['R_error'];          t_err  = fr['T_error']

        pred_q = np.eye(4); pred_q[:3,:3] = R_pred; pred_q[:3,3] = T_pred
        gt_q   = np.eye(4); gt_q[:3,:3]   = R_gt;   gt_q[:3,3]   = T_gt

        # ── load GT YCB mesh (in metres) ───────────────────────────────────────
        mesh_path = os.path.join(YCB_MODELS, obj_folder, "textured_simple.obj")
        gt_mesh   = trimesh.load(mesh_path, force='mesh')

        # ── load frame + K ────────────────────────────────────────────────────
        frame_rgb = load_ho3d_frame(seq_id, fidx)
        K = load_K(seq_id)

        # ── render ────────────────────────────────────────────────────────────
        est_img = render_mesh_on_frame(frame_rgb, gt_mesh, pred_q, K, BLUE)
        gt_img  = render_mesh_on_frame(frame_rgb, gt_mesh, gt_q,   K, GREEN)

        # ── anchor ────────────────────────────────────────────────────────────
        anchor = load_anchor_rgb(obj_folder)

        # ── plot ──────────────────────────────────────────────────────────────
        axes[ri, 0].imshow(anchor)
        axes[ri, 0].set_ylabel(f"{display_name}\n\"{keyword}\"",
                               fontsize=9, rotation=0, ha='right', va='center', labelpad=8)

        axes[ri, 1].imshow(est_img)

        axes[ri, 2].imshow(gt_img)
        axes[ri, 2].set_xlabel(
            f"R_err = {r_err:.1f}°    T_err = {t_err:.2f} cm",
            fontsize=8.5, color='#444444')

        print(f"   R_err={r_err:.1f}°  T_err={t_err:.2f}cm")

    legend_patches = [
        mpatches.Patch(color=np.array(BLUE)/255,  label='Estimated pose  (LLM + YOLOE + Any6D)'),
        mpatches.Patch(color=np.array(GREEN)/255, label='Ground truth pose'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2,
               fontsize=10, framealpha=0.95, facecolor='white',
               edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Qualitative Results — LLM + YOLOE + Any6D on HO3D\n"
        "GT YCB mesh rendered at corrected estimated pose vs ground truth pose",
        fontsize=11, fontweight='bold', color='#111111', y=1.02)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
