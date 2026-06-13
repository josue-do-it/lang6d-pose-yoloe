"""
Qualitative comparison figure — style Any6D paper.

Uses the REAL Any6D debug images (mesh rendered at estimated/GT pose)
overlaid on the actual HO3D query frames.

Columns : Anchor Image | LLM+YOLOE+Any6D Estimated | Ground Truth
Rows    : AP13 (pitcher) | SM1 (mustard) | SB11 (bleach cleanser)

Run inside Docker:
    /opt/conda/envs/Any6D/bin/python3 /workspace/visualization/plot_qualitative_comparison.py
"""
import cv2
import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = "/workspace/results/ho3d_pipeline/run_full"
ANCHOR_DIR  = "/workspace/anchor_results/dexycb_reference_view_ours"
HO3D_ROOT   = "/dataset/ho3d/HO3D_data/evaluation"
OUT_PATH    = "/workspace/results/qualitative_comparison.png"

FRAME_STRIDE = 10

# Best frames (pre-selected by T_error)
BEST_FRAMES = {"AP13": 27, "SM1": 70, "SB11": 113}

# Row definitions
ROWS = [
    ("AP13", "019_pitcher_base",   "Pitcher Base",    "blue pouring container"),
    ("SM1",  "006_mustard_bottle", "Mustard Bottle",  "mustard bottle"),
    ("SB11", "021_bleach_cleanser","Bleach Cleanser", "cleaning bottle"),
]

# ── Panel layout of AP13_img_NNNNN.png (530×1920, 4 panels of 480) ────────────
# Panel 0 : RGB query frame + metrics text
# Panel 1 : GT Mesh on real frame (left) + GT Mesh on dark bg (right)
# Panel 2 : GT Mesh on dark bg  (full panel)
# Panel 3 : EST Mesh on dark bg (full panel)
#
# The dark-bg panels contain the 3D mesh rendered in place.
# We extract the non-black region and blend it onto the real frame.

def get_pipeline_folder(seq_id):
    """Find the pipeline image folder for a sequence (name varies by frame count)."""
    candidates = glob.glob(os.path.join(RESULTS_DIR, f"{seq_id}_img_*_pipeline"))
    if not candidates:
        raise FileNotFoundError(f"No pipeline folder for {seq_id}")
    return candidates[0]

def get_pipeline_panels(seq_id, frame_idx):
    folder = get_pipeline_folder(seq_id)
    path   = os.path.join(folder, f"{seq_id}_img_{frame_idx:05d}.png")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    W = img.shape[1] // 4
    panels = [img[:, i*W:(i+1)*W] for i in range(4)]
    return panels   # list of 4 BGR images (530×480 each)

def load_ho3d_frame(seq_id, eval_frame_idx):
    """Load the actual HO3D frame used by the pipeline (stride=10 sampling)."""
    rgb_dir = os.path.join(HO3D_ROOT, seq_id, "rgb")
    frames  = sorted(
        glob.glob(f"{rgb_dir}/*.jpg") + glob.glob(f"{rgb_dir}/*.png"),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    sampled = frames[::FRAME_STRIDE]
    return cv2.imread(sampled[eval_frame_idx])   # BGR 480×640

def load_anchor_rgb(obj_folder):
    path = os.path.join(ANCHOR_DIR, obj_folder, f"{obj_folder}_img_00000.png")
    img  = cv2.imread(path)
    W    = img.shape[1] // 3
    return img[:, :W, :]   # first panel = raw anchor photo

def blend_mesh_on_frame(frame_bgr, mesh_on_dark, alpha=0.75):
    """
    Overlay a mesh rendered on black background onto a real frame.
    Wherever mesh_on_dark is not black (any channel > threshold),
    we replace/blend with the rendered colour.
    Returns a BGR image the same size as frame_bgr.
    """
    # resize mesh panel to match frame
    h, w = frame_bgr.shape[:2]
    mesh_r = cv2.resize(mesh_on_dark, (w, h), interpolation=cv2.INTER_LINEAR)

    # mask: pixels where the mesh is visible (non-black)
    mask = mesh_r.max(axis=2) > 15          # bool HxW

    out = frame_bgr.copy()
    # blend: alpha*mesh + (1-alpha)*frame where mask is True
    out[mask] = (alpha * mesh_r[mask].astype(float)
                 + (1 - alpha) * frame_bgr[mask].astype(float)).astype(np.uint8)
    return out

def load_metrics(seq_id, frame_idx):
    path = os.path.join(RESULTS_DIR, f"{seq_id}_poses.json")
    with open(path) as f:
        d = json.load(f)
    fr = d['frames'][frame_idx]
    return (fr.get('R_error', float('nan')),
            fr.get('T_error', float('nan')),   # cm
            d['instruction'],
            d['yoloe_prompt'])

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    n_rows, n_cols = len(ROWS), 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.5, n_rows * 3.4))
    fig.patch.set_facecolor('white')
    for ax in axes.flat:
        ax.axis('off')

    col_titles = ["Anchor Image",
                  "LLM + YOLOE + Any6D  (Ours)",
                  "Ground Truth"]
    col_colors = ["#333333", "#1a5276", "#1a7a30"]
    for ci, (t, c) in enumerate(zip(col_titles, col_colors)):
        axes[0, ci].set_title(t, fontsize=12, fontweight='bold',
                              color=c, pad=10)

    for ri, (seq_id, obj_folder, display_name, keyword) in enumerate(ROWS):
        fidx = BEST_FRAMES[seq_id]
        print(f"[{ri+1}/{n_rows}] {seq_id}  frame={fidx}  ({display_name})")

        # load data
        anchor      = load_anchor_rgb(obj_folder)
        frame_bgr   = load_ho3d_frame(seq_id, fidx)
        panels      = get_pipeline_panels(seq_id, fidx)
        r_err, t_err, instr, kw = load_metrics(seq_id, fidx)

        def remove_text(panel):
            """Erase orange debug text (BGR ~0,140,255) from mesh panels."""
            img = panel.copy()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # orange: hue 5-25, high saturation, high value
            lo = np.array([5,  150, 150], dtype=np.uint8)
            hi = np.array([25, 255, 255], dtype=np.uint8)
            text_mask = cv2.inRange(hsv, lo, hi)
            # also catch bright white text
            lo2 = np.array([0, 0, 230], dtype=np.uint8)
            hi2 = np.array([180, 30, 255], dtype=np.uint8)
            white_mask = cv2.inRange(hsv, lo2, hi2)
            mask = cv2.bitwise_or(text_mask, white_mask)
            # dilate slightly to catch anti-aliased edges
            kernel = np.ones((3,3), np.uint8)
            mask   = cv2.dilate(mask, kernel, iterations=1)
            img[mask > 0] = 0
            return img

        p2_clean = remove_text(panels[2])
        p3_clean = remove_text(panels[3])

        # panel 3 = EST mesh on dark bg → overlay on real frame
        est_overlay = blend_mesh_on_frame(frame_bgr, p3_clean, alpha=0.85)
        # panel 2 = GT mesh on dark bg → overlay on real frame
        gt_overlay  = blend_mesh_on_frame(frame_bgr, p2_clean, alpha=0.85)

        # ── column 0: anchor ─────────────────────────────────────────────────
        axes[ri, 0].imshow(cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB))
        axes[ri, 0].set_ylabel(
            f"{display_name}\n\"{kw}\"",
            fontsize=9, rotation=0, ha='right', va='center', labelpad=8)

        # ── column 1: our prediction ─────────────────────────────────────────
        axes[ri, 1].imshow(cv2.cvtColor(est_overlay, cv2.COLOR_BGR2RGB))

        # ── column 2: ground truth ───────────────────────────────────────────
        axes[ri, 2].imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB))
        axes[ri, 2].set_xlabel(
            f"R_err = {r_err:.1f}°    T_err = {t_err:.2f} cm",
            fontsize=8.5, color='#444444')

    # legend
    legend_patches = [
        mpatches.Patch(color='#3A82C4', label='Estimated mesh  (LLM + YOLOE + Any6D)'),
        mpatches.Patch(color='#2EAA50', label='Ground truth mesh'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2,
               fontsize=10, framealpha=0.95, facecolor='white',
               edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Qualitative Results — LLM + YOLOE + Any6D Pipeline on HO3D\n"
        "InstantMesh reconstruction rendered at estimated (Ours) vs ground truth pose",
        fontsize=11, fontweight='bold', color='#111111', y=1.02)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
