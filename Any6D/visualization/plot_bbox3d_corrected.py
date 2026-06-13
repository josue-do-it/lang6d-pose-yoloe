"""
Projette le 3D bounding box du mesh GT aux poses corrigées (pred_q et gt_pose_q).
Bleu = estimé, Vert = GT. Montre 4 frames par séquence (bonne + mauvaise).

Run inside Docker:
    /opt/conda/envs/Any6D/bin/python3 /workspace/visualization/plot_bbox3d_corrected.py
"""
import os, sys, json, glob, pickle
import numpy as np
import cv2
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace')

POSES_DIR  = "/workspace/results/ho3d_pipeline/run_full"
HO3D_ROOT  = "/dataset/ho3d/HO3D_data/evaluation"
YCB_MODELS = "/dataset/ho3d/models"
OUT_PATH   = "/workspace/results/bbox3d_verification.png"
FRAME_STRIDE = 10

ROWS = [
    ("AP13", "019_pitcher_base",   "Pitcher"),
    ("SM1",  "006_mustard_bottle", "Mustard"),
    ("SB11", "021_bleach_cleanser","Bleach"),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def load_K(seq_id):
    files = sorted(glob.glob(os.path.join(HO3D_ROOT, seq_id, "meta", "*.pkl")))
    with open(files[0], 'rb') as f: meta = pickle.load(f, encoding='latin1')
    return meta['camMat'].astype(np.float64)

def load_frame(seq_id, eval_idx):
    rgb_dir = os.path.join(HO3D_ROOT, seq_id, "rgb")
    frames  = sorted(glob.glob(f"{rgb_dir}/*.jpg") + glob.glob(f"{rgb_dir}/*.png"),
                     key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    sampled = frames[::FRAME_STRIDE]
    return cv2.cvtColor(cv2.imread(sampled[eval_idx]), cv2.COLOR_BGR2RGB)

def get_bbox3d_corners(mesh):
    """8 corners of axis-aligned bounding box of mesh vertices (in object frame)."""
    v  = np.asarray(mesh.vertices)
    mn, mx = v.min(axis=0), v.max(axis=0)
    corners = np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ])
    return corners  # (8,3)

def project_corners(corners_3d, pose_4x4, K):
    """Project 3D corners to 2D image pixels."""
    R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
    pts_cam = (R @ corners_3d.T).T + t          # (8,3) in camera frame
    pts_img = (K @ pts_cam.T).T                  # (8,3) homogeneous
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]  # (8,2) pixel coords
    return pts_img.astype(int)

EDGES = [
    (0,1),(1,2),(2,3),(3,0),   # bottom face
    (4,5),(5,6),(6,7),(7,4),   # top face
    (0,4),(1,5),(2,6),(3,7),   # vertical edges
]

def draw_box3d(img_rgb, corners_2d, color_bgr, thickness=2):
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = out.shape[:2]
    for i, j in EDGES:
        p1 = tuple(np.clip(corners_2d[i], [0,0], [W-1,H-1]))
        p2 = tuple(np.clip(corners_2d[j], [0,0], [W-1,H-1]))
        cv2.line(out, p1, p2, color_bgr, thickness, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def draw_axes(img_rgb, pose_4x4, K, scale=0.05, thickness=3):
    """Draw XYZ axes: X=red, Y=green, Z=blue."""
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
    origin = (K @ t).reshape(3)
    origin = (int(origin[0]/origin[2]), int(origin[1]/origin[2]))
    for axis, color in [(np.array([scale,0,0]), (0,0,255)),
                        (np.array([0,scale,0]), (0,255,0)),
                        (np.array([0,0,scale]), (255,0,0))]:
        pt = t + R @ axis
        pt2d = K @ pt
        pt2d = (int(pt2d[0]/pt2d[2]), int(pt2d[1]/pt2d[2]))
        cv2.arrowedLine(out, origin, pt2d, color, thickness,
                        cv2.LINE_AA, tipLength=0.3)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # 4 frames per row: 2 best + 2 worst (by R_error)
    n_show = 4
    fig, axes = plt.subplots(len(ROWS), n_show,
                             figsize=(n_show * 4, len(ROWS) * 3.2))
    fig.patch.set_facecolor('white')
    for ax in axes.flat: ax.axis('off')

    for ri, (seq_id, obj_folder, display_name) in enumerate(ROWS):
        print(f"[{ri+1}/3] {seq_id}")

        with open(os.path.join(POSES_DIR, f"{seq_id}_poses.json")) as f:
            d = json.load(f)
        frames_data = d['frames']

        mesh = trimesh.load(os.path.join(YCB_MODELS, obj_folder, "textured_simple.obj"),
                            force='mesh')
        corners = get_bbox3d_corners(mesh)
        K = load_K(seq_id)

        # sort by R_error, pick 2 best + 2 worst
        sorted_frames = sorted(enumerate(frames_data), key=lambda x: x[1]['R_error'])
        chosen = [sorted_frames[0], sorted_frames[1],
                  sorted_frames[-2], sorted_frames[-1]]

        for ci, (fidx, fr) in enumerate(chosen):
            img = load_frame(seq_id, fidx)

            pred_q = np.eye(4)
            pred_q[:3, :3] = np.array(fr['R_pred'])
            pred_q[:3,  3] = np.array(fr['T_pred'])

            gt_q = np.eye(4)
            gt_q[:3, :3] = np.array(fr['R_gt'])
            gt_q[:3,  3] = np.array(fr['T_gt'])

            # project bbox corners
            c_pred = project_corners(corners, pred_q, K)
            c_gt   = project_corners(corners, gt_q,   K)

            # draw GT box (green) then pred box (blue) then axes
            vis = draw_box3d(img,   c_gt,   (0, 180, 0),   thickness=2)
            vis = draw_box3d(vis,   c_pred, (50, 120, 220), thickness=2)
            vis = draw_axes(vis, pred_q, K, scale=0.05, thickness=3)

            r_err = fr['R_error']
            t_err = fr['T_error']
            label = f"R={r_err:.1f}°  T={t_err:.2f}cm"
            color_txt = (0, 200, 0) if r_err < 30 else (220, 50, 50)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.putText(vis_bgr, label, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_txt[::-1], 2, cv2.LINE_AA)
            vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

            axes[ri, ci].imshow(vis)
            qual = "best" if ci < 2 else "worst"
            axes[ri, ci].set_title(f"frame {fidx} ({qual})", fontsize=8)

        axes[ri, 0].set_ylabel(display_name, fontsize=10,
                               rotation=0, ha='right', va='center', labelpad=8)

    fig.suptitle(
        "3D Bounding Box Verification — Blue=Estimated  Green=GT\n"
        "Columns: 2 best frames (lowest R_error) | 2 worst frames",
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved → {OUT_PATH}")

if __name__ == "__main__":
    main()
