"""
Plot the 3D bounding box of the estimated pose projected onto the image.

Usage:
  python plot_bbox3d.py \
      --json   /workspace/results/infer_pose_bottle/bottle_pose.json \
      --image  /workspace/test_input/bottle.jpg \
      --ply    /workspace/test_input/bottle.ply \
      --K      /workspace/test_input/K.txt \
      --out    /workspace/results/infer_pose_bottle/bottle_bbox3d.png
"""

import argparse, os, json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PLY loader ────────────────────────────────────────────────────────────────
def load_ply(path):
    with open(path, 'rb') as f:
        lines = []
        while True:
            line = f.readline().decode('utf-8', 'ignore').strip()
            lines.append(line)
            if line == 'end_header': break
        n_verts = int(next(l.split()[-1] for l in lines if l.startswith('element vertex')))
        props = [l for l in lines if 'element vertex' not in l and l.startswith('property float')]
        n_props = len(props)
        raw = f.read(n_verts * n_props * 4)
    pts = np.frombuffer(raw[:n_verts*n_props*4], dtype='<f4').reshape(n_verts, n_props)
    xyz = pts[:, :3].copy()
    if np.median(xyz[:, 2]) < 0:
        xyz[:, 2] = -xyz[:, 2]
    return xyz

# ── project Nx3 camera-frame pts → Nx2 pixels ────────────────────────────────
def proj(pts, K):
    u = K[0,0] * pts[:,0] / pts[:,2] + K[0,2]
    v = K[1,1] * pts[:,1] / pts[:,2] + K[1,2]
    return np.stack([u, v], 1).astype(int)

# ── draw 3D bbox (8 corners, 12 edges) on BGR image ──────────────────────────
EDGES = [
    (0,1),(1,2),(2,3),(3,0),   # bottom face
    (4,5),(5,6),(6,7),(7,0+4), # top face
    (0,4),(1,5),(2,6),(3,7),   # verticals
]

def draw_bbox3d(img, corners2d, color=(0,200,255), thickness=3):
    for i, j in EDGES:
        p1 = tuple(corners2d[i])
        p2 = tuple(corners2d[j])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    # draw corners
    for pt in corners2d:
        cv2.circle(img, tuple(pt), 5, color, -1, cv2.LINE_AA)
    return img

def draw_axes(img, R, T, K, length):
    rvec, _ = cv2.Rodrigues(R)
    pts3d = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    try:
        pts2d, _ = cv2.projectPoints(pts3d, rvec, T.reshape(3,1), K, np.zeros(5))
        pts2d = pts2d.astype(int).reshape(-1, 2)
        o = tuple(pts2d[0])
        for pt, col, lbl in zip(pts2d[1:],
                                 [(0,0,220),(0,180,0),(220,60,0)],
                                 ['X','Y','Z']):
            cv2.arrowedLine(img, o, tuple(pt), col, 4, tipLength=0.20, line_type=cv2.LINE_AA)
            cv2.putText(img, lbl, (tuple(pt)[0]+5, tuple(pt)[1]-5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, col, 2, cv2.LINE_AA)
    except Exception:
        pass
    return img

# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    with open(args.json) as f: res = json.load(f)
    R    = np.array(res['R'])
    T    = np.array(res['T'])
    kw   = res['keyword']
    ins  = res['instruction']
    conf = res['yoloe_conf']

    img_bgr = cv2.imread(args.image)
    K       = np.loadtxt(args.K).reshape(3, 3)
    H, W    = img_bgr.shape[:2]

    print("Loading PLY...")
    pts_v = load_ply(args.ply)
    pts_v = pts_v[pts_v[:, 2] > 0.001]

    # ── isolate object points (near centroid T) ──
    dist    = np.linalg.norm(pts_v - T, axis=1)
    thresh  = np.percentile(dist[dist < 0.5], 80) if (dist < 0.5).sum() > 0 else 0.3
    pts_obj = pts_v[dist < thresh]
    if len(pts_obj) < 10:
        pts_obj = pts_v[dist < 0.5]
    print(f"Object points: {len(pts_obj)}")

    # ── compute 3D bbox in object-local frame ──
    # Use bbox saved by infer_pose_single.py (from exact YOLOE mask) if available;
    # otherwise approximate from distance-thresholded pts_obj.
    if 'bbox3d_min' in res and 'bbox3d_max' in res:
        mn = np.array(res['bbox3d_min'])
        mx = np.array(res['bbox3d_max'])
        print(f"BBox from JSON (exact YOLOE mask):")
    else:
        pts_local = (pts_obj - T) @ R
        mn = pts_local.min(axis=0)
        mx = pts_local.max(axis=0)
        print(f"BBox from approximate distance filter:")
    print(f"  {mn.round(3)} → {mx.round(3)}")
    print(f"  Dims (m): {(mx-mn).round(3)}")

    # 8 corners in local frame (standard order: bottom then top)
    corners_local = np.array([
        [mn[0], mn[1], mn[2]],  # 0
        [mx[0], mn[1], mn[2]],  # 1
        [mx[0], mx[1], mn[2]],  # 2
        [mn[0], mx[1], mn[2]],  # 3
        [mn[0], mn[1], mx[2]],  # 4
        [mx[0], mn[1], mx[2]],  # 5
        [mx[0], mx[1], mx[2]],  # 6
        [mn[0], mx[1], mx[2]],  # 7
    ])

    # transform corners to camera frame: p_cam = corners_local @ R.T + T
    corners_cam = corners_local @ R.T + T

    # project to image
    corners2d = proj(corners_cam, K)

    # axis length = ~20% of bbox diagonal
    axis_len = np.linalg.norm(mx - mn) * 0.4

    # centroid pixel
    cx = int(K[0,0]*T[0]/T[2] + K[0,2])
    cy = int(K[1,1]*T[1]/T[2] + K[1,2])

    # ── Figure: 3 panels — original / bbox only / bbox + axes ──
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'3D Bounding Box — "{ins}"  |  keyword: "{kw}"  |  conf={conf:.3f}  |  '
        f'T=[{T[0]:.3f}, {T[1]:.3f}, {T[2]:.3f}] m  |  |T|={np.linalg.norm(T):.3f} m',
        fontsize=11, fontweight='bold', color='#111111', y=1.01
    )

    # Panel 1 — original
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title('(a) Original Image', fontweight='bold')
    axes[0].axis('off')

    # Panel 2 — bbox only
    vis2 = img_bgr.copy()
    draw_bbox3d(vis2, corners2d, color=(0, 200, 255), thickness=3)
    if 0 <= cx < W and 0 <= cy < H:
        cv2.circle(vis2, (cx, cy), 10, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.circle(vis2, (cx, cy), 12, (255,255,255), 2, cv2.LINE_AA)
    # label dimensions
    dims = mx - mn
    label = f'{dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f} m'
    cv2.putText(vis2, label, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,200,255), 2, cv2.LINE_AA)
    axes[1].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('(b) 3D Bounding Box Projection', fontweight='bold')
    axes[1].axis('off')

    # Panel 3 — bbox + pose axes
    vis3 = img_bgr.copy()
    draw_bbox3d(vis3, corners2d, color=(0, 200, 255), thickness=2)
    draw_axes(vis3, R, T, K, length=axis_len)
    if 0 <= cx < W and 0 <= cy < H:
        cv2.circle(vis3, (cx, cy), 10, (0, 220, 220), -1, cv2.LINE_AA)
        cv2.circle(vis3, (cx, cy), 12, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis3, label, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,200,255), 2, cv2.LINE_AA)
    axes[2].imshow(cv2.cvtColor(vis3, cv2.COLOR_BGR2RGB))
    axes[2].set_title('(c) 3D BBox + Pose Axes (X=blue  Y=green  Z=orange)', fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {args.out}")

    # ── also save a clean single-panel for the report ──
    vis_report = img_bgr.copy()
    draw_bbox3d(vis_report, corners2d, color=(0, 200, 255), thickness=3)
    draw_axes(vis_report, R, T, K, length=axis_len)
    if 0 <= cx < W and 0 <= cy < H:
        cv2.circle(vis_report, (cx, cy), 10, (0, 220, 220), -1, cv2.LINE_AA)
    cv2.putText(vis_report, label, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,200,255), 2, cv2.LINE_AA)
    single_out = args.out.replace('.png', '_single.png')
    cv2.imwrite(single_out, vis_report)
    print(f"Saved: {single_out}")

    # ── print summary ──
    print(f"\nBounding box summary:")
    print(f"  Width  (X): {dims[0]:.4f} m")
    print(f"  Height (Y): {dims[1]:.4f} m")
    print(f"  Depth  (Z): {dims[2]:.4f} m")
    print(f"  Centroid T: [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}] m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',  required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--ply',   required=True)
    parser.add_argument('--K',     required=True)
    parser.add_argument('--out',   default='/workspace/results/infer_pose_bottle/bottle_bbox3d.png')
    main(parser.parse_args())
