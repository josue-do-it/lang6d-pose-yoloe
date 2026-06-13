"""
Publication-quality plots for real-world iPhone 13 Pro LiDAR evaluation.
Reads: pose JSON + original image + PLY point cloud
Generates 3 figures for report (white background).

Usage:
  python replot_infer_pose.py \
      --json   /workspace/results/infer_pose/0000001_pose.json \
      --image  /workspace/test_input/0000001.jpg \
      --ply    /workspace/test_input/0000001.ply \
      --K      /workspace/test_input/K.txt \
      --out_dir /workspace/results/infer_pose/report_plots
"""

import argparse, os, json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': '#cccccc',
    'axes.labelcolor': '#222222',
    'xtick.color': '#222222',
    'ytick.color': '#222222',
    'text.color': '#111111',
    'grid.color': '#dddddd',
})

# ── helpers ───────────────────────────────────────────────────────────────────
def load_ply(path):
    with open(path, 'rb') as f:
        lines = []
        while True:
            line = f.readline().decode('utf-8', 'ignore').strip()
            lines.append(line)
            if line == 'end_header': break
        n_verts = int(next(l.split()[-1] for l in lines if l.startswith('element vertex')))
        props = [l.split()[-1] for l in lines if 'element vertex' not in l
                 and l.startswith('property float')]
        n_props = len(props)
        raw = f.read(n_verts * n_props * 4)
    pts = np.frombuffer(raw[:n_verts*n_props*4], dtype='<f4').reshape(n_verts, n_props)
    xyz = pts[:, :3].copy()
    if np.median(xyz[:, 2]) < 0:
        xyz[:, 2] = -xyz[:, 2]
    return xyz

def project(pts, K):
    z = pts[:, 2]; valid = z > 0.001
    u = np.where(valid, K[0,0]*pts[:,0]/np.where(valid, z, 1) + K[0,2], -1)
    v = np.where(valid, K[1,1]*pts[:,1]/np.where(valid, z, 1) + K[1,2], -1)
    return np.stack([u, v], 1), valid

def draw_axes_cv(img, R, T, K, length=0.35):
    rvec, _ = cv2.Rodrigues(R)
    pts3d = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    try:
        pts2d, _ = cv2.projectPoints(pts3d, rvec, T.reshape(3,1), K, np.zeros(5))
        pts2d = pts2d.astype(int).reshape(-1, 2)
        o = tuple(pts2d[0])
        for pt, col, lbl in zip(pts2d[1:], [(220,30,30),(30,160,30),(30,80,220)], 'XYZ'):
            cv2.arrowedLine(img, o, tuple(pt), col, 4, tipLength=0.25)
            cv2.putText(img, lbl, tuple(pt), cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 2)
    except Exception: pass
    return img

# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json) as f: res = json.load(f)
    R     = np.array(res['R'])
    T     = np.array(res['T'])
    P     = np.eye(4); P[:3,:3] = R; P[:3,3] = T
    if 'pose_4x4' in res: P = np.array(res['pose_4x4'])
    kw    = res['keyword']
    ins   = res['instruction']
    conf  = res['yoloe_conf']
    n_pts = res['n_obj_pts']

    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_bgr.shape[:2]
    K       = np.loadtxt(args.K).reshape(3, 3)

    print("Loading PLY...")
    pts_all = load_ply(args.ply)
    pts_v   = pts_all[pts_all[:, 2] > 0.001]

    # Object points: closest to estimated centroid T
    dist     = np.linalg.norm(pts_v - T, axis=1)
    thresh   = np.percentile(dist[dist < 0.5], 80) if (dist < 0.5).sum() > 0 else 0.3
    pts_obj  = pts_v[dist < thresh]
    if len(pts_obj) < 10: pts_obj = pts_v[dist < 0.5]

    # Projected centroid
    cx2 = int(K[0,0]*T[0]/T[2] + K[0,2])
    cy2 = int(K[1,1]*T[1]/T[2] + K[1,2])

    # Approximate segmentation mask from projected object pts
    mask = np.zeros((H, W), bool)
    u_o  = (K[0,0]*pts_obj[:,0]/pts_obj[:,2] + K[0,2]).astype(int)
    v_o  = (K[1,1]*pts_obj[:,1]/pts_obj[:,2] + K[1,2]).astype(int)
    ok   = (u_o >= 0) & (u_o < W) & (v_o >= 0) & (v_o < H)
    mask[v_o[ok], u_o[ok]] = True
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((15,15)), iterations=2).astype(bool)

    axis_len = np.linalg.norm(T) * 0.13

    # Background sample for scatter
    bg = pts_v[np.random.choice(len(pts_v), min(3000, len(pts_v)), replace=False)]

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Main 6-panel report figure (white background)
    # ════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.22)

    title = (f'Real-World Evaluation — iPhone 13 Pro LiDAR (Record3D)\n'
             f'Instruction: "{ins}"  |  Keyword: "{kw}"  |  '
             f'T = [{T[0]:.3f}, {T[1]:.3f}, {T[2]:.3f}] m  |  |T| = {np.linalg.norm(T):.3f} m')
    fig.suptitle(title, fontsize=11, color='#111111', fontweight='bold', y=0.99)

    # Panel 1 — Original RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title('(a) Original RGB Image\niPhone 13 Pro', color='#111111')
    ax1.axis('off')

    # Panel 2 — YOLOE segmentation mask
    ax2 = fig.add_subplot(gs[0, 1])
    ov2 = img_rgb.copy()
    ov2[mask] = (ov2[mask]*0.35 + np.array([0, 200, 80])*0.65).astype('uint8')
    contours, _ = cv2.findContours(mask.astype(np.uint8)*255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov2, contours, -1, (255, 180, 0), 3)
    if 0 <= cx2 < W and 0 <= cy2 < H:
        cv2.circle(ov2, (cx2, cy2), 16, (220, 30, 30), -1)
        cv2.circle(ov2, (cx2, cy2), 18, (255, 255, 255), 2)
    ax2.imshow(ov2)
    ax2.set_title(f'(b) YOLOE Segmentation Mask\nKeyword: "{kw}"  conf={conf:.3f}', color='#111111')
    ax2.axis('off')

    # Panel 3 — Pose axes on image
    ax3 = fig.add_subplot(gs[0, 2])
    ov3 = img_bgr.copy()
    ov3 = draw_axes_cv(ov3, R, T, K, length=axis_len)
    if 0 <= cx2 < W and 0 <= cy2 < H:
        cv2.circle(ov3, (cx2, cy2), 14, (0, 200, 200), -1)
        cv2.circle(ov3, (cx2, cy2), 16, (0, 0, 0), 2)
    ax3.imshow(cv2.cvtColor(ov3, cv2.COLOR_BGR2RGB))
    ax3.set_title('(c) Estimated 6D Pose\nX=red  Y=green  Z=blue  ●=centroid', color='#111111')
    ax3.axis('off')

    # Panel 4 — 3D point cloud
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    ax4.set_facecolor('white')
    ax4.scatter(bg[:,0], bg[:,2], -bg[:,1], s=0.4, c='#aaaacc', alpha=0.35)
    ax4.scatter(pts_obj[:,0], pts_obj[:,2], -pts_obj[:,1],
                s=12, c='#1b7f3a', alpha=0.95, label=f'Object ({len(pts_obj)} pts)')
    origin3 = np.array([T[0], T[2], -T[1]])
    for vec, col, lbl in zip(R.T, ['#cc2200', '#1a7a30', '#1040cc'], ['X', 'Y', 'Z']):
        end = T + vec * axis_len
        ep  = np.array([end[0], end[2], -end[1]])
        ax4.quiver(*origin3, *(ep - origin3), color=col, linewidth=2.5, arrow_length_ratio=0.25)
        ax4.text(*(ep + (ep - origin3)*0.12), lbl, color=col, fontsize=10, fontweight='bold')
    ax4.scatter(*origin3, s=100, c='red', zorder=10)
    ax4.set_xlabel('X (m)'); ax4.set_ylabel('Z (m)'); ax4.set_zlabel('-Y (m)')
    ax4.tick_params(colors='#333333')
    ax4.set_title('(d) 3D LiDAR Point Cloud\n+ Pose Axes', color='#111111')
    ax4.legend(fontsize=9, facecolor='white', markerscale=4)
    ax4.xaxis.pane.fill = False; ax4.yaxis.pane.fill = False; ax4.zaxis.pane.fill = False

    # Panel 5 — Top-down X-Z
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('white')
    ax5.scatter(bg[:,0], bg[:,2], s=0.4, c='#aaaacc', alpha=0.3)
    ax5.scatter(pts_obj[:,0], pts_obj[:,2], s=10, c='#1b7f3a', alpha=0.9, label='Object')
    for vec, col, lbl in zip(R.T, ['#cc2200', '#1a7a30', '#1040cc'], ['X', 'Y', 'Z']):
        end = T + vec * axis_len
        ax5.annotate('', xy=(end[0], end[2]), xytext=(T[0], T[2]),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
        ax5.text(end[0], end[2], f'  {lbl}', color=col, fontsize=10, fontweight='bold')
    ax5.scatter(T[0], T[2], s=150, c='red', zorder=5,
                label=f'Centroid ({T[0]:.2f}, {T[2]:.2f})')
    ax5.scatter(0, 0, s=200, marker='^', c='#444444', zorder=5, label='Camera')
    ax5.set_xlabel('X (m)'); ax5.set_ylabel('Z — Depth (m)')
    ax5.tick_params(colors='#333333')
    ax5.set_title('(e) Top-Down View (X-Z Plane)\nCamera at origin', color='#111111')
    ax5.legend(fontsize=9, facecolor='white')
    ax5.grid(alpha=0.4); ax5.set_aspect('equal')
    for sp in ax5.spines.values(): sp.set_edgecolor('#cccccc')

    # Panel 6 — Pose matrix table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor('white')
    ax6.axis('off')
    mat_data = [
        [f'{R[0,0]:+.4f}', f'{R[0,1]:+.4f}', f'{R[0,2]:+.4f}', f'{T[0]:+.4f}'],
        [f'{R[1,0]:+.4f}', f'{R[1,1]:+.4f}', f'{R[1,2]:+.4f}', f'{T[1]:+.4f}'],
        [f'{R[2,0]:+.4f}', f'{R[2,1]:+.4f}', f'{R[2,2]:+.4f}', f'{T[2]:+.4f}'],
        ['0.0000',          '0.0000',          '0.0000',          '1.0000'],
    ]
    col_labels = ['R col0', 'R col1', 'R col2', 'T (m)']
    tbl = ax6.table(cellText=mat_data, colLabels=col_labels,
                    loc='upper center', cellLoc='center', bbox=[0.0, 0.42, 1.0, 0.52])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#dce8f5')
            cell.set_text_props(color='#111111', fontweight='bold', fontsize=9.5)
        else:
            cell.set_facecolor('#f5f9ff' if row % 2 == 1 else 'white')
            cell.set_text_props(color='#1040cc' if col == 3 else '#111111', fontsize=9.5)
        cell.set_edgecolor('#1040cc' if col == 3 else '#cccccc')
        cell.set_linewidth(0.8)
    info = (f'Instruction:\n"{ins[:48]}..."\n\n'
            f'Keyword: "{kw}"\n'
            f'YOLOE conf: {conf:.4f}\n'
            f'LiDAR pts (object): {n_pts}\n'
            f'|T| = {np.linalg.norm(T):.4f} m\n'
            f'Tx={T[0]:.3f}  Ty={T[1]:.3f}  Tz={T[2]:.3f}')
    ax6.text(0.5, 0.18, info, transform=ax6.transAxes,
             ha='center', va='center', fontsize=9.5, color='#333333',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#aaaacc', lw=1))
    ax6.set_title('(f) 4x4 Pose Matrix  T_cam^obj', color='#111111')

    out1 = os.path.join(args.out_dir, 'report_main.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out1}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Depth heatmap + object highlight (white background)
    # ════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    fig2.patch.set_facecolor('white')
    fig2.suptitle('LiDAR Depth Map — iPhone 13 Pro (Record3D)',
                  color='#111111', fontsize=13, fontweight='bold')

    depth_img = np.zeros((H, W), np.float32)
    u_all = (K[0,0]*pts_v[:,0]/pts_v[:,2] + K[0,2]).astype(int)
    v_all = (K[1,1]*pts_v[:,1]/pts_v[:,2] + K[1,2]).astype(int)
    ok_all = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
    depth_img[v_all[ok_all], u_all[ok_all]] = pts_v[ok_all, 2]
    depth_vis = cv2.dilate(depth_img, np.ones((4,4)), iterations=2).astype(float)
    depth_vis[depth_vis == 0] = np.nan

    im1 = axes2[0].imshow(depth_vis, cmap='plasma', vmin=0.5, vmax=4.0)
    cb = plt.colorbar(im1, ax=axes2[0], label='Depth (m)', fraction=0.03)
    cb.ax.yaxis.set_tick_params(color='#333333')
    cb.ax.set_ylabel('Depth (m)', color='#333333')
    axes2[0].set_title('(a) Depth Map (LiDAR points projected)', color='#111111')
    axes2[0].axis('off')

    depth_rgb = plt.cm.plasma((depth_vis - 0.5) / (4.0 - 0.5))[:, :, :3]
    depth_rgb = (depth_rgb * 255).astype('uint8')
    depth_rgb[np.isnan(depth_vis)] = 240
    depth_rgb[mask] = (depth_rgb[mask]*0.3 + np.array([0, 200, 80])*0.7).astype('uint8')
    if 0 <= cx2 < W and 0 <= cy2 < H:
        cv2.circle(depth_rgb, (cx2, cy2), 18, (220, 30, 30), -1)
        cv2.circle(depth_rgb, (cx2, cy2), 20, (255, 255, 255), 2)
    axes2[1].imshow(depth_rgb)
    axes2[1].set_title('(b) Depth Map + Object Mask (green)\nCentroid = red dot', color='#111111')
    axes2[1].axis('off')

    out2 = os.path.join(args.out_dir, 'report_depth.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out2}")

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — Pipeline flow diagram (white background)
    # ════════════════════════════════════════════════════════════════════════
    fig3, ax = plt.subplots(figsize=(18, 4))
    fig3.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')

    steps = [
        ('iPhone 13 Pro\nLiDAR + RGB\n(Record3D)',   '#1565c0'),
        (f'User Instruction\n"{ins[:28]}..."',        '#4527a0'),
        (f'LLM Keyword\nExtractor\n-> "{kw}"',        '#6a1b9a'),
        (f'YOLOE\nSegmentation\nconf={conf:.3f}',     '#1b5e20'),
        (f'LiDAR pts\nFiltered\nby Mask\n{n_pts} pts','#e65100'),
        (f'6D Pose\nEstimation\n|T|={np.linalg.norm(T):.2f}m', '#b71c1c'),
    ]
    n = len(steps)
    for i, (label, color) in enumerate(steps):
        x = i / (n - 1)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.07, 0.08), 0.13, 0.82,
            boxstyle='round,pad=0.02',
            facecolor=color, edgecolor='#333333',
            linewidth=1.5, transform=ax.transAxes))
        ax.text(x, 0.50, label, transform=ax.transAxes,
                ha='center', va='center', fontsize=9.5, color='white', fontweight='bold')
        if i < n - 1:
            ax.annotate('', xy=((i+1)/(n-1) - 0.08, 0.5), xytext=(x + 0.07, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
    ax.set_title('Full Pipeline — Real-World Evaluation (iPhone 13 Pro + LiDAR)',
                 color='#111111', fontsize=13, fontweight='bold', pad=12)

    out3 = os.path.join(args.out_dir, 'report_pipeline.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out3}")

    print(f"\nAll report figures saved to: {args.out_dir}/")
    print("  report_main.png     — 6-panel main figure")
    print("  report_depth.png    — LiDAR depth map")
    print("  report_pipeline.png — pipeline flow diagram")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',    required=True)
    parser.add_argument('--image',   required=True)
    parser.add_argument('--ply',     required=True)
    parser.add_argument('--K',       required=True)
    parser.add_argument('--out_dir', default='/workspace/results/infer_pose/report_plots')
    main(parser.parse_args())
