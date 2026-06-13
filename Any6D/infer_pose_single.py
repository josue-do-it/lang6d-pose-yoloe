"""
Single-image pose estimation: RGB + PLY (point cloud) + K + text instruction
Pipeline: instruction → LLM keyword → YOLOE mask → project PLY → T (centroid) + R (PCA)
Output: pose 4x4, visualization, JSON

Usage:
  python infer_pose_single.py \
      --image  /path/to/0000001.jpg \
      --ply    /path/to/0000001.ply \
      --K      /path/to/K.txt \
      --instruction "pick me that banana" \
      --out_dir /path/to/output
"""
import argparse, os, sys, re, json
import numpy as np
import cv2
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE

OLLAMA_URL = "http://172.18.0.1:11434/api/generate"
LLM_MODEL  = "mistral:latest"

CALIBRATED_SYSTEM = """\
You are a visual keyword extractor for YOLOE, an open-vocabulary object segmentation model.

Your ONLY job: extract the most precise visual keyword(s) from the user instruction.
Output ONLY the keyword — nothing else. No punctuation, no explanation, no sentence.

IMPORTANT:
- 1 word is BEST when it is specific enough (e.g. "pitcher", "mustard", "banana")
- Use 2-3 words ONLY when 1 word is too ambiguous (e.g. "spam can", "blue pitcher")
- MAXIMUM 3 words
- Always use visual, concrete nouns — NOT functions, NOT actions

Examples (instruction → keyword):
"I'm thirsty, pass me that big blue pitcher" → pitcher
"Hand me the blue jug with the handle" → blue pitcher
"Pass me the flat SPAM tin on the table" → spam can
"Give me the yellow mustard bottle on the tray" → mustard
"Hand me that red power drill on the table" → drill
"Give me that banana please" → banana
"I want the red and white tin can on the left" → soup can
"Pass me the tall white cleaning bottle" → white bottle
"That yellow cardboard box on the desk" → yellow box
"Hand me the screwdriver" → screwdriver
"""

# ── LLM ───────────────────────────────────────────────────────────────────────
def call_llm(instruction: str) -> str:
    try:
        r = requests.post(OLLAMA_URL,
                          json={"model": LLM_MODEL, "stream": False,
                                "system": CALIBRATED_SYSTEM, "prompt": instruction},
                          timeout=30)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        return ""

def parse_llm(raw: str) -> str:
    raw = raw.strip()
    m = re.search(r'["""]([^"""]{1,40})["""]', raw)
    if m: return m.group(1).strip().lower()
    m = re.search(r'→\s*(.+)$', raw)
    if m: return m.group(1).strip().lower()[:40]
    raw = re.sub(r'^(keyword[:\s]+|output[:\s]+|the (object|word) (is|:)\s*)', '', raw, flags=re.I)
    for line in raw.split('\n'):
        line = line.strip().lstrip('-→•*:').strip()
        line = re.sub(r'[^\w\s\'-]', '', line).strip()
        if 0 < len(line.split()) <= 4:
            return line.lower()
    return raw.split('\n')[0][:40].strip().lower()

# ── YOLOE ─────────────────────────────────────────────────────────────────────
_yoloe = None
def get_yoloe():
    global _yoloe
    if _yoloe is None:
        orig = os.getcwd()
        os.chdir('/workspace/yoloe')
        _yoloe = YOLOE("yoloe-26l-seg.pt")
        os.chdir(orig)
    return _yoloe

def yoloe_detect(img_bgr, prompt, H, W):
    model = get_yoloe()
    model.set_classes([prompt], model.get_text_pe([prompt]))
    for conf_th in [0.1, 0.05, 0.03]:
        results = model.predict(img_bgr, conf=conf_th, verbose=False)
        if len(results[0].boxes) > 0 and results[0].masks is not None:
            confs = results[0].boxes.conf.cpu().numpy()
            idx   = int(np.argmax(confs))
            conf  = float(confs[idx])
            m     = results[0].masks.data[idx].cpu().numpy()
            mask  = cv2.resize(m, (W, H)) > 0.5
            return mask, conf
    return None, 0.0

# ── PLY loader ────────────────────────────────────────────────────────────────
def load_ply_vertices(ply_path):
    """Load xyz vertices from binary PLY. Returns (N,3) float32."""
    import struct
    with open(ply_path, 'rb') as f:
        # parse header
        n_verts = 0
        header_end = 0
        lines = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            lines.append(line)
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            if line == 'end_header':
                header_end = f.tell()
                break

        # detect format
        is_binary_le = any('binary_little_endian' in l for l in lines)
        is_binary_be = any('binary_big_endian' in l for l in lines)
        is_ascii = any(l == 'format ascii 1.0' for l in lines)

        # count properties per vertex
        props = []
        in_vertex = False
        for l in lines:
            if l.startswith('element vertex'):
                in_vertex = True
            elif l.startswith('element') and not l.startswith('element vertex'):
                in_vertex = False
            elif in_vertex and l.startswith('property float'):
                props.append(l.split()[-1])

        n_props = len(props)
        xyz_idx = [props.index(p) for p in ['x','y','z'] if p in props]

        f.seek(header_end)
        if is_ascii:
            pts = []
            for _ in range(n_verts):
                vals = list(map(float, f.readline().split()))
                pts.append([vals[i] for i in xyz_idx])
            return np.array(pts, dtype=np.float32)
        else:
            fmt = '<' if is_binary_le else '>'
            fmt += 'f' * n_props
            sz = struct.calcsize(fmt)
            pts = np.zeros((n_verts, 3), dtype=np.float32)
            for i in range(n_verts):
                vals = struct.unpack(fmt, f.read(sz))
                pts[i] = [vals[j] for j in xyz_idx]
            return pts

# ── Pose from point cloud ─────────────────────────────────────────────────────
def pose_from_pointcloud(pts_3d):
    """
    Compute 4x4 pose from a set of 3D points.
    T = centroid
    R = PCA eigenvectors (principal axes of the object)
    """
    if len(pts_3d) < 10:
        return np.eye(4), np.zeros(3)

    centroid = pts_3d.mean(axis=0)
    centered = pts_3d - centroid
    cov = centered.T @ centered / len(pts_3d)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # ensure right-handed coordinate system
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1
    R = eigvecs.T  # rotation: object axes → camera axes
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3]  = centroid
    return pose, centroid

def project_pts(pts_3d, K):
    """Project (N,3) 3D points to (N,2) 2D using K.
    Handles both z>0 (OpenCV) and z<0 (OpenGL/some sensors) conventions."""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    z = pts_3d[:, 2]
    # auto-detect convention: if most z are negative, flip
    if np.median(z) < 0:
        z = -z
        pts_3d = pts_3d.copy(); pts_3d[:,2] = z
    valid = z > 0.001
    u = np.where(valid, fx * pts_3d[:,0] / z + cx, -1)
    v = np.where(valid, fy * pts_3d[:,1] / z + cy, -1)
    return np.stack([u, v], axis=1), valid

# ── Visualization ─────────────────────────────────────────────────────────────
def draw_axes(img, pose, K, length=0.05):
    """Draw 3D coordinate axes on image."""
    R, t = pose[:3,:3], pose[:3,3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3,1)
    pts3d = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    try:
        pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, np.zeros(5))
        pts2d = pts2d.astype(int).reshape(-1,2)
        o = tuple(pts2d[0])
        cv2.arrowedLine(img, o, tuple(pts2d[1]), (255,50,50),  3, tipLength=0.2)
        cv2.arrowedLine(img, o, tuple(pts2d[2]), (50,200,50),  3, tipLength=0.2)
        cv2.arrowedLine(img, o, tuple(pts2d[3]), (50,100,255), 3, tipLength=0.2)
        cv2.putText(img, 'X', tuple(pts2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,50,50),  2)
        cv2.putText(img, 'Y', tuple(pts2d[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,200,50),  2)
        cv2.putText(img, 'Z', tuple(pts2d[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,100,255), 2)
    except Exception:
        pass
    return img

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',       required=True,  help='RGB image path (.jpg/.png)')
    parser.add_argument('--ply',         required=True,  help='Point cloud path (.ply)')
    parser.add_argument('--K',           required=True,  help='Camera intrinsics txt (3x3)')
    parser.add_argument('--instruction', required=True,  help='Natural language instruction')
    parser.add_argument('--prompt',      default=None,   help='Skip LLM, use this YOLOE prompt directly')
    parser.add_argument('--out_dir',     default='./results/infer_pose')
    parser.add_argument('--llm_model',   default='mistral:latest')
    parser.add_argument('--min_pts',     type=int, default=50, help='Min 3D points required for pose')
    args = parser.parse_args()

    LLM_MODEL = args.llm_model
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"ERROR: cannot read image {args.image}"); sys.exit(1)
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    K = np.loadtxt(args.K).reshape(3,3)
    print(f"K loaded: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    print(f"Loading PLY: {args.ply}")
    pts_all = load_ply_vertices(args.ply)
    print(f"  {len(pts_all)} vertices loaded")

    # ── LLM extraction ────────────────────────────────────────────────────────
    if args.prompt:
        keyword = args.prompt
        print(f"Using direct prompt: \"{keyword}\"")
    else:
        print(f"LLM: \"{args.instruction}\"")
        raw = call_llm(args.instruction)
        keyword = parse_llm(raw)
        if not keyword:
            keyword = "object"
        print(f"LLM → keyword: \"{keyword}\"")

    # ── YOLOE detection ───────────────────────────────────────────────────────
    print(f"YOLOE detecting: \"{keyword}\"...")
    mask, conf = yoloe_detect(img_bgr, keyword, H, W)
    if mask is None:
        print("YOLOE: no detection — trying fallback keywords...")
        # try splitting + common synonyms
        fallbacks = keyword.split() + [
            keyword.replace('_',' '),
            keyword.split()[0] if keyword.split() else keyword,
            "eraser", "whiteboard eraser", "board eraser", "felt eraser",
            "sponge", "cleaner", "rubber",
        ]
        for kw in fallbacks:
            if kw == keyword: continue
            mask, conf = yoloe_detect(img_bgr, kw, H, W)
            if mask is not None:
                keyword = kw
                print(f"  fallback keyword \"{kw}\" worked (conf={conf:.3f})")
                break
    if mask is None:
        print("WARNING: YOLOE found nothing. Using full image center region.")
        mask = np.zeros((H,W), bool)
        mask[H//4:3*H//4, W//4:3*W//4] = True
        conf = 0.0
    else:
        print(f"  detected (conf={conf:.3f}, mask={mask.sum()} px)")

    # ── Project PLY → 2D, filter by mask ─────────────────────────────────────
    # handle z sign convention (flip if mostly negative)
    if np.median(pts_all[:,2]) < 0:
        pts_all = pts_all.copy(); pts_all[:,2] = -pts_all[:,2]
        print("  Note: z-flip applied (sensor uses negative-z convention)")

    valid_z = pts_all[:,2] > 0.001
    pts_valid = pts_all[valid_z]

    pts2d, valid_proj = project_pts(pts_valid, K)

    # keep points projected inside image and inside mask
    u = pts2d[:,0].astype(int)
    v = pts2d[:,1].astype(int)
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_proj
    u_clip = np.clip(u, 0, W-1)
    v_clip = np.clip(v, 0, H-1)
    in_mask = mask[v_clip, u_clip] & in_image

    pts_obj = pts_valid[in_mask]
    print(f"Object 3D points: {len(pts_obj)} (from {len(pts_valid)} valid PLY pts)")

    if len(pts_obj) < args.min_pts:
        print(f"WARNING: only {len(pts_obj)} 3D points in mask (min={args.min_pts})")
        print("  Check: does the PLY coordinate system match the image K?")
        print("  Falling back to all valid PLY points centroid.")
        pts_obj = pts_valid

    # Statistical outlier removal: keep points within 2.5σ of centroid per axis
    if len(pts_obj) >= 10:
        mu  = pts_obj.mean(axis=0)
        sig = pts_obj.std(axis=0)
        sig = np.where(sig < 1e-4, 1e-4, sig)
        inliers = np.all(np.abs(pts_obj - mu) <= 2.5 * sig, axis=1)
        if inliers.sum() >= args.min_pts:
            pts_obj = pts_obj[inliers]
            print(f"After outlier removal: {len(pts_obj)} pts")

    # ── Compute pose ──────────────────────────────────────────────────────────
    pose4x4, centroid = pose_from_pointcloud(pts_obj)
    R = pose4x4[:3,:3]
    T = pose4x4[:3, 3]

    print(f"\nPose estimated:")
    print(f"  T (centroid) = [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}]")
    print(f"  Distance from camera = {np.linalg.norm(T):.3f} m")
    print(f"  R =\n{R}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(args.image))[0]

    # pose txt
    np.savetxt(f"{args.out_dir}/{base}_pose.txt", pose4x4)

    # compute 3D bbox in object-local frame from actual masked pts_obj
    pts_local = (pts_obj - T) @ R
    bbox_min  = pts_local.min(axis=0).tolist()
    bbox_max  = pts_local.max(axis=0).tolist()

    # pose JSON
    result = {
        'image':       args.image,
        'ply':         args.ply,
        'instruction': args.instruction,
        'keyword':     keyword,
        'yoloe_conf':  round(conf, 3),
        'n_obj_pts':   int(len(pts_obj)),
        'R':           R.tolist(),
        'T':           T.tolist(),
        'T_norm_m':    round(float(np.linalg.norm(T)), 4),
        'pose_4x4':    pose4x4.tolist(),
        'bbox3d_min':  bbox_min,
        'bbox3d_max':  bbox_max,
    }
    with open(f"{args.out_dir}/{base}_pose.json", 'w') as f:
        json.dump(result, f, indent=2)

    # ── Rich visualization: 6 panels ─────────────────────────────────────────
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f'Instruction: "{args.instruction}"\n'
        f'Keyword: "{keyword}"  |  T=[{T[0]:.3f}, {T[1]:.3f}, {T[2]:.3f}]  '
        f'dist={np.linalg.norm(T):.3f}m  |  conf={conf:.3f}  |  pts={len(pts_obj)}',
        fontsize=11, fontweight='bold')

    # ── Panel 1: Original image ───────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontweight='bold')
    ax1.axis('off')

    # ── Panel 2: YOLOE mask overlay ───────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    ov2 = img_rgb.copy()
    ov2[mask] = (ov2[mask]*0.35 + np.array([0, 220, 80])*0.65).astype('uint8')
    # draw mask contour
    mask_u8 = mask.astype(np.uint8)*255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov2, contours, -1, (255,255,0), 3)
    # draw centroid on 2D
    cx_2d = int(K[0,0]*T[0]/T[2] + K[0,2])
    cy_2d = int(K[1,1]*T[1]/T[2] + K[1,2])
    if 0 <= cx_2d < W and 0 <= cy_2d < H:
        cv2.circle(ov2, (cx_2d, cy_2d), 12, (255,50,50), -1)
        cv2.circle(ov2, (cx_2d, cy_2d), 14, (255,255,255), 2)
    ax2.imshow(ov2)
    ax2.set_title(f'YOLOE Mask (green)\nKeyword: "{keyword}"  conf={conf:.3f}', fontweight='bold')
    ax2.axis('off')

    # ── Panel 3: Pose axes projected on image ────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    ov3 = img_bgr.copy()
    # scale axis length relative to object distance
    axis_len = float(np.linalg.norm(T)) * 0.12
    ov3 = draw_axes(ov3, pose4x4, K, length=axis_len)
    # draw centroid dot
    if 0 <= cx_2d < W and 0 <= cy_2d < H:
        cv2.circle(ov3, (cx_2d, cy_2d), 10, (0, 255, 255), -1)
    ax3.imshow(cv2.cvtColor(ov3, cv2.COLOR_BGR2RGB))
    ax3.set_title('Pose Axes (X=red Y=green Z=blue)\nCentroid=cyan dot', fontweight='bold')
    ax3.axis('off')

    # ── Panel 4: 3D point cloud with pose axes ───────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    # background scene pts (subsample)
    if len(pts_valid) > 2000:
        idx_bg = np.random.choice(len(pts_valid), 2000, replace=False)
        pts_bg = pts_valid[idx_bg]
    else:
        pts_bg = pts_valid
    ax4.scatter(pts_bg[:,0], pts_bg[:,2], -pts_bg[:,1],
                s=0.5, c='lightgray', alpha=0.3, label='Scene')
    # object pts
    if len(pts_obj) > 0:
        ax4.scatter(pts_obj[:,0], pts_obj[:,2], -pts_obj[:,1],
                    s=8, c='limegreen', alpha=0.9, label=f'Object ({len(pts_obj)} pts)')
    # pose axes (scaled)
    origin = np.array([T[0], T[2], -T[1]])
    for vec, col, lbl in zip(R.T, ['red','green','blue'], ['X','Y','Z']):
        end = T + vec * axis_len
        end_plot = np.array([end[0], end[2], -end[1]])
        ax4.quiver(*origin, *(end_plot - origin), color=col, linewidth=2, arrow_length_ratio=0.2)
        ax4.text(*end_plot, lbl, color=col, fontsize=9, fontweight='bold')
    ax4.scatter(*origin, s=80, c='red', zorder=5)
    ax4.set_xlabel('X (m)'); ax4.set_ylabel('Z (m)'); ax4.set_zlabel('-Y (m)')
    ax4.set_title('3D Point Cloud\n+ Pose Axes', fontweight='bold')
    ax4.legend(fontsize=8, markerscale=4)

    # ── Panel 5: Top-down view (X-Z plane) ───────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    if len(pts_bg) > 0:
        ax5.scatter(pts_bg[:,0], pts_bg[:,2], s=0.5, c='lightgray', alpha=0.3)
    if len(pts_obj) > 0:
        ax5.scatter(pts_obj[:,0], pts_obj[:,2], s=6, c='limegreen', alpha=0.8, label='Object')
    # draw axes on top-down
    for vec, col, lbl in zip(R.T, ['red','green','blue'], ['X','Y','Z']):
        end = T + vec * axis_len
        ax5.annotate('', xy=(end[0], end[2]), xytext=(T[0], T[2]),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2))
        ax5.text(end[0], end[2], lbl, color=col, fontsize=9, fontweight='bold')
    ax5.scatter(T[0], T[2], s=100, c='red', zorder=5, label=f'Centroid ({T[0]:.2f},{T[2]:.2f})')
    ax5.scatter(0, 0, s=120, marker='^', c='black', zorder=5, label='Camera')
    ax5.set_xlabel('X (m)'); ax5.set_ylabel('Z — depth (m)')
    ax5.set_title('Top-down View (X-Z)\nCamera at origin', fontweight='bold')
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3); ax5.set_aspect('equal')

    # ── Panel 6: Pose info table ──────────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    rows = [
        ['Instruction', args.instruction[:45]],
        ['Keyword', f'"{keyword}"'],
        ['YOLOE conf', f'{conf:.4f}'],
        ['Object pts', f'{len(pts_obj)}'],
        ['Tx (m)', f'{T[0]:.4f}'],
        ['Ty (m)', f'{T[1]:.4f}'],
        ['Tz (m)', f'{T[2]:.4f}'],
        ['|T| (m)', f'{np.linalg.norm(T):.4f}'],
        ['R row 0', f'[{R[0,0]:.3f}, {R[0,1]:.3f}, {R[0,2]:.3f}]'],
        ['R row 1', f'[{R[1,0]:.3f}, {R[1,1]:.3f}, {R[1,2]:.3f}]'],
        ['R row 2', f'[{R[2,0]:.3f}, {R[2,1]:.3f}, {R[2,2]:.3f}]'],
    ]
    table = ax6.table(cellText=rows, colLabels=['Parameter', 'Value'],
                      cellLoc='left', loc='center', colWidths=[0.35, 0.65])
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#ecf0f1')
    ax6.set_title('Pose Summary', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_viz = f"{args.out_dir}/{base}_pose_viz.png"
    plt.savefig(out_viz, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\nSaved:")
    print(f"  {args.out_dir}/{base}_pose.txt   — 4x4 pose matrix")
    print(f"  {args.out_dir}/{base}_pose.json  — full results")
    print(f"  {out_viz}  — visualization")
