import json, os, cv2, numpy as np, ast, re, requests, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE

YCBV    = '/workspace/dataset/ycbv/test'
MODELS  = '/workspace/dataset/ycbv/models'
ANCHOR  = '/workspace/anchor_results/dexycb_reference_view_ours'
RESULTS = '/workspace/results/ycbv_pipeline'
os.makedirs(RESULTS, exist_ok=True)

OUR_IDS = {2, 3, 4, 5, 9, 13, 14}
PROMPTS = {
    5:  'I want to grab the yellow bottle',
    13: 'can you hand me the blue pitcher',
    4:  'give me the soup can',
    3:  'I need the sugar box',
    2:  'hand me the cracker box',
    14: 'get the bleach bottle',
    9:  'pick up the meat can',
}
NAMES = {
    2: '003_cracker_box',     3: '004_sugar_box',
    4: '005_tomato_soup_can', 5: '006_mustard_bottle',
    9: '010_potted_meat_can', 13: '019_pitcher_base',
    14: '021_bleach_cleanser',
}

# ── Step 1: LLaMA extracts prompts ───────────────────────────────
def extract_prompts(instruction):
    r = requests.post('http://172.18.0.1:11434/api/generate',
        json={'model': 'pose-extractor', 'prompt': instruction, 'stream': False},
        timeout=30)
    raw = r.json()['response'].strip()
    try:
        p = ast.literal_eval(raw)
        if isinstance(p, list): return p
    except: pass
    return re.findall(r'"([^"]+)"', raw) or [raw]

# ── Step 2: YOLOE detects and generates mask ─────────────────────
_model = None
def detect(image_bgr, prompts):
    global _model
    if _model is None:
        orig = os.getcwd()
        os.chdir('/workspace/yoloe')
        _model = YOLOE('/workspace/yoloe/yoloe-26l-seg.pt')
        os.chdir(orig)
    H, W = image_bgr.shape[:2]
    best = None
    for prompt in prompts:
        _model.set_classes([prompt], _model.get_text_pe([prompt]))
        for thr in [0.1, 0.05, 0.03]:
            res = _model.predict(image_bgr, conf=thr, verbose=False)
            if not len(res[0].boxes): continue
            score = res[0].boxes.conf[0].item()
            if best and score <= best['score']: continue
            mask = cv2.resize(res[0].masks.data[0].cpu().numpy(), (W, H)) > 0.5
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            m = mask.astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
            m = cv2.erode(m, k, iterations=1)
            ov = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
            ov[m > 127] = ov[m > 127] * 0.4 + np.array([0, 200, 0]) * 0.6
            best = {'prompt': prompt, 'score': score,
                    'mask': m > 127, 'overlay': ov,
                    'mask_px': int((m > 127).sum()),
                    'bbox': res[0].boxes.xyxy[0].cpu().numpy().tolist()}
            break
    return best

# ── Step 3: Any6D estimates pose ─────────────────────────────────
def estimate_pose(img_rgb, depth, mask, K, obj_name):
    import sys, os
    sys.path.insert(0, '/workspace')
    sys.path.insert(0, '/workspace/foundationpose')
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    import trimesh
    from Any6D import Any6D

    mesh_path = f'{ANCHOR}/{obj_name}/center_mesh_{obj_name}.obj'
    if not os.path.exists(mesh_path):
        return None

    mesh = trimesh.load(mesh_path)
    est  = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=None, debug=0)
    pred_pose = est.register_any6d(
        K=K, rgb=img_rgb, depth=depth,
        ob_mask=mask, iteration=5, name=obj_name)
    return pred_pose

# ── Step 4: Metrics ───────────────────────────────────────────────
def rot_err(R_p, R_g):
    R_e = R_p.T @ R_g
    return float(np.degrees(np.arccos(np.clip((np.trace(R_e) - 1) / 2, -1, 1))))

def add_metric(T_pred, T_gt, obj_id):
    import trimesh
    mesh_path = f'{MODELS}/obj_{obj_id:06d}.ply'
    if not os.path.exists(mesh_path): return None, None
    mesh = trimesh.load(mesh_path)
    pts  = np.array(mesh.vertices)
    pp   = (T_pred[:3, :3] @ pts.T).T + T_pred[:3, 3]
    pg   = (T_gt[:3, :3]   @ pts.T).T + T_gt[:3, 3]
    add  = np.linalg.norm(pp - pg, axis=1).mean()
    diam = np.linalg.norm(pts.max(0) - pts.min(0))
    return add, diam

# ── IoU mask metric ───────────────────────────────────────────────
def mask_iou(mask_pred, mask_gt):
    inter = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return float(inter / union) if union > 0 else 0.0

# ── Load eval set ─────────────────────────────────────────────────
selected = json.load(open('/workspace/results/ycbv_eval_set.json'))

print('='*100)
print('FULL PIPELINE EVALUATION: LLaMA -> YOLOE -> Any6D on YCB-V BOP')
print('='*100)
print(f'{"Frame":18} {"Object":22} {"Prompt->Det":28} {"Score":6} {"Mask px":8} {"IoU":5}')
print('-'*100)

all_logs = []
for s in selected:
    img_bgr = cv2.imread(s['img_path'])
    if img_bgr is None: continue

    dep = cv2.imread(s['dep_path'], cv2.IMREAD_UNCHANGED)
    dep_m = dep.astype(np.float32) * s['depth_scale'] / 1000.0 if dep is not None else None

    K_flat = s['K_flat']
    K = np.array(K_flat).reshape(3, 3)

    # GT pose
    T_gt = np.eye(4)
    T_gt[:3, :3] = np.array(s['gt_R']).reshape(3, 3)
    T_gt[:3,  3] = np.array(s['gt_t'])

    # GT mask (mask_visib)
    frame_int = int(s['frame'])
    mask_gt_path = f"{YCBV}/{s['scene']}/mask_visib/{frame_int:06d}_000000.png"
    mask_gt = None
    if os.path.exists(mask_gt_path):
        mg = cv2.imread(mask_gt_path, cv2.IMREAD_GRAYSCALE)
        mask_gt = mg > 127 if mg is not None else None

    # Step 1 LLaMA
    prompts = extract_prompts(s['prompt'])

    # Step 2 YOLOE
    det = detect(img_bgr, prompts)

    # IoU with GT mask
    iou = 0.0
    if det is not None and mask_gt is not None:
        iou = mask_iou(det['mask'], mask_gt)

    tag     = f"{s['scene']}/{s['frame']:>6}"
    det_str = f"{det['prompt'][:15]}" if det else 'NO DET'
    sc_str  = f"{det['score']:.2f}" if det else '-'
    px_str  = str(det['mask_px']) if det else '-'
    iou_str = f'{iou:.3f}' if det else '-'

    print(f"{tag:18} {s['obj_name'][:22]:22} {det_str:28} {sc_str:6} {px_str:8} {iou_str:5}")

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"[{s['scene']}] {s['prompt']}", fontsize=10)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    axes[0].imshow(img_rgb)
    axes[0].set_title('RGB input'); axes[0].axis('off')

    if dep_m is not None:
        dep_m[dep_m < 0.001] = np.nan
        im = axes[1].imshow(dep_m, cmap='jet',
                             vmin=np.nanmin(dep_m), vmax=np.nanmax(dep_m))
        plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title('Depth (m)'); axes[1].axis('off')

    if det:
        axes[2].imshow(det['overlay'])
        axes[2].set_title(f"YOLOE: {det['prompt']} (s={det['score']:.2f}, IoU={iou:.3f})")
    else:
        axes[2].imshow(img_rgb)
        axes[2].set_title('No detection')
    axes[2].axis('off')

    plt.tight_layout()
    fname = f"{s['scene']}_{s['frame']}_{s['obj_id']}.png"
    plt.savefig(f'{RESULTS}/{fname}', dpi=80)
    plt.close()

    all_logs.append({
        'scene':        s['scene'],
        'frame':        s['frame'],
        'obj_id':       s['obj_id'],
        'obj_name':     s['obj_name'],
        'instruction':  s['prompt'],
        'prompts':      prompts,
        'detected':     det is not None,
        'det_prompt':   det['prompt'] if det else None,
        'det_score':    round(det['score'], 3) if det else 0,
        'mask_px':      det['mask_px'] if det else 0,
        'mask_iou':     round(iou, 3),
        'K':            K_flat,
        'gt_R':         s['gt_R'],
        'gt_t':         s['gt_t'],
    })

# ── Summary ──────────────────────────────────────────────────────
print('='*100)
det_count  = sum(1 for r in all_logs if r['detected'])
mean_iou   = np.mean([r['mask_iou'] for r in all_logs if r['detected']]) if det_count > 0 else 0
mean_score = np.mean([r['det_score'] for r in all_logs if r['detected']]) if det_count > 0 else 0

print(f'Detection rate : {det_count}/{len(all_logs)}')
print(f'Mean IoU       : {mean_iou:.3f}')
print(f'Mean YOLOE conf: {mean_score:.3f}')

summary = {
    'total_frames':    len(all_logs),
    'detected':        det_count,
    'detection_rate':  f'{det_count}/{len(all_logs)}',
    'mean_mask_iou':   round(float(mean_iou), 3),
    'mean_yoloe_score': round(float(mean_score), 3),
    'per_frame':       all_logs,
}

with open(f'{RESULTS}/eval_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Saved: {RESULTS}/eval_results.json')
print(f'Viz  : {RESULTS}/*.png')
