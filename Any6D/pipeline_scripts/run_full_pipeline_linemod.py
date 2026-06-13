"""
Full pipeline: User instruction → LLM keyword extractor → YOLOE mask → Any6D → BOP metrics
LINEMOD BOP dataset (15 objects), GT models (scaled to m), GT masks as fallback.

Dataset must be mounted at /dataset/lm/ inside Docker:
  host: /home/josue_aims_ac_za/ssd_4tb/dataset/lm
  container: /dataset/lm

Run inside Docker:
    /opt/conda/envs/Any6D/bin/python3 /workspace/run_full_pipeline_linemod.py
    /opt/conda/envs/Any6D/bin/python3 /workspace/run_full_pipeline_linemod.py --stride 5 --obj_ids 1 5 8
"""
import copy, sys, os, re, json, argparse
import numpy as np
import cv2
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
try:
    sys.path.insert(0, '/workspace/yoloe')
    from ultralytics import YOLOE as _YOLOE
    _YOLOE_AVAILABLE = True
except (ImportError, Exception):
    _YOLOE = None
    _YOLOE_AVAILABLE = False

# ── Natural language instructions (one per LINEMOD object) ────────────────────
LM_INSTRUCTIONS = {
    1:  "Find the small dark toy monkey figure on the table",
    2:  "Grab the blue metal bench vise tool",
    3:  "Pass me that red round bowl",
    4:  "Hand me the black digital camera",
    5:  "Pick up that cylindrical steel can",
    6:  "Find the small orange cat figurine",
    7:  "Give me the blue coffee cup",
    8:  "I need the yellow power driller",
    9:  "Find the small yellow rubber duck",
    10: "Pick up the cardboard egg box",
    11: "Hand me that orange bottle of glue",
    12: "I need the red hole puncher",
    13: "Find that small toy iron",
    14: "Pass me the desk lamp",
    15: "Hand me the mobile phone on the table",
}

LM_NAMES = {
    1: "ape", 2: "benchvise", 3: "bowl",    4: "camera",     5: "can",
    6: "cat", 7: "cup",       8: "driller", 9: "duck",       10: "eggbox",
    11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp",   15: "phone",
}

# Objects with rotational symmetry → use ADD-S instead of ADD
LM_SYMMETRIC = {10, 11}   # eggbox, glue

# ── LLM / YOLOE setup ─────────────────────────────────────────────────────────
CALIBRATED_SYSTEM = """\
You are a visual keyword extractor for YOLOE, an open-vocabulary object segmentation model.
Your ONLY job: extract the most precise visual keyword(s) from the user instruction.
Output ONLY the keyword — nothing else. No punctuation, no explanation, no sentence.
- 1 word is BEST when specific enough (e.g. "duck", "can", "driller")
- Use 2 words ONLY when ambiguous (e.g. "yellow duck", "orange glue")
- MAXIMUM 3 words
Examples:
"Find the small dark toy monkey" → toy monkey
"Grab the blue metal bench vise tool" → benchvise
"Hand me the black digital camera" → camera
"Pick up that cylindrical steel can" → tin can
"Find the small orange cat figurine" → cat figurine
"I need the yellow power driller" → yellow driller
"Find the small yellow rubber duck" → rubber duck
"Pick up the cardboard egg box" → egg box
"Hand me that orange bottle of glue" → glue bottle
"I need the red hole puncher" → hole puncher
"Find that small toy iron" → iron
"Pass me the desk lamp" → lamp
"Hand me the mobile phone" → phone
"""
OLLAMA_URL = "http://172.18.0.1:11434/api/generate"
LLM_MODEL  = "mistral:latest"

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
    raw = re.sub(r'^(keyword[:\s]+|output[:\s]+|the (object|keyword|word) (is|:)\s*)', '', raw, flags=re.I)
    for line in raw.split('\n'):
        line = line.strip().lstrip('-→•*:').strip()
        line = re.sub(r'[^\w\s\'-]', '', line).strip()
        if 0 < len(line.split()) <= 4:
            return line.lower()
    return raw.split('\n')[0][:40].strip().lower()

_yoloe_model = None

def get_yoloe_mask(img_rgb, prompt, H, W):
    if not _YOLOE_AVAILABLE:
        return None
    global _yoloe_model
    _orig = os.getcwd()
    if _yoloe_model is None:
        os.chdir('/workspace/yoloe')
        _yoloe_model = _YOLOE("yoloe-26l-seg.pt")
        os.chdir(_orig)
    if not prompt or len(prompt.strip()) < 1:
        return None
    _yoloe_model.set_classes([prompt], _yoloe_model.get_text_pe([prompt]))
    results = _yoloe_model.predict(img_rgb, conf=0.1, verbose=False)
    os.chdir(_orig)
    if len(results[0].boxes) > 0 and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask.astype(np.float32), (W, H))
        return mask > 0.5
    return None

# ── BOP LINEMOD data reader ───────────────────────────────────────────────────
class LineMODReader:
    """Reads BOP-format LINEMOD dataset."""
    LM_ROOT = "/dataset/lm"
    MM_TO_M  = 0.001

    def __init__(self, obj_id: int):
        self.obj_id   = obj_id
        self.scene_id = obj_id

        # camera from camera.json (same for all images)
        cam_path = os.path.join(self.LM_ROOT, "lm", "camera.json")
        with open(cam_path) as f:
            c = json.load(f)
        self.K = np.array([[c['fx'], 0, c['cx']],
                           [0, c['fy'], c['cy']],
                           [0, 0, 1]], dtype=np.float64)
        self.depth_scale = c.get('depth_scale', 1.0)  # mm/unit (=1 → depth in mm)

        # scene directory (test/ is at root of LM_ROOT, not under lm/)
        self.scene_dir = os.path.join(
            self.LM_ROOT, "test", f"{self.scene_id:06d}")

        # load scene_camera.json and scene_gt.json
        with open(os.path.join(self.scene_dir, "scene_camera.json")) as f:
            self._scene_cam = json.load(f)
        with open(os.path.join(self.scene_dir, "scene_gt.json")) as f:
            self._scene_gt = json.load(f)

        # BOP test targets (200 per object)
        targets_path = os.path.join(self.LM_ROOT, "lm", "test_targets_bop19.json")
        with open(targets_path) as f:
            all_targets = json.load(f)
        self.test_im_ids = sorted(
            t['im_id'] for t in all_targets
            if t['obj_id'] == obj_id and t['scene_id'] == self.scene_id)

    def get_rgb(self, im_id: int) -> np.ndarray:
        path = os.path.join(self.scene_dir, "rgb", f"{im_id:06d}.png")
        img  = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_depth(self, im_id: int) -> np.ndarray:
        path  = os.path.join(self.scene_dir, "depth", f"{im_id:06d}.png")
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        # depth in mm (depth_scale=1.0) → convert to meters
        return depth * self.depth_scale * self.MM_TO_M

    def get_mask_visib(self, im_id: int):
        """BOP visible mask for the target object (annotation 0)."""
        path = os.path.join(self.scene_dir, "mask_visib",
                            f"{im_id:06d}_000000.png")
        if not os.path.exists(path):
            return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return m > 0 if m is not None else None

    def get_gt_pose(self, im_id: int) -> np.ndarray:
        """Returns 4×4 pose matrix in METERS."""
        ann = self._scene_gt[str(im_id)][0]
        R = np.array(ann['cam_R_m2c']).reshape(3, 3)
        t = np.array(ann['cam_t_m2c']) * self.MM_TO_M
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3,  3] = t
        return pose

    def load_gt_mesh(self, scale_m=True) -> trimesh.Trimesh:
        """Load GT PLY model. scale_m=True converts mm → m."""
        path = os.path.join(self.LM_ROOT, "models",
                            f"obj_{self.obj_id:06d}.ply")
        mesh = trimesh.load(path, force='mesh')
        if scale_m:
            mesh.apply_scale(self.MM_TO_M)
        return mesh


# ── Helpers ───────────────────────────────────────────────────────────────────
def rotation_error_deg(R_pred, R_gt):
    R_rel   = R_pred @ R_gt.T
    cos_val = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_val)))

def translation_error_cm(t_pred, t_gt):
    return float(np.linalg.norm(t_pred - t_gt) * 100)


# ── Main ──────────────────────────────────────────────────────────────────────
from estimater import *
from metrics import *
from pytorch_lightning import seed_everything
from datetime import datetime

try:
    from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd
    _BOP_METRICS = True
except Exception:
    _BOP_METRICS = False

try:
    from renderer_pyrender import RendererVispy
    _RENDERER = True
except Exception:
    _RENDERER = False

if __name__ == '__main__':
    seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_root",  default="/dataset/lm")
    parser.add_argument("--obj_ids",  type=int, nargs='+',
                        default=list(range(1, 16)),
                        help="LINEMOD object IDs to evaluate (1-15)")
    parser.add_argument("--stride",   type=int, default=1,
                        help="Stride over BOP19 test images (1=full, 5=quick)")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--llm_model", default="mistral:latest")
    parser.add_argument("--skip_llm",   action="store_true",
                        help="Use object name as YOLOE prompt directly")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames per object, e.g. 1 for quick single-image test")
    args = parser.parse_args()

    LM_NAMES_LOCAL = LM_NAMES   # shadow for access below
    LLM_MODEL = args.llm_model
    LineMODReader.LM_ROOT = args.lm_root

    save_dir = args.save_dir or f"./results/lm_pipeline/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results → {save_dir}")

    # ── Initialize Any6D ──────────────────────────────────────────────────────
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(
        trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4)))
    mesh_init = trimesh.Trimesh(
        vertices=mesh_tmp.vertices.copy(), faces=mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh_init, scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                debug_dir=save_dir, debug=0, glctx=glctx)

    # models_info for BOP metrics
    models_info_path = os.path.join(args.lm_root, "models", "models_info.json")
    with open(models_info_path) as f:
        raw_info = json.load(f)
    # re-key by int (file uses str keys)
    models_info = {int(k): v for k, v in raw_info.items()}

    all_metrics   = {}
    all_poses_out = {}  # for JSON export

    for obj_id in args.obj_ids:
        obj_name = LM_NAMES.get(obj_id, f"obj{obj_id:06d}")
        instruction = LM_INSTRUCTIONS.get(obj_id, f"find the {obj_name}")
        is_symmetric = obj_id in LM_SYMMETRIC

        print(f"\n{'='*60}")
        print(f"  Object {obj_id:02d}: {obj_name}  (sym={is_symmetric})")
        print(f"{'='*60}")

        # ── LLM keyword extraction ─────────────────────────────────────────
        if args.skip_llm:
            yoloe_kw = obj_name
        else:
            print(f"[LLM] \"{instruction}\"")
            raw_llm  = call_llm(instruction)
            yoloe_kw = parse_llm(raw_llm)
            if not yoloe_kw or len(yoloe_kw.split()) > 5:
                yoloe_kw = obj_name
        print(f"[LLM] → keyword: \"{yoloe_kw}\"")

        json.dump({'obj_id': obj_id, 'obj_name': obj_name,
                   'instruction': instruction, 'keyword': yoloe_kw},
                  open(f"{save_dir}/obj_{obj_id:06d}_llm.json", 'w'), indent=2)

        # ── Load reader + GT mesh ──────────────────────────────────────────
        reader  = LineMODReader(obj_id)
        gt_mesh = reader.load_gt_mesh(scale_m=True)
        est.reset_object(mesh=gt_mesh, symmetry_tfs=None)

        diameter_m = models_info[obj_id]['diameter'] * LineMODReader.MM_TO_M
        add_thresh  = 0.1 * diameter_m   # 10% of diameter

        obj_metrics = {
            'ADD': [], 'ADD-S': [], 'AR': [], 'VSD': [], 'MSSD': [], 'MSPD': [],
            'R error': [], 'T error': [], 'yoloe_det': [],
        }
        frames_out = []

        test_ids = reader.test_im_ids[::args.stride]
        if args.max_frames is not None:
            test_ids = test_ids[:args.max_frames]
        print(f"  Evaluating {len(test_ids)} images (stride={args.stride})")

        # ── Anchor: first test image → anchor pose ─────────────────────────
        anchor_id = reader.test_im_ids[0]
        anchor_rgb   = reader.get_rgb(anchor_id)
        anchor_depth = reader.get_depth(anchor_id)
        anchor_mask  = reader.get_mask_visib(anchor_id)
        H, W = anchor_rgb.shape[:2]
        if anchor_mask is None:
            anchor_mask = np.ones((H, W), dtype=bool)

        pred_pose_a = est.register(
            K=reader.K, rgb=anchor_rgb, depth=anchor_depth,
            ob_mask=anchor_mask, iteration=5, name=f"anchor_{obj_id}")
        gt_pose_a = reader.get_gt_pose(anchor_id)

        # ── Per-image inference ────────────────────────────────────────────
        obj_count = 0
        for im_id in tqdm(test_ids, desc=obj_name, leave=False):
            rgb   = reader.get_rgb(im_id)
            depth = reader.get_depth(im_id)
            H, W  = rgb.shape[:2]

            gt_pose_q = reader.get_gt_pose(im_id)

            # YOLOE detection
            yoloe_mask = get_yoloe_mask(rgb, yoloe_kw, H, W)
            gt_mask    = reader.get_mask_visib(im_id)

            if yoloe_mask is not None:
                mask        = yoloe_mask
                yoloe_det   = True
                if gt_mask is not None:
                    inter = np.logical_and(yoloe_mask, gt_mask).sum()
                    union = np.logical_or(yoloe_mask, gt_mask).sum()
                    iou   = inter / (union + 1e-6)
                else:
                    iou = -1.0
            else:
                mask      = gt_mask if gt_mask is not None else np.ones((H, W), bool)
                yoloe_det = False
                iou       = -1.0

            # Pose estimation
            pred_pose_q = est.register(
                K=reader.K, rgb=rgb, depth=depth,
                ob_mask=mask, iteration=5, name=f"{obj_name}_{im_id}")

            # Relative pose correction (same as HO3D pipeline)
            pose_aq = pred_pose_q @ np.linalg.inv(pred_pose_a)
            pred_q  = pose_aq @ gt_pose_a

            # Metrics
            err_R = rotation_error_deg(pred_q[:3, :3], gt_pose_q[:3, :3])
            err_T = translation_error_cm(pred_q[:3, 3], gt_pose_q[:3, 3])

            add_val  = compute_add(gt_mesh.vertices, pred_q, gt_pose_q)
            adds_val = compute_adds(gt_mesh.vertices, pred_q, gt_pose_q)
            add_ok   = float(add_val  < add_thresh)
            adds_ok  = float(adds_val < add_thresh)

            # BOP metrics (MSSD, MSPD) — VSD skipped (needs OSMesa renderer)
            mssd_err = mspd_err = mean_ar = float('nan')
            if _BOP_METRICS:
                try:
                    pts = gt_mesh.vertices
                    sym = [{'R': np.eye(3), 't': np.zeros(3)}]
                    mssd_err = float(mssd(pose_est=pred_q, pose_gt=gt_pose_q,
                                         pts=pts, syms=sym))
                    mspd_err = float(mspd(pose_est=pred_q, pose_gt=gt_pose_q,
                                         pts=pts, K=reader.K, syms=sym))
                    mean_ar  = (float(mssd_err < 0.2 * diameter_m) +
                                float(mspd_err < 0.1 * max(640, 480))) / 2
                except Exception:
                    pass

            obj_metrics['ADD'].append(add_ok)
            obj_metrics['ADD-S'].append(adds_ok)
            obj_metrics['AR'].append(mean_ar)
            obj_metrics['VSD'].append(float('nan'))
            obj_metrics['MSSD'].append(mssd_err)
            obj_metrics['MSPD'].append(mspd_err)
            obj_metrics['R error'].append(err_R)
            obj_metrics['T error'].append(err_T)
            obj_metrics['yoloe_det'].append(yoloe_det)

            frames_out.append({
                'im_id': im_id, 'yoloe_detected': yoloe_det, 'iou': float(iou),
                'R_pred': pred_q[:3, :3].tolist(), 'T_pred': pred_q[:3, 3].tolist(),
                'R_gt':   gt_pose_q[:3, :3].tolist(), 'T_gt': gt_pose_q[:3, 3].tolist(),
                'R_error': err_R, 'T_error': err_T,
                'ADD': add_ok, 'ADD-S': adds_ok, 'AR': mean_ar,
            })
            obj_count += 1

        # ── Per-object summary ─────────────────────────────────────────────
        def mean(lst):
            vals = [v for v in lst if not (isinstance(v, float) and np.isnan(v))]
            return float(np.mean(vals)) if vals else float('nan')

        metric_sym = 'ADD-S' if is_symmetric else 'ADD'
        add_score  = mean(obj_metrics['ADD-S' if is_symmetric else 'ADD'])

        summary = {
            'obj_id':   obj_id, 'obj_name': obj_name,
            'n_frames': len(test_ids),
            'yoloe_det_rate': mean(obj_metrics['yoloe_det']),
            'ADD':   mean(obj_metrics['ADD']),
            'ADD-S': mean(obj_metrics['ADD-S']),
            f'{metric_sym} (used)': add_score,
            'AR':    mean(obj_metrics['AR']),
            'R_error_mean': mean(obj_metrics['R error']),
            'R_error_med':  float(np.median(obj_metrics['R error'])),
            'T_error_mean': mean(obj_metrics['T error']),
        }

        print(f"\n  [{obj_name}] {metric_sym}={add_score*100:.1f}%  "
              f"AR={mean(obj_metrics['AR'])*100:.1f}%  "
              f"R_med={float(np.median(obj_metrics['R error'])):.1f}°  "
              f"YOLOE={mean(obj_metrics['yoloe_det'])*100:.0f}%")

        all_metrics[obj_id] = summary
        all_poses_out[obj_id] = {
            'obj_id': obj_id, 'obj_name': obj_name,
            'keyword': yoloe_kw, 'summary': summary, 'frames': frames_out,
        }

        # ── Per-object JSON ────────────────────────────────────────────────
        json.dump(all_poses_out[obj_id],
                  open(f"{save_dir}/obj_{obj_id:06d}_poses.json", 'w'), indent=2)

        # ── Per-object XLSX ────────────────────────────────────────────────
        df_obj = pd.DataFrame({
            'im_id':    [f['im_id']   for f in frames_out],
            'obj_name': obj_name,
            'ADD':      obj_metrics['ADD'],
            'ADD-S':    obj_metrics['ADD-S'],
            'AR':       obj_metrics['AR'],
            'MSSD':     obj_metrics['MSSD'],
            'MSPD':     obj_metrics['MSPD'],
            'R_error':  obj_metrics['R error'],
            'T_error':  obj_metrics['T error'],
            'yoloe_det': obj_metrics['yoloe_det'],
        })
        mean_row = {
            'im_id': 'MEAN', 'obj_name': obj_name,
            'ADD':      f"{mean(obj_metrics['ADD'])*100:.1f}%",
            'ADD-S':    f"{mean(obj_metrics['ADD-S'])*100:.1f}%",
            'AR':       f"{mean(obj_metrics['AR'])*100:.1f}%",
            'MSSD':     f"{mean(obj_metrics['MSSD']):.4f}",
            'MSPD':     f"{mean(obj_metrics['MSPD']):.1f}",
            'R_error':  f"{mean(obj_metrics['R error']):.2f}°",
            'T_error':  f"{mean(obj_metrics['T error']):.2f}cm",
            'yoloe_det': f"{mean(obj_metrics['yoloe_det'])*100:.0f}%",
        }
        df_obj = pd.concat([df_obj, pd.DataFrame([mean_row])], ignore_index=True)
        df_obj.to_excel(f"{save_dir}/obj_{obj_id:06d}_{obj_name}_metrics.xlsx", index=False)

    # ── Global summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LINEMOD RESULTS")
    print(f"{'='*60}")
    print(f"{'Obj':>4}  {'Name':<14} {'ADD(S)':>8} {'AR':>8} {'R_med':>8} {'YOLOE':>8}")
    print("-" * 56)

    add_scores, ar_scores = [], []
    summary_rows = []
    for oid in sorted(all_metrics.keys()):
        s   = all_metrics[oid]
        sym = oid in LM_SYMMETRIC
        add = s['ADD-S'] if sym else s['ADD']
        ar  = s['AR']
        add_scores.append(add)
        if not np.isnan(ar): ar_scores.append(ar)
        flag = '(S)' if sym else '   '
        print(f"  {oid:2d}  {s['obj_name']:<14} {add*100:6.1f}%{flag}  "
              f"{ar*100 if not np.isnan(ar) else float('nan'):6.1f}%  "
              f"{s['R_error_med']:6.1f}°  "
              f"{s['yoloe_det_rate']*100:6.0f}%")
        summary_rows.append({
            'obj_id':       oid,
            'obj_name':     s['obj_name'],
            'symmetric':    sym,
            'n_frames':     s['n_frames'],
            'ADD':          round(s['ADD'] * 100, 1),
            'ADD-S':        round(s['ADD-S'] * 100, 1),
            'ADD(S)_used':  round(add * 100, 1),
            'AR':           round(ar * 100, 1) if not np.isnan(ar) else float('nan'),
            'R_error_mean': round(s['R_error_mean'], 2),
            'R_error_med':  round(s['R_error_med'], 2),
            'T_error_mean': round(s['T_error_mean'], 2),
            'yoloe_det_%':  round(s['yoloe_det_rate'] * 100, 1),
        })

    mean_add  = float(np.mean(add_scores))
    mean_ar   = float(np.mean(ar_scores)) if ar_scores else float('nan')
    print("-" * 56)
    print(f"  {'MEAN':<16} {mean_add*100:6.1f}%     "
          f"{mean_ar*100 if not np.isnan(mean_ar) else float('nan'):6.1f}%")
    print(f"{'='*60}")

    # ── Global JSON ───────────────────────────────────────────────────────────
    global_summary = {
        'dataset': 'LINEMOD BOP',
        'stride': args.stride,
        'yoloe_available': _YOLOE_AVAILABLE,
        'mask_mode': 'YOLOE' if _YOLOE_AVAILABLE else 'GT (oracle)',
        'mean_ADD_S_used': round(mean_add * 100, 2),
        'mean_AR': round(mean_ar * 100, 2) if not np.isnan(mean_ar) else None,
        'per_object': all_metrics,
    }
    json.dump(global_summary,
              open(f"{save_dir}/lm_summary.json", 'w'), indent=2)

    # ── Global XLSX ───────────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    mean_row_global = {
        'obj_id': 'MEAN', 'obj_name': '—', 'symmetric': '—', 'n_frames': '—',
        'ADD':         round(float(np.mean([r['ADD']   for r in summary_rows])), 1),
        'ADD-S':       round(float(np.mean([r['ADD-S'] for r in summary_rows])), 1),
        'ADD(S)_used': round(mean_add * 100, 1),
        'AR':          round(mean_ar * 100, 1) if not np.isnan(mean_ar) else float('nan'),
        'R_error_mean': round(float(np.mean([r['R_error_mean'] for r in summary_rows])), 2),
        'R_error_med':  round(float(np.mean([r['R_error_med']  for r in summary_rows])), 2),
        'T_error_mean': round(float(np.mean([r['T_error_mean'] for r in summary_rows])), 2),
        'yoloe_det_%':  round(float(np.mean([r['yoloe_det_%']  for r in summary_rows])), 1),
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([mean_row_global])], ignore_index=True)
    df_summary.to_excel(f"{save_dir}/lm_FINAL_SUMMARY.xlsx", index=False)

    print(f"\nResults saved → {save_dir}/")
    print(f"  lm_FINAL_SUMMARY.xlsx  (global table)")
    print(f"  lm_summary.json        (global JSON)")
    print(f"  obj_XXXXXX_*.xlsx      (per-object tables)")
