"""
Full pipeline YCB-Video BOP → ADD/ADD-S metrics.

Structure:
  For each of 21 YCB objects:
    LLM → keyword → YOLOE (fallback GT mask)
    For each scene where the object appears (from test_targets_bop19.json):
      anchor = first im_id → est.register()
      For each subsequent im_id (stride):
        est.register() → pred_pose_q
        relative pose correction → pred_q
        ADD / ADD-S metrics
  Save per-object JSON + global XLSX summary

Run inside Docker:
    /opt/conda/envs/Any6D/bin/python3 /workspace/run_full_pipeline_ycbv.py [--stride 1] [--skip_llm]
"""
import os, sys, re, json, glob, argparse, collections, copy
from datetime import datetime
import numpy as np
import cv2
import trimesh
import requests
import torch
import nvdiffrast.torch as dr
from tqdm import tqdm

sys.path.insert(0, '/workspace')

# ── YOLOE (optional) ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLOE as _YOLOE
    _YOLOE_AVAILABLE = True
except Exception:
    _YOLOE_AVAILABLE = False

_yoloe_model = None

def get_yoloe_mask(img_rgb, prompt, H, W):
    global _yoloe_model
    if not _YOLOE_AVAILABLE or not prompt:
        return None
    if _yoloe_model is None:
        _yoloe_model = _YOLOE('/workspace/yoloe/yoloe-26l-seg.pt')
    try:
        _yoloe_model.set_classes([prompt], _yoloe_model.get_text_pe([prompt]))
        results = _yoloe_model.predict(img_rgb, verbose=False)
        if results and results[0].masks is not None and len(results[0].masks) > 0:
            m = results[0].masks.data[0].cpu().numpy()
            m = cv2.resize(m.astype(np.uint8), (W, H)) > 0
            return m
    except Exception:
        pass
    return None

# ── LLM setup ─────────────────────────────────────────────────────────────────
CALIBRATED_SYSTEM = """\
You are a visual keyword extractor for YOLOE, an open-vocabulary object segmentation model.
Your ONLY job: extract the most precise visual keyword(s) from the user instruction.
Output ONLY the keyword — nothing else. No punctuation, no explanation, no sentence.
- 1 word is BEST when specific enough (e.g. "banana", "mug", "scissors")
- Use 2 words ONLY when ambiguous (e.g. "mustard bottle", "power drill")
- MAXIMUM 3 words
Examples:
"Find the master chef can on the table" → chef can
"Find the cracker box" → cracker box
"Find the sugar box" → sugar box
"Hand me the tomato soup can" → soup can
"Find the mustard bottle on the table" → mustard bottle
"Pick up the tuna fish can" → tuna can
"Find the pudding box" → pudding box
"Hand me the gelatin box" → gelatin box
"Pick up the potted meat can" → meat can
"Find the banana" → banana
"Hand me the pitcher" → pitcher
"Grab the bleach cleanser" → bleach bottle
"Pass me the bowl" → bowl
"Hand me the mug" → mug
"I need the power drill" → power drill
"Find the wood block" → wood block
"Hand me the scissors" → scissors
"Find the large marker" → marker
"Hand me the large clamp" → clamp
"Find the foam brick" → foam brick
"""

OLLAMA_URL = "http://172.18.0.1:11434/api/generate"
LLM_MODEL  = "mistral:latest"

def call_llm(instruction: str) -> str:
    try:
        r = requests.post(OLLAMA_URL, timeout=30,
                          json={"model": LLM_MODEL, "stream": False,
                                "system": CALIBRATED_SYSTEM, "prompt": instruction},
                          headers={"Content-Type": "application/json"})
        return r.json().get("response", instruction).strip()
    except Exception:
        return instruction

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

# ── YCB object names ──────────────────────────────────────────────────────────
YCBV_NAMES = {
    1:  "master chef can",    2:  "cracker box",
    3:  "sugar box",          4:  "tomato soup can",
    5:  "mustard bottle",     6:  "tuna fish can",
    7:  "pudding box",        8:  "gelatin box",
    9:  "potted meat can",    10: "banana",
    11: "pitcher base",       12: "bleach cleanser",
    13: "bowl",               14: "mug",
    15: "power drill",        16: "wood block",
    17: "scissors",           18: "large marker",
    19: "large clamp",        20: "extra large clamp",
    21: "foam brick",
}

# symmetric objects → use ADD-S instead of ADD
YCBV_SYMMETRIC = {1, 13, 16, 18, 19, 20, 21}

YCBV_ROOT = "/dataset/ycbv"
MM_TO_M   = 0.001

# ── BOP YCB-Video reader ──────────────────────────────────────────────────────
class YCBVReader:
    """Reads BOP-format YCB-Video dataset for a given (scene_id, obj_id)."""

    def __init__(self, scene_id: int, obj_id: int):
        self.scene_id  = scene_id
        self.obj_id    = obj_id
        self.scene_dir = os.path.join(YCBV_ROOT, "test", f"{scene_id:06d}")

        with open(os.path.join(self.scene_dir, "scene_camera.json")) as f:
            self._scene_cam = json.load(f)
        with open(os.path.join(self.scene_dir, "scene_gt.json")) as f:
            self._scene_gt = json.load(f)

        # K and depth_scale from first frame (constant within a scene)
        first_key = list(self._scene_cam.keys())[0]
        km = self._scene_cam[first_key]['cam_K']
        self.K = np.array([[km[0], km[1], km[2]],
                           [km[3], km[4], km[5]],
                           [km[6], km[7], km[8]]], dtype=np.float64)
        self.depth_scale = self._scene_cam[first_key].get('depth_scale', 0.1)

        # annotation index for obj_id per frame
        self._ann_idx = {}
        for im_id_str, anns in self._scene_gt.items():
            im_id = int(im_id_str)
            for idx, ann in enumerate(anns):
                if ann['obj_id'] == obj_id:
                    self._ann_idx[im_id] = idx
                    break

    def im_ids_with_obj(self):
        return sorted(self._ann_idx.keys())

    def get_rgb(self, im_id: int) -> np.ndarray:
        path = os.path.join(self.scene_dir, "rgb", f"{im_id:06d}.png")
        img  = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_depth(self, im_id: int) -> np.ndarray:
        path  = os.path.join(self.scene_dir, "depth", f"{im_id:06d}.png")
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        return depth * self.depth_scale * MM_TO_M   # → meters

    def get_mask_visib(self, im_id: int):
        ann_idx = self._ann_idx.get(im_id, 0)
        path = os.path.join(self.scene_dir, "mask_visib",
                            f"{im_id:06d}_{ann_idx:06d}.png")
        if not os.path.exists(path):
            return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return (m > 0) if m is not None else None

    def get_gt_pose(self, im_id: int) -> np.ndarray:
        ann_idx = self._ann_idx.get(im_id, 0)
        ann = self._scene_gt[str(im_id)][ann_idx]
        R   = np.array(ann['cam_R_m2c']).reshape(3, 3)
        t   = np.array(ann['cam_t_m2c']) * MM_TO_M
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3,  3] = t
        return pose


# ── Metric helpers ────────────────────────────────────────────────────────────
def rotation_error_deg(R_pred, R_gt):
    R_rel   = R_pred @ R_gt.T
    cos_val = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_val)))

def translation_error_cm(t_pred, t_gt):
    return float(np.linalg.norm(t_pred - t_gt) * 100)

def mean_valid(lst):
    vals = [v for v in lst if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float('nan')


# ── Imports (heavy — after arg parse) ─────────────────────────────────────────
from estimater import *
from metrics import *
from pytorch_lightning import seed_everything

try:
    from bop_toolkit_lib.pose_error_custom import mssd, mspd
    _BOP_METRICS = True
except Exception:
    _BOP_METRICS = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride",    type=int, default=1,
                        help="Frame stride within BOP19 targets (default=1)")
    parser.add_argument("--skip_llm",  action="store_true")
    parser.add_argument("--llm_model", default="mistral:latest")
    parser.add_argument("--obj_ids",   type=int, nargs='+', default=None,
                        help="Subset of obj_ids to run (default: all 1-21)")
    args = parser.parse_args()
    global LLM_MODEL
    LLM_MODEL = args.llm_model

    seed_everything(0)

    run_name = f"stride{args.stride}"
    save_dir  = f"/workspace/results/ycbv_pipeline/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir : {save_dir}")
    print(f"YOLOE    : {_YOLOE_AVAILABLE}  |  BOP metrics: {_BOP_METRICS}")
    print(f"Stride   : {args.stride}  |  LLM: {'skip' if args.skip_llm else LLM_MODEL}\n")

    # models_info (diameters)
    with open(f"{YCBV_ROOT}/models/models_info.json") as f:
        models_info = {int(k): v for k, v in json.load(f).items()}

    # test targets index: obj_id → scene_id → [im_id]
    with open(f"{YCBV_ROOT}/ycbv/test_targets_bop19.json") as f:
        all_targets = json.load(f)
    obj_scene_frames = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in all_targets:
        obj_scene_frames[t['obj_id']][t['scene_id']].append(t['im_id'])
    for ob in obj_scene_frames:
        for sc in obj_scene_frames[ob]:
            obj_scene_frames[ob][sc] = sorted(set(obj_scene_frames[ob][sc]))

    obj_ids = args.obj_ids if args.obj_ids else sorted(YCBV_NAMES.keys())

    # ── Estimator (initialised with first PLY, reset per object) ──────────────
    glctx     = dr.RasterizeCudaContext()
    mesh_init = trimesh.load(f"{YCBV_ROOT}/models/obj_000001.ply", force='mesh')
    mesh_init.apply_scale(MM_TO_M)
    est = Any6D(mesh=mesh_init, scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                debug_dir=save_dir, debug=0, glctx=glctx)

    all_summaries = []

    for obj_id in obj_ids:
        obj_name     = YCBV_NAMES.get(obj_id, f"obj{obj_id:02d}")
        is_sym       = obj_id in YCBV_SYMMETRIC
        diameter_m   = models_info[obj_id]['diameter'] * MM_TO_M
        add_thresh   = 0.1 * diameter_m

        print(f"\n{'='*60}")
        print(f"obj{obj_id:02d}  {obj_name}  (sym={is_sym}  diam={diameter_m*100:.1f}cm)")

        # ── LLM instruction ───────────────────────────────────────────────────
        instruction = f"Find the {obj_name} on the table"
        if args.skip_llm:
            yoloe_kw = obj_name
        else:
            print(f"[LLM] \"{instruction}\"")
            yoloe_kw = parse_llm(call_llm(instruction))
        print(f"[LLM] → \"{yoloe_kw}\"")

        json.dump({'obj_id': obj_id, 'obj_name': obj_name,
                   'instruction': instruction, 'keyword': yoloe_kw},
                  open(f"{save_dir}/obj_{obj_id:06d}_llm.json", 'w'), indent=2)

        # ── Mesh ──────────────────────────────────────────────────────────────
        gt_mesh = trimesh.load(f"{YCBV_ROOT}/models/obj_{obj_id:06d}.ply", force='mesh')
        gt_mesh.apply_scale(MM_TO_M)
        est.reset_object(mesh=gt_mesh, symmetry_tfs=None)

        # ── Loop over scenes ──────────────────────────────────────────────────
        all_frames = []
        obj_add = []; obj_adds = []; obj_ar = []; obj_re = []; obj_te = []

        for scene_id in sorted(obj_scene_frames[obj_id].keys()):
            im_ids_all = obj_scene_frames[obj_id][scene_id]
            im_ids     = im_ids_all[::args.stride]
            reader     = YCBVReader(scene_id, obj_id)

            # anchor: first BOP19 target frame in this scene
            anchor_id    = im_ids_all[0]
            anchor_rgb   = reader.get_rgb(anchor_id)
            anchor_depth = reader.get_depth(anchor_id)
            anchor_mask  = reader.get_mask_visib(anchor_id)
            H, W = anchor_rgb.shape[:2]
            if anchor_mask is None:
                anchor_mask = np.ones((H, W), dtype=bool)

            pred_pose_a = est.register(
                K=reader.K, rgb=anchor_rgb, depth=anchor_depth,
                ob_mask=anchor_mask, iteration=5,
                name=f"anc_{obj_id}_{scene_id}")
            gt_pose_a = reader.get_gt_pose(anchor_id)

            print(f"  sc{scene_id:05d}: {len(im_ids)} frames  anchor={anchor_id}")

            for im_id in tqdm(im_ids, desc=f"sc{scene_id:05d}", leave=False):
                rgb   = reader.get_rgb(im_id)
                depth = reader.get_depth(im_id)
                H, W  = rgb.shape[:2]
                gt_pose_q = reader.get_gt_pose(im_id)

                # detection (YOLOE or GT fallback)
                yoloe_mask = get_yoloe_mask(rgb, yoloe_kw, H, W)
                gt_mask    = reader.get_mask_visib(im_id)
                if yoloe_mask is not None:
                    mask = yoloe_mask; yoloe_det = True
                    iou  = (float(np.logical_and(yoloe_mask, gt_mask).sum() /
                                  (np.logical_or(yoloe_mask, gt_mask).sum() + 1e-6))
                            if gt_mask is not None else -1.0)
                else:
                    mask = gt_mask if gt_mask is not None else np.ones((H, W), bool)
                    yoloe_det = False; iou = -1.0

                # pose estimation + relative correction
                pred_pose_q = est.register(
                    K=reader.K, rgb=rgb, depth=depth, ob_mask=mask,
                    iteration=5, name=f"{obj_name}_{scene_id}_{im_id}")
                pose_aq = pred_pose_q @ np.linalg.inv(pred_pose_a)
                pred_q  = pose_aq @ gt_pose_a

                # metrics
                err_R  = rotation_error_deg(pred_q[:3, :3], gt_pose_q[:3, :3])
                err_T  = translation_error_cm(pred_q[:3, 3], gt_pose_q[:3, 3])
                add_v  = float(compute_add(gt_mesh.vertices,  pred_q, gt_pose_q) < add_thresh)
                adds_v = float(compute_adds(gt_mesh.vertices, pred_q, gt_pose_q) < add_thresh)

                mean_ar = float('nan')
                if _BOP_METRICS:
                    try:
                        sym = [{'R': np.eye(3), 't': np.zeros(3)}]
                        ms  = float(mssd(pose_est=pred_q, pose_gt=gt_pose_q,
                                         pts=gt_mesh.vertices, syms=sym))
                        mp  = float(mspd(pose_est=pred_q, pose_gt=gt_pose_q,
                                         pts=gt_mesh.vertices, K=reader.K, syms=sym))
                        mean_ar = (float(ms < 0.2 * diameter_m) +
                                   float(mp < 0.1 * max(W, H))) / 2
                    except Exception:
                        pass

                obj_add.append(add_v); obj_adds.append(adds_v)
                obj_ar.append(mean_ar); obj_re.append(err_R); obj_te.append(err_T)

                all_frames.append({
                    'scene_id': scene_id, 'im_id': im_id,
                    'yoloe_detected': yoloe_det, 'iou': float(iou),
                    'R_pred': pred_q[:3, :3].tolist(), 'T_pred': pred_q[:3, 3].tolist(),
                    'R_gt':   gt_pose_q[:3, :3].tolist(), 'T_gt': gt_pose_q[:3, 3].tolist(),
                    'R_error': err_R, 'T_error': err_T,
                    'ADD': add_v, 'ADD-S': adds_v, 'AR': mean_ar,
                })

        # ── Per-object summary ─────────────────────────────────────────────────
        metric_sym = 'ADD-S' if is_sym else 'ADD'
        add_score  = mean_valid(obj_adds if is_sym else obj_add)

        summary = {
            'obj_id': obj_id, 'obj_name': obj_name, 'symmetric': is_sym,
            'instruction': instruction, 'keyword': yoloe_kw,
            'n_frames': len(all_frames),
            'ADD':            round(mean_valid(obj_add)  * 100, 1),
            'ADD-S':          round(mean_valid(obj_adds) * 100, 1),
            f'{metric_sym} (used)': round(add_score * 100, 1),
            'AR':             round(mean_valid(obj_ar) * 100, 1),
            'R_error_mean':   round(mean_valid(obj_re), 2),
            'R_error_med':    round(float(np.median([v for v in obj_re if not np.isnan(v)])), 2)
                              if any(not np.isnan(v) for v in obj_re) else float('nan'),
            'T_error_mean':   round(mean_valid(obj_te), 2),
            'yoloe_det_%':    round(sum(f['yoloe_detected'] for f in all_frames) /
                                    max(len(all_frames), 1) * 100, 1),
        }
        all_summaries.append(summary)

        print(f"  → {metric_sym}={summary[f'{metric_sym} (used)']}%  "
              f"R_med={summary['R_error_med']}°  T_mean={summary['T_error_mean']}cm")

        json.dump({
            'obj_id': obj_id, 'obj_name': obj_name,
            'instruction': instruction, 'keyword': yoloe_kw,
            'summary': summary, 'frames': all_frames,
        }, open(f"{save_dir}/obj_{obj_id:06d}_poses.json", 'w'), indent=2)

    # ── Global XLSX/JSON ───────────────────────────────────────────────────────
    try:
        import pandas as pd
        add_used = [r.get('ADD-S (used)', r.get('ADD (used)', float('nan')))
                    for r in all_summaries]
        mean_row = {
            'obj_id': 'MEAN', 'obj_name': '—', 'symmetric': '—',
            'n_frames':      sum(r['n_frames'] for r in all_summaries),
            'ADD':           round(mean_valid([r['ADD']          for r in all_summaries]), 1),
            'ADD-S':         round(mean_valid([r['ADD-S']        for r in all_summaries]), 1),
            'ADD(S) used':   round(mean_valid(add_used), 1),
            'AR':            round(mean_valid([r['AR']           for r in all_summaries]), 1),
            'R_error_mean':  round(mean_valid([r['R_error_mean'] for r in all_summaries]), 2),
            'R_error_med':   round(mean_valid([r['R_error_med']  for r in all_summaries]), 2),
            'T_error_mean':  round(mean_valid([r['T_error_mean'] for r in all_summaries]), 2),
            'yoloe_det_%':   round(mean_valid([r['yoloe_det_%']  for r in all_summaries]), 1),
        }
        pd.DataFrame(all_summaries + [mean_row]).to_excel(
            f"{save_dir}/ycbv_SUMMARY.xlsx", index=False)
        json.dump({
            'dataset': 'YCB-Video BOP', 'run': run_name,
            'mean_ADD(S)': mean_row['ADD(S) used'],
            'per_object': {r['obj_name']: r for r in all_summaries},
        }, open(f"{save_dir}/ycbv_SUMMARY.json", 'w'), indent=2)
        print(f"\nMEAN ADD(S) = {mean_row['ADD(S) used']}%")
        print(f"Saved → {save_dir}/ycbv_SUMMARY.xlsx")
    except Exception as e:
        print(f"XLSX failed: {e}")

    print(f"\nDone. Results in {save_dir}/")


if __name__ == "__main__":
    main()
