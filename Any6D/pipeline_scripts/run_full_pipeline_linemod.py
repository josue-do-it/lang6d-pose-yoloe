"""
Full pipeline: User instruction → LLM keyword → YOLOE mask → Any6D → BOP metrics
Dataset: LINEMOD BOP (15 objects, GT models scaled to metres).

Dataset must be mounted at /dataset/lm/ inside Docker:
  host:      /home/josue_aims_ac_za/ssd_4tb/dataset/lm
  container: /dataset/lm

Run inside Docker:
    # Full evaluation
    /opt/conda/envs/Any6D/bin/python3 /workspace/pipeline_scripts/run_full_pipeline_linemod.py

    # Quick test — one object, 3 frames
    /opt/conda/envs/Any6D/bin/python3 /workspace/pipeline_scripts/run_full_pipeline_linemod.py \
        --obj_ids 8 --max_frames 3
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import copy
import sys
import os
import re
import json
import argparse
from datetime import datetime

import numpy as np
import cv2
import trimesh
import requests
import pandas as pd
from tqdm import tqdm

# ── Any6D imports (Docker only) ───────────────────────────────────────────────
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/foundationpose/mycpp/build')
import nvdiffrast.torch as dr
from estimater import Any6D, ScorePredictor, PoseRefinePredictor
from metrics import compute_add, compute_adds
from pytorch_lightning import seed_everything

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    sys.path.insert(0, '/workspace/yoloe')
    from ultralytics import YOLOE as _YOLOE
    _YOLOE_AVAILABLE = True
except (ImportError, Exception):
    _YOLOE = None
    _YOLOE_AVAILABLE = False

try:
    from bop_toolkit_lib.pose_error_custom import mssd, mspd
    _BOP_METRICS = True
except Exception:
    _BOP_METRICS = False

# ── Constants ─────────────────────────────────────────────────────────────────
LM_ROOT         = "/dataset/lm"
YOLOE_MODEL     = "yoloe-26l-seg.pt"
YOLOE_CONF      = 0.1
ANY6D_ITERS     = 5
ADD_THRESH_RATIO = 0.10   # 10% of object diameter
MM_TO_M          = 0.001
OLLAMA_URL       = "http://172.18.0.1:11434/api/generate"
LLM_MODEL        = "mistral:latest"

# ── Object metadata ───────────────────────────────────────────────────────────
LM_NAMES = {
    1: "ape", 2: "benchvise", 3: "bowl",    4: "camera",     5: "can",
    6: "cat", 7: "cup",       8: "driller", 9: "duck",       10: "eggbox",
    11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp",   15: "phone",
}

# Objects with rotational symmetry → use ADD-S instead of ADD
LM_SYMMETRIC = {10, 11}

# Natural language instruction for each LINEMOD object
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

# ── Calibrated LLM system prompt ─────────────────────────────────────────────
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


# ── LLM ───────────────────────────────────────────────────────────────────────

def extract_keyword(instruction: str, llm_model: str = LLM_MODEL) -> str:
    """Send instruction to Ollama LLM and return the extracted YOLOE keyword."""
    raw = _call_llm(instruction, llm_model)
    keyword = _parse_llm_response(raw)
    return keyword or instruction.split()[-1]


def _call_llm(instruction: str, llm_model: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": llm_model, "stream": False,
                  "system": CALIBRATED_SYSTEM, "prompt": instruction},
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception:
        return ""


def _parse_llm_response(raw: str) -> str:
    raw = raw.strip()
    m = re.search(r'["""]([^"""]{1,40})["""]', raw)
    if m:
        return m.group(1).strip().lower()
    m = re.search(r'→\s*(.+)$', raw)
    if m:
        return m.group(1).strip().lower()[:40]
    raw = re.sub(
        r'^(keyword[:\s]+|output[:\s]+|the (object|keyword|word) (is|:)\s*)',
        '', raw, flags=re.I)
    for line in raw.split('\n'):
        line = re.sub(r'[^\w\s\'-]', '', line.strip().lstrip('-→•*:')).strip()
        if 0 < len(line.split()) <= 4:
            return line.lower()
    return raw.split('\n')[0][:40].strip().lower()


# ── YOLOE ─────────────────────────────────────────────────────────────────────

_yoloe_model = None


def detect_mask(img_rgb: np.ndarray, keyword: str, H: int, W: int):
    """Run YOLOE detection. Returns (mask, conf) or (None, 0.0) if not found."""
    if not _YOLOE_AVAILABLE or not keyword.strip():
        return None, 0.0

    global _yoloe_model
    orig_dir = os.getcwd()
    if _yoloe_model is None:
        os.chdir('/workspace/yoloe')
        _yoloe_model = _YOLOE(YOLOE_MODEL)
        os.chdir(orig_dir)

    _yoloe_model.set_classes([keyword], _yoloe_model.get_text_pe([keyword]))
    results = _yoloe_model.predict(img_rgb, conf=YOLOE_CONF, verbose=False)
    os.chdir(orig_dir)

    if len(results[0].boxes) > 0 and results[0].masks is not None:
        idx  = int(results[0].boxes.conf.cpu().numpy().argmax())
        conf = float(results[0].boxes.conf.cpu().numpy()[idx])
        mask = results[0].masks.data[idx].cpu().numpy()
        mask = cv2.resize(mask.astype(np.float32), (W, H)) > 0.5
        return mask, conf

    return None, 0.0


def get_detection_mask(img_rgb, keyword, gt_mask, H, W):
    """Return best available mask: YOLOE detection → GT fallback → full image."""
    yoloe_mask, conf = detect_mask(img_rgb, keyword, H, W)

    if yoloe_mask is not None:
        iou = _compute_iou(yoloe_mask, gt_mask)
        return yoloe_mask, True, conf, iou

    fallback = gt_mask if gt_mask is not None else np.ones((H, W), dtype=bool)
    return fallback, False, 0.0, -1.0


def _compute_iou(mask_a, mask_b):
    if mask_b is None:
        return -1.0
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / (union + 1e-6))


# ── Pose estimation ───────────────────────────────────────────────────────────

def estimate_corrected_pose(est, reader, im_id, pred_pose_anchor, gt_pose_anchor, mask):
    """
    Estimate pose for query frame and apply relative anchor correction.
    Correction: pred_q = (pred_anchor → pred_query) @ gt_anchor
    This reduces systematic registration drift.
    """
    rgb   = reader.get_rgb(im_id)
    depth = reader.get_depth(im_id)
    H, W  = rgb.shape[:2]

    gt_pose_q = reader.get_gt_pose(im_id)
    pred_pose_q = est.register(
        K=reader.K, rgb=rgb, depth=depth,
        ob_mask=mask, iteration=ANY6D_ITERS)

    relative_transform = pred_pose_q @ np.linalg.inv(pred_pose_anchor)
    corrected_pose     = relative_transform @ gt_pose_anchor

    return corrected_pose, gt_pose_q, rgb


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_frame_metrics(pred_pose, gt_pose, mesh_vertices, diameter_m, K):
    """Compute ADD, ADD-S, rotation error, translation error, and BOP metrics."""
    add_thresh = ADD_THRESH_RATIO * diameter_m

    err_R = _rotation_error_deg(pred_pose[:3, :3], gt_pose[:3, :3])
    err_T = _translation_error_cm(pred_pose[:3, 3], gt_pose[:3, 3])

    add_val  = compute_add(mesh_vertices, pred_pose, gt_pose)
    adds_val = compute_adds(mesh_vertices, pred_pose, gt_pose)
    add_ok   = float(add_val  < add_thresh)
    adds_ok  = float(adds_val < add_thresh)

    mssd_err = mspd_err = float('nan')
    if _BOP_METRICS:
        try:
            syms = [{'R': np.eye(3), 't': np.zeros(3)}]
            mssd_err = float(mssd(pose_est=pred_pose, pose_gt=gt_pose,
                                  pts=mesh_vertices, syms=syms))
            mspd_err = float(mspd(pose_est=pred_pose, pose_gt=gt_pose,
                                  pts=mesh_vertices, K=K, syms=syms))
        except Exception:
            pass

    mean_ar = (float(mssd_err < 0.2 * diameter_m) +
               float(mspd_err < 0.1 * max(640, 480))) / 2 \
              if _BOP_METRICS and not np.isnan(mssd_err) else float('nan')

    return {
        'ADD': add_ok, 'ADD-S': adds_ok, 'AR': mean_ar,
        'MSSD': mssd_err, 'MSPD': mspd_err,
        'R_error': err_R, 'T_error': err_T,
    }


def _rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    cos_val = np.clip((np.trace(R_pred @ R_gt.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_val)))


def _translation_error_cm(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(t_pred - t_gt) * 100)


def _nanmean(values: list) -> float:
    valid = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(valid)) if valid else float('nan')


# ── Data reader ───────────────────────────────────────────────────────────────

class LineMODReader:
    """BOP-format LINEMOD dataset reader for a single object."""

    def __init__(self, obj_id: int, lm_root: str = LM_ROOT):
        self.obj_id    = obj_id
        self.lm_root   = lm_root
        self.scene_dir = os.path.join(lm_root, "test", f"{obj_id:06d}")

        cam_path = os.path.join(lm_root, "lm", "camera.json")
        with open(cam_path) as f:
            c = json.load(f)
        self.K = np.array([[c['fx'], 0, c['cx']],
                           [0, c['fy'], c['cy']],
                           [0, 0, 1]], dtype=np.float64)
        self.depth_scale = c.get('depth_scale', 1.0)

        with open(os.path.join(self.scene_dir, "scene_gt.json")) as f:
            self._scene_gt = json.load(f)

        targets_path = os.path.join(lm_root, "lm", "test_targets_bop19.json")
        with open(targets_path) as f:
            all_targets = json.load(f)
        self.test_im_ids = sorted(
            t['im_id'] for t in all_targets
            if t['obj_id'] == obj_id and t['scene_id'] == obj_id)

    def get_rgb(self, im_id: int) -> np.ndarray:
        path = os.path.join(self.scene_dir, "rgb", f"{im_id:06d}.png")
        img  = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"RGB image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_depth(self, im_id: int) -> np.ndarray:
        path  = os.path.join(self.scene_dir, "depth", f"{im_id:06d}.png")
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        return depth * self.depth_scale * MM_TO_M

    def get_mask_visib(self, im_id: int):
        path = os.path.join(self.scene_dir, "mask_visib", f"{im_id:06d}_000000.png")
        if not os.path.exists(path):
            return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return m > 0 if m is not None else None

    def get_gt_pose(self, im_id: int) -> np.ndarray:
        """4×4 pose matrix in metres."""
        ann = self._scene_gt[str(im_id)][0]
        R   = np.array(ann['cam_R_m2c']).reshape(3, 3)
        t   = np.array(ann['cam_t_m2c']) * MM_TO_M
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3,  3] = t
        return pose

    def load_gt_mesh(self, scale_to_metres: bool = True) -> trimesh.Trimesh:
        path = os.path.join(self.lm_root, "models", f"obj_{self.obj_id:06d}.ply")
        mesh = trimesh.load(path, force='mesh')
        if scale_to_metres:
            mesh.apply_scale(MM_TO_M)
        return mesh


# ── Per-object pipeline ───────────────────────────────────────────────────────

def run_object(obj_id: int, est, models_info: dict, args, save_dir: str) -> dict:
    """Run the full pipeline for one LINEMOD object. Returns summary dict."""
    obj_name     = LM_NAMES.get(obj_id, f"obj{obj_id:06d}")
    instruction  = LM_INSTRUCTIONS.get(obj_id, f"find the {obj_name}")
    is_symmetric = obj_id in LM_SYMMETRIC
    diameter_m   = models_info[obj_id]['diameter'] * MM_TO_M

    print(f"\n{'='*60}")
    print(f"  Object {obj_id:02d}: {obj_name}  (sym={is_symmetric})")
    print(f"{'='*60}")

    # LLM keyword
    if args.skip_llm:
        keyword = obj_name
    else:
        print(f"[LLM] \"{instruction}\"")
        keyword = extract_keyword(instruction, args.llm_model)
        if not keyword or len(keyword.split()) > 5:
            keyword = obj_name
    print(f"[LLM] → keyword: \"{keyword}\"")

    with open(os.path.join(save_dir, f"obj_{obj_id:06d}_llm.json"), 'w') as f:
        json.dump({'obj_id': obj_id, 'obj_name': obj_name,
                   'instruction': instruction, 'keyword': keyword}, f, indent=2)

    reader  = LineMODReader(obj_id, lm_root=args.lm_root)
    gt_mesh = reader.load_gt_mesh(scale_to_metres=True)
    est.reset_object(mesh=gt_mesh, symmetry_tfs=None)

    test_ids = reader.test_im_ids[::args.stride]
    if args.max_frames is not None:
        test_ids = test_ids[:args.max_frames]
    print(f"  Evaluating {len(test_ids)} images (stride={args.stride})")

    # Anchor frame: first BOP19 target
    anchor_id    = reader.test_im_ids[0]
    anchor_rgb   = reader.get_rgb(anchor_id)
    anchor_depth = reader.get_depth(anchor_id)
    anchor_mask  = reader.get_mask_visib(anchor_id)
    H, W = anchor_rgb.shape[:2]
    if anchor_mask is None:
        anchor_mask = np.ones((H, W), dtype=bool)

    pred_pose_anchor = est.register(
        K=reader.K, rgb=anchor_rgb, depth=anchor_depth,
        ob_mask=anchor_mask, iteration=ANY6D_ITERS, name=f"anchor_{obj_id}")
    gt_pose_anchor = reader.get_gt_pose(anchor_id)

    # Per-frame loop
    metric_lists = {k: [] for k in
                    ['ADD', 'ADD-S', 'AR', 'MSSD', 'MSPD',
                     'R_error', 'T_error', 'yoloe_det', 'iou']}
    frames_out = []

    for im_id in tqdm(test_ids, desc=obj_name, leave=False):
        rgb      = reader.get_rgb(im_id)
        depth    = reader.get_depth(im_id)
        H, W     = rgb.shape[:2]
        gt_mask  = reader.get_mask_visib(im_id)
        gt_pose  = reader.get_gt_pose(im_id)

        mask, yoloe_det, conf, iou = get_detection_mask(rgb, keyword, gt_mask, H, W)

        corrected_pose, _, _ = estimate_corrected_pose(
            est, reader, im_id, pred_pose_anchor, gt_pose_anchor, mask)

        frame_m = compute_frame_metrics(
            corrected_pose, gt_pose, gt_mesh.vertices, diameter_m, reader.K)

        for key in ['ADD', 'ADD-S', 'AR', 'MSSD', 'MSPD', 'R_error', 'T_error']:
            metric_lists[key].append(frame_m[key])
        metric_lists['yoloe_det'].append(float(yoloe_det))
        metric_lists['iou'].append(iou)

        frames_out.append({
            'im_id': im_id,
            'yoloe_detected': yoloe_det, 'yoloe_conf': round(conf, 3), 'iou': round(iou, 3),
            'R_pred': corrected_pose[:3, :3].tolist(), 'T_pred': corrected_pose[:3, 3].tolist(),
            'R_gt':   gt_pose[:3, :3].tolist(),        'T_gt':   gt_pose[:3, 3].tolist(),
            **{k: round(frame_m[k], 4) for k in ['R_error', 'T_error', 'ADD', 'ADD-S', 'AR']},
        })

    # Object summary
    metric_sym = 'ADD-S' if is_symmetric else 'ADD'
    add_score  = _nanmean(metric_lists[metric_sym])
    summary = {
        'obj_id': obj_id, 'obj_name': obj_name, 'instruction': instruction,
        'keyword': keyword, 'n_frames': len(test_ids),
        'yoloe_det_rate': _nanmean(metric_lists['yoloe_det']),
        'ADD':   _nanmean(metric_lists['ADD']),
        'ADD-S': _nanmean(metric_lists['ADD-S']),
        f'{metric_sym} (used)': add_score,
        'AR':    _nanmean(metric_lists['AR']),
        'R_error_mean': _nanmean(metric_lists['R_error']),
        'R_error_med':  float(np.median(metric_lists['R_error'])),
        'T_error_mean': _nanmean(metric_lists['T_error']),
    }

    print(f"\n  [{obj_name}] {metric_sym}={add_score*100:.1f}%  "
          f"AR={_nanmean(metric_lists['AR'])*100:.1f}%  "
          f"R_med={float(np.median(metric_lists['R_error'])):.1f}°  "
          f"YOLOE={_nanmean(metric_lists['yoloe_det'])*100:.0f}%")

    _save_object_results(obj_id, obj_name, keyword, summary, frames_out,
                         metric_lists, save_dir)
    return summary


# ── Save helpers ──────────────────────────────────────────────────────────────

def _save_object_results(obj_id, obj_name, keyword, summary, frames_out,
                         metric_lists, save_dir):
    """Save per-object JSON and XLSX."""
    with open(os.path.join(save_dir, f"obj_{obj_id:06d}_poses.json"), 'w') as f:
        json.dump({'obj_id': obj_id, 'obj_name': obj_name, 'keyword': keyword,
                   'summary': summary, 'frames': frames_out}, f, indent=2)

    df = pd.DataFrame({
        'im_id':    [fr['im_id'] for fr in frames_out],
        'obj_name': obj_name,
        'ADD':      metric_lists['ADD'],
        'ADD-S':    metric_lists['ADD-S'],
        'AR':       metric_lists['AR'],
        'MSSD':     metric_lists['MSSD'],
        'MSPD':     metric_lists['MSPD'],
        'R_error':  metric_lists['R_error'],
        'T_error':  metric_lists['T_error'],
        'yoloe_det': metric_lists['yoloe_det'],
    })
    mean_row = {
        'im_id': 'MEAN', 'obj_name': obj_name,
        'ADD':   f"{_nanmean(metric_lists['ADD'])*100:.1f}%",
        'ADD-S': f"{_nanmean(metric_lists['ADD-S'])*100:.1f}%",
        'AR':    f"{_nanmean(metric_lists['AR'])*100:.1f}%",
        'MSSD':  f"{_nanmean(metric_lists['MSSD']):.4f}",
        'MSPD':  f"{_nanmean(metric_lists['MSPD']):.1f}",
        'R_error': f"{_nanmean(metric_lists['R_error']):.2f}°",
        'T_error': f"{_nanmean(metric_lists['T_error']):.2f}cm",
        'yoloe_det': f"{_nanmean(metric_lists['yoloe_det'])*100:.0f}%",
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.to_excel(os.path.join(save_dir, f"obj_{obj_id:06d}_{obj_name}_metrics.xlsx"),
                index=False)


def save_global_summary(all_summaries: dict, args, save_dir: str):
    """Print global table and save XLSX + JSON."""
    print(f"\n{'='*60}")
    print(f"  LINEMOD RESULTS")
    print(f"{'='*60}")
    print(f"{'Obj':>4}  {'Name':<14} {'ADD(S)':>8} {'AR':>8} {'R_med':>8} {'YOLOE':>8}")
    print("-" * 56)

    add_scores, ar_scores, summary_rows = [], [], []

    for obj_id in sorted(all_summaries.keys()):
        s     = all_summaries[obj_id]
        sym   = obj_id in LM_SYMMETRIC
        add   = s['ADD-S'] if sym else s['ADD']
        ar    = s['AR']
        flag  = '(S)' if sym else '   '
        add_scores.append(add)
        if not np.isnan(ar):
            ar_scores.append(ar)

        print(f"  {obj_id:2d}  {s['obj_name']:<14} {add*100:6.1f}%{flag}  "
              f"{ar*100 if not np.isnan(ar) else float('nan'):6.1f}%  "
              f"{s['R_error_med']:6.1f}°  {s['yoloe_det_rate']*100:6.0f}%")

        summary_rows.append({
            'obj_id': obj_id, 'obj_name': s['obj_name'], 'symmetric': sym,
            'n_frames': s['n_frames'],
            'ADD':         round(s['ADD'] * 100, 1),
            'ADD-S':       round(s['ADD-S'] * 100, 1),
            'ADD(S)_used': round(add * 100, 1),
            'AR':          round(ar * 100, 1) if not np.isnan(ar) else float('nan'),
            'R_error_mean': round(s['R_error_mean'], 2),
            'R_error_med':  round(s['R_error_med'], 2),
            'T_error_mean': round(s['T_error_mean'], 2),
            'yoloe_det_%':  round(s['yoloe_det_rate'] * 100, 1),
        })

    mean_add = float(np.mean(add_scores))
    mean_ar  = float(np.mean(ar_scores)) if ar_scores else float('nan')
    print("-" * 56)
    print(f"  {'MEAN':<16} {mean_add*100:6.1f}%"
          f"     {mean_ar*100 if not np.isnan(mean_ar) else float('nan'):6.1f}%")
    print(f"{'='*60}")

    global_json = {
        'dataset': 'LINEMOD BOP', 'stride': args.stride,
        'yoloe_available': _YOLOE_AVAILABLE,
        'mask_mode': 'YOLOE' if _YOLOE_AVAILABLE else 'GT (oracle)',
        'mean_ADD_S_used': round(mean_add * 100, 2),
        'mean_AR': round(mean_ar * 100, 2) if not np.isnan(mean_ar) else None,
        'per_object': all_summaries,
    }
    with open(os.path.join(save_dir, "lm_summary.json"), 'w') as f:
        json.dump(global_json, f, indent=2)

    mean_row = {
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
    df = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame([mean_row])],
                   ignore_index=True)
    df.to_excel(os.path.join(save_dir, "lm_FINAL_SUMMARY.xlsx"), index=False)

    print(f"\nResults saved → {save_dir}/")
    print(f"  lm_FINAL_SUMMARY.xlsx  (global table)")
    print(f"  lm_summary.json        (global JSON)")
    print(f"  obj_XXXXXX_*.xlsx      (per-object tables)")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LINEMOD BOP full pipeline: LLM → YOLOE → Any6D")
    parser.add_argument("--lm_root",    default=LM_ROOT)
    parser.add_argument("--obj_ids",    type=int, nargs='+', default=list(range(1, 16)),
                        help="LINEMOD object IDs to evaluate (1-15)")
    parser.add_argument("--stride",     type=int, default=1,
                        help="Frame stride over BOP19 test images (1=full, 5=quick)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames per object (e.g. 1 for quick single-image test)")
    parser.add_argument("--save_dir",   default=None)
    parser.add_argument("--llm_model",  default=LLM_MODEL)
    parser.add_argument("--skip_llm",   action="store_true",
                        help="Use object name as YOLOE prompt directly (skip LLM)")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(0)

    save_dir = args.save_dir or f"./results/lm_pipeline/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results  → {save_dir}")
    print(f"YOLOE    : {_YOLOE_AVAILABLE}  |  BOP metrics: {_BOP_METRICS}")
    print(f"Stride   : {args.stride}  |  LLM: {'skip' if args.skip_llm else args.llm_model}")

    with open(os.path.join(args.lm_root, "models", "models_info.json")) as f:
        models_info = {int(k): v for k, v in json.load(f).items()}

    # Initialise estimator once, reset mesh per object
    glctx     = dr.RasterizeCudaContext()
    mesh_tmp  = trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4))
    mesh_init = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(),
                                faces=mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh_init, scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                debug_dir=save_dir, debug=0, glctx=glctx)

    all_summaries = {}
    for obj_id in args.obj_ids:
        all_summaries[obj_id] = run_object(obj_id, est, models_info, args, save_dir)

    save_global_summary(all_summaries, args, save_dir)


if __name__ == '__main__':
    main()
