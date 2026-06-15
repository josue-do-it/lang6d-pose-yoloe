"""
Full pipeline: User instruction → LLM keyword → YOLOE mask → Any6D → BOP metrics
Dataset: YCB-Video BOP (21 household objects, ~900 test scenes).

Pipeline overview
-----------------
For each of the 21 YCB objects:
  1. LLM extracts a YOLOE keyword from a natural language instruction using the
     dataset-specific CALIBRATED_SYSTEM prompt (few-shot examples for YCB objects).
  2. YOLOE segments the object in each frame.  Falls back to GT mask on failure.
  3. Any6D registers the anchor (first frame per scene), then corrects every
     subsequent frame pose relative to the anchor:
         corrected = (pred_q @ inv(pred_anchor)) @ gt_anchor
  4. ADD / ADD-S metrics are computed per frame.  Symmetric objects (bowls, cans,
     foam brick) use ADD-S.
  5. Per-object JSON and a global XLSX summary are written to the output directory.

Dataset layout (inside Docker)
-------------------------------
/dataset/ycbv/
    models/obj_000001.ply … obj_000021.ply
    models_info.json
    test/000048/ … test/000059/                ← 12 BOP test scenes
        rgb/, depth/, mask_visib/, scene_gt.json, scene_camera.json
    test_targets_bop19.json                    ← official frame list

  Mount:  host /home/josue_aims_ac_za/ssd_4tb/dataset/ycbv → container /dataset/ycbv

YOLOE parameters for this dataset
-----------------------------------
conf=None (YOLOE's built-in default ~0.25), conf_fallbacks=(), use_first_det=True.
YCB objects appear on cluttered real-world backgrounds where YOLOE was trained,
so the default threshold works better than the low 0.1 used for LINEMOD.
use_first_det=True matches the original YCBV pipeline implementation.

Run inside Docker
-----------------
    # Full evaluation (all 21 objects)
    /opt/conda/envs/Any6D/bin/python3 /workspace/pipeline_scripts/run_full_pipeline_ycbv.py

    # Quick test — object 5 (mustard bottle), scene 52, 3 frames
    /opt/conda/envs/Any6D/bin/python3 /workspace/pipeline_scripts/run_full_pipeline_ycbv.py \\
        --obj_ids 5 --scene_id 52 --max_frames 3
"""
import os, sys, json, argparse, collections
from datetime import datetime
import numpy as np
import trimesh
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, '/workspace')
import nvdiffrast.torch as dr
from estimater import Any6D, ScorePredictor, PoseRefinePredictor
from pytorch_lightning import seed_everything

from core.constants import MM_TO_M, ADD_THRESH_RATIO
from core.llm import call_llm, parse_llm
from core.detection import get_detection_mask
from core.pose_utils import estimate_corrected_pose
from core.metrics_utils import nanmean
from core.io_utils import save_json
from core.readers import YCBVReader

# ── YCB object metadata ───────────────────────────────────────────────────────
YCBV_ROOT = "/dataset/ycbv"

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

YCBV_SYMMETRIC = {1, 13, 16, 18, 19, 20, 21}

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


def _extract_keyword(instruction: str, llm_model: str) -> str:
    raw = call_llm(instruction, CALIBRATED_SYSTEM, llm_model)
    return parse_llm(raw) or instruction.split()[-1]


# ── Optional BOP metrics ──────────────────────────────────────────────────────
try:
    from bop_toolkit_lib.pose_error_custom import mssd, mspd
    _BOP_METRICS = True
except Exception:
    _BOP_METRICS = False


def _compute_bop_ar(pred_q, gt_pose, mesh_vertices, diameter_m, K, W, H):
    if not _BOP_METRICS:
        return float('nan')
    try:
        sym = [{'R': np.eye(3), 't': np.zeros(3)}]
        ms  = float(mssd(pose_est=pred_q, pose_gt=gt_pose,
                         pts=mesh_vertices, syms=sym))
        mp  = float(mspd(pose_est=pred_q, pose_gt=gt_pose,
                         pts=mesh_vertices, K=K, syms=sym))
        return (float(ms < 0.2 * diameter_m) + float(mp < 0.1 * max(W, H))) / 2
    except Exception:
        return float('nan')


from metrics import compute_add, compute_adds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride",     type=int, default=1)
    parser.add_argument("--skip_llm",   action="store_true")
    parser.add_argument("--llm_model",  default="mistral:latest")
    parser.add_argument("--obj_ids",    type=int, nargs='+', default=None)
    parser.add_argument("--scene_id",   type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    seed_everything(0)

    run_name = f"stride{args.stride}"
    save_dir  = f"/workspace/results/ycbv_pipeline/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir : {save_dir}")
    print(f"BOP metrics: {_BOP_METRICS}")
    print(f"Stride   : {args.stride}  |  LLM: {'skip' if args.skip_llm else args.llm_model}\n")

    with open(f"{YCBV_ROOT}/models/models_info.json") as f:
        models_info = {int(k): v for k, v in json.load(f).items()}

    with open(f"{YCBV_ROOT}/ycbv/test_targets_bop19.json") as f:
        all_targets = json.load(f)
    obj_scene_frames = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in all_targets:
        obj_scene_frames[t['obj_id']][t['scene_id']].append(t['im_id'])
    for ob in obj_scene_frames:
        for sc in obj_scene_frames[ob]:
            obj_scene_frames[ob][sc] = sorted(set(obj_scene_frames[ob][sc]))

    obj_ids = args.obj_ids if args.obj_ids else sorted(YCBV_NAMES.keys())

    glctx     = dr.RasterizeCudaContext()
    mesh_init = trimesh.load(f"{YCBV_ROOT}/models/obj_000001.ply", force='mesh')
    mesh_init.apply_scale(MM_TO_M)
    est = Any6D(mesh=mesh_init, scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                debug_dir=save_dir, debug=0, glctx=glctx)

    all_summaries = []

    for obj_id in obj_ids:
        obj_name   = YCBV_NAMES.get(obj_id, f"obj{obj_id:02d}")
        is_sym     = obj_id in YCBV_SYMMETRIC
        diameter_m = models_info[obj_id]['diameter'] * MM_TO_M
        add_thresh = ADD_THRESH_RATIO * diameter_m

        print(f"\n{'='*60}")
        print(f"obj{obj_id:02d}  {obj_name}  (sym={is_sym}  diam={diameter_m*100:.1f}cm)")

        instruction = f"Find the {obj_name} on the table"
        if args.skip_llm:
            yoloe_kw = obj_name
        else:
            print(f"[LLM] \"{instruction}\"")
            yoloe_kw = _extract_keyword(instruction, args.llm_model)
        print(f"[LLM] → \"{yoloe_kw}\"")

        save_json(f"{save_dir}/obj_{obj_id:06d}_llm.json",
                  {'obj_id': obj_id, 'obj_name': obj_name,
                   'instruction': instruction, 'keyword': yoloe_kw})

        gt_mesh = trimesh.load(f"{YCBV_ROOT}/models/obj_{obj_id:06d}.ply", force='mesh')
        gt_mesh.apply_scale(MM_TO_M)
        est.reset_object(mesh=gt_mesh, symmetry_tfs=None)

        all_frames = []
        obj_add = []; obj_adds = []; obj_ar = []; obj_re = []; obj_te = []

        scenes = sorted(obj_scene_frames[obj_id].keys())
        if args.scene_id is not None:
            scenes = [s for s in scenes if s == args.scene_id]

        for scene_id in scenes:
            im_ids_all = obj_scene_frames[obj_id][scene_id]
            im_ids     = im_ids_all[::args.stride]
            if args.max_frames is not None:
                im_ids = im_ids[:args.max_frames]
            reader = YCBVReader(scene_id, obj_id)

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
                rgb     = reader.get_rgb(im_id)
                depth   = reader.get_depth(im_id)
                H, W    = rgb.shape[:2]
                gt_pose = reader.get_gt_pose(im_id)
                gt_mask = reader.get_mask_visib(im_id)

                mask, yoloe_det, _, iou = get_detection_mask(
                    rgb, yoloe_kw, gt_mask, H, W,
                    conf=None,          # use YOLOE built-in default (original behaviour)
                    use_first_det=True) # original took masks.data[0]

                pred_q = estimate_corrected_pose(
                    est, reader.K, rgb, depth, mask, pred_pose_a, gt_pose_a,
                    name=f"{obj_name}_{scene_id}_{im_id}")

                err_R  = float(np.degrees(np.arccos(np.clip(
                    (np.trace(pred_q[:3,:3] @ gt_pose[:3,:3].T) - 1) / 2, -1, 1))))
                err_T  = float(np.linalg.norm(pred_q[:3,3] - gt_pose[:3,3]) * 100)
                add_v  = float(compute_add(gt_mesh.vertices,  pred_q, gt_pose) < add_thresh)
                adds_v = float(compute_adds(gt_mesh.vertices, pred_q, gt_pose) < add_thresh)
                mean_ar = _compute_bop_ar(pred_q, gt_pose, gt_mesh.vertices,
                                          diameter_m, reader.K, W, H)

                obj_add.append(add_v); obj_adds.append(adds_v)
                obj_ar.append(mean_ar); obj_re.append(err_R); obj_te.append(err_T)

                all_frames.append({
                    'scene_id': scene_id, 'im_id': im_id,
                    'yoloe_detected': yoloe_det, 'iou': float(iou),
                    'R_pred': pred_q[:3,:3].tolist(), 'T_pred': pred_q[:3,3].tolist(),
                    'R_gt':   gt_pose[:3,:3].tolist(), 'T_gt': gt_pose[:3,3].tolist(),
                    'R_error': err_R, 'T_error': err_T,
                    'ADD': add_v, 'ADD-S': adds_v, 'AR': mean_ar,
                })

        metric_sym = 'ADD-S' if is_sym else 'ADD'
        add_score  = nanmean(obj_adds if is_sym else obj_add)

        summary = {
            'obj_id': obj_id, 'obj_name': obj_name, 'symmetric': is_sym,
            'instruction': instruction, 'keyword': yoloe_kw,
            'n_frames': len(all_frames),
            'ADD':            round(nanmean(obj_add)  * 100, 1),
            'ADD-S':          round(nanmean(obj_adds) * 100, 1),
            f'{metric_sym} (used)': round(add_score * 100, 1),
            'AR':             round(nanmean(obj_ar) * 100, 1),
            'R_error_mean':   round(nanmean(obj_re), 2),
            'R_error_med':    round(float(np.median(
                [v for v in obj_re if not np.isnan(v)])), 2)
                              if any(not np.isnan(v) for v in obj_re) else float('nan'),
            'T_error_mean':   round(nanmean(obj_te), 2),
            'yoloe_det_%':    round(sum(f['yoloe_detected'] for f in all_frames) /
                                    max(len(all_frames), 1) * 100, 1),
        }
        all_summaries.append(summary)

        print(f"  → {metric_sym}={summary[f'{metric_sym} (used)']}%  "
              f"R_med={summary['R_error_med']}°  T_mean={summary['T_error_mean']}cm")

        save_json(f"{save_dir}/obj_{obj_id:06d}_poses.json", {
            'obj_id': obj_id, 'obj_name': obj_name,
            'instruction': instruction, 'keyword': yoloe_kw,
            'summary': summary, 'frames': all_frames,
        })

    # ── Global XLSX/JSON ───────────────────────────────────────────────────────
    try:
        add_used = [r.get('ADD-S (used)', r.get('ADD (used)', float('nan')))
                    for r in all_summaries]
        mean_row = {
            'obj_id': 'MEAN', 'obj_name': '—', 'symmetric': '—',
            'n_frames':     sum(r['n_frames'] for r in all_summaries),
            'ADD':          round(nanmean([r['ADD']          for r in all_summaries]), 1),
            'ADD-S':        round(nanmean([r['ADD-S']        for r in all_summaries]), 1),
            'ADD(S) used':  round(nanmean(add_used), 1),
            'AR':           round(nanmean([r['AR']           for r in all_summaries]), 1),
            'R_error_mean': round(nanmean([r['R_error_mean'] for r in all_summaries]), 2),
            'R_error_med':  round(nanmean([r['R_error_med']  for r in all_summaries]), 2),
            'T_error_mean': round(nanmean([r['T_error_mean'] for r in all_summaries]), 2),
            'yoloe_det_%':  round(nanmean([r['yoloe_det_%']  for r in all_summaries]), 1),
        }
        pd.DataFrame(all_summaries + [mean_row]).to_excel(
            f"{save_dir}/ycbv_SUMMARY.xlsx", index=False)
        save_json(f"{save_dir}/ycbv_SUMMARY.json", {
            'dataset': 'YCB-Video BOP', 'run': run_name,
            'mean_ADD(S)': mean_row['ADD(S) used'],
            'per_object': {r['obj_name']: r for r in all_summaries},
        })
        print(f"\nMEAN ADD(S) = {mean_row['ADD(S) used']}%")
        print(f"Saved → {save_dir}/ycbv_SUMMARY.xlsx")
    except Exception as e:
        print(f"XLSX failed: {e}")

    print(f"\nDone. Results in {save_dir}/")


if __name__ == "__main__":
    main()
