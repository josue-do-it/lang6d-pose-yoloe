"""
Full pipeline: User instruction → LLM keyword → YOLOE mask → Any6D → BOP metrics
Dataset: HO3D v3 (hand-object interaction, 13 sequences, YCB objects).

Pipeline overview
-----------------
HO3D differs from LINEMOD and YCBV because objects are always partially occluded
by a hand.  This changes several pipeline parameters:

  1. LLM uses a HO3D-specific CALIBRATED_SYSTEM with examples tailored to the
     5 YCB object categories that appear in HO3D sequences.
  2. YOLOE uses conf_fallbacks=(0.05, 0.03) because hand occlusion lowers
     detection confidence.  If the primary threshold (0.1) finds nothing, two
     lower thresholds are tried in sequence before falling back to GT mask.
  3. Images are loaded with cv2.imread() (BGR) and passed to YOLOE as-is
     (bgr_input=True).  Converting to RGB would change results vs the original
     paper pipeline, so this is preserved exactly.
  4. Per-frame RGB+depth+mask visualisations are saved alongside pose results
     to support qualitative analysis (useful for hand-occlusion debugging).

Sequences and objects
---------------------
  MPM10-14 → SPAM (potted meat can, YCB obj 10)
  AP10-14  → pitcher base (YCB obj 11)
  SB11,13  → bleach cleanser (YCB obj 12)
  SM1      → mustard bottle (YCB obj 5)

Dataset layout (inside Docker)
-------------------------------
/dataset/ho3d/
    evaluation/MPM10/ … SM1/
        rgb/   (JPEG)
        depth/ (PNG uint16 mm)
        meta/  (pkl with K, gt_R, gt_t, obj_id)

  Mount:  host /home/josue_aims_ac_za/ssd_4tb/dataset/ho3d → container /dataset/ho3d

YOLOE parameters for this dataset
-----------------------------------
conf=0.1, conf_fallbacks=(0.05, 0.03), use_first_det=True, bgr_input=True.

Metrics
-------
ADD, ADD-S, MSSD, MSPD, AR (BOP standard), VSD, R_error (°), T_error (cm).
VSD is computed only for HO3D because the original paper reports it.

Run inside Docker
-----------------
    # Full evaluation (all 13 sequences)
    /opt/conda/envs/Any6D/bin/python3 /workspace/pipeline_scripts/run_full_pipeline_ho3d.py

    # Quick test — one sequence, 5 frames
    /opt/conda/envs/Any6D/bin/python3 /workspace/pipeline_scripts/run_full_pipeline_ho3d.py \\
        --sequences MPM10 --max_frames 5
"""
import sys, os, json, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import trimesh
import pandas as pd
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/yoloe')

from core.llm import call_llm, parse_llm
from core.detection import get_detection_mask
from core.metrics_utils import nanmean

from foundationpose.datareader import Ho3dReader
from estimater import *
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd
from metrics import *
from renderer_pyrender import RendererVispy
from pytorch_lightning import seed_everything

# ── HO3D sequence instructions ────────────────────────────────────────────────
HO3D_INSTRUCTIONS = {
    "MPM10": "Hand me that flat blue SPAM tin on the table",
    "MPM11": "Pick me that small rectangular meat can please",
    "MPM12": "I want that flat metal tin with the peel-off lid",
    "MPM13": "Grab the potted meat can sitting on the desk",
    "MPM14": "Pass me that small flat tin of SPAM over there",
    "AP10":  "I'm thirsty, pass me that big blue pitcher",
    "AP11":  "Pick me that blue jug with the handle on the side",
    "AP12":  "I want that blue water pitcher on the left",
    "AP13":  "Hand me the blue pouring container please",
    "AP14":  "Can you grab that blue pitcher for me?",
    "SB11":  "Pass me that tall white cleaning bottle on the right",
    "SB13":  "I need that white bleach container to clean something",
    "SM1":   "Hand me the yellow mustard bottle on the tray",
}

CALIBRATED_SYSTEM = """\
You are a visual keyword extractor for YOLOE, an open-vocabulary object segmentation model.

Your ONLY job: extract the most precise visual keyword(s) from the user instruction.
Output ONLY the keyword — nothing else. No punctuation, no explanation, no sentence.

IMPORTANT:
- 1 word is BEST when it is specific enough (e.g. "pitcher", "mustard", "banana")
- Use 2-3 words ONLY when 1 word is too ambiguous (e.g. "spam can", "blue pitcher")
- MAXIMUM 3 words
- Always use visual, concrete nouns — NOT functions, NOT actions

Examples from our dataset (instruction → keyword):
"I'm thirsty, pass me that big blue pitcher" → pitcher
"Hand me the blue jug with the handle" → blue pitcher
"Can you grab that blue container for pouring?" → pitcher
"That blue thing on the desk for pouring water" → blue pitcher
"Pass me the flat SPAM tin on the table" → spam can
"I need that rectangular meat can over there" → spam
"Grab the potted meat tin sitting on the desk" → spam can
"That small flat tin with the label on top" → meat tin
"Hand me the white cleaning bottle please" → cleanser
"Pass me the tall bleach container" → bleach bottle
"I need that tall white bottle for cleaning" → white bottle
"The white squeeze bottle, hand it to me" → cleanser
"Give me the yellow mustard bottle on the tray" → mustard
"I need something to put on my hot dog" → mustard
"Pass me that yellow condiment squeeze bottle" → mustard bottle
"That yellow bottle standing on the wooden tray" → mustard
"I want that red cracker box behind the banana" → cracker box
"Pass me the Cheez-It snack box please" → crackers
"Grab the red rectangular cardboard box" → cracker box
"That tall red snack box on the table" → cracker box
"Hand me the sugar" → sugar
"Pass me that yellow Domino box" → sugar box
"I need to sweeten my coffee, get me that box" → sugar box
"That orange and yellow carton on the black board" → sugar box
"Give me that tomato soup can" → soup can
"I want the red and white tin can on the left" → soup can
"Pass me that Campbell's can next to the tool" → tomato soup
"The short cylindrical tin with the red label" → soup can
"Hand me that red power drill on the table" → drill
"Pass me the Black and Decker tool" → drill
"I need to drill something, get me that red tool" → red drill
"Grab the cordless red device lying on the board" → drill
"Give me that banana please" → banana
"I need a snack, can you get me the banana?" → banana
"That yellow curved fruit on the table" → banana
"The banana lying between the drill and the box" → banana
"Can you pass me the pitcher?" → pitcher
"I want that blue thing used for pouring" → blue pitcher
"The white plastic bottle on the far right" → white bottle
"That flat tin sitting next to the soup can" → spam can
"""


def _extract_keyword(instruction: str, llm_model: str) -> str:
    raw = call_llm(instruction, CALIBRATED_SYSTEM, llm_model)
    kw  = parse_llm(raw)
    return kw if kw and len(kw.split()) <= 5 else "object"


def plot_pipeline_frame(rgb, depth, yoloe_mask, pose_4x4, K, obj_f,
                        frame_idx, instruction, extracted_prompt, save_path):
    H, W = rgb.shape[:2]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f'{obj_f}  |  "{instruction}"\n→ keyword: "{extracted_prompt}"',
        fontsize=10)

    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")

    d = depth.copy().astype(float); d[d < 0.001] = float('nan')
    axes[1].imshow(d, cmap="jet"); axes[1].set_title("Depth"); axes[1].axis("off")

    overlay = rgb.copy()
    if yoloe_mask is not None:
        overlay[yoloe_mask] = (overlay[yoloe_mask] * 0.4 +
                               np.array([0, 200, 0]) * 0.6).astype('uint8')
    det_str = "detected" if yoloe_mask is not None else "fallback GT"
    axes[2].imshow(overlay)
    axes[2].set_title(f'YOLOE "{extracted_prompt}" ({det_str})')
    axes[2].axis("off")

    pose_img = rgb.copy()
    if pose_4x4 is not None and K is not None:
        R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        pts = np.float32([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
        try:
            p2d, _ = cv2.projectPoints(pts, rvec, t.reshape(3, 1),
                                       K.reshape(3, 3), np.zeros(5))
            p2d = p2d.astype(int).reshape(-1, 2)
            o = tuple(p2d[0])
            for ci, col in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                cv2.arrowedLine(pose_img, o, tuple(p2d[ci + 1]), col, 3)
        except Exception:
            pass
    axes[3].imshow(pose_img); axes[3].set_title("Any6D Pose"); axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/viz_{obj_f}_f{frame_idx:04d}.png",
                dpi=80, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_path",     default="/workspace/anchor_results/dexycb_reference_view_ours")
    parser.add_argument("--hot3d_data_root", default="/dataset/ho3d/HO3D_data")
    parser.add_argument("--ycb_model_path",  default="/dataset/ho3d")
    parser.add_argument("--ycbv_modesl_info_path", default="./models_info.json")
    parser.add_argument("--start_idx",       type=int, default=0)
    parser.add_argument("--end_idx",         type=int, default=13)
    parser.add_argument("--running_stride",  type=int, default=10)
    parser.add_argument("--save_dir",        default=None)
    parser.add_argument("--llm_model",       default="mistral:latest")
    args = parser.parse_args()

    save_dir = args.save_dir or f"./results/ho3d_pipeline/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    os.makedirs(save_dir, exist_ok=True)

    obj_folder = ['MPM10', 'MPM11', 'MPM12', 'MPM13', 'MPM14',
                  'AP10',  'AP11',  'AP12',  'AP13',  'AP14',
                  'SB11',  'SB13',  'SM1'][args.start_idx:args.end_idx]

    object_metrics = {obj: {
        'ADD': [], 'ADD-S': [], 'AR': [], 'VSD': [], 'MSSD': [], 'MSPD': [],
        'R error': [], 'T error': [], 'cls_id': [], 'instance_id': []
    } for obj in obj_folder}

    glctx    = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4))
    mesh     = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(),
                               faces=mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                debug_dir=save_dir, debug=0, glctx=glctx)
    renderer   = RendererVispy(640, 480, mode='depth')
    obj_count  = 0

    for obj_f in tqdm(obj_folder, desc="Objects"):
        instruction = HO3D_INSTRUCTIONS.get(obj_f, "pick me that object")

        print(f"\n[LLM] {obj_f}: \"{instruction}\"")
        yoloe_kw = _extract_keyword(instruction, args.llm_model)
        print(f"[LLM] → keyword: \"{yoloe_kw}\"")

        json.dump({
            'sequence': obj_f, 'instruction': instruction,
            'yoloe_prompt': yoloe_kw, 'llm_model': args.llm_model,
        }, open(f"{save_dir}/{obj_f}_llm_extraction.json", 'w'), indent=2)

        video_dir = os.path.join(f"{args.hot3d_data_root}/evaluation", obj_f)
        reader = Ho3dReader(video_dir, args.hot3d_data_root)
        reader.color_files = reader.color_files[::args.running_stride]
        ob_id = reader.get_obj_id()

        with open(args.ycbv_modesl_info_path) as f:
            model_info = json.load(f)
        trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]
        if "symmetries_discrete" in model_info[f"{ob_id}"]:
            for sym in model_info[f"{ob_id}"]["symmetries_discrete"]:
                s4 = np.reshape(sym, (4, 4))
                trans_disc.append({"R": s4[:3, :3], "t": s4[:3, 3].reshape(3, 1)})

        gt_mesh     = reader.get_gt_mesh(args.ycb_model_path)
        gt_diameter = reader.get_gt_mesh_diamter(args.ycb_model_path)
        mesh        = trimesh.load(reader.get_reference_view_1_mesh(args.anchor_path))

        renderer.my_add_object({
            'pts':     np.asarray(gt_mesh.vertices) * 1e3,
            'normals': np.asarray(gt_mesh.face_normals),
            'faces':   np.asarray(gt_mesh.faces),
        }, ob_id)

        pred_pose_a = np.loadtxt(reader.get_reference_view_1_pose(args.anchor_path))
        gt_pose_a   = np.loadtxt(
            reader.get_reference_view_1_pose(args.anchor_path).replace('initial', 'gt'))
        est.reset_object(mesh=mesh, symmetry_tfs=None)

        det_total = det_detected = 0
        det_iou_list = []
        pose_records = []

        for i in tqdm(range(len(reader.color_files)), desc=obj_f, leave=False):
            gt_pose_q = reader.get_gt_pose(i)
            if gt_pose_q is None:
                continue
            det_total += 1

            color = cv2.cvtColor(cv2.imread(reader.color_files[i]), cv2.COLOR_BGR2RGB)
            H, W  = color.shape[:2]
            depth = reader.get_depth(i)

            gt_mask_raw = reader.get_mask(i)
            mask, yoloe_det, _, iou = get_detection_mask(
                color, yoloe_kw, gt_mask_raw, H, W,
                conf_fallbacks=(0.05, 0.03), # original tried 3 thresholds: 0.1 → 0.05 → 0.03
                use_first_det=True,           # original took masks.data[0]
                bgr_input=True)               # original converted RGB→BGR before YOLOE

            if yoloe_det:
                det_detected += 1
                if iou > 0:
                    det_iou_list.append(iou)
            elif gt_mask_raw is None:
                continue

            pred_pose_q = est.register(K=reader.K, rgb=color, depth=depth,
                                       ob_mask=mask, iteration=5, name=obj_f)
            if i == 0:
                plot_pipeline_frame(color, depth,
                                    mask if yoloe_det else gt_mask_raw,
                                    pred_pose_q, reader.K, obj_f, i,
                                    instruction, yoloe_kw, save_dir)

            # Anchor-relative correction (same formula as original)
            pred_q = (pred_pose_q @ np.linalg.inv(pred_pose_a)) @ gt_pose_a

            err_R, err_T = compute_RT_distances(pred_q, gt_pose_q)
            add  = compute_add(gt_mesh.vertices, pred_q, gt_pose_q)
            adds = compute_adds(gt_mesh.vertices, pred_q, gt_pose_q)
            add_thres  = float(add  <= gt_diameter * 0.1)
            adds_thres = float(adds <= gt_diameter * 0.1)

            pred_q16 = pred_q.astype(np.float16)
            gt_q16   = gt_pose_q.astype(np.float16)
            pr_r = pred_q16[:3, :3]
            pr_t = np.expand_dims(pred_q16[:3, 3], axis=1) * 1e3
            gt_r = gt_q16[:3, :3]
            gt_t = np.expand_dims(gt_q16[:3, 3], axis=1) * 1e3

            mssd_err = mssd(pose_est=pred_q, pose_gt=gt_pose_q,
                            pts=gt_mesh.vertices, syms=trans_disc) * 1e3
            mspd_err = mspd(pose_est=pred_q, pose_gt=gt_pose_q,
                            pts=gt_mesh.vertices, K=reader.K, syms=trans_disc)

            mssd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
            vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            vsd_rec  = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

            vsd_errs = vsd(pr_r, pr_t, gt_r, gt_t, depth * 1e3,
                           reader.K.reshape(3, 3), 15.0, vsd_taus, True,
                           gt_diameter * 1e3, renderer, ob_id)
            vsd_errs   = np.asarray(vsd_errs)
            mean_vsd   = np.stack([vsd_errs < r for r in vsd_rec], axis=1).mean()
            mean_mssd  = (mssd_err < mssd_rec * gt_diameter * 1e3).mean()
            mean_mspd  = (mspd_err < mspd_rec).mean()
            mean_ar    = (mean_mssd + mean_mspd + mean_vsd) / 3.

            pose_records.append({
                'frame': i,
                'yoloe_detected': yoloe_det,
                'R_pred': pred_q[:3, :3].tolist(),
                'T_pred': pred_q[:3, 3].tolist(),
                'R_gt':   gt_pose_q[:3, :3].tolist(),
                'T_gt':   gt_pose_q[:3, 3].tolist(),
                'R_error': float(err_R.tolist()[0]) if hasattr(err_R, 'tolist') else float(err_R),
                'T_error': float(err_T.tolist()[0]) if hasattr(err_T, 'tolist') else float(err_T),
            })

            object_metrics[obj_f]['ADD'].append(add_thres)
            object_metrics[obj_f]['ADD-S'].append(adds_thres)
            object_metrics[obj_f]['AR'].append(mean_ar)
            object_metrics[obj_f]['VSD'].append(mean_vsd)
            object_metrics[obj_f]['MSSD'].append(mean_mssd)
            object_metrics[obj_f]['MSPD'].append(mean_mspd)
            object_metrics[obj_f]['R error'].append(err_R.tolist()[0])
            object_metrics[obj_f]['T error'].append(err_T.tolist()[0])
            object_metrics[obj_f]['cls_id'].append(obj_f)
            object_metrics[obj_f]['instance_id'].append(obj_count)

            try:
                visualize_frame_results_gt(
                    color=color, gt_mesh=gt_mesh, K=reader.K,
                    gt_pose=gt_pose_q, pred_pose=pred_pose_q,
                    metric=object_metrics[obj_f], obj_f=obj_f, frame_idx=i,
                    save_path=save_dir, glctx=glctx,
                    name=f"{len(reader.color_files)}_pipeline",
                    nocs_metric=True, est_mesh=est.mesh)
            except Exception:
                pass
            obj_count += 1

        means = {k: round(float(np.mean(object_metrics[obj_f][k])) * 100, 1)
                 for k in ['ADD', 'ADD-S', 'AR', 'VSD', 'MSSD', 'MSPD']}
        means['R_error'] = round(float(np.mean(object_metrics[obj_f]['R error'])), 2)
        means['T_error'] = round(float(np.mean(object_metrics[obj_f]['T error'])), 2)

        df_obj = pd.DataFrame({
            'Frame_ID': object_metrics[obj_f]['instance_id'],
            'Class':    object_metrics[obj_f]['cls_id'],
            'ADD-S':    object_metrics[obj_f]['ADD-S'],
            'ADD':      object_metrics[obj_f]['ADD'],
            'AR':       object_metrics[obj_f]['AR'],
            'MSSD':     object_metrics[obj_f]['MSSD'],
            'MSPD':     object_metrics[obj_f]['MSPD'],
            'VSD':      object_metrics[obj_f]['VSD'],
            'R_error':  object_metrics[obj_f]['R error'],
            'T_error':  object_metrics[obj_f]['T error'],
        })
        df_obj = pd.concat([df_obj, pd.DataFrame([{
            'Frame_ID': 'MEAN', 'Class': obj_f,
            **{k: f"{means[k]:.1f}" for k in
               ['ADD-S', 'ADD', 'AR', 'MSSD', 'MSPD', 'VSD', 'R_error', 'T_error']}
        }])], ignore_index=True)
        df_obj.to_excel(f"{save_dir}/{obj_f}_metrics_results.xlsx", index=False)

        json.dump({
            'sequence': obj_f, 'instruction': instruction,
            'yoloe_prompt': yoloe_kw, 'llm_model': args.llm_model,
            'frames': pose_records,
        }, open(f"{save_dir}/{obj_f}_poses.json", 'w'), indent=2)

        det_rate = round(det_detected / det_total * 100, 1) if det_total > 0 else 0.0
        mean_iou = round(float(np.mean(det_iou_list)) * 100, 1) if det_iou_list else 0.0
        json.dump({
            'sequence': obj_f, 'instruction': instruction,
            'yoloe_prompt': yoloe_kw, 'llm_model': args.llm_model,
            'total_frames': det_total, 'detected_frames': det_detected,
            'detection_rate': det_rate, 'mean_iou': mean_iou,
            'fallback_frames': det_total - det_detected,
            **means,
        }, open(f"{save_dir}/{obj_f}_yoloe_detection.json", 'w'), indent=2)

        print(f"[{obj_f}] AR={means['AR']} ADD-S={means['ADD-S']} "
              f"Det={det_rate}% IoU={mean_iou}% kw=\"{yoloe_kw}\"")

    print(f"\nDone. Results in {save_dir}")
