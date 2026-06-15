"""
main_pipeline.py — Run the full pipeline on a single image.

This file centralises all calibrated instructions (per dataset) and lets you
test the complete pipeline (LLM → YOLOE → Any6D) on one image and get back
the 4×4 pose matrix + a visualisation figure.

Run inside Docker:
    # Minimal (no anchor):
    /opt/conda/envs/Any6D/bin/python3 /workspace/pipeline_scripts/main_pipeline.py \
        --image       /path/to/rgb.png \
        --depth       /path/to/depth.png \
        --mesh        /path/to/object.ply \
        --K           572.4 0 325.2 0 572.4 242.0 0 0 1 \
        --instruction "Hand me the mustard bottle"

    # With anchor correction:
    /opt/conda/envs/Any6D/bin/python3 /workspace/pipeline_scripts/main_pipeline.py \
        --image        /path/to/rgb.png \
        --depth        /path/to/depth.png \
        --mesh         /path/to/object.ply \
        --K            572.4 0 325.2 0 572.4 242.0 0 0 1 \
        --instruction  "Hand me the mustard bottle" \
        --anchor_image /path/to/anchor_rgb.png \
        --anchor_depth /path/to/anchor_depth.png \
        --anchor_mask  /path/to/anchor_mask.png \
        --anchor_pose  /path/to/gt_anchor_pose.txt

    # Use a dataset instruction (e.g. LINEMOD object 8 = driller):
    /opt/conda/envs/Any6D/bin/python3 /workspace/pipeline_scripts/main_pipeline.py \
        --image   /path/to/rgb.png \
        --depth   /path/to/depth.png \
        --mesh    /dataset/lm/models/obj_000008.ply \
        --K       572.4 0 325.2 0 572.4 242.0 0 0 1 \
        --dataset lm --obj_id 8

Outputs (saved to --out_dir, default /tmp/main_pipeline_out):
    pose_4x4.txt      — 4×4 pose matrix (text format)
    result.json       — full summary (keyword, pose, confidence)
    viz_pipeline.png  — 4-panel figure: RGB | Depth | YOLOE mask | Pose axes
"""

import sys, os, argparse
import numpy as np
import cv2
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/foundationpose/mycpp/build')
import nvdiffrast.torch as dr
from estimater import Any6D, ScorePredictor, PoseRefinePredictor
from pytorch_lightning import seed_everything

from core.llm import call_llm, parse_llm
from core.detection import get_detection_mask
from core.pose_utils import estimate_corrected_pose
from core.io_utils import save_result
from core.constants import MM_TO_M, ANY6D_ITERS


# ── Calibrated system prompts (per dataset) ───────────────────────────────────

SYSTEM_PROMPTS = {

    'lm': """\
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
""",

    'ycbv': """\
You are a visual keyword extractor for YOLOE, an open-vocabulary object segmentation model.
Your ONLY job: extract the most precise visual keyword(s) from the user instruction.
Output ONLY the keyword — nothing else. No punctuation, no explanation, no sentence.
- 1 word is BEST when specific enough (e.g. "banana", "mug", "scissors")
- Use 2 words ONLY when ambiguous (e.g. "mustard bottle", "power drill")
- MAXIMUM 3 words
Examples:
"Find the master chef can on the table" → chef can
"Find the cracker box" → cracker box
"Find the mustard bottle on the table" → mustard bottle
"Pick up the tuna fish can" → tuna can
"Find the banana" → banana
"Hand me the mug" → mug
"I need the power drill" → power drill
"Find the wood block" → wood block
"Hand me the scissors" → scissors
"Find the foam brick" → foam brick
""",

    'ho3d': """\
You are a visual keyword extractor for YOLOE, an open-vocabulary object segmentation model.
Your ONLY job: extract the most precise visual keyword(s) from the user instruction.
Output ONLY the keyword — nothing else. No punctuation, no explanation, no sentence.
- 1 word is BEST when specific enough (e.g. "pitcher", "mustard", "banana")
- Use 2-3 words ONLY when 1 word is too ambiguous (e.g. "spam can", "blue pitcher")
- MAXIMUM 3 words
Examples:
"I'm thirsty, pass me that big blue pitcher" → pitcher
"Hand me the blue jug with the handle" → blue pitcher
"Pass me the flat SPAM tin on the table" → spam can
"Grab the potted meat tin sitting on the desk" → spam can
"Hand me the white cleaning bottle please" → cleanser
"Pass me the tall bleach container" → bleach bottle
"Give me the yellow mustard bottle on the tray" → mustard
"Can you pass me the pitcher?" → pitcher
""",

}


# ── Calibrated instructions per dataset and object ────────────────────────────

DATASET_INSTRUCTIONS = {

    'lm': {
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
    },

    'ycbv': {
        1:  "Find the master chef can on the table",
        2:  "Find the cracker box",
        3:  "Find the sugar box",
        4:  "Hand me the tomato soup can",
        5:  "Find the mustard bottle on the table",
        6:  "Pick up the tuna fish can",
        7:  "Find the pudding box",
        8:  "Hand me the gelatin box",
        9:  "Pick up the potted meat can",
        10: "Find the banana",
        11: "Hand me the pitcher",
        12: "Grab the bleach cleanser",
        13: "Pass me the bowl",
        14: "Hand me the mug",
        15: "I need the power drill",
        16: "Find the wood block",
        17: "Hand me the scissors",
        18: "Find the large marker",
        19: "Hand me the large clamp",
        20: "Find the extra large clamp",
        21: "Find the foam brick",
    },

    'ho3d': {
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
    },

}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_depth(path: str, scale: float = MM_TO_M) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    return depth.astype(np.float32) * scale


def _build_K(vals: list) -> np.ndarray:
    """Accept 4 values (fx fy cx cy) or 9 values (row-major 3×3 matrix)."""
    if len(vals) == 4:
        fx, fy, cx, cy = vals
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if len(vals) == 9:
        return np.array(vals, dtype=np.float64).reshape(3, 3)
    raise ValueError(f"--K expects 4 or 9 values, got {len(vals)}")


def _clip_endpoint(o, p, W, H):
    ox, oy = float(o[0]), float(o[1])
    px, py = float(p[0]), float(p[1])
    dx, dy = px - ox, py - oy
    t = 1.0
    margin = 8
    if dx > 0:   t = min(t, (W - 1 - margin - ox) / dx)
    elif dx < 0: t = min(t, (margin - ox) / dx)
    if dy > 0:   t = min(t, (H - 1 - margin - oy) / dy)
    elif dy < 0: t = min(t, (margin - oy) / dy)
    if t < 1.0:
        t *= 0.6
    t = max(0.0, min(1.0, t))
    return (int(ox + t * dx), int(oy + t * dy))


def _draw_axes(img_rgb: np.ndarray, pose: np.ndarray, K: np.ndarray,
               axis_size: float = 0.25) -> np.ndarray:
    overlay = img_rgb.copy()
    H, W = overlay.shape[:2]
    try:
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        tvec = pose[:3, 3].reshape(3, 1)
        ax3d = np.float32([[0,0,0],[axis_size,0,0],[0,axis_size,0],[0,0,axis_size]])
        pts2d, _ = cv2.projectPoints(ax3d, rvec, tvec, K.reshape(3,3), np.zeros(5))
        pts2d = pts2d.astype(int).reshape(-1, 2)
        o  = tuple(pts2d[0])
        px = _clip_endpoint(o, tuple(pts2d[1]), W, H)
        py = _clip_endpoint(o, tuple(pts2d[2]), W, H)
        pz = _clip_endpoint(o, tuple(pts2d[3]), W, H)
        th, tip = 4, 0.10
        for pt, col in [(px,(220,30,30)),(py,(30,210,30)),(pz,(30,30,220))]:
            cv2.arrowedLine(overlay, o, pt, (0,0,0), th+3, tipLength=tip, line_type=cv2.LINE_AA)
            cv2.arrowedLine(overlay, o, pt, col,     th,   tipLength=tip, line_type=cv2.LINE_AA)
        cv2.circle(overlay, o, 8, (255,255,255), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, o, 8, (30,30,30),    2,  lineType=cv2.LINE_AA)
    except Exception as e:
        print(f'[axes] {e}')
    return overlay


def _save_visualisation(rgb, depth, mask, pose, K, keyword,
                        instruction, yoloe_detected, out_path):
    """4-panel figure: RGB | Depth | YOLOE mask overlay | Pose axes."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f'Instruction: "{instruction}"\n'
        f'→ YOLOE keyword: "{keyword}"  |  detected: {yoloe_detected}',
        fontsize=10)

    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")

    d = depth.copy().astype(float); d[d < 0.001] = float('nan')
    axes[1].imshow(d, cmap="jet"); axes[1].set_title("Depth"); axes[1].axis("off")

    overlay = rgb.copy()
    if mask is not None:
        overlay[mask] = (overlay[mask] * 0.4 +
                         np.array([0, 200, 0]) * 0.6).astype(np.uint8)
    src = "YOLOE" if yoloe_detected else "GT/fallback"
    axes[2].imshow(overlay); axes[2].set_title(f"Mask ({src})"); axes[2].axis("off")

    pose_img = rgb.copy()
    pose_text = ""
    if pose is not None and K is not None:
        R, t = pose[:3, :3], pose[:3, 3]
        pose_img = _draw_axes(pose_img, pose, K)
        cos_val = np.clip((np.trace(R) - 1) / 2, -1, 1)
        r_deg = float(np.degrees(np.arccos(cos_val)))
        pose_text = (
            f"tx={t[0]*100:+.1f}cm  ty={t[1]*100:+.1f}cm  tz={t[2]*100:+.1f}cm\n"
            f"R∠identity={r_deg:.1f}°"
        )
    axes[3].imshow(pose_img)
    axes[3].set_title(f"Any6D Pose (X=R G=Y B=Z)\n{pose_text}", fontsize=9)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualisation → {out_path}")


def _print_pose(pose: np.ndarray):
    print("\n── Pose 4×4 (metres) ──")
    for row in pose:
        print("  " + "  ".join(f"{v:+.6f}" for v in row))
    t = pose[:3, 3]
    R = pose[:3, :3]
    cos_val = np.clip((np.trace(R) - 1) / 2, -1, 1)
    print(f"\n  Translation: [{t[0]*100:+.2f}cm, {t[1]*100:+.2f}cm, {t[2]*100:+.2f}cm]")
    print(f"  Rotation from identity: {np.degrees(np.arccos(cos_val)):.1f}°")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full pipeline on a single image")
    p.add_argument("--image",       required=True,  help="RGB image (.png/.jpg)")
    p.add_argument("--depth",       required=True,  help="Depth image (.png)")
    p.add_argument("--mesh",        required=True,  help="Object mesh (.ply)")
    p.add_argument("--K",           type=float, nargs='+', required=True,
                   help="Camera intrinsics: 4 values (fx fy cx cy) or 9 (3×3 row-major)")
    p.add_argument("--instruction", default=None,
                   help="Natural language instruction")
    p.add_argument("--dataset",     choices=['lm', 'ycbv', 'ho3d'], default=None,
                   help="Dataset to select calibrated prompt and instruction")
    p.add_argument("--obj_id",      default=None,
                   help="Object ID or sequence key (used with --dataset)")
    p.add_argument("--depth_scale", type=float, default=MM_TO_M,
                   help=f"Depth→metres scale factor (default: {MM_TO_M})")
    p.add_argument("--anchor_image", default=None, help="Anchor RGB image")
    p.add_argument("--anchor_depth", default=None, help="Anchor depth image")
    p.add_argument("--anchor_mask",  default=None, help="Anchor binary mask")
    p.add_argument("--anchor_pose",  default=None, help="Anchor GT pose (.txt 4×4)")
    p.add_argument("--skip_llm",    action="store_true",
                   help="Use instruction directly as YOLOE keyword (skip LLM)")
    p.add_argument("--llm_model",   default="mistral:latest")
    p.add_argument("--out_dir",     default="/tmp/main_pipeline_out")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(0)
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve instruction
    instruction = args.instruction
    if instruction is None:
        if args.dataset and args.obj_id:
            key = int(args.obj_id) if args.obj_id.isdigit() else args.obj_id
            instruction = DATASET_INSTRUCTIONS.get(args.dataset, {}).get(key)
            if instruction is None:
                raise ValueError(
                    f"obj_id '{args.obj_id}' not found in dataset '{args.dataset}'")
        else:
            raise ValueError("Provide --instruction or (--dataset + --obj_id)")

    system_prompt = SYSTEM_PROMPTS.get(args.dataset or 'ycbv')
    print(f"\nInstruction : \"{instruction}\"")

    # LLM keyword extraction
    if args.skip_llm:
        keyword = instruction
        print(f"[LLM skip] keyword = \"{keyword}\"")
    else:
        raw     = call_llm(instruction, system_prompt, args.llm_model)
        keyword = parse_llm(raw) or instruction.split()[-1]
        print(f"[LLM] raw: \"{raw}\"")
        print(f"[LLM] → keyword: \"{keyword}\"")

    # Load data
    K     = _build_K(args.K)
    rgb   = _load_rgb(args.image)
    depth = _load_depth(args.depth, args.depth_scale)
    H, W  = rgb.shape[:2]
    print(f"\nImage: {W}×{H}  |  Mesh: {args.mesh}")

    # YOLOE detection
    mask, yoloe_det, conf, _ = get_detection_mask(rgb, keyword, None, H, W)
    if yoloe_det:
        print(f"[YOLOE] Detected  (conf={conf:.2f})")
    else:
        print("[YOLOE] Not detected — using full-image fallback mask")

    # Load mesh and initialise estimator
    mesh = trimesh.load(args.mesh, force='mesh')
    if mesh.vertices.max() > 10:
        mesh.apply_scale(MM_TO_M)  # mm → m
        print(f"[Mesh] Vertices max > 10 → applied scale ×{MM_TO_M} (mm→m)")

    glctx = dr.RasterizeCudaContext()
    est   = Any6D(mesh=mesh, scorer=ScorePredictor(),
                  refiner=PoseRefinePredictor(),
                  debug_dir=args.out_dir, debug=0, glctx=glctx)

    # Pose estimation
    has_anchor = all([args.anchor_image, args.anchor_depth, args.anchor_pose])

    if has_anchor:
        anc_rgb   = _load_rgb(args.anchor_image)
        anc_depth = _load_depth(args.anchor_depth, args.depth_scale)
        anc_mask  = None
        if args.anchor_mask:
            m = cv2.imread(args.anchor_mask, cv2.IMREAD_GRAYSCALE)
            anc_mask = (m > 0) if m is not None else None
        if anc_mask is None:
            aH, aW = anc_rgb.shape[:2]
            anc_mask = np.ones((aH, aW), dtype=bool)

        gt_pose_anchor   = np.loadtxt(args.anchor_pose)
        pred_pose_anchor = est.register(
            K=K, rgb=anc_rgb, depth=anc_depth,
            ob_mask=anc_mask, iteration=ANY6D_ITERS, name="anchor")

        pose = estimate_corrected_pose(
            est, K, rgb, depth, mask,
            pred_pose_anchor, gt_pose_anchor, name="query")
        print("\n[Pose] Relative anchor correction applied")
    else:
        pose = est.register(
            K=K, rgb=rgb, depth=depth,
            ob_mask=mask, iteration=ANY6D_ITERS, name="query")
        print("\n[Pose] Direct estimation (no anchor)")

    # Print and save results
    _print_pose(pose)

    pose_path = os.path.join(args.out_dir, "pose_4x4.txt")
    np.savetxt(pose_path, pose, fmt="%.8f")
    print(f"\nPose saved → {pose_path}")

    save_result(args.out_dir, {
        'instruction':       instruction,
        'keyword':           keyword,
        'yoloe_detected':    yoloe_det,
        'yoloe_conf':        float(conf),
        'anchor_correction': has_anchor,
        'pose_4x4':          pose.tolist(),
    })

    _save_visualisation(
        rgb, depth,
        mask if yoloe_det else None,
        pose, K, keyword, instruction, yoloe_det,
        os.path.join(args.out_dir, "viz_pipeline.png"))

    print(f"\nDone. Results → {args.out_dir}/")


if __name__ == "__main__":
    main()
