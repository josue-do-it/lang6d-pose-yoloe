"""
Test rapide : compare register() vs register_any6d() sur quelques frames de AP13.

Run inside Docker:
    /opt/conda/envs/Any6D/bin/python3 /workspace/test_register_vs_any6d.py
"""
import copy, sys, os, json, glob
import numpy as np

sys.path.insert(0, '/workspace/yoloe')

from foundationpose.datareader import Ho3dReader
from estimater import *
from pytorch_lightning import seed_everything
import nvdiffrast.torch as dr
import trimesh

# ── Config ─────────────────────────────────────────────────────────────────────
SEQ          = "AP13"
ANCHOR_PATH  = "/workspace/anchor_results/dexycb_reference_view_ours"
HO3D_ROOT    = "/dataset/ho3d/HO3D_data"
YCB_PATH     = "/dataset/ho3d"
MODELS_INFO  = "./models_info.json"
STRIDE       = 200         # frames à tester : ~8 frames rapide
OUT_DIR      = "/workspace/results/test_register_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Setup ──────────────────────────────────────────────────────────────────────
seed_everything(0)

video_dir = os.path.join(HO3D_ROOT, "evaluation", SEQ)
reader    = Ho3dReader(video_dir, HO3D_ROOT)
reader.color_files = reader.color_files[::STRIDE]

mesh_path = reader.get_reference_view_1_mesh(ANCHOR_PATH)
mesh      = trimesh.load(mesh_path)
glctx     = dr.RasterizeCudaContext()

with open(MODELS_INFO) as f:
    model_info = json.load(f)

ob_id     = reader.get_obj_id()
gt_mesh   = reader.get_gt_mesh(YCB_PATH)

# ── Helpers ────────────────────────────────────────────────────────────────────
def rotation_error_deg(R_pred, R_gt):
    R_rel = R_pred @ R_gt.T
    cos_val = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_val)))

def translation_error_cm(t_pred, t_gt):
    return float(np.linalg.norm(t_pred - t_gt) * 100)

def run_estimator(method_name, register_fn):
    """Run one method on all sampled frames, return list of (R_err, T_err)."""
    seed_everything(0)
    method_dir = os.path.join(OUT_DIR, method_name)
    os.makedirs(method_dir, exist_ok=True)          # register_any6d needs this
    mesh_copy = trimesh.load(mesh_path)
    est = Any6D(mesh=mesh_copy,
                scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                debug_dir=method_dir,
                debug=0, glctx=glctx)
    est.reset_object(mesh=mesh_copy, symmetry_tfs=None)
    pred_pose_a = np.loadtxt(reader.get_reference_view_1_pose(ANCHOR_PATH))

    results = []
    for i in range(len(reader.color_files)):
        gt_pose = reader.get_gt_pose(i)
        if gt_pose is None:
            continue
        color = cv2.cvtColor(cv2.imread(reader.color_files[i]), cv2.COLOR_BGR2RGB)
        H, W  = color.shape[:2]
        depth = reader.get_depth(i)
        mask  = reader.get_mask(i)
        if mask is None:
            continue
        mask = mask.astype(bool)

        print(f"  [{method_name}] frame {i} ...", end=" ", flush=True)
        pose = register_fn(est, color, depth, mask, reader.K, i)

        R_pred = pose[:3, :3]
        t_pred = pose[:3,  3]
        R_gt   = gt_pose[:3, :3]
        t_gt   = gt_pose[:3,  3]

        r_err = rotation_error_deg(R_pred, R_gt)
        t_err = translation_error_cm(t_pred, t_gt)
        results.append({'frame': i, 'R_error': r_err, 'T_error': t_err})
        print(f"R={r_err:.1f}°  T={t_err:.2f}cm")

    return results


def fn_register(est, color, depth, mask, K, i):
    return est.register(K=K, rgb=color, depth=depth,
                        ob_mask=mask, iteration=5, name=SEQ)

def fn_register_any6d(est, color, depth, mask, K, i):
    return est.register_any6d(K=K, rgb=color, depth=depth,
                               ob_mask=mask, iteration=5, name=SEQ)


# ── Run both methods ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Séquence : {SEQ}   stride={STRIDE}  → {len(reader.color_files)} frames")
print(f"{'='*60}\n")

print(">>> register() (FoundationPose standard)")
res_fp = run_estimator("register", fn_register)

print("\n>>> register_any6d() (Any6D complet)")
res_a6 = run_estimator("register_any6d", fn_register_any6d)

# ── Résumé ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  COMPARAISON — {SEQ}")
print(f"{'='*60}")
print(f"{'Frame':<8} {'register() R':>14} {'register() T':>14} {'any6d R':>12} {'any6d T':>12} {'ΔR':>10} {'ΔT':>10}")
print("-"*82)

for fp, a6 in zip(res_fp, res_a6):
    delta_r = fp['R_error'] - a6['R_error']
    delta_t = fp['T_error'] - a6['T_error']
    sign_r  = "↓" if delta_r > 0 else "↑"
    sign_t  = "↓" if delta_t > 0 else "↑"
    print(f"  {fp['frame']:<6} {fp['R_error']:>12.1f}°  {fp['T_error']:>12.2f}cm"
          f"  {a6['R_error']:>10.1f}°  {a6['T_error']:>10.2f}cm"
          f"  {sign_r}{abs(delta_r):>7.1f}°  {sign_t}{abs(delta_t):>7.2f}cm")

r_fp   = [r['R_error'] for r in res_fp]
r_a6   = [r['R_error'] for r in res_a6]
t_fp   = [r['T_error'] for r in res_fp]
t_a6   = [r['T_error'] for r in res_a6]

print("-"*82)
print(f"  {'MEAN':<6} {sum(r_fp)/len(r_fp):>12.1f}°  {sum(t_fp)/len(t_fp):>12.2f}cm"
      f"  {sum(r_a6)/len(r_a6):>10.1f}°  {sum(t_a6)/len(t_a6):>10.2f}cm")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out = {'seq': SEQ, 'stride': STRIDE,
       'register':      res_fp,
       'register_any6d': res_a6}
with open(os.path.join(OUT_DIR, "comparison.json"), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSauvegardé → {OUT_DIR}/comparison.json")
