"""
Export tous les résultats disponibles (HO3D + LINEMOD) en XLSX et JSON consolidés.
Lit les fichiers *_poses.json existants — n'a pas besoin de relancer l'évaluation.

Run inside Docker:
    /opt/conda/envs/Any6D/bin/python3 /workspace/visualization/export_results_xlsx_json.py
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, '/workspace')
from estimater import *   # imports pandas as pd via Utils.py

# ── Paths ──────────────────────────────────────────────────────────────────────
HO3D_DIR  = "/workspace/results/ho3d_pipeline/run_full"
LM_DIR    = "/workspace/results/lm_pipeline/full_stride5"
OUT_DIR   = "/workspace/results/exports"
os.makedirs(OUT_DIR, exist_ok=True)

LM_SYMMETRIC = {10, 11}  # eggbox, glue

def nanmean(lst):
    vals = [v for v in lst if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float('nan')


# ══════════════════════════════════════════════════════════════════════════════
# HO3D  — reads from 0_FINAL_SUMMARY.json (already aggregated)
# ══════════════════════════════════════════════════════════════════════════════
print("=== HO3D ===")
ho3d_summary_path = f"{HO3D_DIR}/0_FINAL_SUMMARY.json"
ho3d_rows = []
ho3d_frames_all = []

if os.path.exists(ho3d_summary_path):
    with open(ho3d_summary_path) as f:
        ho3d_summary_raw = json.load(f)

    # 0_FINAL_SUMMARY.json is a list of dicts (one per sequence + MEAN)
    ho3d_rows = [r for r in ho3d_summary_raw if r.get('Object') != 'MEAN']

    # Rename columns for consistency
    col_map = {
        'Object':       'sequence',
        'ADD':          'ADD',
        'ADD-S':        'ADD-S',
        'AR':           'AR',
        'VSD':          'VSD',
        'MSSD':         'MSSD',
        'MSPD':         'MSPD',
        'R_error':      'R_error_mean',
        'T_error':      'T_error_mean',
        'Det_Rate (%)': 'yoloe_det_%',
        'Mean_IoU (%)': 'mean_iou_%',
        'Total_frames': 'n_frames',
        'Prompt':       'keyword',
    }
    ho3d_clean = []
    for r in ho3d_rows:
        row = {col_map.get(k, k): v for k, v in r.items()}
        seq = row.get('sequence', '')
        # load per-frame R_error from poses json for median
        poses_path = f"{HO3D_DIR}/{seq}_poses.json"
        r_med = float('nan')
        if os.path.exists(poses_path):
            with open(poses_path) as pf:
                pd_data = json.load(pf)
            r_errs = [fr.get('R_error', float('nan')) for fr in pd_data.get('frames', [])]
            valid  = [v for v in r_errs if not np.isnan(v)]
            r_med  = round(float(np.median(valid)), 2) if valid else float('nan')
            # also collect frames
            for fr in pd_data.get('frames', []):
                ho3d_frames_all.append({
                    'sequence': seq,
                    'frame':    fr.get('frame', ''),
                    'R_error':  fr.get('R_error'),
                    'T_error':  fr.get('T_error'),
                    'yoloe_detected': fr.get('yoloe_detected'),
                })
        row['R_error_med'] = r_med
        ho3d_clean.append(row)
        print(f"  {seq}: ADD-S={row.get('ADD-S')}%  ADD={row.get('ADD')}%  "
              f"AR={row.get('AR')}%  R_med={r_med}°  YOLOE={row.get('yoloe_det_%')}%")

    # MEAN row from file
    mean_raw = next((r for r in ho3d_summary_raw if r.get('Object') == 'MEAN'), None)
    if mean_raw:
        mean_clean = {col_map.get(k, k): v for k, v in mean_raw.items()}
        mean_clean['R_error_med'] = round(nanmean([r['R_error_med'] for r in ho3d_clean]), 2)
        ho3d_clean.append(mean_clean)

    df_ho3d = pd.DataFrame(ho3d_clean)
    df_ho3d.to_excel(f"{OUT_DIR}/HO3D_SUMMARY.xlsx", index=False)
    print(f"  → {OUT_DIR}/HO3D_SUMMARY.xlsx")

    json.dump({
        'dataset': 'HO3D', 'n_sequences': len(ho3d_rows),
        'mean_ADD':   mean_raw.get('ADD')   if mean_raw else None,
        'mean_ADD-S': mean_raw.get('ADD-S') if mean_raw else None,
        'mean_AR':    mean_raw.get('AR')    if mean_raw else None,
        'mean_VSD':   mean_raw.get('VSD')   if mean_raw else None,
        'per_sequence': {r['sequence']: r for r in ho3d_clean if r.get('sequence') != 'MEAN'},
    }, open(f"{OUT_DIR}/HO3D_SUMMARY.json", 'w'), indent=2)
    print(f"  → {OUT_DIR}/HO3D_SUMMARY.json")

    if ho3d_frames_all:
        pd.DataFrame(ho3d_frames_all).to_excel(f"{OUT_DIR}/HO3D_ALL_FRAMES.xlsx", index=False)
        print(f"  → {OUT_DIR}/HO3D_ALL_FRAMES.xlsx")
else:
    print("  No HO3D summary found.")


# ══════════════════════════════════════════════════════════════════════════════
# LINEMOD
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== LINEMOD ===")
lm_files = sorted(glob.glob(f"{LM_DIR}/obj_*_poses.json"))
lm_rows  = []
lm_frames_all = []

LM_NAMES = {
    1:'ape', 2:'benchvise', 3:'bowl', 4:'camera', 5:'can',
    6:'cat', 7:'cup', 8:'driller', 9:'duck', 10:'eggbox',
    11:'glue', 12:'holepuncher', 13:'iron', 14:'lamp', 15:'phone',
}

for fpath in lm_files:
    with open(fpath) as f:
        d = json.load(f)

    obj_id   = d.get('obj_id', 0)
    obj_name = d.get('obj_name', LM_NAMES.get(obj_id, f'obj{obj_id:02d}'))
    sym      = obj_id in LM_SYMMETRIC
    frames   = d.get('frames', [])
    if not frames:
        continue

    add_vals   = [fr.get('ADD',   float('nan')) for fr in frames]
    adds_vals  = [fr.get('ADD-S', float('nan')) for fr in frames]
    ar_vals    = [fr.get('AR',    float('nan')) for fr in frames]
    r_errs     = [fr.get('R_error', float('nan')) for fr in frames]
    t_errs     = [fr.get('T_error', float('nan')) for fr in frames]
    yoloe_dets = [fr.get('yoloe_detected', False) for fr in frames]

    add_used = nanmean(adds_vals) if sym else nanmean(add_vals)

    row = {
        'obj_id':      obj_id,
        'obj_name':    obj_name,
        'symmetric':   sym,
        'n_frames':    len(frames),
        'ADD':         round(nanmean(add_vals)  * 100, 1),
        'ADD-S':       round(nanmean(adds_vals) * 100, 1),
        'ADD(S)_used': round(add_used           * 100, 1),
        'AR':          round(nanmean(ar_vals)   * 100, 1),
        'R_error_mean':round(nanmean(r_errs),   2),
        'R_error_med': round(float(np.median([v for v in r_errs if not np.isnan(v)])), 2)
                       if any(not np.isnan(v) for v in r_errs) else float('nan'),
        'T_error_mean':round(nanmean(t_errs),   2),
        'yoloe_det_%': round(sum(yoloe_dets) / len(yoloe_dets) * 100, 1) if yoloe_dets else 0.0,
    }
    lm_rows.append(row)

    for fr in frames:
        lm_frames_all.append({
            'obj_id': obj_id, 'obj_name': obj_name,
            'im_id': fr.get('im_id'),
            'ADD': fr.get('ADD'), 'ADD-S': fr.get('ADD-S'), 'AR': fr.get('AR'),
            'R_error': fr.get('R_error'), 'T_error': fr.get('T_error'),
            'yoloe_detected': fr.get('yoloe_detected'), 'iou': fr.get('iou'),
        })

    sym_tag = '(S)' if sym else '   '
    print(f"  obj{obj_id:02d} {obj_name:<14}: ADD(S)={row['ADD(S)_used']}%{sym_tag}  "
          f"R_med={row['R_error_med']}°")

if lm_rows:
    add_used_scores = [r['ADD(S)_used'] for r in lm_rows]
    mean_row_lm = {
        'obj_id': 'MEAN', 'obj_name': '—', 'symmetric': '—',
        'n_frames': sum(r['n_frames'] for r in lm_rows),
        'ADD':         round(nanmean([r['ADD']          for r in lm_rows]), 1),
        'ADD-S':       round(nanmean([r['ADD-S']        for r in lm_rows]), 1),
        'ADD(S)_used': round(nanmean(add_used_scores),  1),
        'AR':          round(nanmean([r['AR']           for r in lm_rows]), 1),
        'R_error_mean':round(nanmean([r['R_error_mean'] for r in lm_rows]), 2),
        'R_error_med': round(nanmean([r['R_error_med']  for r in lm_rows]), 2),
        'T_error_mean':round(nanmean([r['T_error_mean'] for r in lm_rows]), 2),
        'yoloe_det_%': round(nanmean([r['yoloe_det_%']  for r in lm_rows]), 1),
    }
    df_lm = pd.DataFrame(lm_rows + [mean_row_lm])
    df_lm.to_excel(f"{OUT_DIR}/LINEMOD_SUMMARY.xlsx", index=False)
    print(f"  → {OUT_DIR}/LINEMOD_SUMMARY.xlsx")

    json.dump({
        'dataset': 'LINEMOD BOP', 'n_objects': len(lm_rows),
        'mask_mode': 'GT (oracle)',
        'mean_ADD(S)': round(nanmean(add_used_scores), 1),
        'mean_ADD':    round(nanmean([r['ADD']   for r in lm_rows]), 1),
        'mean_ADD-S':  round(nanmean([r['ADD-S'] for r in lm_rows]), 1),
        'per_object':  {r['obj_name']: r for r in lm_rows},
    }, open(f"{OUT_DIR}/LINEMOD_SUMMARY.json", 'w'), indent=2)
    print(f"  → {OUT_DIR}/LINEMOD_SUMMARY.json")

    df_lm_frames = pd.DataFrame(lm_frames_all)
    df_lm_frames.to_excel(f"{OUT_DIR}/LINEMOD_ALL_FRAMES.xlsx", index=False)
    print(f"  → {OUT_DIR}/LINEMOD_ALL_FRAMES.xlsx")
else:
    print("  No LINEMOD results found.")


# ══════════════════════════════════════════════════════════════════════════════
# Combined JSON (both datasets)
# ══════════════════════════════════════════════════════════════════════════════
ho3d_clean_noMean = [r for r in ho3d_clean if r.get('sequence') != 'MEAN'] if 'ho3d_clean' in dir() else []
combined = {
    'HO3D': {
        'n_sequences': len(ho3d_rows),
        'mean_ADD':   mean_raw.get('ADD')   if (os.path.exists(ho3d_summary_path) and 'mean_raw' in dir() and mean_raw) else None,
        'mean_ADD-S': mean_raw.get('ADD-S') if (os.path.exists(ho3d_summary_path) and 'mean_raw' in dir() and mean_raw) else None,
        'mean_AR':    mean_raw.get('AR')    if (os.path.exists(ho3d_summary_path) and 'mean_raw' in dir() and mean_raw) else None,
        'per_sequence': {r['sequence']: r for r in ho3d_clean_noMean},
    },
    'LINEMOD': {
        'n_objects': len(lm_rows),
        'mask_mode': 'GT (oracle)',
        'mean_ADD(S)': round(nanmean([r['ADD(S)_used'] for r in lm_rows]), 1) if lm_rows else None,
        'per_object': {r['obj_name']: r for r in lm_rows},
    },
}
json.dump(combined, open(f"{OUT_DIR}/ALL_RESULTS.json", 'w'), indent=2)
print(f"\n  → {OUT_DIR}/ALL_RESULTS.json  (combined)")
print(f"\nDone. All files in {OUT_DIR}/")
