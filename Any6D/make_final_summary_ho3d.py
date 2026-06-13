"""
Final summary: BOP pose metrics + YOLOE detection stats + prompts → one Excel + JSON
"""
import argparse, os, json
import numpy as np
import pandas as pd

OBJECTS = ['MPM10', 'MPM11', 'MPM12', 'MPM13', 'MPM14',
           'AP10', 'AP11', 'AP12', 'AP13', 'AP14',
           'SB11', 'SB13', 'SM1']

POSE_METRICS = ['ADD-S', 'ADD', 'AR', 'MSSD', 'MSPD', 'VSD']
ERROR_METRICS = ['R_error', 'T_error']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    rows = []
    missing_pose, missing_det = [], []

    for obj in OBJECTS:
        pose_path = os.path.join(args.results_dir, f'{obj}_metrics_results.xlsx')
        det_path  = os.path.join(args.results_dir, f'{obj}_yoloe_detection.json')

        row = {'Object': obj}

        # --- YOLOE detection stats ---
        if os.path.exists(det_path):
            with open(det_path) as f:
                det = json.load(f)
            row['Prompt']         = det.get('prompt', '')
            row['Det_Rate (%)']   = det.get('detection_rate', 0)
            row['Mean_IoU (%)']   = det.get('mean_iou', 0)
            row['Detected_frames']= det.get('detected_frames', 0)
            row['Total_frames']   = det.get('total_frames', 0)
            row['Fallback_frames']= det.get('fallback_frames', 0)
        else:
            missing_det.append(obj)
            row.update({'Prompt': '', 'Det_Rate (%)': '', 'Mean_IoU (%)': '',
                        'Detected_frames': '', 'Total_frames': '', 'Fallback_frames': ''})

        # --- BOP pose metrics ---
        if os.path.exists(pose_path):
            df = pd.read_excel(pose_path)
            df_frames = df[df['Frame_ID'] != 'MEAN'].copy()
            for col in POSE_METRICS + ERROR_METRICS:
                df_frames[col] = pd.to_numeric(df_frames[col], errors='coerce')
            for m in POSE_METRICS:
                row[m] = round(df_frames[m].mean() * 100, 1)
            for m in ERROR_METRICS:
                row[m] = round(df_frames[m].mean(), 2)
        else:
            missing_pose.append(obj)
            for m in POSE_METRICS + ERROR_METRICS:
                row[m] = ''

        rows.append(row)
        print(f"{obj:6s} | prompt={row.get('Prompt','?'):25s} | det={row.get('Det_Rate (%)','?')}% "
              f"iou={row.get('Mean_IoU (%)','?')}% | AR={row.get('AR','?')} ADD-S={row.get('ADD-S','?')}")

    df_all = pd.DataFrame(rows)

    # Global mean row (numeric cols only)
    num_cols = ['Det_Rate (%)', 'Mean_IoU (%)'] + POSE_METRICS + ERROR_METRICS
    mean_row = {'Object': 'MEAN', 'Prompt': '—'}
    for col in num_cols:
        vals = pd.to_numeric(df_all[col], errors='coerce').dropna()
        mean_row[col] = round(vals.mean(), 1) if len(vals) > 0 else ''
    df_all = pd.concat([df_all, pd.DataFrame([mean_row])], ignore_index=True)

    # Save Excel
    out_xlsx = os.path.join(args.results_dir, '0_FINAL_SUMMARY.xlsx')
    df_all.to_excel(out_xlsx, index=False)
    print(f"\nExcel saved: {out_xlsx}")

    # Save JSON
    out_json = os.path.join(args.results_dir, '0_FINAL_SUMMARY.json')
    with open(out_json, 'w') as f:
        json.dump(df_all.to_dict(orient='records'), f, indent=2)
    print(f"JSON  saved: {out_json}")

    # LaTeX row
    m = mean_row
    print(f"\nLaTeX: MEAN & {m.get('AR','')} & {m.get('VSD','')} & {m.get('MSSD','')} & "
          f"{m.get('MSPD','')} & {m.get('ADD-S','')} & {m.get('Det_Rate (%)','')} & {m.get('Mean_IoU (%)','')} \\\\")

    if missing_pose: print(f"\nWARN missing pose xlsx: {missing_pose}")
    if missing_det:  print(f"WARN missing detection json: {missing_det}")
