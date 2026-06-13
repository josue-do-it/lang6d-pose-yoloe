import argparse
import os
import numpy as np
import pandas as pd

OBJECTS = ['MPM10', 'MPM11', 'MPM12', 'MPM13', 'MPM14',
           'AP10', 'AP11', 'AP12', 'AP13', 'AP14',
           'SB11', 'SB13', 'SM1']

METRICS = ['ADD-S', 'ADD', 'AR', 'MSSD', 'MSPD', 'VSD']
ERROR_METRICS = ['R_error', 'T_error']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    data = []
    all_frames = []
    missing = []

    for obj in OBJECTS:
        path = os.path.join(args.results_dir, f'{obj}_metrics_results.xlsx')
        if not os.path.exists(path):
            missing.append(obj)
            print(f'WARNING: missing {obj}')
            continue

        df = pd.read_excel(path)
        # Drop the MEAN summary row at the bottom
        df_frames = df[df['Frame_ID'] != 'MEAN'].copy()
        for col in METRICS + ERROR_METRICS:
            df_frames[col] = pd.to_numeric(df_frames[col], errors='coerce')

        means = {m: df_frames[m].mean() * 100 for m in METRICS}
        errors = {m: df_frames[m].mean() for m in ERROR_METRICS}
        row = {'Class_ID': obj}
        row.update({m: f"{means[m]:.1f}" for m in METRICS})
        row.update({m: f"{errors[m]:.2f}" for m in ERROR_METRICS})
        data.append(row)

        all_frames.append(df_frames)
        print(f'{obj}: AR={means["AR"]:.1f}  ADD-S={means["ADD-S"]:.1f}  VSD={means["VSD"]:.1f}  R_err={errors["R_error"]:.2f}  T_err={errors["T_error"]:.2f}')

    if not data:
        print('No results found.')
        raise SystemExit(1)

    # Per-object summary
    df_summary = pd.DataFrame(data)

    # Global mean across all objects
    all_df = pd.concat(all_frames, ignore_index=True)
    overall = {m: all_df[m].mean() * 100 for m in METRICS}
    overall_errors = {m: all_df[m].mean() for m in ERROR_METRICS}
    mean_row = {'Class_ID': 'MEAN'}
    mean_row.update({m: f"{overall[m]:.1f}" for m in METRICS})
    mean_row.update({m: f"{overall_errors[m]:.2f}" for m in ERROR_METRICS})
    df_summary = pd.concat([df_summary, pd.DataFrame([mean_row])], ignore_index=True)

    out_path = os.path.join(args.results_dir, '0_mean_all_metrics_classes_results.xlsx')
    df_summary.to_excel(out_path, index=False)
    print(f'\nGlobal mean saved to {out_path}')

    latex = (f"MEAN & {overall['AR']:.1f} & {overall['VSD']:.1f} & "
             f"{overall['MSSD']:.1f} & {overall['MSPD']:.1f} & {overall['ADD-S']:.1f} & - \\\\")
    print('\nLatex row:')
    print(latex)

    if missing:
        print(f'\nMissing objects (not included in mean): {missing}')
