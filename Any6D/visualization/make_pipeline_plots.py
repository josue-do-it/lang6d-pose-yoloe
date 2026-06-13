"""
Generate 6+ plots for HO3D pipeline results and 6+ for YCB-V pipeline results.
Reads from:
  HO3D: --ho3d_dir  (contains *_yoloe_detection.json + *_metrics_results.xlsx)
  YCB-V: --ycbv_dir (contains 0_per_instruction.xlsx + 0_summary.xlsx)
"""
import argparse, os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'figure.dpi': 120})

POSE_METRICS = ['ADD-S','ADD','AR','MSSD','MSPD','VSD']
OBJ_COLORS   = {
    '010_potted_meat_can': '#e07b39',
    '019_pitcher_base':    '#3a7fc1',
    '021_bleach_cleanser': '#5ab55e',
    '006_mustard_bottle':  '#d4a017',
}
SEQ_TO_OBJ = {
    'MPM10':'010_potted_meat_can','MPM11':'010_potted_meat_can',
    'MPM12':'010_potted_meat_can','MPM13':'010_potted_meat_can',
    'MPM14':'010_potted_meat_can',
    'AP10':'019_pitcher_base','AP11':'019_pitcher_base',
    'AP12':'019_pitcher_base','AP13':'019_pitcher_base','AP14':'019_pitcher_base',
    'SB11':'021_bleach_cleanser','SB13':'021_bleach_cleanser',
    'SM1':'006_mustard_bottle',
}

# ══════════════════════════════════════════════════════════════════════════════
# HO3D PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def load_ho3d(ho3d_dir):
    records = []
    seqs = ['MPM10','MPM11','MPM12','MPM13','MPM14',
            'AP10','AP11','AP12','AP13','AP14','SB11','SB13','SM1']
    for seq in seqs:
        det_path  = os.path.join(ho3d_dir, f'{seq}_yoloe_detection.json')
        pose_path = os.path.join(ho3d_dir, f'{seq}_metrics_results.xlsx')
        if not os.path.exists(det_path):
            continue
        with open(det_path) as f:
            det = json.load(f)
        row = {'seq': seq, 'object': SEQ_TO_OBJ.get(seq, seq)}
        row['instruction']   = det.get('instruction', det.get('prompt',''))
        row['keyword']       = det.get('yoloe_prompt', det.get('prompt',''))
        row['det_rate']      = det.get('detection_rate', 0)
        row['mean_iou']      = det.get('mean_iou', 0)
        row['total_frames']  = det.get('total_frames', 0)
        row['detected_frames'] = det.get('detected_frames', 0)
        if os.path.exists(pose_path):
            df = pd.read_excel(pose_path)
            df = df[df['Frame_ID'] != 'MEAN'].copy()
            for m in POSE_METRICS + ['R_error','T_error']:
                df[m] = pd.to_numeric(df[m], errors='coerce')
            for m in POSE_METRICS:
                row[m] = round(df[m].mean()*100, 1)
            row['R_error'] = round(df['R_error'].mean(), 2)
            row['T_error'] = round(df['T_error'].mean(), 2)
        records.append(row)
    return pd.DataFrame(records)

def plot_ho3d(ho3d_dir, out_dir):
    df = load_ho3d(ho3d_dir)
    if df.empty:
        print("No HO3D data found."); return
    os.makedirs(out_dir, exist_ok=True)
    seqs = df['seq'].tolist()
    obj_types = df['object'].unique()

    # ── Plot 1: BOP metrics bar (AR, ADD-S, VSD) per sequence ─────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(seqs)); width = 0.25
    colors_m = ['#2196f3','#4caf50','#ff9800']
    for i, m in enumerate(['AR','ADD-S','VSD']):
        if m in df.columns:
            ax.bar(x + i*width, df[m].fillna(0), width, label=m, color=colors_m[i], alpha=0.85)
    ax.set_xticks(x + width); ax.set_xticklabels(seqs, rotation=45, ha='right')
    ax.set_ylabel('Score (%)'); ax.set_title('HO3D Pipeline — BOP Metrics per Sequence')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    # color background by object
    for i, seq in enumerate(seqs):
        obj = SEQ_TO_OBJ.get(seq, '')
        c   = OBJ_COLORS.get(obj, '#cccccc')
        ax.axvspan(i-0.4, i+0.9, alpha=0.06, color=c)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/HO3D_1_bop_metrics_per_seq.png", bbox_inches='tight')
    plt.close(); print("Saved: HO3D_1_bop_metrics_per_seq.png")

    # ── Plot 2: Detection rate + IoU per sequence ──────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.2, df['det_rate'].fillna(0), 0.35, label='Det. Rate (%)', color='#7c4dff', alpha=0.85)
    ax.bar(x + 0.2, df['mean_iou'].fillna(0), 0.35, label='Mean IoU (%)',  color='#00bcd4', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(seqs, rotation=45, ha='right')
    ax.set_ylabel('%'); ax.set_title('HO3D Pipeline — Detection Rate & IoU vs XMem per Sequence')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/HO3D_2_detection_per_seq.png", bbox_inches='tight')
    plt.close(); print("Saved: HO3D_2_detection_per_seq.png")

    # ── Plot 3: Det.Rate vs AR scatter (sequence labels) ──────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in df.iterrows():
        c = OBJ_COLORS.get(row['object'], '#999')
        ax.scatter(row['det_rate'], row.get('AR', 0), s=120, color=c, zorder=3)
        ax.annotate(row['seq'], (row['det_rate'], row.get('AR',0)),
                    textcoords='offset points', xytext=(5,3), fontsize=8)
    patches = [mpatches.Patch(color=c, label=o.replace('0','').lstrip('_').replace('_',' '))
               for o, c in OBJ_COLORS.items()]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel('Detection Rate (%)'); ax.set_ylabel('AR (%)')
    ax.set_title('HO3D Pipeline — Detection Rate vs Pose AR')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/HO3D_3_detrate_vs_ar.png", bbox_inches='tight')
    plt.close(); print("Saved: HO3D_3_detrate_vs_ar.png")

    # ── Plot 4: Extracted keyword + IoU annotation per sequence ───────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(seqs, df['mean_iou'].fillna(0),
                   color=[OBJ_COLORS.get(SEQ_TO_OBJ.get(s,''),'#999') for s in seqs],
                   alpha=0.8)
    for i, (_, row) in enumerate(df.iterrows()):
        kw = row.get('keyword','?')
        ax.text(row['mean_iou']+0.5, i, f'  "{kw}"', va='center', fontsize=8)
    ax.set_xlabel('Mean IoU vs XMem (%)')
    ax.set_title('HO3D Pipeline — LLM Extracted Keyword & YOLOE IoU per Sequence')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/HO3D_4_keyword_iou.png", bbox_inches='tight')
    plt.close(); print("Saved: HO3D_4_keyword_iou.png")

    # ── Plot 5: Full metrics heatmap (sequences × metrics) ────────────────────
    heat_cols = ['AR','ADD-S','VSD','MSSD','MSPD','ADD','det_rate','mean_iou']
    heat_cols = [c for c in heat_cols if c in df.columns]
    heat_data = df.set_index('seq')[heat_cols].fillna(0).values
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(heat_cols))); ax.set_xticklabels(heat_cols, rotation=30, ha='right')
    ax.set_yticks(range(len(seqs))); ax.set_yticklabels(df['seq'].tolist())
    for i in range(len(seqs)):
        for j in range(len(heat_cols)):
            ax.text(j, i, f'{heat_data[i,j]:.0f}', ha='center', va='center',
                    fontsize=8, color='black' if 20 < heat_data[i,j] < 80 else 'white')
    plt.colorbar(im, ax=ax, label='Score (%)')
    ax.set_title('HO3D Pipeline — Full Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/HO3D_5_heatmap.png", bbox_inches='tight')
    plt.close(); print("Saved: HO3D_5_heatmap.png")

    # ── Plot 6: Per-object-type box/bar summary ────────────────────────────────
    obj_groups = df.groupby('object')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ['AR','det_rate','mean_iou']):
        vals = [obj_groups.get_group(o)[metric].values
                for o in obj_types if o in obj_groups.groups]
        obj_labels = [o.split('_')[-1] if '_' in o else o for o in obj_types
                      if o in obj_groups.groups]
        bp = ax.boxplot(vals, labels=obj_labels, patch_artist=True)
        for patch, obj in zip(bp['boxes'], [o for o in obj_types if o in obj_groups.groups]):
            patch.set_facecolor(OBJ_COLORS.get(obj, '#999'))
        ax.set_title(f'{metric} by Object Type'); ax.set_ylabel('%')
        ax.tick_params(axis='x', rotation=15); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('HO3D Pipeline — Distribution by Object Type', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/HO3D_6_by_object_type.png", bbox_inches='tight')
    plt.close(); print("Saved: HO3D_6_by_object_type.png")

    # ── Plot 7: Radar chart per object type ───────────────────────────────────
    radar_metrics = ['AR','ADD-S','VSD','det_rate','mean_iou']
    radar_metrics = [m for m in radar_metrics if m in df.columns]
    N = len(radar_metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for obj in obj_types:
        if obj not in obj_groups.groups:
            continue
        vals = [obj_groups.get_group(obj)[m].mean() for m in radar_metrics]
        vals += vals[:1]
        c = OBJ_COLORS.get(obj, '#999')
        ax.plot(angles, vals, 'o-', linewidth=2, color=c)
        ax.fill(angles, vals, alpha=0.08, color=c)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics)
    ax.set_ylim(0, 100); ax.set_title('HO3D Pipeline — Radar Chart per Object Type', pad=20)
    patches = [mpatches.Patch(color=OBJ_COLORS.get(o,'#999'),
                              label=o.split('_')[-1] if '_' in o else o)
               for o in obj_types if o in obj_groups.groups]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"{out_dir}/HO3D_7_radar.png", bbox_inches='tight')
    plt.close(); print("Saved: HO3D_7_radar.png")

    # ── Plot 8: R_error and T_error per sequence ──────────────────────────────
    if 'R_error' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        c_list = [OBJ_COLORS.get(SEQ_TO_OBJ.get(s,''),'#999') for s in seqs]
        ax1.bar(seqs, df['R_error'].fillna(0), color=c_list, alpha=0.85)
        ax1.set_xticklabels(seqs, rotation=45, ha='right')
        ax1.set_ylabel('R_error (°)'); ax1.set_title('Rotation Error per Sequence')
        ax1.grid(axis='y', alpha=0.3)
        ax2.bar(seqs, df['T_error'].fillna(0), color=c_list, alpha=0.85)
        ax2.set_xticklabels(seqs, rotation=45, ha='right')
        ax2.set_ylabel('T_error (mm)'); ax2.set_title('Translation Error per Sequence')
        ax2.grid(axis='y', alpha=0.3)
        plt.suptitle('HO3D Pipeline — Pose Errors (R & T)', fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/HO3D_8_errors.png", bbox_inches='tight')
        plt.close(); print("Saved: HO3D_8_errors.png")

    print(f"\nHO3D: {len(df)} sequences plotted → {out_dir}")

# ══════════════════════════════════════════════════════════════════════════════
# YCB-V PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_ycbv(ycbv_dir, out_dir):
    per_instr_path = os.path.join(ycbv_dir, '0_per_instruction.xlsx')
    summary_path   = os.path.join(ycbv_dir, '0_summary.xlsx')
    if not os.path.exists(per_instr_path):
        print("No YCB-V per_instruction.xlsx found."); return
    os.makedirs(out_dir, exist_ok=True)

    df_pi  = pd.read_excel(per_instr_path)
    df_sum = pd.read_excel(summary_path)
    df_sum = df_sum[df_sum['Object'] != 'MEAN'].copy()
    objects = df_sum['Object'].tolist()

    # ── Plot 1: IoU heatmap (objects × instructions) ──────────────────────────
    iou_matrix = np.zeros((len(objects), 5))
    for i, obj in enumerate(objects):
        rows = df_pi[df_pi['object'] == obj].sort_values('instr_idx')
        for j, (_, r) in enumerate(rows.iterrows()):
            if j < 5:
                iou_matrix[i, j] = r.get('iou', 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(iou_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(5)); ax.set_xticklabels([f'Instr {i}' for i in range(5)])
    ax.set_yticks(range(len(objects))); ax.set_yticklabels(objects, fontsize=9)
    for i in range(len(objects)):
        for j in range(5):
            v = iou_matrix[i,j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, color='black' if 0.2 < v < 0.8 else 'white')
    plt.colorbar(im, ax=ax, label='IoU')
    ax.set_title(f'YCB-V Pipeline — IoU Heatmap (Objects × Instructions)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/YCBV_1_iou_heatmap.png", bbox_inches='tight')
    plt.close(); print("Saved: YCBV_1_iou_heatmap.png")

    # ── Plot 2: Det. Rate per object ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(objects)))
    bars = ax.bar(objects, df_sum['Det_Rate(%)'].fillna(0), color=colors, alpha=0.85)
    ax.set_xticklabels(objects, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Detection Rate (%)'); ax.set_ylim(0, 110)
    ax.axhline(100, ls='--', color='gray', alpha=0.5)
    for bar, val in zip(bars, df_sum['Det_Rate(%)'].fillna(0)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val:.0f}%', ha='center', fontsize=9)
    ax.set_title('YCB-V Pipeline — Detection Rate per Object')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/YCBV_2_det_rate.png", bbox_inches='tight')
    plt.close(); print("Saved: YCBV_2_det_rate.png")

    # ── Plot 3: Mean IoU + Prompt Accuracy grouped bar ────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(objects))
    ax.bar(x-0.2, df_sum['Mean_IoU(%)'].fillna(0),   0.35, label='Mean IoU (%)',     color='#2196f3', alpha=0.85)
    ax.bar(x+0.2, df_sum['Prompt_Acc(%)'].fillna(0), 0.35, label='Prompt Acc (%)',   color='#ff5722', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(objects, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('%'); ax.legend()
    ax.set_title('YCB-V Pipeline — IoU & Prompt Accuracy per Object')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/YCBV_3_iou_accuracy.png", bbox_inches='tight')
    plt.close(); print("Saved: YCBV_3_iou_accuracy.png")

    # ── Plot 4: IoU violin per object ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    violin_data = [df_pi[df_pi['object']==obj]['iou'].dropna().values for obj in objects]
    vparts = ax.violinplot(violin_data, positions=range(len(objects)), showmedians=True)
    for i, pc in enumerate(vparts['bodies']):
        pc.set_facecolor(plt.cm.tab10(i / len(objects)))
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(objects))); ax.set_xticklabels(objects, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('IoU'); ax.set_ylim(-0.05, 1.05)
    ax.set_title('YCB-V Pipeline — IoU Distribution per Object (5 instructions)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/YCBV_4_iou_violin.png", bbox_inches='tight')
    plt.close(); print("Saved: YCBV_4_iou_violin.png")

    # ── Plot 5: Keyword length vs IoU scatter ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df_pi.iterrows():
        kw = str(row.get('keyword',''))
        n_words = len(kw.split())
        iou_val = row.get('iou', 0)
        ax.scatter(n_words, iou_val, alpha=0.6,
                   color=plt.cm.tab10(objects.index(row['object'])/len(objects))
                   if row['object'] in objects else '#999')
    ax.set_xlabel('Keyword length (words)')
    ax.set_ylabel('IoU')
    ax.set_xticks([1,2,3,4,5])
    ax.set_title('YCB-V Pipeline — Keyword Length vs IoU\n(1 word often best)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/YCBV_5_kw_length_vs_iou.png", bbox_inches='tight')
    plt.close(); print("Saved: YCBV_5_kw_length_vs_iou.png")

    # ── Plot 6: Best keyword per object (horizontal bar) ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    best_rows = []
    for obj in objects:
        rows_o = df_pi[df_pi['object']==obj]
        if rows_o.empty:
            continue
        best = rows_o.loc[rows_o['iou'].idxmax()]
        best_rows.append({'object': obj, 'iou': best['iou'], 'keyword': best.get('keyword','?')})
    df_best = pd.DataFrame(best_rows)
    bars = ax.barh(df_best['object'], df_best['iou']*100,
                   color=plt.cm.tab10(np.linspace(0,1,len(df_best))), alpha=0.85)
    for i, row in df_best.iterrows():
        ax.text(row['iou']*100 + 0.5, i if isinstance(i,int) else list(df_best.index).index(i),
                f'  "{row["keyword"]}"', va='center', fontsize=9)
    ax.set_xlabel('Best IoU (%)'); ax.set_xlim(0, 120)
    ax.set_title('YCB-V Pipeline — Best Keyword & IoU per Object')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/YCBV_6_best_keyword.png", bbox_inches='tight')
    plt.close(); print("Saved: YCBV_6_best_keyword.png")

    # ── Plot 7: 4-metric summary radar ────────────────────────────────────────
    radar_m = ['Det_Rate(%)','Mean_IoU(%)','Prompt_Acc(%)','Conf_Score(%)']
    radar_m = [m for m in radar_m if m in df_sum.columns]
    N = len(radar_m)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(objects)))
    for i, obj in enumerate(objects):
        row_s = df_sum[df_sum['Object']==obj]
        if row_s.empty:
            continue
        vals = [row_s[m].values[0] for m in radar_m]; vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=1.5, color=colors[i], label=obj.split('_')[-1])
        ax.fill(angles, vals, alpha=0.05, color=colors[i])
    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [m.replace('(%)','').replace('_',' ') for m in radar_m])
    ax.set_ylim(0, 100)
    ax.set_title('YCB-V Pipeline — Radar per Object', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/YCBV_7_radar.png", bbox_inches='tight')
    plt.close(); print("Saved: YCBV_7_radar.png")

    # ── Plot 8: Instruction-level IoU bar (all 45) ────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    x_vals, y_vals, c_list, labels = [], [], [], []
    idx = 0
    for oi, obj in enumerate(objects):
        rows_o = df_pi[df_pi['object']==obj].sort_values('instr_idx')
        for _, row in rows_o.iterrows():
            x_vals.append(idx); y_vals.append(row.get('iou',0)*100)
            c_list.append(plt.cm.tab10(oi/len(objects)))
            labels.append(f"{obj.split('_')[-1][:6]}\n{row.get('keyword','?')[:8]}")
            idx += 1
    ax.bar(x_vals, y_vals, color=c_list, alpha=0.8, width=0.7)
    ax.set_xticks(x_vals); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('IoU (%)'); ax.axhline(30, ls='--', color='red', alpha=0.5, label='IoU=0.3 threshold')
    ax.legend(); ax.set_title('YCB-V Pipeline — IoU for All 45 Instructions')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/YCBV_8_all45_iou.png", bbox_inches='tight')
    plt.close(); print("Saved: YCBV_8_all45_iou.png")

    print(f"\nYCB-V: {len(objects)} objects plotted → {out_dir}")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ho3d_dir',  default=None, help='HO3D pipeline results dir')
    parser.add_argument('--ycbv_dir',  default=None, help='YCB-V pipeline results dir')
    parser.add_argument('--out_dir',   default='/workspace/results/pipeline_plots')
    args = parser.parse_args()

    if args.ho3d_dir:
        print("=" * 50)
        print("Generating HO3D plots...")
        print("=" * 50)
        plot_ho3d(args.ho3d_dir, os.path.join(args.out_dir, 'ho3d'))

    if args.ycbv_dir:
        print("=" * 50)
        print("Generating YCB-V plots...")
        print("=" * 50)
        plot_ycbv(args.ycbv_dir, os.path.join(args.out_dir, 'ycbv'))

    if not args.ho3d_dir and not args.ycbv_dir:
        print("Usage: python make_pipeline_plots.py --ho3d_dir <path> --ycbv_dir <path>")
