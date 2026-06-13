"""
Publication-quality comparison table: YOLOE+Any6D  vs  LLM+YOLOE+Any6D
White background, suitable for paper.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

OUT = '/workspace/results/pipeline_plots/comparison_table.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
# "YOLOE + Any6D" = Any6D with YOLOE masks (from paper Table 1 "Ours" column)
A = {   # seq: (ADD-S, ADD, AR)
    'AP10':  (100.0, 16.2, 22.2),
    'AP11':  (100.0, 73.8, 59.0),
    'AP12':  ( 99.4, 48.8, 28.3),
    'AP13':  (100.0, 74.4, 45.0),
    'AP14':  (100.0, 35.6, 29.7),
    'SM1':   ( 86.5, 34.8, 27.8),
    'SB11':  (100.0, 86.8, 68.9),
    'SB13':  ( 99.4, 64.1, 54.6),
    'MPM10': ( 98.7, 26.8, 31.3),
    'MPM11': (100.0,  3.2, 32.4),
    'MPM12': (100.0,  1.3, 23.5),
    'MPM13': (100.0, 15.9, 30.9),
    'MPM14': ( 98.7, 43.9, 44.0),
}

# "LLM + YOLOE + Any6D" = our full pipeline results
B = {   # seq: (Det%, IoU%, ADD-S, ADD, AR)
    'AP10':  (100.0, 64.5,  72.5,  2.5, 15.7),
    'AP11':  (100.0, 84.1,  93.8, 56.9, 47.0),
    'AP12':  ( 82.5, 91.9, 100.0, 16.9, 33.0),
    'AP13':  (100.0, 93.8, 100.0, 57.5, 48.6),
    'AP14':  ( 86.2, 58.4,  76.9, 13.1, 24.4),
    'SM1':   ( 92.1, 89.9,  97.8,  3.4, 32.9),
    'SB11':  (100.0, 59.4,  65.3, 58.7, 46.5),
    'SB13':  (  1.2, 94.3, 100.0, 73.7, 67.2),
    'MPM10': (100.0, 81.4,  91.7, 31.2, 33.8),
    'MPM11': ( 32.5, 87.0,  99.4, 10.2, 30.1),
    'MPM12': ( 63.7, 91.4,  88.5, 15.9, 24.7),
    'MPM13': ( 95.5, 78.6,  93.6, 10.2, 38.0),
    'MPM14': ( 91.1, 77.0,  84.7, 22.3, 47.1),
}

SEQS = list(A.keys())
N    = len(SEQS)

# means
a_mean = tuple(np.mean([[A[s][i] for s in SEQS] for i in range(3)], axis=1))
b_mean = (
    np.mean([B[s][0] for s in SEQS]),
    np.mean([B[s][1] for s in SEQS]),
    np.mean([B[s][2] for s in SEQS]),
    np.mean([B[s][3] for s in SEQS]),
    np.mean([B[s][4] for s in SEQS]),
)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')

# column layout
# Seq | YOLOE+Any6D (ADD-S, ADD, AR) | LLM+YOLOE+Any6D (Det%, IoU%, ADD-S, ADD, AR)
cols = ['Seq.',
        'ADD-S', 'ADD', 'AR',           # YOLOE+Any6D
        'Det(%)', 'IoU(%)', 'ADD-S', 'ADD', 'AR']  # LLM+YOLOE+Any6D

header_groups = {
    'Sequence': (0, 0),
    'YOLOE + Any6D': (1, 3),
    'LLM + YOLOE + Any6D  (Ours)': (4, 8),
}

rows = []
for s in SEQS:
    a = A[s]; b = B[s]
    rows.append([s,
                 f'{a[0]:.1f}', f'{a[1]:.1f}', f'{a[2]:.1f}',
                 f'{b[0]:.1f}', f'{b[1]:.1f}', f'{b[2]:.1f}', f'{b[3]:.1f}', f'{b[4]:.1f}'])
# mean row
rows.append(['MEAN',
             f'{a_mean[0]:.1f}', f'{a_mean[1]:.1f}', f'{a_mean[2]:.1f}',
             f'{b_mean[0]:.1f}', f'{b_mean[1]:.1f}',
             f'{b_mean[2]:.1f}', f'{b_mean[3]:.1f}', f'{b_mean[4]:.1f}'])

# ── draw table ────────────────────────────────────────────────────────────────
n_rows = len(rows) + 2   # +2 for two header rows
n_cols = len(cols)

# cell dimensions (normalised axes units)
CW = [0.072, 0.068, 0.068, 0.068, 0.072, 0.072, 0.072, 0.068, 0.068]
CH = 0.072
x0, y0 = 0.01, 0.01

# compute x positions
xs = [x0]
for w in CW[:-1]: xs.append(xs[-1] + w)
total_w = sum(CW)

HEADER_BG1 = '#2c3e50'
HEADER_BG2 = '#1a6b9a'
HEADER_BG3 = '#1a5276'
ROW_ODD    = '#f8f9fa'
ROW_EVEN   = 'white'
MEAN_BG    = '#dce8f5'
BEST_COL   = '#1a5276'
WORSE_COL  = '#922b21'

def cell(ax, x, y, w, h, text, bg='white', tc='#111111', fs=9,
         bold=False, border=True, ha='center', va='center'):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=bg,
                                edgecolor='#cccccc', linewidth=0.6,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x + w/2 if ha=='center' else x+0.005,
            y + h/2, text,
            ha=ha, va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal',
            color=tc, transform=ax.transAxes, clip_on=False)

total_h = n_rows * CH
# flip y: draw top → bottom
def ry(row_idx): return y0 + total_h - (row_idx+1)*CH

# ── Group header row (row 0) ──────────────────────────────────────────────────
# "Sequence" spanning col 0
cell(ax, xs[0], ry(0), CW[0], CH*2, 'Seq.', bg=HEADER_BG1, tc='white', fs=9.5, bold=True)
# YOLOE+Any6D spanning cols 1-3
w_a = sum(CW[1:4])
cell(ax, xs[1], ry(0), w_a, CH, 'YOLOE + Any6D', bg=HEADER_BG2, tc='white', fs=9.5, bold=True)
# LLM+YOLOE+Any6D spanning cols 4-8
w_b = sum(CW[4:])
cell(ax, xs[4], ry(0), w_b, CH, 'LLM + YOLOE + Any6D  (Ours Full Pipeline)',
     bg=HEADER_BG3, tc='white', fs=9.5, bold=True)

# ── Sub-header row (row 1) ────────────────────────────────────────────────────
sub_labels_a = ['ADD-S', 'ADD', 'AR']
sub_labels_b = ['Det(%)', 'IoU(%)', 'ADD-S', 'ADD', 'AR']
for i, lbl in enumerate(sub_labels_a):
    cell(ax, xs[1+i], ry(1), CW[1+i], CH, lbl, bg=HEADER_BG2, tc='white', fs=9, bold=True)
for i, lbl in enumerate(sub_labels_b):
    cell(ax, xs[4+i], ry(1), CW[4+i], CH, lbl, bg=HEADER_BG3, tc='white', fs=9, bold=True)

# ── Data rows ─────────────────────────────────────────────────────────────────
for ri, row in enumerate(rows):
    y = ry(ri + 2)
    is_mean = (row[0] == 'MEAN')
    bg_base = MEAN_BG if is_mean else (ROW_ODD if ri % 2 == 0 else ROW_EVEN)

    for ci, (val, w, xc) in enumerate(zip(row, CW, xs)):
        bg = bg_base
        tc = '#111111'
        bold = is_mean

        if not is_mean and ci > 0:
            try:
                v = float(val)
                # compare LLM pipeline (cols 6,7,8) vs YOLOE+Any6D (cols 1,2,3)
                if ci in (6, 7, 8):   # ADD-S, ADD, AR of LLM pipeline
                    ref_ci = ci - 5   # corresponding col in YOLOE+Any6D
                    ref_v  = float(row[ref_ci])
                    diff   = v - ref_v
                    if diff >= 0:
                        bg = '#d5f5e3'  # green tint — better or equal
                    else:
                        bg = '#fadbd8'  # red tint — worse
            except: pass

            # highlight Det% < 60 in orange
            if ci == 4:
                try:
                    if float(val) < 60: bg = '#fdebd0'
                except: pass

        cell(ax, xc, y, w, CH, val, bg=bg, tc=tc, fs=9, bold=bold)

# ── Colour legend ─────────────────────────────────────────────────────────────
legend_x, legend_y = xs[4], ry(N + 3) + 0.01
patches = [
    mpatches.Patch(facecolor='#d5f5e3', edgecolor='#999', label='Better than YOLOE+Any6D'),
    mpatches.Patch(facecolor='#fadbd8', edgecolor='#999', label='Worse than YOLOE+Any6D'),
    mpatches.Patch(facecolor='#fdebd0', edgecolor='#999', label='Low detection rate (<60%)'),
    mpatches.Patch(facecolor=MEAN_BG,   edgecolor='#999', label='Mean row'),
]
ax.legend(handles=patches, loc='lower right', bbox_to_anchor=(1.0, -0.02),
          fontsize=8.5, framealpha=0.9, ncol=4,
          facecolor='white', edgecolor='#cccccc')

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    'Table — HO3D Pose Estimation: YOLOE + Any6D  vs  LLM + YOLOE + Any6D\n'
    'Metrics: AUC of ADD-S, ADD, AR (↑ higher is better).  '
    'Det(%) = YOLOE detection rate.  IoU(%) = mean mask IoU.',
    fontsize=10.5, fontweight='bold', color='#111111', pad=14)

ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUT}")

# ── Also make a delta figure ──────────────────────────────────────────────────
OUT2 = '/workspace/results/pipeline_plots/comparison_delta.png'

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.patch.set_facecolor('white')
fig2.suptitle('Δ Metric: (LLM+YOLOE+Any6D) − (YOLOE+Any6D)  per sequence\n'
              'Positive = our full pipeline improves over YOLOE+Any6D baseline',
              fontsize=11, fontweight='bold', color='#111111')

metrics = [('ADD-S', 2, 2), ('ADD', 3, 3), ('AR', 4, 4)]
for ax2, (mname, ai, bi) in zip(axes, [('ADD-S', 0, 2), ('ADD', 1, 3), ('AR', 2, 4)]):
    deltas = [B[s][bi] - A[s][ai] for s in SEQS]
    colors = ['#1a7a30' if d >= 0 else '#c0392b' for d in deltas]
    bars = ax2.barh(SEQS, deltas, color=colors, edgecolor='white', linewidth=0.5)
    ax2.axvline(0, color='#333333', linewidth=1.2)
    ax2.set_title(f'Δ {mname}', fontweight='bold', fontsize=11)
    ax2.set_xlabel(f'Δ {mname} (pp)', fontsize=9)
    ax2.tick_params(labelsize=9)
    ax2.set_facecolor('white')
    for sp in ax2.spines.values(): sp.set_edgecolor('#cccccc')
    ax2.grid(axis='x', alpha=0.3)
    # annotate
    for bar, d in zip(bars, deltas):
        ax2.text(d + (0.5 if d >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                 f'{d:+.1f}', ha='left' if d >= 0 else 'right', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUT2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUT2}")
