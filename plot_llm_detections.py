"""
Visual grid: YOLOE detection per LLM — 3 models x 7 objects
Complex user instructions — shows mask overlay + bbox + prompt + confidence
"""
import os, json, cv2, numpy as np, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RESULTS_DIR = os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/results/llm_comparison')
ANCHOR_BASE = os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/anchor_results/dexycb_reference_view_ours')

MODELS = ['llama3.2', 'mistral', 'qwen2.5:3b']
MODEL_LABELS = {
    'llama3.2':   'LLaMA 3.2 3B',
    'mistral':    'Mistral 7B',
    'qwen2.5:3b': 'Qwen 2.5 3B',
}
MODEL_COLORS = {
    'llama3.2':   '#5B8BD4',
    'mistral':    '#E8843A',
    'qwen2.5:3b': '#4BAF7A',
}
OBJECTS = [
    '006_mustard_bottle',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '010_potted_meat_can',
    '019_pitcher_base',
    '021_bleach_cleanser',
]
OBJ_LABELS = {
    '006_mustard_bottle':   'mustard bottle',
    '003_cracker_box':      'cracker box',
    '004_sugar_box':        'sugar box',
    '005_tomato_soup_can':  'tomato soup can',
    '010_potted_meat_can':  'potted meat can',
    '019_pitcher_base':     'pitcher base',
    '021_bleach_cleanser':  'bleach cleanser',
}

# Load results
with open(f'{RESULTS_DIR}/raw_results.json') as f:
    results = json.load(f)

# Best result per object per model (highest confidence detected)
best = {}
for obj in OBJECTS:
    best[obj] = {}
    for model in MODELS:
        best_conf = -1
        best_row  = None
        for row in results:
            if row['gt_object'] != obj:
                continue
            e = row['extractions'].get(model, {})
            if e.get('detected') and e.get('confidence', 0) > best_conf:
                best_conf = e['confidence']
                best_row  = row
        best[obj][model] = best_row

# Load YOLOE
sys.path.insert(0, os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/yoloe'))
orig = os.getcwd()
os.chdir(os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/yoloe'))
from ultralytics import YOLOE
_yoloe = YOLOE('yoloe-26l-seg.pt')
os.chdir(orig)

def detect(img_bgr, prompt):
    H, W = img_bgr.shape[:2]
    _yoloe.set_classes([prompt], _yoloe.get_text_pe([prompt]))
    for thr in [0.1, 0.05, 0.03]:
        res = _yoloe.predict(img_bgr, conf=thr, verbose=False)
        if len(res[0].boxes):
            conf = float(res[0].boxes.conf[0].item())
            box  = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
            mask = cv2.resize(res[0].masks.data[0].cpu().numpy(),
                              (W, H)) > 0.5
            return conf, box, mask
    return None, None, None

# ── Figure: 7 rows × 3 cols ───────────────────────────────────────────────────
fig, axes = plt.subplots(len(OBJECTS), len(MODELS),
                         figsize=(5 * len(MODELS), 3.8 * len(OBJECTS)))
fig.patch.set_facecolor('#0f0f0f')

for ri, obj in enumerate(OBJECTS):
    img_path = f'{ANCHOR_BASE}/{obj}/color.png'
    img_bgr  = cv2.imread(img_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W     = img_rgb.shape[:2]

    # Find a representative complex instruction for this object
    sample_instr = ''
    for row in results:
        if row['gt_object'] == obj:
            sample_instr = row['instruction']
            break

    for ci, model in enumerate(MODELS):
        ax    = axes[ri][ci]
        color = MODEL_COLORS[model]
        ax.set_facecolor('#0f0f0f')

        # Column header
        if ri == 0:
            ax.set_title(MODEL_LABELS[model], color=color,
                         fontsize=12, pad=8)

        # Row: object name left
        if ci == 0:
            ax.set_ylabel(OBJ_LABELS[obj], color='#cccccc',
                          fontsize=9, labelpad=5)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        best_row = best[obj][model]

        if best_row is None:
            ax.imshow(img_rgb)
            # Show the instruction that failed
            instr_short = sample_instr[:42] + '…' if len(sample_instr) > 42 else sample_instr
            ax.text(W/2, H/2 - 12, 'no detection',
                    ha='center', va='center', fontsize=10,
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.35',
                              facecolor='#991111', alpha=0.92))
            ax.text(5, H - 7,
                    f'instruction: "{instr_short}"',
                    color='#aaaaaa', fontsize=6.5,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='#0f0f0f', alpha=0.8))
            continue

        instruction = best_row['instruction']
        prompt      = best_row['extractions'][model]['prompt']
        conf        = best_row['extractions'][model]['confidence']

        det_conf, box, mask = detect(img_bgr, prompt)

        overlay = img_rgb.copy()
        if mask is not None:
            hex_col = color.lstrip('#')
            r, g, b = tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))
            colored = np.zeros_like(overlay)
            colored[:, :] = [r, g, b]
            overlay[mask] = (overlay[mask] * 0.38 +
                             colored[mask] * 0.62).astype(np.uint8)

        ax.imshow(overlay)

        if box is not None:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # Instruction (top, grey, small)
        instr_short = instruction[:44] + '…' if len(instruction) > 44 else instruction
        ax.text(5, 10,
                f'"{instr_short}"',
                color='#bbbbbb', fontsize=6.2, va='top',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='#0f0f0f', alpha=0.78))

        # Extracted prompt (bottom left, white)
        prompt_short = prompt[:32] + '…' if len(prompt) > 32 else prompt
        ax.text(5, H - 7,
                f'prompt: "{prompt_short}"',
                color='white', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='#0f0f0f', alpha=0.82))

        # Confidence (top right, colored)
        ax.text(W - 5, 10, f'{conf:.2f}',
                color=color, fontsize=10,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='#0f0f0f', alpha=0.82))

plt.suptitle(
    'YOLOE detection per LLM on complex user instructions\n'
    'grey text = user instruction   white text = extracted prompt   number = confidence',
    color='white', fontsize=11, y=1.003)

plt.tight_layout(pad=0.5)
out = f'{RESULTS_DIR}/llm_detection_grid.png'
plt.savefig(out, dpi=110, bbox_inches='tight', facecolor='#0f0f0f')
plt.close()
print(f'Saved: {out}')
