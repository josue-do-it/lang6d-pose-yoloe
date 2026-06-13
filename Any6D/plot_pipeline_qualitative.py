"""
Publication-quality qualitative figure — style ORYON/Any6D papers
Shows for each object:
  Anchor image | YOLOE mask | Pose axes overlay
with instruction + prompt labels
"""
import os, cv2, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RESULTS_DIR = '/workspace/results/pipeline_test'
ANCHOR_DIR  = '/workspace/anchor_results/dexycb_reference_view_ours'

EXAMPLES = [
    ('006_mustard_bottle',   'I want to grab the yellow mustard bottle'),
    ('005_tomato_soup_can',  'give me the soup can'),
    ('003_cracker_box',      'hand me the cracker box'),
    ('021_bleach_cleanser',  'get the bleach bottle'),
    ('019_pitcher_base',     'can you hand me the blue pitcher'),
    ('010_potted_meat_can',  'pick up the meat can'),
    ('004_sugar_box',        'I need the sugar box'),
]

OBJ_SHORT = {
    '006_mustard_bottle':   'mustard bottle',
    '005_tomato_soup_can':  'tomato soup can',
    '003_cracker_box':      'cracker box',
    '021_bleach_cleanser':  'bleach cleanser',
    '019_pitcher_base':     'pitcher base',
    '010_potted_meat_can':  'potted meat can',
    '004_sugar_box':        'sugar box',
}

n = len(EXAMPLES)
fig, axes = plt.subplots(n, 3, figsize=(13, 3.2 * n))
fig.patch.set_facecolor('white')

# Column headers
col_titles = ['anchor image', 'YOLOE segmentation', 'estimated pose']
for ci, title in enumerate(col_titles):
    axes[0][ci].set_title(title, fontsize=10, pad=6,
                           color='#222222', fontweight='normal')

for ri, (obj, instruction) in enumerate(EXAMPLES):
    # Load pipeline figure panels from saved PNG
    fig_path = f'{RESULTS_DIR}/{obj}_pipeline.png'
    anchor_path = f'{ANCHOR_DIR}/{obj}/color.png'

    anchor = cv2.imread(anchor_path)
    anchor_rgb = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB) if anchor is not None else np.zeros((480,640,3),dtype=np.uint8)

    if os.path.exists(fig_path):
        # Load the 4-panel figure and crop panels 3 and 4
        full = cv2.imread(fig_path)
        full_rgb = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
        H, W = full_rgb.shape[:2]
        pw = W // 4
        panel_mask = full_rgb[:, 2*pw:3*pw, :]
        panel_pose = full_rgb[:, 3*pw:,     :]
    else:
        panel_mask = np.zeros_like(anchor_rgb)
        panel_pose = np.zeros_like(anchor_rgb)

    # Load JSON for prompt info
    json_path = f'{RESULTS_DIR}/{obj}_result.json'
    prompt = ''
    score  = 0
    t_str  = ''
    if os.path.exists(json_path):
        d = json.load(open(json_path))
        prompt = d.get('detected_prompt', '')
        score  = d.get('yoloe_score', 0)
        t = d.get('translation_m')
        if t:
            t_str = f't=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})m'

    for ci, (panel, color) in enumerate([
        (anchor_rgb,  '#333333'),
        (panel_mask,  '#2a8a3e'),
        (panel_pose,  '#2255aa'),
    ]):
        ax = axes[ri][ci]
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#dddddd')
            spine.set_linewidth(0.8)

    # Object label — left of row
    axes[ri][0].set_ylabel(OBJ_SHORT[obj], fontsize=9,
                           color='#333333', labelpad=6)

    # Instruction below anchor
    instr_short = instruction[:38] + '…' if len(instruction) > 38 else instruction
    axes[ri][0].text(0.02, 0.04, f'"{instr_short}"',
                     transform=axes[ri][0].transAxes,
                     fontsize=7, color='white', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2',
                               facecolor='#333333', alpha=0.75))

    # Prompt + score on mask panel
    if prompt:
        axes[ri][1].text(0.02, 0.04, f'"{prompt}"  {score:.2f}',
                         transform=axes[ri][1].transAxes,
                         fontsize=7.5, color='white', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='#1a5c2a', alpha=0.82))

    # Translation on pose panel
    if t_str:
        axes[ri][2].text(0.02, 0.04, t_str,
                         transform=axes[ri][2].transAxes,
                         fontsize=7.5, color='white', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='#1a3366', alpha=0.82))

plt.suptitle(
    'Qualitative results — open-vocabulary 6D pose estimation pipeline\n'
    'LLaMA 3.2  →  YOLOE  →  Any6D',
    fontsize=11, y=1.01, color='#111111')

plt.tight_layout(pad=0.5, h_pad=0.4)
out = f'{RESULTS_DIR}/qualitative_results.png'
plt.savefig(out, dpi=130, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
