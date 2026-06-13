"""
Test 4 prompt extraction methods on 5 objects x 1 frame each.
Methods:
  0. Raw          — pass full user instruction directly to YOLOE
  1. Heuristic    — regex/keyword extraction (no LLM)
  2. LLM basic    — LLM with no system prompt
  3. LLM calibrated — LLM with YOLOE-calibrated system prompt + better parser

Saves: results/test_extraction/results.json + grid_<obj>.png per object
"""

import os, sys, re, json, cv2, requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE

OLLAMA_URL  = "http://172.18.0.1:11434/api/generate"
LLM_MODEL   = "mistral:latest"          # best from prior experiment
EVAL_PATH   = "/workspace/dataset/ycbv_eval/eval_set_9x5.json"
IMG_BASE    = "/workspace/dataset/ycbv_eval"
OUT_DIR     = "/workspace/results/test_extraction"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 5 objects: pick diverse scenes + hardest instruction per object ────────────
SELECTED = [
    ("019_pitcher_base",    1),   # "Grab the tall blue plastic container with the handle..."
    ("005_tomato_soup_can", 1),   # "I want something warm for lunch, pass me that soup can."
    ("banana",              3),   # "I need a snack, can you get me that banana..."
    ("006_mustard_bottle",  3),   # "I need something to put on my hot dog..."
    ("010_potted_meat_can", 2),   # "That small flat tin that looks like it has a peel-off top..."
]

# ── Calibrated system prompt: built from actual high/low IoU results ──────────
CALIBRATED_SYSTEM = """\
You are a YOLOE visual object detector prompt extractor.
YOLOE is an open-vocabulary segmentation model trained on COCO/LVIS.
It works best with SHORT, CONCRETE, VISUAL descriptions (2-5 words).

GOOD examples — these gave high detection accuracy:
  "blue pitcher"                    ← color + object name
  "red and white tin can"           ← color + material + shape
  "tall white cleaning bottle"      ← size + color + category
  "yellow mustard bottle"           ← color + specific name
  "campbell's can"                  ← brand name works too
  "red snack box"                   ← color + category
  "yellow banana"                   ← color + name

BAD examples — these failed:
  "big blue bottle"                 ✗ too generic, no specific shape
  "soup can"                        ✗ missing color/visual detail
  "household cleaner"               ✗ functional, not visual
  "the object name is 'X'."        ✗ too verbose
  "something to pour drinks with"   ✗ describes function, not appearance

RULES:
1. Output ONLY the visual description — nothing else
2. 2-5 words maximum
3. Include color if visible
4. Use the most distinctive visual feature
5. NO punctuation, NO explanation, NO extra words
"""

# ── Heuristic extraction ──────────────────────────────────────────────────────
COLOR_WORDS = {'red','blue','yellow','green','white','black','orange','brown',
               'purple','pink','gray','grey','silver','gold','dark','light'}
OBJECT_WORDS = {'bottle','can','box','container','jar','tube','bag','tin','carton',
                'pitcher','jug','drill','banana','tool','device','cleaner','bottle'}
SKIP_WORDS   = {'grab','pass','get','hand','bring','need','want','please','the','a',
                'an','that','this','those','me','my','us','you','it','is','are',
                'on','at','in','of','to','for','with','and','or','but','can','i',
                'something','anything','there','here','next','near','beside','between',
                'left','right','side','middle','sitting','lying','standing','looking',
                'desk','table','board','keyboard','floor','shelf','tray'}

def heuristic_extract(instruction: str) -> str:
    words = re.findall(r"[a-zA-Z'&-]+", instruction.lower())
    kept = []
    for w in words:
        if w in SKIP_WORDS or len(w) <= 2:
            continue
        kept.append(w)
    # prefer color + noun combos
    result_words = []
    for w in kept:
        if w in COLOR_WORDS or w in OBJECT_WORDS:
            result_words.append(w)
        elif len(result_words) < 2 and len(w) > 4:
            result_words.append(w)
    if not result_words:
        result_words = kept[:3]
    return ' '.join(result_words[:5])

# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(instruction: str, system_prompt: str = None) -> str:
    payload = {"model": LLM_MODEL, "stream": False}
    if system_prompt:
        payload["system"] = system_prompt
    payload["prompt"] = instruction
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"

def parse_llm_output(raw: str) -> str:
    """Robust parser: handles quoted output, verbose prefix, arrows, JSON."""
    raw = raw.strip()
    # extract quoted text if present (most common llama/mistral verbosity)
    m = re.search(r'["“”]([^"“”]{2,40})["“”]', raw)
    if m:
        return m.group(1).strip().lower()
    # strip common prefixes
    raw_clean = re.sub(
        r'^(the object (name )?is|output|result|answer|extracted|description|prompt)[:\s]+',
        '', raw, flags=re.I
    )
    # take first non-empty line
    for line in raw_clean.split('\n'):
        line = line.strip().lstrip('-→•*:').strip()
        line = re.sub(r'\s*[→|]\s*.*$', '', line)   # remove trailing arrows
        line = re.sub(r'[^\w\s\'-]', '', line).strip()  # remove punctuation
        if 1 <= len(line.split()) <= 6:
            return line.lower()
    return raw_clean.split('\n')[0][:50].strip().lower()

# ── YOLOE ─────────────────────────────────────────────────────────────────────
_model = None
def get_model():
    global _model
    if _model is None:
        orig = os.getcwd()
        os.chdir('/workspace/yoloe')
        _model = YOLOE("yoloe-26l-seg.pt")
        os.chdir(orig)
    return _model

def yoloe_detect(img_bgr, prompt, H, W):
    if not prompt or len(prompt.strip()) < 2:
        return None, 0.0
    model = get_model()
    model.set_classes([prompt], model.get_text_pe([prompt]))
    for conf_th in [0.1, 0.05, 0.03]:
        results = model.predict(img_bgr, conf=conf_th, verbose=False)
        if len(results[0].boxes) > 0 and results[0].masks is not None:
            confs = results[0].boxes.conf.cpu().numpy()
            idx   = int(np.argmax(confs))
            conf  = float(confs[idx])
            m     = results[0].masks.data[idx].cpu().numpy()
            mask  = cv2.resize(m, (W, H)) > 0.5
            return mask, conf
    return None, 0.0

def compute_iou(pred, gt):
    if pred is None:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# ── Load eval set ─────────────────────────────────────────────────────────────
with open(EVAL_PATH) as f:
    eval_set = json.load(f)
obj_map = {o['obj_name']: o for o in eval_set['objects']}

# ── Run experiment ────────────────────────────────────────────────────────────
METHOD_NAMES = ["Raw", "Heuristic", "LLM basic", "LLM calibrated"]
all_results  = []

print(f"\n{'Object':<22} {'Instruction (truncated)':<42} {'Method':<15} {'Prompt':<30} {'IoU':>6}")
print("─" * 120)

for obj_name, instr_idx in SELECTED:
    obj = obj_map[obj_name]
    instruction = obj['instructions'][instr_idx]
    img_path  = os.path.join(IMG_BASE, obj['img_path'])
    mask_path = os.path.join(IMG_BASE, obj['mask_path'])

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  MISSING IMAGE: {img_path}")
        continue
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gt_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask     = gt_mask_raw > 0 if gt_mask_raw is not None else np.zeros((H, W), bool)

    row = {
        'object': obj_name,
        'instruction': instruction,
        'instruction_idx': instr_idx,
        'methods': []
    }

    # build prompts for each method
    methods_prompts = []
    # 0. Raw
    methods_prompts.append(("Raw",            instruction))
    # 1. Heuristic
    methods_prompts.append(("Heuristic",      heuristic_extract(instruction)))
    # 2. LLM basic (no system)
    raw_basic = call_llm(instruction, system_prompt=None)
    methods_prompts.append(("LLM basic",      parse_llm_output(raw_basic)))
    # 3. LLM calibrated
    raw_calib = call_llm(instruction, system_prompt=CALIBRATED_SYSTEM)
    methods_prompts.append(("LLM calibrated", parse_llm_output(raw_calib)))

    masks_for_plot = []
    for method_name, prompt in methods_prompts:
        mask, conf = yoloe_detect(img_bgr, prompt, H, W)
        iou = compute_iou(mask, gt_mask)
        row['methods'].append({
            'method': method_name,
            'prompt': prompt,
            'iou':    round(iou, 3),
            'conf':   round(conf, 3),
            'detected': mask is not None
        })
        masks_for_plot.append(mask)
        instr_short = instruction[:40] + '…' if len(instruction) > 40 else instruction
        print(f"  {obj_name:<20} {instr_short:<42} {method_name:<15} {prompt:<30} {iou:>6.3f}")

    all_results.append(row)

    # ── Plot grid: RGB + GT + 4 method masks ──────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{obj_name}\nInstruction: \"{instruction}\"", fontsize=11, wrap=True)
    axes = axes.flatten()

    # col 0: original RGB
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis('off')

    # col 1: GT mask
    gt_overlay = img_rgb.copy()
    gt_overlay[gt_mask] = (gt_overlay[gt_mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    axes[1].imshow(gt_overlay)
    axes[1].set_title("GT Mask (green)")
    axes[1].axis('off')

    # cols 2-5: each method
    colors_rgb = [(0, 120, 255), (255, 140, 0), (200, 0, 200), (255, 50, 50)]
    for mi, (method_name, prompt) in enumerate(methods_prompts):
        ax  = axes[2 + mi]
        iou = row['methods'][mi]['iou']
        mask = masks_for_plot[mi]
        overlay = img_rgb.copy()
        if mask is not None:
            c = colors_rgb[mi]
            overlay[mask] = (overlay[mask] * 0.45 + np.array(c) * 0.55).astype(np.uint8)
        # GT border
        gt_border = cv2.dilate(gt_mask.astype(np.uint8), np.ones((3,3)), iterations=2).astype(bool)
        gt_border = gt_border & ~gt_mask
        overlay[gt_border] = [0, 220, 0]
        ax.imshow(overlay)
        det_str = "✓" if mask is not None else "✗"
        ax.set_title(f"{method_name} {det_str}\n\"{prompt[:28]}\"\nIoU={iou:.3f}", fontsize=9)
        ax.axis('off')

    # legend
    patches = [mpatches.Patch(color='#00dc00', label='GT boundary')]
    fig.legend(handles=patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    safe_name = obj_name.replace('/', '_')
    out_path = os.path.join(OUT_DIR, f"grid_{safe_name}.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → saved {out_path}")
    print()

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_path = os.path.join(OUT_DIR, "results.json")
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2)

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "═" * 80)
print(f"{'SUMMARY':^80}")
print("═" * 80)
print(f"{'Object':<22}  {'Raw':>8}  {'Heuristic':>9}  {'LLM basic':>9}  {'LLM calib':>9}")
print("─" * 80)
for row in all_results:
    ious = [m['iou'] for m in row['methods']]
    best = max(range(len(ious)), key=lambda i: ious[i])
    vals = [f"{v:.3f}" + ("*" if i == best else " ") for i, v in enumerate(ious)]
    print(f"  {row['object']:<20}  {vals[0]:>8}  {vals[1]:>9}  {vals[2]:>9}  {vals[3]:>9}")

print("─" * 80)
# per-method mean
means = {}
for method_idx in range(4):
    vals = [r['methods'][method_idx]['iou'] for r in all_results]
    means[METHOD_NAMES[method_idx]] = np.mean(vals)

print(f"  {'MEAN':<20}  {means['Raw']:>8.3f}  {means['Heuristic']:>9.3f}  {means['LLM basic']:>9.3f}  {means['LLM calibrated']:>9.3f}")
print("═" * 80)
print(f"\nResults saved to {OUT_DIR}/")
print(f"  - results.json")
print(f"  - grid_<object>.png  (5 files)")
