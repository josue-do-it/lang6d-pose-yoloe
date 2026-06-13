"""
Full pipeline for YCB-V: User instruction → LLM keyword extractor → YOLOE detection metrics
9 objects x 5 instructions = 45 evaluations.
Metrics: Det. Rate, Conf. Score, Mean IoU (vs GT mask), Prompt Accuracy (IoU > 0.3)
Uses the calibrated system prompt with 40 examples.
"""
import os, re, sys, json, cv2, requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE

EVAL_PATH   = "/workspace/dataset/ycbv_eval/eval_set_9x5.json"
IMG_BASE    = "/workspace/dataset/ycbv_eval"
OLLAMA_URL  = "http://172.18.0.1:11434/api/generate"
LLM_MODEL   = "llama3.2:3b"   # llama with calibrated system
OUT_DIR     = "/workspace/results/ycbv_pipeline"
IOU_THRESH  = 0.3
os.makedirs(OUT_DIR, exist_ok=True)

# ── Same calibrated system prompt ─────────────────────────────────────────────
CALIBRATED_SYSTEM = """\
You are a visual keyword extractor for YOLOE, an open-vocabulary object segmentation model.

Your ONLY job: extract the most precise visual keyword(s) from the user instruction.
Output ONLY the keyword — nothing else. No punctuation, no explanation, no sentence.

IMPORTANT:
- 1 word is BEST when it is specific enough (e.g. "pitcher", "mustard", "banana")
- Use 2-3 words ONLY when 1 word is too ambiguous (e.g. "spam can", "blue pitcher")
- MAXIMUM 3 words
- Always use visual, concrete nouns — NOT functions, NOT actions

Examples from our dataset (instruction → keyword):
"I'm thirsty, pass me that big blue pitcher" → pitcher
"Hand me the blue jug with the handle" → blue pitcher
"Can you grab that blue container for pouring?" → pitcher
"That blue thing on the desk for pouring water" → blue pitcher
"Pass me the flat SPAM tin on the table" → spam can
"I need that rectangular meat can over there" → spam
"Grab the potted meat tin sitting on the desk" → spam can
"That small flat tin with the label on top" → meat tin
"Hand me the white cleaning bottle please" → cleanser
"Pass me the tall bleach container" → bleach bottle
"I need that tall white bottle for cleaning" → white bottle
"The white squeeze bottle, hand it to me" → cleanser
"Give me the yellow mustard bottle on the tray" → mustard
"I need something to put on my hot dog" → mustard
"Pass me that yellow condiment squeeze bottle" → mustard bottle
"That yellow bottle standing on the wooden tray" → mustard
"I want that red cracker box behind the banana" → cracker box
"Pass me the Cheez-It snack box please" → crackers
"Grab the red rectangular cardboard box" → cracker box
"That tall red snack box on the table" → cracker box
"Hand me the sugar" → sugar
"Pass me that yellow Domino box" → sugar box
"I need to sweeten my coffee, get me that box" → sugar box
"That orange and yellow carton on the black board" → sugar box
"Give me that tomato soup can" → soup can
"I want the red and white tin can on the left" → soup can
"Pass me that Campbell's can next to the tool" → tomato soup
"The short cylindrical tin with the red label" → soup can
"Hand me that red power drill on the table" → drill
"Pass me the Black and Decker tool" → drill
"I need to drill something, get me that red tool" → red drill
"Grab the cordless red device lying on the board" → drill
"Give me that banana please" → banana
"I need a snack, can you get me the banana?" → banana
"That yellow curved fruit on the table" → banana
"The banana lying between the drill and the box" → banana
"Can you pass me the pitcher?" → pitcher
"I want that blue thing used for pouring" → blue pitcher
"The white plastic bottle on the far right" → white bottle
"That flat tin sitting next to the soup can" → spam can
"""

def call_llm(instruction: str) -> str:
    try:
        r = requests.post(OLLAMA_URL,
                          json={"model": LLM_MODEL, "stream": False,
                                "system": CALIBRATED_SYSTEM, "prompt": instruction},
                          timeout=30)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return ""

def parse_llm(raw: str) -> str:
    raw = raw.strip()
    m = re.search(r'["""]([^"""]{1,40})["""]', raw)
    if m:
        return m.group(1).strip().lower()
    m = re.search(r'→\s*(.+)$', raw)
    if m:
        return m.group(1).strip().lower()[:40]
    raw = re.sub(r'^(keyword[:\s]+|output[:\s]+|the (object|keyword|word) (is|:)\s*)', '', raw, flags=re.I)
    for line in raw.split('\n'):
        line = line.strip().lstrip('-→•*:').strip()
        line = re.sub(r'[^\w\s\'-]', '', line).strip()
        if 0 < len(line.split()) <= 4:
            return line.lower()
    return raw.split('\n')[0][:40].strip().lower()

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
    if not prompt or len(prompt.strip()) < 1:
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
    return float(inter)/float(union) if union > 0 else 0.0

# ── Load eval set ─────────────────────────────────────────────────────────────
with open(EVAL_PATH) as f:
    eval_set = json.load(f)

print(f"Running {LLM_MODEL} + calibrated system on {len(eval_set['objects'])} objects x 5 instructions")
print("="*70)

all_rows  = []
obj_masks = {}  # keep masks for plots

for obj in eval_set['objects']:
    obj_name   = obj['obj_name']
    img_path   = os.path.join(IMG_BASE, obj['img_path'])
    mask_path  = os.path.join(IMG_BASE, obj['mask_path'])

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  MISSING: {img_path}")
        continue
    H, W    = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gt_raw  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = gt_raw > 0 if gt_raw is not None else np.zeros((H,W), bool)

    print(f"\n{obj_name}")
    obj_masks[obj_name] = {'img_rgb': img_rgb, 'gt_mask': gt_mask,
                           'preds': [], 'keywords': []}

    for instr_idx, instruction in enumerate(obj['instructions']):
        raw_llm  = call_llm(instruction)
        keyword  = parse_llm(raw_llm)
        if not keyword or len(keyword.split()) > 5:
            keyword = obj_name.split('_')[-1] if '_' in obj_name else obj_name

        pred_mask, conf = yoloe_detect(img_bgr, keyword, H, W)
        iou  = compute_iou(pred_mask, gt_mask)
        acc  = float(iou >= IOU_THRESH)

        obj_masks[obj_name]['preds'].append(pred_mask)
        obj_masks[obj_name]['keywords'].append(keyword)

        row = {
            'object':      obj_name,
            'instr_idx':   instr_idx,
            'instruction': instruction,
            'llm_raw':     raw_llm,
            'keyword':     keyword,
            'detected':    pred_mask is not None,
            'conf':        round(conf, 3),
            'iou':         round(iou, 3),
            'accurate':    acc,
        }
        all_rows.append(row)
        status = "✓" if pred_mask is not None else "✗"
        print(f"  [{instr_idx}] \"{instruction[:40]:40s}\" → \"{keyword:20s}\" "
              f"{status} iou={iou:.3f}")

# ── Per-object summary ────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Object':<22} {'Det.Rate':>8} {'Conf':>6} {'IoU':>6} {'Acc(>0.3)':>10} {'Best keyword'}")
print("-"*70)

summary_rows = []
for obj_name in [o['obj_name'] for o in eval_set['objects']]:
    rows = [r for r in all_rows if r['object'] == obj_name]
    if not rows:
        continue
    det_rate = round(np.mean([r['detected'] for r in rows])*100, 1)
    conf_m   = round(np.mean([r['conf']     for r in rows])*100, 1)
    iou_m    = round(np.mean([r['iou']      for r in rows])*100, 1)
    acc_m    = round(np.mean([r['accurate'] for r in rows])*100, 1)
    best_kw  = max(rows, key=lambda r: r['iou'])['keyword']
    print(f"  {obj_name:<20} {det_rate:>8.1f} {conf_m:>6.1f} {iou_m:>6.1f} {acc_m:>10.1f}  \"{best_kw}\"")
    summary_rows.append({
        'Object': obj_name,
        'Det_Rate(%)': det_rate,
        'Conf_Score(%)': conf_m,
        'Mean_IoU(%)': iou_m,
        'Prompt_Acc(%)': acc_m,
        'Best_Keyword': best_kw,
        'LLM_Model': LLM_MODEL,
    })

print("-"*70)
global_det  = round(np.mean([r['detected'] for r in all_rows])*100,1)
global_conf = round(np.mean([r['conf']     for r in all_rows])*100,1)
global_iou  = round(np.mean([r['iou']      for r in all_rows])*100,1)
global_acc  = round(np.mean([r['accurate'] for r in all_rows])*100,1)
print(f"  {'MEAN':<20} {global_det:>8.1f} {global_conf:>6.1f} {global_iou:>6.1f} {global_acc:>10.1f}")
summary_rows.append({
    'Object':'MEAN','Det_Rate(%)':global_det,'Conf_Score(%)':global_conf,
    'Mean_IoU(%)':global_iou,'Prompt_Acc(%)':global_acc,
    'Best_Keyword':'—','LLM_Model':LLM_MODEL,
})

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame(summary_rows).to_excel(f"{OUT_DIR}/0_summary.xlsx", index=False)
pd.DataFrame(all_rows).drop(columns=['llm_raw']).to_excel(f"{OUT_DIR}/0_per_instruction.xlsx", index=False)

# save JSON (no masks)
json.dump({'llm_model': LLM_MODEL, 'iou_threshold': IOU_THRESH,
           'summary': summary_rows, 'per_instruction': [
               {k:v for k,v in r.items() if k != 'llm_raw'} for r in all_rows
           ]},
          open(f"{OUT_DIR}/0_FINAL_SUMMARY.json",'w'), indent=2)

print(f"\nSaved to {OUT_DIR}/")
print("  0_summary.xlsx  |  0_per_instruction.xlsx  |  0_FINAL_SUMMARY.json")

# ── Visual grid: best vs worst per object ─────────────────────────────────────
print("\nGenerating detection grids...")
for obj_name, data in obj_masks.items():
    img_rgb  = data['img_rgb']
    gt_mask  = data['gt_mask']
    preds    = data['preds']
    keywords = data['keywords']
    rows_obj = [r for r in all_rows if r['object'] == obj_name]

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f"{obj_name} — {LLM_MODEL} + calibrated system prompt", fontsize=13)

    for col_i, (r, pred, kw) in enumerate(zip(rows_obj, preds, keywords)):
        for row_i, (ax, overlay_mask, label) in enumerate([
            (axes[0, col_i], gt_mask,  "GT"),
            (axes[1, col_i], pred,     "YOLOE"),
        ]):
            ov = img_rgb.copy()
            if overlay_mask is not None:
                color = [0,220,0] if label=="GT" else [0,100,255]
                ov[overlay_mask] = (ov[overlay_mask]*0.45 +
                                    np.array(color)*0.55).astype('uint8')
            ax.imshow(ov)
            if label == "GT":
                ax.set_title(f'[{col_i}] "{r["instruction"][:30]}..."', fontsize=7)
            else:
                det_s = "✓" if pred is not None else "✗"
                ax.set_title(f'kw="{kw}" {det_s} IoU={r["iou"]:.2f}', fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    safe = obj_name.replace('/', '_')
    plt.savefig(f"{OUT_DIR}/grid_{safe}.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  saved grid_{safe}.png")

print("\nDone.")
