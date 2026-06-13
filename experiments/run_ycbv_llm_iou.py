"""
Compare LLM-generated YOLOE prompts vs hand-crafted baseline on YCB-V BOP test frames.
Metric: IoU between YOLOE segmentation mask and GT mask (mask_visib).

Models compared: manual baseline, llama3.2:3b, mistral, qwen2.5:3b, pose-extractor
Output: results/ycbv_llm_iou/llm_iou_comparison.json + .xlsx
"""
import json, os, sys, cv2, re, requests, numpy as np
from datetime import datetime

sys.path.insert(0, '/home/josue_aims_ac_za/open-vocabulary-6d-pose-yoloe/Any6D/yoloe')
from ultralytics import YOLOE as _YOLOE

# ── Paths ─────────────────────────────────────────────────────────────────────
YCBV_TEST  = '/home/josue_aims_ac_za/open-vocabulary-6d-pose-yoloe/Any6D/dataset/ycb-v/test'
EVAL_SET   = '/home/josue_aims_ac_za/open-vocabulary-6d-pose-yoloe/Any6D/results/ycbv_mini_eval.json'
RESULTS    = '/home/josue_aims_ac_za/open-vocabulary-6d-pose-yoloe/results/ycbv_llm_iou'
OLLAMA_URL = 'http://localhost:11434/api/generate'
os.makedirs(RESULTS, exist_ok=True)

# ── System prompt (validated — all 4 models return valid Python list) ─────────
YOLOE_SYSTEM_PROMPT = """You are a visual descriptor generator for YOLOE, an open-vocabulary object detection model used in robotics.

Given a user instruction, identify the target object and return a Python list of 1 to 3 short visual descriptors ordered from MOST to LEAST specific.
YOLOE will try each descriptor in order — keep them SHORT (2 to 3 words max).

Rules:
- Return ONLY the Python list, nothing else. Start immediately with [
- Each descriptor: 2 to 3 words maximum, visual and concrete
- First descriptor: brand name + object type OR color + object type (most specific)
- Last descriptor: simple object name (fallback)
- Include brand name if well-known (domino, campbell, spam, ajax)
- No verbs, no actions, no scene context, no explanations

Examples:
Instruction: "grab that yellow squeeze bottle used for seasoning"
Output: ["yellow mustard bottle", "mustard bottle", "condiment bottle"]

Instruction: "bring me that flat blue tin with meat inside"
Output: ["spam can", "blue metal tin", "canned meat"]

Instruction: "pass me the orange cardboard box with sweet powder"
Output: ["domino sugar box", "orange sugar box", "sugar box"]

Instruction: "grab the small red round metal can with soup"
Output: ["tomato soup can", "small red can", "soup can"]

Instruction: "bring that large blue plastic container for pouring"
Output: ["blue plastic pitcher", "water pitcher", "blue jug"]

Instruction: "get the tall white plastic bottle for cleaning"
Output: ["ajax white bottle", "white cleanser bottle", "cleaning bottle"]

Instruction: "pass the rectangular cardboard box with crackers"
Output: ["cracker box", "nabisco cracker box", "snack box"]

Instruction: "grab the yellow curved fruit from the bowl"
Output: ["yellow banana", "ripe banana", "banana"]

Instruction: "bring me that white ceramic cup with a handle"
Output: ["white coffee mug", "ceramic mug", "coffee cup"]"""

# ── One instruction per object (representative, slightly descriptive) ─────────
OBJ_INSTRUCTIONS = {
    '003_cracker_box':     'I want a snack — bring me that rectangular cardboard box with crackers',
    '004_sugar_box':       'I need to sweeten my coffee, bring that orange cardboard box with white powder inside',
    '005_tomato_soup_can': 'I want something warm for lunch, grab that small round red metal can',
    '006_mustard_bottle':  'I want to season my food, pass me that yellow squeeze bottle',
    '010_potted_meat_can': 'I need protein — grab that flat blue rectangular tin with meat',
    '019_pitcher_base':    'I am thirsty, bring that large blue plastic container with a handle for pouring',
    '021_bleach_cleanser': 'I need to clean the sink, bring me that tall white plastic bottle',
}

# ── Baseline hand-crafted prompts ─────────────────────────────────────────────
MANUAL_PROMPTS = {
    '003_cracker_box':     'cracker box',
    '004_sugar_box':       'domino sugar box',
    '005_tomato_soup_can': 'small red can',
    '006_mustard_bottle':  'yellow mustard bottle',
    '010_potted_meat_can': 'spam box',
    '019_pitcher_base':    'large blue pitcher',
    '021_bleach_cleanser': 'ajax white bottle',
}

MODELS = ['llama3.2:3b', 'mistral:latest', 'qwen2.5:3b', 'pose-extractor:latest']

# ── Load eval set (paths already corrected in ycbv_mini_eval.json) ───────────
eval_set = json.load(open(EVAL_SET))

# ── YOLOE (loaded once) ───────────────────────────────────────────────────────
_model = None
def get_model():
    global _model
    if _model is None:
        orig = os.getcwd()
        os.chdir('/home/josue_aims_ac_za/open-vocabulary-6d-pose-yoloe/Any6D/yoloe')
        _model = _YOLOE('yoloe-26l-seg.pt')
        os.chdir(orig)
    return _model

def detect(img_bgr, prompt):
    m = get_model()
    H, W = img_bgr.shape[:2]
    m.set_classes([prompt], m.get_text_pe([prompt]))
    for thr in [0.10, 0.05, 0.03]:
        res = m.predict(img_bgr, conf=thr, verbose=False)
        if not len(res[0].boxes):
            continue
        score = float(res[0].boxes.conf[0])
        mask  = cv2.resize(res[0].masks.data[0].cpu().numpy(), (W, H)) > 0.5
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.erode(mask, k, iterations=1)
        return {'mask': mask > 127, 'score': score}
    return None

def iou(pred, gt):
    if pred is None or gt is None:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0

# ── LLM extraction (with validated system prompt) ────────────────────────────
def extract_descriptors(model, instruction):
    payload = {
        'model':   model,
        'system':  YOLOE_SYSTEM_PROMPT,
        'prompt':  instruction,
        'stream':  False,
        'options': {'temperature': 0.1, 'num_predict': 60}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
        raw = r.json().get('response', '').strip()
    except Exception as e:
        return None, None, str(e)

    # Parse Python list
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if m:
        try:
            lst = json.loads(m.group(0).replace("'", '"'))
            if lst and isinstance(lst, list):
                return lst, lst[0], raw
        except Exception:
            pass

    # Fallback: first 3 words
    clean = re.sub(r'[\[\]"\'{}]', '', raw).strip()
    first = ' '.join(clean.split()[:3])
    return None, first if first else None, raw

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"Evaluating {len(eval_set)} frames × {len(MODELS)+1} sources")
print("=" * 65)

results = []

for entry in eval_set:
    obj = entry['obj_name']
    img = cv2.imread(entry['img_path'])
    mg  = cv2.imread(entry['mask_gt_path'], cv2.IMREAD_GRAYSCALE)
    gt  = (mg > 127) if mg is not None else None

    print(f"\n[{entry['scene']} f{entry['frame']}] {obj}")

    row = {
        'scene':      entry['scene'],
        'frame':      entry['frame'],
        'obj_name':   obj,
        'scene_type': entry['scene_type'],
        'results':    {}
    }

    # Baseline manual
    prompt = MANUAL_PROMPTS[obj]
    det = detect(img, prompt)
    row['results']['manual'] = {
        'prompt':    prompt,
        'detected':  det is not None,
        'score':     round(det['score'], 3) if det else 0.0,
        'iou':       round(iou(det['mask'] if det else None, gt), 3)
    }
    print(f"  manual           '{prompt}' → IoU={row['results']['manual']['iou']:.3f}")

    # LLM models — cascade through full list until YOLOE detects
    instruction = OBJ_INSTRUCTIONS[obj]
    for model in MODELS:
        desc_list, _, raw = extract_descriptors(model, instruction)

        # Build cascade: LLM list + manual fallback at end
        candidates = desc_list if desc_list else []
        if MANUAL_PROMPTS[obj] not in candidates:
            candidates = candidates + [MANUAL_PROMPTS[obj]]

        det, used_prompt = None, candidates[0] if candidates else MANUAL_PROMPTS[obj]
        tried = []
        for candidate in candidates:
            det = detect(img, candidate)
            tried.append(candidate)
            if det is not None:
                used_prompt = candidate
                break

        row['results'][model] = {
            'instruction':  instruction,
            'full_list':    desc_list,
            'used_prompt':  used_prompt,
            'tried':        tried,
            'raw_llm':      raw,
            'detected':     det is not None,
            'score':        round(det['score'], 3) if det else 0.0,
            'iou':          round(iou(det['mask'] if det else None, gt), 3)
        }
        tag = '✓' if det else '✗'
        cascade_info = f"(tried {len(tried)})" if len(tried) > 1 else ""
        print(f"  {model:22} '{used_prompt}' {tag} IoU={row['results'][model]['iou']:.3f} {cascade_info}")

    results.append(row)

# ── Aggregate ─────────────────────────────────────────────────────────────────
sources = ['manual'] + MODELS
summary = {}
for src in sources:
    ious_s = [r['results'][src]['iou'] for r in results]
    det_s  = [r['results'][src]['detected'] for r in results]
    by_obj = {}
    for r in results:
        by_obj.setdefault(r['obj_name'], []).append(r['results'][src]['iou'])
    summary[src] = {
        'mean_iou':       round(float(np.mean(ious_s)), 3),
        'detection_rate': f"{sum(det_s)}/{len(det_s)}",
        'per_object':     {obj: round(float(np.mean(v)), 3) for obj, v in by_obj.items()}
    }

# ── Print table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'Source':24} {'Mean IoU':10} {'Det Rate'}")
print("-" * 45)
for src, s in summary.items():
    label = src.replace(':latest', '')
    print(f"  {label:22} {s['mean_iou']:.3f}      {s['detection_rate']}")

print("\nPer-object IoU:")
objs = list(summary['manual']['per_object'].keys())
labels = [s.replace(':latest', '') for s in sources]
print(f"  {'Object':28} " + "  ".join(f"{l[:10]:10}" for l in labels))
print("  " + "-" * 75)
for obj in objs:
    row_str = f"  {obj:28} " + "  ".join(f"{summary[src]['per_object'].get(obj, 0):.3f}     " for src in sources)
    print(row_str)

# ── Save ──────────────────────────────────────────────────────────────────────
ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
out = {
    'timestamp':        ts,
    'n_frames':         len(results),
    'models_compared':  sources,
    'summary':          summary,
    'per_frame':        results
}
json_path = os.path.join(RESULTS, 'llm_iou_comparison.json')
json.dump(out, open(json_path, 'w'), indent=2)
print(f"\nSaved → {json_path}")

try:
    import pandas as pd
    rows_xl = []
    for r in results:
        row_xl = {
            'scene':      r['scene'],
            'frame':      r['frame'],
            'object':     r['obj_name'],
            'scene_type': r['scene_type']
        }
        for src in sources:
            lbl = src.replace(':latest', '')
            row_xl[f'{lbl}_prompt'] = (r['results'][src].get('used_prompt') or
                                       r['results'][src].get('prompt', ''))
            row_xl[f'{lbl}_iou']    = r['results'][src]['iou']
            row_xl[f'{lbl}_score']  = r['results'][src]['score']
        rows_xl.append(row_xl)
    df = pd.DataFrame(rows_xl)
    xl_path = os.path.join(RESULTS, 'llm_iou_comparison.xlsx')
    df.to_excel(xl_path, index=False)
    print(f"Saved → {xl_path}")
except Exception as e:
    print(f"Excel skipped: {e}")
