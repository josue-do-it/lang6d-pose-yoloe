"""
9-object × 5-instruction LLM+YOLOE experiment.
Models: llama3.2:3b, mistral:latest, qwen2.5:3b, pose-extractor:latest
Metric: IoU between YOLOE segmentation and GT mask (mask_visib)
System prompt: user-defined object-name extractor (1-3 words, no list)
"""
import json, os, sys, cv2, re, requests, numpy as np
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_SET_PATH = '/home/josue_aims_ac_za/open-vocabulary-6d-pose-yoloe/results/ycbv_llm_iou/eval_set_9x5.json'

# Image base path: try SSD first, fall back to Docker workspace dataset
SSD_BASE    = '/home/josue_aims_ac_za/ssd_4tb/dataset/ycbv/test'
DOCKER_BASE = '/workspace/dataset/ycb-v/test'

OLLAMA_URL  = 'http://172.18.0.1:11434/api/generate'
RESULTS_DIR = os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/results/ycbv_llm_iou')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── User system prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an object name extractor for a robot vision system.

CONTEXT:
A user gives a natural language instruction to a robot. Your job is to extract only the name of the target object so it can be passed to an object detector.

RULES:
1. Identify the target object mentioned in the instruction.
2. Discard all task-related expressions, spatial qualifiers, and contextual information.
3. Return only a concise noun phrase of 1 to 3 words.
4. Do not explain. Do not add punctuation. Return only the object name.

EXAMPLES:
I want to drink water, bring me the bottle → bottle
Bring me the banana on the table → banana
Find the red cordless drill and grasp it → red drill
Pick up the yellow mustard next to the cup → yellow mustard
I need to clean the sink, bring the white bottle → white bottle
That blue jug for pouring water, hand it to me → blue pitcher
Grab the flat rectangular tin with meat inside → spam can
I want a snack, bring that rectangular cardboard box → cracker box
I need to sweeten my coffee, bring that container → sugar box
The round tin with the red and white label → soup can
Pass me the tall white cleaning bottle → bleach cleanser
That yellow squeeze thing on the counter → mustard bottle
I am thirsty, bring the large blue pouring container → blue pitcher
I want something warm for lunch, the cylindrical tin → soup can
Grab the Domino branded box in orange → sugar box
The crunchy biscuits in a box on the table → cracker box
I need protein, grab that flat tin with meat → meat can
Bring me the blue container with a handle → blue pitcher
That cleaning product in the white plastic bottle → bleach cleanser
The snack packaging that looks like a small cereal box → cracker box"""

MODELS = ['llama3.2:3b', 'mistral:latest', 'qwen2.5:3b', 'pose-extractor:latest']

# ── Resolve image/mask base path ──────────────────────────────────────────────
def resolve_base():
    # try SSD
    if os.path.isdir(SSD_BASE):
        return SSD_BASE
    # try Docker workspace dataset
    if os.path.isdir(DOCKER_BASE):
        return DOCKER_BASE
    raise FileNotFoundError(
        f"Dataset not found at:\n  {SSD_BASE}\n  {DOCKER_BASE}"
    )

BASE = resolve_base()
print(f"Using dataset base: {BASE}")

# ── Load eval set ─────────────────────────────────────────────────────────────
eval_set = json.load(open(EVAL_SET_PATH))
objects  = eval_set['objects']

# ── YOLOE (loaded once) ───────────────────────────────────────────────────────
sys.path.insert(0, '/workspace/yoloe')
_model = None

def get_model():
    global _model
    if _model is None:
        orig = os.getcwd()
        os.chdir('/workspace/yoloe')
        from ultralytics import YOLOE as _YOLOE
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

# ── LLM extraction ────────────────────────────────────────────────────────────
def extract_descriptor(model, instruction):
    payload = {
        'model':   model,
        'system':  SYSTEM_PROMPT,
        'prompt':  instruction,
        'stream':  False,
        'options': {'temperature': 0.1, 'num_predict': 20}
    }
    try:
        r   = requests.post(OLLAMA_URL, json=payload, timeout=30)
        raw = r.json().get('response', '').strip()
    except Exception as e:
        return None, str(e)

    # Clean: remove arrows, punctuation, take first 3 words
    clean = re.sub(r'[→\-–\|].*', '', raw)        # remove anything after arrow
    clean = re.sub(r'[^\w\s]', '', clean).strip()  # remove punctuation
    clean = ' '.join(clean.split()[:3])            # max 3 words
    return clean.lower() if clean else None, raw

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\nRunning: {len(objects)} objects × 5 instructions × {len(MODELS)} models")
print("=" * 70)

all_results = []

for obj_entry in objects:
    obj_name     = obj_entry['obj_name']
    img_full     = os.path.join(BASE, obj_entry['img_path'])
    mask_full    = os.path.join(BASE, obj_entry['mask_path'])
    instructions = obj_entry['instructions']

    img  = cv2.imread(img_full)
    mg   = cv2.imread(mask_full, cv2.IMREAD_GRAYSCALE)
    gt   = (mg > 127) if mg is not None else None

    if img is None:
        print(f"\n[SKIP] {obj_name} — image not found: {img_full}")
        continue
    if gt is None:
        print(f"\n[SKIP] {obj_name} — mask not found: {mask_full}")
        continue

    print(f"\n{'─'*70}")
    print(f"  {obj_name}  ({len(instructions)} instructions)")

    obj_result = {
        'obj_name':     obj_name,
        'img_path':     img_full,
        'mask_path':    mask_full,
        'instructions': []
    }

    for instr in instructions:
        print(f"\n  Instruction: {instr[:65]}")
        instr_result = {'instruction': instr, 'models': {}}

        for model in MODELS:
            descriptor, raw = extract_descriptor(model, instr)

            if not descriptor:
                instr_result['models'][model] = {
                    'prompt': None, 'raw_llm': raw,
                    'detected': False, 'iou': 0.0, 'conf': 0.0
                }
                print(f"    {model:25} → [FAILED TO EXTRACT]")
                continue

            det     = detect(img, descriptor)
            iou_val = iou(det['mask'] if det else None, gt)
            conf    = det['score'] if det else 0.0

            instr_result['models'][model] = {
                'prompt':   descriptor,
                'raw_llm':  raw,
                'detected': det is not None,
                'iou':      round(iou_val, 4),
                'conf':     round(conf, 4)
            }
            tag = '✓' if det else '✗'
            print(f"    {model:25} → \"{descriptor}\"  {tag}  IoU={iou_val:.3f}  conf={conf:.3f}")

        obj_result['instructions'].append(instr_result)

    all_results.append(obj_result)

# ── Aggregate ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

agg = {m: {'iou_list': [], 'det': 0, 'total': 0} for m in MODELS}
per_obj_iou = {m: {} for m in MODELS}

for obj_res in all_results:
    obj = obj_res['obj_name']
    for instr_res in obj_res['instructions']:
        for m in MODELS:
            mr = instr_res['models'].get(m, {})
            v  = mr.get('iou', 0.0)
            agg[m]['iou_list'].append(v)
            agg[m]['total'] += 1
            if mr.get('detected'):
                agg[m]['det'] += 1
            per_obj_iou[m].setdefault(obj, []).append(v)

summary = {}
print(f"\n{'Model':28} {'Mean IoU':10} {'Det Rate':12} {'Acc (IoU>0.5)'}")
print("-" * 65)
for m in MODELS:
    lst   = agg[m]['iou_list']
    mean  = round(float(np.mean(lst)), 4) if lst else 0.0
    dr    = f"{agg[m]['det']}/{agg[m]['total']}"
    acc   = sum(1 for v in lst if v >= 0.5)
    label = m.replace(':latest', '').replace(':3b', '')
    print(f"  {label:26} {mean:.4f}    {dr:12} {acc}/{len(lst)}")
    summary[m] = {
        'mean_iou': mean,
        'detection_rate': dr,
        'acc_50': f"{acc}/{len(lst)}",
        'per_object': {obj: round(float(np.mean(v)), 4) for obj, v in per_obj_iou[m].items()}
    }

print("\nPer-object mean IoU:")
objs = [o['obj_name'] for o in all_results]
labels = [m.replace(':latest','').replace(':3b','') for m in MODELS]
print(f"  {'Object':30} " + "  ".join(f"{l[:10]:>10}" for l in labels))
print("  " + "-" * 75)
for obj in objs:
    row = f"  {obj:30} "
    row += "  ".join(f"{summary[m]['per_object'].get(obj, 0):.4f}    " for m in MODELS)
    print(row)

# ── Save ──────────────────────────────────────────────────────────────────────
ts  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
out = {
    'timestamp':       ts,
    'system_prompt':   SYSTEM_PROMPT,
    'n_objects':       len(all_results),
    'n_instructions':  5,
    'models':          MODELS,
    'summary':         summary,
    'results':         all_results
}

json_path = os.path.join(RESULTS_DIR, 'llm_9x5_results.json')
json.dump(out, open(json_path, 'w'), indent=2)
print(f"\nSaved → {json_path}")
print("Done.")
