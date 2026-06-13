"""
3 LLMs × 3 System Prompts × 45 instructions comparison
Models  : llama3.2:3b, mistral:latest, qwen2.5:3b
Prompts : (1) no system prompt  (2) simple extractor  (3) structured JSON extractor
Metrics : Det. Rate, Conf. Score, Mean IoU (vs XMem GT), Prompt Acc. (IoU > 0.3)
"""
import os, sys, json, re, requests, cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_SET_PATH = '/workspace/dataset/ycbv_eval/eval_set_9x5.json'
SSD_BASE      = '/workspace/dataset/ycbv_eval'
DOCKER_BASE   = '/workspace/dataset/ycbv_eval'
OLLAMA_URL    = 'http://172.18.0.1:11434/api/generate'
RESULTS_DIR   = '/workspace/results/llm_3x3_comparison'
os.makedirs(RESULTS_DIR, exist_ok=True)

IOU_THRESHOLD = 0.3

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = ['llama3.2:3b', 'mistral:latest', 'qwen2.5:3b']

# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {

    'no_system': None,   # raw model, no system description

    'simple_extractor': """You are an object name extractor for a robot vision system.

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
The snack packaging that looks like a small cereal box → cracker box""",

    'structured_json': """You are an object name extractor for a robot vision system.

Given a natural language instruction, extract the following fields:
A) Object name (1-3 words, lowercase)
B) Visual descriptors: colour, shape, material, brand
E) Object category: condiment / food / container / tool / electronic / fruit / other

Return ONLY a JSON object. No explanation. No markdown.

{
  "A": "object name",
  "B": ["descriptor1", "descriptor2"],
  "E": "category"
}

EXAMPLES:
{"instruction": "I want to season my hot dog, can you get the thing I need?", "A": "mustard bottle", "B": ["yellow", "squeeze", "plastic"], "E": "condiment"}
{"instruction": "That yellow squeeze thing on the counter, can you pass it?", "A": "mustard bottle", "B": ["yellow", "squeeze"], "E": "condiment"}
{"instruction": "The tall plastic container used for condiments, bring it here", "A": "condiment bottle", "B": ["tall", "plastic"], "E": "condiment"}
{"instruction": "Grab the French brand condiment in the yellow container", "A": "mustard bottle", "B": ["yellow", "french brand", "heinz"], "E": "condiment"}
{"instruction": "That round tin can with the soup inside, can you reach it?", "A": "soup can", "B": ["round", "metal", "campbell"], "E": "food"}
{"instruction": "The metal cylinder with the red and white label, please", "A": "tomato soup can", "B": ["cylindrical", "metal", "red", "white"], "E": "food"}
{"instruction": "I want a snack, can you bring me that rectangular cardboard box?", "A": "cracker box", "B": ["rectangular", "cardboard", "nabisco"], "E": "food"}
{"instruction": "That thing like a cereal box but smaller, with snacks", "A": "cracker box", "B": ["small", "rectangular", "cardboard"], "E": "food"}
{"instruction": "That Domino branded box in orange, what is in it is sweet", "A": "sugar box", "B": ["orange", "cardboard", "domino"], "E": "food"}
{"instruction": "Grab whatever has the white granules used for baking and drinks", "A": "sugar box", "B": ["white", "cardboard"], "E": "food"}
{"instruction": "I need protein, grab that flat rectangular tin with meat", "A": "potted meat can", "B": ["flat", "rectangular", "metal", "spam"], "E": "food"}
{"instruction": "The blue labelled canned meat product on the table", "A": "potted meat can", "B": ["blue", "flat", "metal"], "E": "food"}
{"instruction": "I am thirsty, can you bring the large vessel for pouring water?", "A": "pitcher", "B": ["large", "blue", "plastic"], "E": "container"}
{"instruction": "That thing that looks like a jug, it is blue coloured", "A": "pitcher base", "B": ["blue", "plastic", "ceramic"], "E": "container"}
{"instruction": "Wide-mouthed blue container for serving beverages", "A": "pitcher base", "B": ["blue", "wide", "glass"], "E": "container"}
{"instruction": "I need to clean the sink, bring me the cleaning product bottle", "A": "bleach cleanser", "B": ["white", "plastic", "tall", "ajax"], "E": "tool"}
{"instruction": "That cleaning product in the white plastic bottle", "A": "bleach cleanser", "B": ["white", "plastic"], "E": "tool"}
{"instruction": "I need the Ajax or similar cleaning agent in a white bottle", "A": "ajax cleanser", "B": ["white", "plastic", "ajax"], "E": "tool"}
{"instruction": "Find the red cordless drill and grasp it", "A": "red drill", "B": ["red", "plastic", "metal"], "E": "tool"}
{"instruction": "Bring me the banana on the table", "A": "banana", "B": ["yellow", "curved", "small"], "E": "fruit"}""",
}

PROMPT_LABELS = {
    'no_system':        'No System',
    'simple_extractor': 'Simple Extractor',
    'structured_json':  'Structured JSON',
}

# ── LLM call ───────────────────────────────────────────────────────────────────
def call_ollama(model, instruction, system_prompt):
    if system_prompt:
        payload = {
            'model': model,
            'system': system_prompt,
            'prompt': instruction,
            'stream': False,
            'options': {'temperature': 0.1, 'num_predict': 40}
        }
    else:
        payload = {
            'model': model,
            'prompt': f"Extract the object name to grab (2-4 words max): {instruction}",
            'stream': False,
            'options': {'temperature': 0.1, 'num_predict': 20}
        }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
        return r.json().get('response', '').strip()
    except Exception as e:
        return f"ERROR:{e}"

def parse_prompt(raw, prompt_key):
    """Extract YOLOE-usable string from LLM response."""
    raw = raw.strip().strip('"').strip("'")
    if prompt_key == 'structured_json':
        # try to parse JSON and extract field "A"
        try:
            # find JSON block
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                return str(obj.get('A', raw)).strip().lower()
        except Exception:
            pass
        # fallback: grep for "A": "..."
        m = re.search(r'"A"\s*:\s*"([^"]+)"', raw)
        if m:
            return m.group(1).strip().lower()
    # for no_system and simple_extractor: clean up and take first line
    first_line = raw.split('\n')[0].strip()
    # remove arrows, colons, bullet points
    first_line = re.sub(r'^[→\-\*•:]+\s*', '', first_line)
    first_line = re.sub(r'\s*→.*$', '', first_line)
    return first_line[:60].strip().lower()

# ── YOLOE ──────────────────────────────────────────────────────────────────────
_yoloe_model = None

def get_yoloe_model():
    global _yoloe_model
    if _yoloe_model is None:
        orig = os.getcwd()
        os.chdir('/workspace/yoloe')
        _yoloe_model = YOLOE("yoloe-26l-seg.pt")
        os.chdir(orig)
    return _yoloe_model

def yoloe_detect(img_bgr, prompt, H, W):
    model = get_yoloe_model()
    model.set_classes([prompt], model.get_text_pe([prompt]))
    best_mask, best_conf = None, 0.0
    for conf_th in [0.1, 0.05, 0.03]:
        results = model.predict(img_bgr, conf=conf_th, verbose=False)
        if len(results[0].boxes) > 0:
            confs = results[0].boxes.conf.cpu().numpy()
            idx   = int(np.argmax(confs))
            best_conf = float(confs[idx])
            m = results[0].masks.data[idx].cpu().numpy()
            best_mask = cv2.resize(m, (W, H)) > 0.5
            break
    return best_mask, best_conf

def compute_iou(mask_pred, mask_gt):
    if mask_pred is None or mask_gt is None:
        return 0.0
    inter = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# ── Load eval set ──────────────────────────────────────────────────────────────
with open(EVAL_SET_PATH) as f:
    eval_set = json.load(f)

def resolve_path(rel):
    p = os.path.join(SSD_BASE, rel)
    if os.path.exists(p):
        return p
    return os.path.join(DOCKER_BASE, rel)

# ── Main experiment ────────────────────────────────────────────────────────────
results = {}   # results[prompt_key][model] = list of per-instruction dicts

for prompt_key, system_prompt in SYSTEM_PROMPTS.items():
    results[prompt_key] = {}
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  {PROMPT_LABELS[prompt_key]}  |  {model}")
        print(f"{'='*60}")
        model_results = []

        for obj_entry in eval_set['objects']:
            obj_name  = obj_entry['obj_name']
            img_path  = resolve_path(obj_entry['img_path'])
            mask_path = resolve_path(obj_entry['mask_path'])

            if not os.path.exists(img_path):
                print(f"  MISSING image: {img_path}")
                continue

            img_bgr = cv2.imread(img_path)
            H, W    = img_bgr.shape[:2]

            gt_mask = None
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0)

            for instr in obj_entry['instructions']:
                raw_resp = call_ollama(model, instr, system_prompt)
                yoloe_prompt = parse_prompt(raw_resp, prompt_key)

                pred_mask, conf = yoloe_detect(img_bgr, yoloe_prompt, H, W)
                iou = compute_iou(pred_mask, gt_mask)

                row = {
                    'obj_name':     obj_name,
                    'instruction':  instr,
                    'llm_raw':      raw_resp,
                    'yoloe_prompt': yoloe_prompt,
                    'detected':     pred_mask is not None,
                    'conf':         round(conf, 4),
                    'iou':          round(iou, 4),
                    'prompt_acc':   iou >= IOU_THRESHOLD,
                    'img_path':     img_path,
                    'mask_path':    mask_path,
                    'pred_mask':    pred_mask,   # kept in memory for plots
                    'gt_mask':      gt_mask,
                }
                model_results.append(row)
                status = f"✓ iou={iou:.2f}" if pred_mask is not None else "✗ no det"
                print(f"  [{obj_name:20s}] \"{yoloe_prompt:25s}\" → {status}")

        results[prompt_key][model] = model_results

# ── Save raw results (strip numpy arrays before serialization) ─────────────────
SKIP_KEYS = {'pred_mask', 'gt_mask'}
serializable = {
    pk: {
        model: [{k: v for k, v in r.items() if k not in SKIP_KEYS} for r in rows]
        for model, rows in models_dict.items()
    }
    for pk, models_dict in results.items()
}
raw_out = os.path.join(RESULTS_DIR, 'raw_results.json')
with open(raw_out, 'w') as f:
    json.dump(serializable, f, indent=2)
print(f"\nRaw results saved: {raw_out}")

# ── Aggregate metrics ──────────────────────────────────────────────────────────
summary_rows = []
for prompt_key in SYSTEM_PROMPTS:
    for model in MODELS:
        rows = results[prompt_key].get(model, [])
        if not rows:
            continue
        n         = len(rows)
        det_rate  = sum(r['detected']    for r in rows) / n * 100
        conf_mean = np.mean([r['conf']   for r in rows]) * 100
        iou_mean  = np.mean([r['iou']    for r in rows]) * 100
        acc       = sum(r['prompt_acc']  for r in rows) / n * 100
        summary_rows.append({
            'System Prompt':    PROMPT_LABELS[prompt_key],
            'Model':            model,
            'Det. Rate (%)':    round(det_rate,  1),
            'Conf. Score (%)':  round(conf_mean, 1),
            'Mean IoU (%)':     round(iou_mean,  1),
            'Prompt Acc. (%)':  round(acc,        1),
            'N':                n,
        })
        print(f"{PROMPT_LABELS[prompt_key]:20s} | {model:15s} | "
              f"Det={det_rate:.1f}%  Conf={conf_mean:.1f}%  "
              f"IoU={iou_mean:.1f}%  Acc={acc:.1f}%")

df_summary = pd.DataFrame(summary_rows)
xlsx_out = os.path.join(RESULTS_DIR, '0_summary.xlsx')
df_summary.to_excel(xlsx_out, index=False)
print(f"\nSummary saved: {xlsx_out}")

# ── Per-object breakdown ───────────────────────────────────────────────────────
obj_rows = []
for prompt_key in SYSTEM_PROMPTS:
    for model in MODELS:
        rows = results[prompt_key].get(model, [])
        by_obj = {}
        for r in rows:
            by_obj.setdefault(r['obj_name'], []).append(r)
        for obj, obj_rows_list in by_obj.items():
            n = len(obj_rows_list)
            obj_rows.append({
                'System Prompt':   PROMPT_LABELS[prompt_key],
                'Model':           model,
                'Object':          obj,
                'Det. Rate (%)':   round(sum(r['detected']   for r in obj_rows_list)/n*100, 1),
                'Conf. Score (%)': round(np.mean([r['conf']  for r in obj_rows_list])*100,  1),
                'Mean IoU (%)':    round(np.mean([r['iou']   for r in obj_rows_list])*100,  1),
                'Prompt Acc. (%)': round(sum(r['prompt_acc'] for r in obj_rows_list)/n*100, 1),
            })

pd.DataFrame(obj_rows).to_excel(os.path.join(RESULTS_DIR, '0_per_object.xlsx'), index=False)

# ── Plots ──────────────────────────────────────────────────────────────────────
MODEL_COLORS  = {'llama3.2:3b': '#5B8BD4', 'mistral:latest': '#E8843A', 'qwen2.5:3b': '#4BAF7A'}
PROMPT_KEYS   = list(SYSTEM_PROMPTS.keys())
METRIC_KEYS   = ['Det. Rate (%)', 'Conf. Score (%)', 'Mean IoU (%)', 'Prompt Acc. (%)']
METRIC_LABELS = ['Det. Rate (%)', 'Conf. Score (%)', 'Mean IoU (vs XMem GT) (%)', 'Prompt Acc. (IoU>0.3) (%)']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
x = np.arange(len(PROMPT_KEYS))
width = 0.25

for ax_i, (metric, label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS)):
    ax = axes[ax_i]
    for m_i, model in enumerate(MODELS):
        vals = []
        for pk in PROMPT_KEYS:
            row = df_summary[(df_summary['System Prompt'] == PROMPT_LABELS[pk]) &
                             (df_summary['Model'] == model)]
            vals.append(float(row[metric].values[0]) if len(row) > 0 else 0)
        bars = ax.bar(x + m_i * width, vals, width, label=model,
                      color=MODEL_COLORS[model], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels([PROMPT_LABELS[pk] for pk in PROMPT_KEYS], fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(label, fontsize=10, fontweight='bold')

plt.suptitle('LLM × System Prompt Comparison — YOLOE Detection\n'
             '3 models × 3 system prompts × 45 instructions (9 objects × 5)',
             fontsize=12, y=1.01)
plt.tight_layout()
plot_out = os.path.join(RESULTS_DIR, '0_comparison_plot.png')
plt.savefig(plot_out, dpi=150, bbox_inches='tight')
print(f"Plot saved: {plot_out}")

# ── Plot 2: IoU heatmap per object × model (best system prompt per model) ──────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
all_objs = [o['obj_name'] for o in eval_set['objects']]
for ax, model in zip(axes2, MODELS):
    heat = []
    for pk in PROMPT_KEYS:
        row_vals = []
        rows_m = results[pk].get(model, [])
        for obj in all_objs:
            obj_r = [r for r in rows_m if r['obj_name'] == obj]
            row_vals.append(np.mean([r['iou'] for r in obj_r]) * 100 if obj_r else 0)
        heat.append(row_vals)
    heat = np.array(heat)
    im = ax.imshow(heat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(all_objs)))
    ax.set_xticklabels([o.replace('_', '\n') for o in all_objs], fontsize=7)
    ax.set_yticks(range(len(PROMPT_KEYS)))
    ax.set_yticklabels([PROMPT_LABELS[pk] for pk in PROMPT_KEYS], fontsize=8)
    ax.set_title(model, fontsize=10, fontweight='bold')
    for i in range(len(PROMPT_KEYS)):
        for j in range(len(all_objs)):
            ax.text(j, i, f'{heat[i,j]:.0f}', ha='center', va='center', fontsize=7,
                    color='black' if 20 < heat[i,j] < 80 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('Mean IoU (%) per Object × System Prompt — by Model', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '1_iou_heatmap.png'), dpi=150, bbox_inches='tight')
print("Saved: 1_iou_heatmap.png")

# ── Plot 3: Radar chart — model × system prompt on 4 metrics ───────────────────
from matplotlib.patches import FancyArrowPatch
RADAR_METRICS = ['Det. Rate (%)', 'Conf. Score (%)', 'Mean IoU (%)', 'Prompt Acc. (%)']
angles = np.linspace(0, 2 * np.pi, len(RADAR_METRICS), endpoint=False).tolist()
angles += angles[:1]
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
for ax, prompt_key in zip(axes3, PROMPT_KEYS):
    for model in MODELS:
        row = df_summary[(df_summary['System Prompt'] == PROMPT_LABELS[prompt_key]) &
                         (df_summary['Model'] == model)]
        if len(row) == 0:
            continue
        vals = [float(row[m].values[0]) / 100 for m in RADAR_METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, color=MODEL_COLORS[model], linewidth=2, label=model)
        ax.fill(angles, vals, color=MODEL_COLORS[model], alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Det.Rate', 'Conf.', 'IoU', 'Acc.'], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(PROMPT_LABELS[prompt_key], fontsize=11, pad=15, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.suptitle('Radar: 4 Metrics per Model × System Prompt', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '2_radar_chart.png'), dpi=150, bbox_inches='tight')
print("Saved: 2_radar_chart.png")

# ── Plot 4: 5 FAILURE cases (lowest IoU, detected=False or IoU<0.1) ────────────
def make_overlay(img_bgr, pred_mask, gt_mask):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    vis = img_rgb.copy()
    if gt_mask is not None:
        vis[gt_mask] = (vis[gt_mask] * 0.5 + np.array([0, 0, 200]) * 0.5).astype(np.uint8)
    if pred_mask is not None:
        vis[pred_mask] = (vis[pred_mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
    return vis

# collect all rows with masks across all prompts/models
all_rows_with_img = []
for pk in PROMPT_KEYS:
    for model in MODELS:
        for r in results[pk].get(model, []):
            if r.get('img_path') and os.path.exists(r['img_path']):
                all_rows_with_img.append((pk, model, r))

# failures: no detection OR very low IoU
failures = sorted(
    [(pk, m, r) for pk, m, r in all_rows_with_img if not r['detected'] or r['iou'] < 0.1],
    key=lambda x: x[2]['iou']
)[:5]

fig4, axes4 = plt.subplots(1, 5, figsize=(25, 5))
fig4.suptitle('5 Failure Cases (No Detection or IoU < 0.1)\n'
              'Blue = GT mask  |  Green = YOLOE mask', fontsize=11)
for ax, (pk, model, r) in zip(axes4, failures):
    img_bgr = cv2.imread(r['img_path'])
    vis = make_overlay(img_bgr, r.get('pred_mask'), r.get('gt_mask'))
    ax.imshow(vis)
    instr_short = r['instruction'][:45] + '…' if len(r['instruction']) > 45 else r['instruction']
    ax.set_title(
        f"{r['obj_name'].replace('_',' ')}\n"
        f"Model: {model.split(':')[0]}\n"
        f"Prompt: \"{PROMPT_LABELS[pk]}\"\n"
        f"YOLOE: \"{r['yoloe_prompt']}\"\n"
        f"IoU={r['iou']:.2f}  Det={'✓' if r['detected'] else '✗'}",
        fontsize=6.5, loc='left')
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '3_failure_cases.png'), dpi=150, bbox_inches='tight')
print("Saved: 3_failure_cases.png")

# ── Plot 5: 5 SUCCESS cases (highest IoU, detected=True, IoU>0.6) ──────────────
successes = sorted(
    [(pk, m, r) for pk, m, r in all_rows_with_img if r['detected'] and r['iou'] > 0.5],
    key=lambda x: -x[2]['iou']
)[:5]

fig5, axes5 = plt.subplots(1, 5, figsize=(25, 5))
fig5.suptitle('5 Best Success Cases (Highest IoU)\n'
              'Blue = GT mask  |  Green = YOLOE mask', fontsize=11)
for ax, (pk, model, r) in zip(axes5, successes):
    img_bgr = cv2.imread(r['img_path'])
    vis = make_overlay(img_bgr, r.get('pred_mask'), r.get('gt_mask'))
    ax.imshow(vis)
    ax.set_title(
        f"{r['obj_name'].replace('_',' ')}\n"
        f"Model: {model.split(':')[0]}\n"
        f"Prompt: \"{PROMPT_LABELS[pk]}\"\n"
        f"YOLOE: \"{r['yoloe_prompt']}\"\n"
        f"IoU={r['iou']:.2f}  Conf={r['conf']:.2f}",
        fontsize=6.5, loc='left')
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '4_success_cases.png'), dpi=150, bbox_inches='tight')
print("Saved: 4_success_cases.png")

# ── Plot 6: IoU distribution violin per system prompt ─────────────────────────
fig6, axes6 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, pk in zip(axes6, PROMPT_KEYS):
    data_per_model = []
    labels = []
    for model in MODELS:
        ious = [r['iou'] * 100 for r in results[pk].get(model, [])]
        data_per_model.append(ious)
        labels.append(model.split(':')[0])
    parts = ax.violinplot(data_per_model, positions=range(len(MODELS)), showmedians=True, showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(MODEL_COLORS.values())[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(PROMPT_LABELS[pk], fontsize=10, fontweight='bold')
    ax.set_ylabel('IoU (%)', fontsize=9)
    ax.set_ylim(0, 105)
    ax.axhline(30, color='red', linestyle='--', linewidth=1, alpha=0.5, label='IoU=0.3 threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=7)
plt.suptitle('IoU Distribution per Model × System Prompt\n(red dashed = 0.3 threshold)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '5_iou_violin.png'), dpi=150, bbox_inches='tight')
print("Saved: 5_iou_violin.png")

# ── save final JSON summary ────────────────────────────────────────────────────
final_json = {
    'experiment': '3 LLMs × 3 System Prompts × 45 instructions',
    'models': MODELS,
    'system_prompts': list(SYSTEM_PROMPTS.keys()),
    'iou_threshold': IOU_THRESHOLD,
    'summary': df_summary.to_dict(orient='records'),
    'per_object': pd.DataFrame(obj_rows).to_dict(orient='records'),
}
with open(os.path.join(RESULTS_DIR, '0_FINAL_SUMMARY.json'), 'w') as f:
    # strip non-serializable keys before saving
    clean = json.loads(json.dumps(final_json, default=str))
    json.dump(clean, f, indent=2)
print("Saved: 0_FINAL_SUMMARY.json")

print("\nDone. All results in:", RESULTS_DIR)
print("\n=== FINAL SUMMARY ===")
print(df_summary.to_string(index=False))
