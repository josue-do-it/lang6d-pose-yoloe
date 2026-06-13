"""
LLM Prompt Extraction Comparison
Compares LLaMA 3.2, Mistral 7B, Qwen2.5:3B and Gemini on 50 user instructions
Measures YOLOE detection rate and confidence score per extracted prompt
"""
import os, sys, json, requests, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/yoloe')

OLLAMA_URL  = 'http://localhost:11434/api/generate'
GEMINI_KEY  = os.environ.get('GEMINI_API_KEY', '')  # set before running
RESULTS_DIR = os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/results/llm_comparison')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 50 user instructions covering all 7 objects ──────────────────────────────
USER_INSTRUCTIONS = [
    # mustard bottle (006)
    "Can you grab the yellow mustard bottle for me?",
    "I need the mustard condiment on the table",
    "Pass me that yellow squeeze bottle",
    "Get the mustard sauce container please",
    "Hand me the yellow condiment bottle",
    "Fetch the French's mustard from the shelf",
    "I want the yellow plastic bottle with the label",
    # cracker box (003)
    "Give me the cracker box on the counter",
    "Can you pick up that rectangular snack box?",
    "I need the box of crackers",
    "Grab the Cheez-It box please",
    "Pass me that cardboard snack box",
    "Get the cracker package from the table",
    "Hand me the rectangular food box",
    # sugar box (004)
    "I need the sugar container",
    "Can you get the Domino sugar box?",
    "Pass me that orange sugar package",
    "Grab the sugar carton from the shelf",
    "Get me the sweetener box",
    "Hand me the sugar supply",
    "Fetch the orange box with sugar",
    # tomato soup can (005)
    "Give me the can of tomato soup",
    "I need that red soup can",
    "Can you grab the Campbell's can?",
    "Pass me the cylindrical soup container",
    "Get the tomato soup from the shelf",
    "Hand me that tin can with soup",
    "Fetch the red and white soup tin",
    # potted meat can (010)
    "Give me the SPAM can please",
    "I need that rectangular meat tin",
    "Can you grab the canned meat?",
    "Pass me the blue label can",
    "Get the spam container from the table",
    "Hand me that flat rectangular tin",
    "Fetch the potted meat can",
    # pitcher base (019)
    "Give me the blue pitcher",
    "I need that large blue jug",
    "Can you grab the water pitcher?",
    "Pass me the blue plastic jug",
    "Get the large blue container",
    "Hand me that blue pouring vessel",
    "Fetch the blue water jug from the table",
    # bleach cleanser (021)
    "Give me the cleaning bottle",
    "I need the white Ajax bottle",
    "Can you grab the cleanser?",
    "Pass me the tall white cleaning product",
    "Get the detergent bottle from the shelf",
    "Hand me that white plastic cleaner",
    "Fetch the bleach container",
]

# GT object per instruction (0-indexed, maps to obj_name)
GT_OBJECTS = (
    ['006_mustard_bottle'] * 7 +
    ['003_cracker_box']    * 7 +
    ['004_sugar_box']      * 7 +
    ['005_tomato_soup_can']* 7 +
    ['010_potted_meat_can']* 7 +
    ['019_pitcher_base']   * 7 +
    ['021_bleach_cleanser']* 7
)

ANCHOR_IMGS = {
    '006_mustard_bottle':   os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/anchor_results/dexycb_reference_view_ours/006_mustard_bottle/color.png'),
    '003_cracker_box':      os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/anchor_results/dexycb_reference_view_ours/003_cracker_box/color.png'),
    '004_sugar_box':        os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/anchor_results/dexycb_reference_view_ours/004_sugar_box/color.png'),
    '005_tomato_soup_can':  os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/anchor_results/dexycb_reference_view_ours/005_tomato_soup_can/color.png'),
    '010_potted_meat_can':  os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/anchor_results/dexycb_reference_view_ours/010_potted_meat_can/color.png'),
    '019_pitcher_base':     os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/anchor_results/dexycb_reference_view_ours/019_pitcher_base/color.png'),
    '021_bleach_cleanser':  os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/anchor_results/dexycb_reference_view_ours/021_bleach_cleanser/color.png'),
}

SYSTEM_PROMPT = (
    "You are a robot arm assistant. Extract a short object description "
    "(2-5 words) from the user instruction that can be used as a text prompt "
    "for an object detector. Output ONLY the object description, nothing else. "
    "Example: 'Can you grab the yellow mustard bottle?' → 'yellow mustard bottle'"
)

# ── LLM extraction functions ──────────────────────────────────────────────────
def extract_ollama(model, instruction):
    payload = {
        'model': model,
        'prompt': f"Output ONLY 2-4 words describing the object to grab, nothing else: {instruction}",
        'stream': False,
        'options': {'temperature': 0.1, 'num_predict': 15}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp = r.json().get('response', '').strip().strip('"').strip("'")
        return resp[:60]
    except Exception as e:
        return f"ERROR: {e}"

def extract_gemini(instruction):
    if not GEMINI_KEY:
        return "NO_API_KEY"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"Extract the object name (2-5 words only, no explanation): {instruction}"}]}],
        "generationConfig": {"temperature": 0.1}
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        candidates = r.json().get('candidates', [])
        if candidates:
            return candidates[0]['content']['parts'][0]['text'].strip().strip('"').strip("'")[:60]
        return "NO_RESPONSE"
    except Exception as e:
        return f"ERROR: {e}"

# ── YOLOE detection ───────────────────────────────────────────────────────────
import cv2
_yoloe_model = None
def detect_yoloe(img_path, prompt):
    global _yoloe_model
    if _yoloe_model is None:
        from ultralytics import YOLOE as YOLOE
        orig = os.getcwd()
        os.chdir(os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/yoloe'))
        _yoloe_model = YOLOE('yoloe-26l-seg.pt')
        os.chdir(orig)
    img = cv2.imread(img_path)
    if img is None:
        return {'detected': False, 'confidence': 0.0, 'prompt': prompt}
    H, W = img.shape[:2]
    _yoloe_model.set_classes([prompt], _yoloe_model.get_text_pe([prompt]))
    for thr in [0.1, 0.05, 0.03]:
        res = _yoloe_model.predict(img, conf=thr, verbose=False)
        if len(res[0].boxes):
            conf = float(res[0].boxes.conf[0].item())
            return {'detected': True, 'confidence': round(conf, 3), 'prompt': prompt}
    return {'detected': False, 'confidence': 0.0, 'prompt': prompt}

# ── Main loop ─────────────────────────────────────────────────────────────────
MODELS = {
    'llama3.2': lambda x: extract_ollama('llama3.2:3b', x),
    'mistral':  lambda x: extract_ollama('mistral', x),
    'qwen2.5:3b': lambda x: extract_ollama('qwen2.5:3b', x),
    'gemini':   extract_gemini,
}

# Check which Ollama models are available
try:
    avail = requests.get('http://172.18.0.1:11434/api/tags', timeout=5).json()
    avail_names = [m['name'] for m in avail.get('models', [])]
    print('Available Ollama models:', avail_names)
except:
    avail_names = []
    print('Cannot reach Ollama')

results = []
print(f'\nRunning comparison on {len(USER_INSTRUCTIONS)} instructions × {len(MODELS)} models')
print('='*80)

for i, (instruction, gt_obj) in enumerate(zip(USER_INSTRUCTIONS, GT_OBJECTS)):
    img_path = ANCHOR_IMGS[gt_obj]
    row = {
        'id': i,
        'instruction': instruction,
        'gt_object': gt_obj,
        'extractions': {}
    }
    print(f'\n[{i+1:02d}/50] {instruction[:55]}...' if len(instruction)>55 else f'\n[{i+1:02d}/50] {instruction}')
    print(f'       GT: {gt_obj}')

    for model_name, extract_fn in MODELS.items():
        # skip unavailable ollama models
        if model_name != 'gemini' and not any(model_name in n for n in avail_names):
            print(f'  {model_name:15} SKIPPED (not installed)')
            row['extractions'][model_name] = {
                'prompt': 'NOT_INSTALLED',
                'detected': False,
                'confidence': 0.0
            }
            continue

        prompt = extract_fn(instruction)
        det    = detect_yoloe(img_path, prompt) if 'ERROR' not in prompt and 'NOT_INSTALLED' not in prompt and 'NO_API' not in prompt else {'detected': False, 'confidence': 0.0, 'prompt': prompt}
        row['extractions'][model_name] = {
            'prompt':     prompt,
            'detected':   det['detected'],
            'confidence': det['confidence'],
        }
        status = '✓' if det['detected'] else '✗'
        print(f'  {model_name:15} → "{prompt[:35]:35}" {status} conf={det["confidence"]:.3f}')
        time.sleep(0.1)

    results.append(row)

# Save raw results
with open(f'{RESULTS_DIR}/raw_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {RESULTS_DIR}/raw_results.json')

# ── Analysis ──────────────────────────────────────────────────────────────────
print('\n' + '='*80)
print('SUMMARY PER MODEL')
print('='*80)

summary = {}
for model_name in MODELS:
    det_rates = []
    confs     = []
    for row in results:
        e = row['extractions'].get(model_name, {})
        det_rates.append(1 if e.get('detected') else 0)
        confs.append(e.get('confidence', 0.0))
    summary[model_name] = {
        'detection_rate': round(np.mean(det_rates), 3),
        'mean_confidence': round(np.mean(confs), 3),
        'mean_conf_detected': round(np.mean([c for c,d in zip(confs,det_rates) if d]), 3) if any(det_rates) else 0,
        'n_detected': sum(det_rates),
        'n_total': len(det_rates),
    }
    print(f'{model_name:15} det_rate={summary[model_name]["detection_rate"]:.3f}  '
          f'mean_conf={summary[model_name]["mean_confidence"]:.3f}  '
          f'({summary[model_name]["n_detected"]}/{summary[model_name]["n_total"]} detected)')

# Per-object summary
print('\nPER OBJECT DETECTION RATE')
objects = list(ANCHOR_IMGS.keys())
obj_summary = {obj: {m: [] for m in MODELS} for obj in objects}
for row in results:
    obj = row['gt_object']
    for m in MODELS:
        e = row['extractions'].get(m, {})
        obj_summary[obj][m].append(1 if e.get('detected') else 0)

print(f"{'Object':25}", end='')
for m in MODELS:
    print(f' {m[:10]:>10}', end='')
print()
print('-'*70)
for obj in objects:
    print(f'{obj[:25]:25}', end='')
    for m in MODELS:
        rate = np.mean(obj_summary[obj][m]) if obj_summary[obj][m] else 0
        print(f' {rate:>10.2f}', end='')
    print()

with open(f'{RESULTS_DIR}/summary.json', 'w') as f:
    json.dump({'per_model': summary, 'per_object': {o: {m: float(np.mean(v)) for m,v in d.items()} for o,d in obj_summary.items()}}, f, indent=2)

# ── Plots ─────────────────────────────────────────────────────────────────────
colors = {'llama3.2': '#5B8BD4', 'mistral': '#E8843A', 'qwen2.5:3b': '#4BAF7A', 'gemini': '#9B6BB5'}

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

# 1. Detection rate bar chart
ax1 = fig.add_subplot(gs[0, 0])
models_list = list(MODELS.keys())
det_rates   = [summary[m]['detection_rate'] for m in models_list]
bars = ax1.bar(models_list, det_rates, color=[colors[m] for m in models_list], width=0.5)
ax1.set_ylim(0, 1.0)
ax1.set_ylabel('detection rate')
ax1.set_title('detection rate per model (50 instructions)')
for bar, val in zip(bars, det_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom', fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 2. Mean confidence bar chart
ax2 = fig.add_subplot(gs[0, 1])
mean_confs = [summary[m]['mean_confidence'] for m in models_list]
bars2 = ax2.bar(models_list, mean_confs, color=[colors[m] for m in models_list], width=0.5)
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('mean confidence score')
ax2.set_title('mean yoloe confidence per model')
for bar, val in zip(bars2, mean_confs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Per-object detection rate heatmap
ax3 = fig.add_subplot(gs[1, :])
obj_labels  = [o.replace('_', ' ') for o in objects]
data_matrix = np.array([[float(np.mean(obj_summary[obj][m])) for m in models_list] for obj in objects])
im = ax3.imshow(data_matrix, aspect='auto', cmap='YlGn', vmin=0, vmax=1)
ax3.set_xticks(range(len(models_list)))
ax3.set_xticklabels(models_list, fontsize=10)
ax3.set_yticks(range(len(objects)))
ax3.set_yticklabels(obj_labels, fontsize=9)
ax3.set_title('detection rate per object per model')
plt.colorbar(im, ax=ax3, fraction=0.02)
for i in range(len(objects)):
    for j in range(len(models_list)):
        ax3.text(j, i, f'{data_matrix[i,j]:.2f}', ha='center', va='center', fontsize=9,
                 color='black' if data_matrix[i,j] < 0.7 else 'white')

# 4. Confidence distribution per model (violin or box)
ax4 = fig.add_subplot(gs[2, 0])
conf_data = []
for m in models_list:
    confs = [row['extractions'].get(m,{}).get('confidence',0) for row in results]
    conf_data.append(confs)
vp = ax4.violinplot(conf_data, positions=range(len(models_list)), showmedians=True, showmeans=False)
for pc, m in zip(vp['bodies'], models_list):
    pc.set_facecolor(colors[m])
    pc.set_alpha(0.7)
ax4.set_xticks(range(len(models_list)))
ax4.set_xticklabels(models_list, fontsize=9)
ax4.set_ylabel('confidence score')
ax4.set_title('confidence score distribution')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# 5. Detection rate over instruction index (rolling)
ax5 = fig.add_subplot(gs[2, 1])
window = 7
for m in models_list:
    det_series = [1 if results[i]['extractions'].get(m,{}).get('detected') else 0 for i in range(len(results))]
    rolling    = np.convolve(det_series, np.ones(window)/window, mode='valid')
    ax5.plot(range(len(rolling)), rolling, color=colors[m], label=m, linewidth=1.5)
ax5.set_xlabel('instruction index')
ax5.set_ylabel('detection rate (7-window avg)')
ax5.set_title('detection stability across instructions')
ax5.legend(fontsize=8, frameon=False)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

plt.suptitle('LLM prompt extraction comparison — YOLOE detection performance\n50 user instructions × 4 models', fontsize=12, y=0.98)
plt.savefig(f'{RESULTS_DIR}/llm_comparison_plots.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'\nSaved plot: {RESULTS_DIR}/llm_comparison_plots.png')
print('Done.')
