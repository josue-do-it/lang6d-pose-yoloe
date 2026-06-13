import os, sys, ast, re, json, requests, cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE

OLLAMA_URL   = 'http://172.18.0.1:11434/api/generate'
OLLAMA_MODEL = 'pose-extractor'
YOLOE_MODEL  = '/workspace/yoloe/yoloe-26l-seg.pt'
ANCHOR_DIR   = '/workspace/anchor_results/dexycb_reference_view_ours'
RESULTS_DIR  = '/workspace/results/pipeline'

def extract_prompts(instruction):
    payload = {'model': OLLAMA_MODEL, 'prompt': instruction, 'stream': False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=30)
    raw = r.json()['response'].strip()
    try:
        p = ast.literal_eval(raw)
        if isinstance(p, list): return [str(x) for x in p]
    except: pass
    return re.findall(r'"([^"]+)"', raw) or [raw]

_model = None
def detect(image_bgr, prompts, conf=0.1):
    global _model
    if _model is None:
        orig = os.getcwd()
        os.chdir('/workspace/yoloe')
        _model = YOLOE(YOLOE_MODEL)
        os.chdir(orig)
    H, W = image_bgr.shape[:2]
    best = None
    for prompt in prompts:
        _model.set_classes([prompt], _model.get_text_pe([prompt]))
        for thr in [conf, 0.05, 0.03]:
            res = _model.predict(image_bgr, conf=thr, verbose=False)
            if not len(res[0].boxes): continue
            score = res[0].boxes.conf[0].item()
            if best and score <= best['score']: continue
            mask = cv2.resize(res[0].masks.data[0].cpu().numpy(),(W,H)) > 0.5
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            m = mask.astype(np.uint8)*255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
            m = cv2.erode(m, k, iterations=1)
            ov = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
            ov[m>127] = ov[m>127]*0.4 + np.array([0,200,0])*0.6
            best = {'prompt': prompt, 'score': score, 'mask': m>127,
                    'overlay': ov, 'mask_px': int((m>127).sum()),
                    'bbox': res[0].boxes.xyxy[0].cpu().numpy().tolist()}
            break
    return best

def run_pipeline(instruction, obj_folder, verbose=True):
    print('\n' + '='*60)
    img_path = ANCHOR_DIR + '/' + obj_folder + '/color.png'
    img = cv2.imread(img_path)
    if img is None:
        print('[ERROR] Not found: ' + img_path)
        return None
    prompts = extract_prompts(instruction)
    if verbose:
        print('[LLaMA] ' + instruction)
        print('        -> ' + str(prompts))
    result = detect(img, prompts)
    if result is None:
        print('[YOLOE] No detection')
        return None
    if verbose:
        print('[YOLOE] ' + result['prompt'] + ' score=' + str(round(result['score'],3)) + ' mask=' + str(result['mask_px']) + ' px')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = {'instruction': instruction, 'object': obj_folder,
           'prompts': prompts, 'detected_prompt': result['prompt'],
           'score': round(result['score'],4), 'mask_pixels': result['mask_px'],
           'bbox': result['bbox']}
    with open(RESULTS_DIR + '/' + obj_folder[:25] + '.json', 'w') as jf:
        json.dump(log, jf, indent=2)
    print('[JSON] Saved: ' + RESULTS_DIR + '/' + obj_folder[:25] + '.json')
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    fig.suptitle(instruction, fontsize=11)
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original image')
    axes[0].axis('off')
    axes[1].imshow(result['overlay'])
    axes[1].set_title('YOLOE: ' + result['prompt'] + ' (' + str(round(result['score'],2)) + ')')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + '/' + obj_folder[:25] + '.png', dpi=100)
    plt.close()
    print('[VIZ] Saved: ' + RESULTS_DIR + '/' + obj_folder[:25] + '.png')
    result['log'] = log
    return result

TESTS = [
    ('I want to grab the yellow bottle',  '006_mustard_bottle'),
    ('can you hand me the blue pitcher',  '019_pitcher_base'),
    ('give me the soup can',              '005_tomato_soup_can'),
    ('I need the sugar box',              '004_sugar_box'),
    ('hand me the cracker box',           '003_cracker_box'),
    ('get the bleach bottle',             '021_bleach_cleanser'),
    ('pick up the meat can',              '010_potted_meat_can'),
]

all_logs = []
for instruction, obj in TESTS:
    r = run_pipeline(instruction, obj)
    if r: all_logs.append(r['log'])

with open(RESULTS_DIR + '/pipeline_summary.json', 'w') as jf:
    json.dump(all_logs, jf, indent=2)
print('\n[SUMMARY] ' + str(len(all_logs)) + '/7 detected')
print('[SUMMARY] Saved: ' + RESULTS_DIR + '/pipeline_summary.json')
