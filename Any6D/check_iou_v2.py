import json, os, cv2, numpy as np, sys
sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE as _YOLOE

YCBV    = '/workspace/dataset/ycbv/test'
RESULTS = '/workspace/results/ycbv_pipeline'
os.makedirs(RESULTS, exist_ok=True)
selected = json.load(open('/workspace/results/ycbv_eval_set_v2.json'))

orig = os.getcwd()
os.chdir('/workspace/yoloe')
model = _YOLOE('yoloe-26l-seg.pt')
os.chdir(orig)

def detect(img_bgr, prompt):
    H, W = img_bgr.shape[:2]
    model.set_classes([prompt], model.get_text_pe([prompt]))
    for thr in [0.1, 0.05, 0.03]:
        res = model.predict(img_bgr, conf=thr, verbose=False)
        if not len(res[0].boxes): continue
        score = res[0].boxes.conf[0].item()
        mask = cv2.resize(res[0].masks.data[0].cpu().numpy(), (W, H)) > 0.5
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = mask.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.erode(m, k, iterations=1)
        return {'mask': m > 127, 'score': score}
    return None

def iou(p, g):
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union > 0 else 0.0

logs = []
ious = []
for s in selected:
    img_bgr = cv2.imread(s['img_path'])
    mg = cv2.imread(s['mask_gt_path'], cv2.IMREAD_GRAYSCALE)
    mask_gt = mg > 127 if mg is not None else None
    det = detect(img_bgr, s['prompt'])
    iou_val = iou(det['mask'], mask_gt) if (det and mask_gt is not None) else 0.0
    ious.append(iou_val)
    logs.append({
        'scene': s['scene'], 'frame': s['frame'],
        'obj_id': s['obj_id'], 'obj_name': s['obj_name'],
        'yoloe_prompt': s['prompt'],
        'scene_type': s['scene_type'],
        'detected': det is not None,
        'yoloe_score': round(det['score'], 3) if det else 0,
        'mask_iou': round(iou_val, 3),
    })

obj_ious = {}
for l in logs:
    obj_ious.setdefault(l['obj_name'], []).append(l['mask_iou'])

print('=== FINAL YOLOE MASK RESULTS ON YCB-V BOP ===')
print(f"{'Object':25} {'Mean IoU':10} {'N'}")
print('-'*42)
for obj, vals in obj_ious.items():
    print(f"{obj[:25]:25} {np.mean(vals):.3f}      {len(vals)}")
print('-'*42)
print(f"{'MEAN':25} {np.mean(ious):.3f}      {len(ious)}")
print(f"Detection rate: {sum(1 for i in ious if i > 0)}/{len(ious)}")

clean_ious     = [ious[i] for i, s in enumerate(selected) if s['scene_type'] == 'clean']
cluttered_ious = [ious[i] for i, s in enumerate(selected) if s['scene_type'] == 'cluttered']

summary = {
    'dataset': 'YCB-V BOP',
    'n_frames': len(logs),
    'detection_rate': f"{sum(1 for i in ious if i > 0)}/{len(ious)}",
    'mean_iou_overall': round(float(np.mean(ious)), 3),
    'mean_iou_clean': round(float(np.mean(clean_ious)), 3),
    'mean_iou_cluttered': round(float(np.mean(cluttered_ious)), 3),
    'per_object': {obj: round(float(np.mean(vals)), 3) for obj, vals in obj_ious.items()},
    'per_frame': logs,
}
with open(f'{RESULTS}/yoloe_mask_final.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSaved: {RESULTS}/yoloe_mask_final.json')