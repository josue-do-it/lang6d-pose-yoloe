import cv2, numpy as np, os, sys

BASE_RGB  = '/workspace/tmp_sugar'
OUT       = '/workspace/results/ycbv_pipeline'
YCBV_049  = '/workspace/tmp_049'

sys.path.insert(0, '/workspace/yoloe')
os.chdir('/workspace/yoloe')
from ultralytics import YOLOE as _YOLOE
model = _YOLOE('yoloe-26l-seg.pt')
os.chdir('/workspace')

PROMPT = 'spam box'

def detect(img_bgr, prompt):
    H, W = img_bgr.shape[:2]
    model.set_classes([prompt], model.get_text_pe([prompt]))
    for thr in [0.1, 0.05, 0.03]:
        res = model.predict(img_bgr, conf=thr, verbose=False)
        if not len(res[0].boxes): continue
        score = res[0].boxes.conf[0].item()
        mask = cv2.resize(res[0].masks.data[0].cpu().numpy(), (W, H)) > 0.5
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        m = mask.astype(np.uint8)*255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.erode(m, k, iterations=1)
        return {'mask': m>127, 'score': score}
    return None

def iou(p, g):
    inter = np.logical_and(p,g).sum()
    union = np.logical_or(p,g).sum()
    return float(inter/union) if union>0 else 0.0

# frame 1: mask index 0 = obj_id=3 cracker, index 1=obj_id=6 mustard, index 2=obj_id=9 meat_can, index 3=obj_id=13 pitcher
# frame 36: same indices
for frame_num, frame_str, mask_idx in [(1,'000001',2),(36,'000036',2)]:
    img    = cv2.imread(YCBV_049 + '/rgb/' + frame_str + '.png')
    gt_m   = cv2.imread(YCBV_049 + '/mask_visib/' + '%06d_%06d.png' % (frame_num, mask_idx), cv2.IMREAD_GRAYSCALE)
    gt_mask = gt_m > 127
    print('Frame %d: img=%s gt_pixels=%d' % (frame_num, str(img.shape), gt_mask.sum()))

    det = detect(img, PROMPT)
    iou_val = iou(det['mask'], gt_mask) if det else 0.0
    score   = det['score'] if det else 0.0
    print('  YOLOE: score=%.3f IoU=%.3f' % (score, iou_val))

    ct_gt, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    p1 = img.copy()
    cv2.drawContours(p1, ct_gt, -1, (0,255,0), 3)
    cv2.putText(p1, 'RGB + GT contour', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    p2 = img.copy()
    p2[gt_mask] = (p2[gt_mask]*0.4 + np.array([0,180,0])*0.6).astype(np.uint8)
    cv2.drawContours(p2, ct_gt, -1, (0,255,0), 2)
    cv2.putText(p2, 'GT mask (010_potted_meat_can)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    p3 = img.copy()
    if det is not None:
        pred = det['mask']
        tp = np.logical_and(pred, gt_mask)
        fp = np.logical_and(pred, ~gt_mask)
        fn = np.logical_and(~pred, gt_mask)
        p3[tp] = (p3[tp]*0.3 + np.array([0,220,220])*0.7).astype(np.uint8)
        p3[fp] = (p3[fp]*0.3 + np.array([50,50,255])*0.7).astype(np.uint8)
        p3[fn] = (p3[fn]*0.3 + np.array([0,180,0])*0.7).astype(np.uint8)
        ct_p, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p3, ct_p, -1, (0,220,220), 2)
    label1 = '"spam box"  score=%.3f  IoU=%.3f' % (score, iou_val)
    cv2.putText(p3, label1, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    cv2.putText(p3, 'cyan=TP  blue=FP  green=FN', (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    canvas = np.hstack([p1, p2, p3])
    out = OUT + '/000049_' + str(frame_num) + '_9_meatviz.png'
    cv2.imwrite(out, canvas)
    print('  Saved: ' + out)
