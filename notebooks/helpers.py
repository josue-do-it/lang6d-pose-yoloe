import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# ─────────────────────────────────────────────────────────────
# YOLOE Helper Functions
# ─────────────────────────────────────────────────────────────

def extract_best_detection(result, scene_rgb):
    """Extract best bbox, mask and conf from a YOLOE result."""
    H, W = scene_rgb.shape[:2]
    if len(result.boxes) == 0:
        return None, None, None
    best_idx  = result.boxes.conf.argmax().item()
    bbox_xyxy = result.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    conf      = result.boxes.conf[best_idx].item()
    if result.masks is not None:
        raw_mask = result.masks.data[best_idx].cpu().numpy()
        ob_mask  = (cv2.resize(raw_mask, (W, H)) > 0.5).astype(np.bool_)
    else:
        ob_mask = np.zeros((H, W), dtype=np.bool_)
        x1, y1, x2, y2 = bbox_xyxy
        ob_mask[y1:y2, x1:x2] = True
    return bbox_xyxy, ob_mask, conf


def yoloe_text_prompt(model, scene_path, text_prompts, conf=0.25, iou=0.45):
    """
    Run YOLOE text prompt.
    Returns: bbox_xyxy, ob_mask, conf, result
    """
    scene_rgb = cv2.cvtColor(cv2.imread(scene_path), cv2.COLOR_BGR2RGB)
    
    model.set_classes(text_prompts, model.get_text_pe(text_prompts))
    results = model.predict(scene_path, conf=conf, iou=iou, verbose=False)
    result  = results[0]
    
    print(f'[Text Prompt] "{text_prompts}" → {len(result.boxes)} detections')
    bbox_xyxy, ob_mask, best_conf = extract_best_detection(result, scene_rgb)
    
    if bbox_xyxy is None:
        raise ValueError(f'No object detected for: {text_prompts}')
    
    print(f'  Best: conf={best_conf:.3f}, bbox={bbox_xyxy}')
    print(f'  Mask: {ob_mask.sum()} pixels')
    return bbox_xyxy, ob_mask, best_conf, result


def yoloe_visual_prompt(model, scene_path, anchor_path, anchor_bbox=None, conf=0.25):
    """
    Run YOLOE visual prompt.
    anchor_bbox: [x1,y1,x2,y2] in anchor image. None = full image.
    Returns: bbox_xyxy, ob_mask, conf, result
    """
    scene_rgb  = cv2.cvtColor(cv2.imread(scene_path),  cv2.COLOR_BGR2RGB)
    anchor_rgb = cv2.cvtColor(cv2.imread(anchor_path), cv2.COLOR_BGR2RGB)
    h_a, w_a   = anchor_rgb.shape[:2]
    
    bbox = np.array([anchor_bbox]) if anchor_bbox is not None else np.array([[0, 0, w_a, h_a]])
    
    results = model.predict(
        scene_path,
        refer_image=anchor_path,
        visual_prompts={'cls': [0], 'bboxes': bbox},
        predictor=YOLOEVPSegPredictor,
        conf=conf
    )
    result = results[0]
    
    print(f'[Visual Prompt] anchor_bbox={bbox[0]} → {len(result.boxes)} detections')
    bbox_xyxy, ob_mask, best_conf = extract_best_detection(result, scene_rgb)
    
    if bbox_xyxy is None:
        raise ValueError('No object detected!')
    
    print(f'  Best: conf={best_conf:.3f}, bbox={bbox_xyxy}')
    print(f'  Mask: {ob_mask.sum()} pixels')
    return bbox_xyxy, ob_mask, best_conf, result


def yoloe_free_prompt(model_pf, scene_path, conf=0.25):
    """
    Run YOLOE prompt-free.
    Returns: bbox_xyxy, ob_mask, conf, result
    """
    scene_rgb = cv2.cvtColor(cv2.imread(scene_path), cv2.COLOR_BGR2RGB)
    
    results = model_pf.predict(scene_path, conf=conf, verbose=False)
    result  = results[0]
    
    classes = [result.names[int(c)] for c in result.boxes.cls]
    print(f'[Free Prompt] → {len(result.boxes)} detections: {classes}')
    bbox_xyxy, ob_mask, best_conf = extract_best_detection(result, scene_rgb)
    
    if bbox_xyxy is None:
        raise ValueError('No object detected!')
    
    print(f'  Best: conf={best_conf:.3f}, bbox={bbox_xyxy}')
    print(f'  Mask: {ob_mask.sum()} pixels')
    return bbox_xyxy, ob_mask, best_conf, result


def visualize_detection(scene_path, anchor_path, bbox_xyxy, ob_mask, conf, title='YOLOE Detection'):
    """Visualize detection results."""
    scene_rgb  = cv2.cvtColor(cv2.imread(scene_path),  cv2.COLOR_BGR2RGB)
    anchor_rgb = cv2.cvtColor(cv2.imread(anchor_path), cv2.COLOR_BGR2RGB)
    
    scene_box = scene_rgb.copy()
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(scene_box, (x1,y1), (x2,y2), (0,255,0), 3)
    
    mask_vis = scene_rgb.copy()
    mask_vis[ob_mask] = (mask_vis[ob_mask]*0.4 + np.array([0,255,0])*0.6).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(anchor_rgb);  axes[0].set_title('Anchor');                       axes[0].axis('off')
    axes[1].imshow(scene_box);   axes[1].set_title(f'Detection (conf={conf:.2f})'); axes[1].axis('off')
    axes[2].imshow(mask_vis);    axes[2].set_title('Mask → ob_mask for Any6D');     axes[2].axis('off')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()