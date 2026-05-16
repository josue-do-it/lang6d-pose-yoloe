"""
yoloe_helpers.py
Shared YOLOE detection helpers.
Used by: 06_pipeline_yoloe_any6d, any6d_yoloe_pipeline.py
"""

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


def extract_best_detection(result, scene_rgb):
    """
    Extract the highest-confidence detection from a YOLOE result.

    Args:
        result    : YOLOE prediction result object
        scene_rgb : (H, W, 3) scene image (used for shape)
    Returns:
        bbox_xyxy : [x1, y1, x2, y2] or None
        ob_mask   : (H, W) boolean mask or None
        conf      : float confidence or None
    """
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
    Run YOLOE with a text prompt on a scene image.

    Args:
        model        : loaded YOLOE model (on cuda)
        scene_path   : path to scene image
        text_prompts : str or list of str
        conf         : confidence threshold
        iou          : IoU threshold for NMS
    Returns:
        bbox_xyxy : [x1, y1, x2, y2]
        ob_mask   : (H, W) boolean mask
        conf      : float
        result    : raw YOLOE result object
    """
    scene_rgb = cv2.cvtColor(cv2.imread(scene_path), cv2.COLOR_BGR2RGB)
    prompts   = [text_prompts] if isinstance(text_prompts, str) else text_prompts

    model.set_classes(prompts, model.get_text_pe(prompts))
    results = model.predict(scene_path, conf=conf, iou=iou, verbose=False)
    result  = results[0]

    print(f'[Text prompt] "{prompts}" → {len(result.boxes)} detection(s)')
    bbox_xyxy, ob_mask, best_conf = extract_best_detection(result, scene_rgb)

    if bbox_xyxy is None:
        raise ValueError(f'No object detected for: {prompts}')

    print(f'  conf={best_conf:.3f}  bbox={bbox_xyxy}  mask={ob_mask.sum()} px')
    return bbox_xyxy, ob_mask, best_conf, result


def yoloe_visual_prompt(model, scene_path, anchor_path, anchor_bbox=None, conf=0.25):
    """
    Run YOLOE with a visual prompt (anchor image) on a scene image.

    Args:
        model        : loaded YOLOE model (on cuda)
        scene_path   : path to scene image
        anchor_path  : path to anchor reference image
        anchor_bbox  : [x1, y1, x2, y2] in anchor image. None = full image.
        conf         : confidence threshold
    Returns:
        bbox_xyxy : [x1, y1, x2, y2]
        ob_mask   : (H, W) boolean mask
        conf      : float
        result    : raw YOLOE result object
    """
    scene_rgb  = cv2.cvtColor(cv2.imread(scene_path),  cv2.COLOR_BGR2RGB)
    anchor_rgb = cv2.cvtColor(cv2.imread(anchor_path), cv2.COLOR_BGR2RGB)
    h_a, w_a   = anchor_rgb.shape[:2]
    bbox       = np.array([anchor_bbox]) if anchor_bbox is not None else np.array([[0, 0, w_a, h_a]])

    results = model.predict(
        scene_path,
        refer_image=anchor_path,
        visual_prompts={'cls': [0], 'bboxes': bbox},
        predictor=YOLOEVPSegPredictor,
        conf=conf
    )
    result = results[0]

    print(f'[Visual prompt] anchor_bbox={bbox[0]} → {len(result.boxes)} detection(s)')
    bbox_xyxy, ob_mask, best_conf = extract_best_detection(result, scene_rgb)

    if bbox_xyxy is None:
        raise ValueError('No object detected!')

    print(f'  conf={best_conf:.3f}  bbox={bbox_xyxy}  mask={ob_mask.sum()} px')
    return bbox_xyxy, ob_mask, best_conf, result


def yoloe_free_prompt(model_pf, scene_path, conf=0.25):
    """
    Run YOLOE in prompt-free mode on a scene image.

    Args:
        model_pf   : loaded prompt-free YOLOE model (on cuda)
        scene_path : path to scene image
        conf       : confidence threshold
    Returns:
        bbox_xyxy : [x1, y1, x2, y2]
        ob_mask   : (H, W) boolean mask
        conf      : float
        result    : raw YOLOE result object
    """
    scene_rgb = cv2.cvtColor(cv2.imread(scene_path), cv2.COLOR_BGR2RGB)
    results   = model_pf.predict(scene_path, conf=conf, verbose=False)
    result    = results[0]

    classes = [result.names[int(c)] for c in result.boxes.cls]
    print(f'[Free prompt] → {len(result.boxes)} detection(s): {classes}')
    bbox_xyxy, ob_mask, best_conf = extract_best_detection(result, scene_rgb)

    if bbox_xyxy is None:
        raise ValueError('No object detected!')

    print(f'  conf={best_conf:.3f}  bbox={bbox_xyxy}  mask={ob_mask.sum()} px')
    return bbox_xyxy, ob_mask, best_conf, result
