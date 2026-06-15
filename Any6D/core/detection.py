"""
YOLOE open-vocabulary segmentation shared across all dataset pipelines.

Singleton pattern
-----------------
YOLOE weights (~300 MB) take ~10 seconds to load onto the GPU.  Loading them
once per process and reusing the same instance across all frames cuts wall-clock
time from ~10 s/frame to ~0.1 s/frame for the detection step.

Dataset-specific differences
-----------------------------
Each dataset requires different YOLOE call parameters because the pipelines
were independently developed before this shared module was created.  The
differences are preserved exactly to guarantee identical results:

  Dataset  │ conf        │ conf_fallbacks  │ use_first_det │ bgr_input
  ─────────┼─────────────┼─────────────────┼───────────────┼──────────
  LINEMOD  │ 0.1         │ ()              │ False         │ False
  YCBV     │ None        │ ()              │ True          │ False
  HO3D     │ 0.1         │ (0.05, 0.03)   │ True          │ True

- YCBV passes conf=None because its original pipeline used YOLOE's built-in
  threshold (~0.25), which works better for YCB objects on natural backgrounds.
- HO3D uses conf_fallbacks because hand occlusion makes objects hard to detect
  at the primary threshold; retrying with lower values recovers more frames.
- HO3D passes bgr_input=True because its images are loaded with cv2.imread()
  which returns BGR, and the original pipeline passed them to YOLOE as-is.
"""
import os
from typing import Optional
import numpy as np
import cv2

from .constants import YOLOE_MODEL, YOLOE_CONF

try:
    import sys as _sys
    _sys.path.insert(0, '/workspace/yoloe')
    from ultralytics import YOLOE as _YOLOE
    _YOLOE_AVAILABLE = True
except Exception:
    _YOLOE = None
    _YOLOE_AVAILABLE = False

# Module-level singleton.  Populated on first call to detect_mask().
_yoloe_model = None


def detect_mask(
    img_rgb: np.ndarray,
    keyword: str,
    H: int,
    W: int,
    conf: Optional[float] = YOLOE_CONF,
    conf_fallbacks: tuple = (),
    use_first_det: bool = False,
    bgr_input: bool = False,
) -> tuple:
    """
    Run YOLOE open-vocabulary segmentation and return a binary mask.

    Args:
        img_rgb:        Input image in RGB order, shape (H, W, 3) uint8.
        keyword:        1-3 word text prompt (e.g. "yellow mustard bottle").
                        Empty or whitespace-only strings return (None, 0.0).
        H, W:           Output mask dimensions.  YOLOE predicts at its own
                        internal resolution; the mask is resized back to (H, W).
        conf:           YOLOE minimum confidence threshold.
                        Pass None to use YOLOE's built-in default (~0.25).
        conf_fallbacks: Extra thresholds tried in sequence if the primary
                        threshold finds nothing.  E.g. (0.05, 0.03) for HO3D
                        where heavy hand occlusion lowers detection confidence.
        use_first_det:  If True, return masks.data[0] (the first detection by
                        internal YOLOE order).  If False, return the detection
                        with the highest confidence score.  YCBV and HO3D use
                        True to match their original implementation.
        bgr_input:      If True, convert img_rgb from RGB to BGR before passing
                        to YOLOE.  Required for HO3D because its pipeline loads
                        images with cv2.imread() (which returns BGR) and passes
                        them directly — converting to RGB would change results.

    Returns:
        mask:  Boolean array (H, W), or None if nothing detected.
        score: YOLOE confidence of the chosen detection, or 0.0 if none.
    """
    if not _YOLOE_AVAILABLE or not keyword.strip():
        return None, 0.0

    global _yoloe_model
    orig_dir = os.getcwd()
    if _yoloe_model is None:
        # YOLOE's predict() changes the working directory internally; we
        # restore it after loading to avoid breaking relative path assumptions.
        os.chdir('/workspace/yoloe')
        _yoloe_model = _YOLOE(YOLOE_MODEL)
        os.chdir(orig_dir)

    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if bgr_input else img_rgb
    _yoloe_model.set_classes([keyword], _yoloe_model.get_text_pe([keyword]))

    for c in [conf] + list(conf_fallbacks):
        predict_kwargs = {'verbose': False}
        if c is not None:
            predict_kwargs['conf'] = c
        results = _yoloe_model.predict(img, **predict_kwargs)
        os.chdir(orig_dir)  # restore after predict() side-effect

        if len(results[0].boxes) > 0 and results[0].masks is not None:
            if use_first_det:
                idx   = 0
                score = float(results[0].boxes.conf.cpu().numpy()[0])
            else:
                idx   = int(results[0].boxes.conf.cpu().numpy().argmax())
                score = float(results[0].boxes.conf.cpu().numpy()[idx])
            mask = results[0].masks.data[idx].cpu().numpy()
            mask = cv2.resize(mask.astype(np.float32), (W, H)) > 0.5
            return mask, score

    return None, 0.0


def get_detection_mask(
    img_rgb: np.ndarray,
    keyword: str,
    gt_mask: Optional[np.ndarray],
    H: int,
    W: int,
    conf: Optional[float] = YOLOE_CONF,
    conf_fallbacks: tuple = (),
    use_first_det: bool = False,
    bgr_input: bool = False,
) -> tuple:
    """
    Detect the target object and return the mask used for pose estimation.

    This is the main entry point for all pipeline scripts.  It wraps
    detect_mask() and adds two behaviours:
      1. Computes IoU against a ground-truth mask when one is available
         (used to measure detection quality independently of pose accuracy).
      2. Falls back to the GT mask (or a full-image mask) when YOLOE fails,
         so that pose metrics can still be computed for every frame even when
         detection rate is low (as it is for some LINEMOD objects).

    Args:
        img_rgb:  RGB image, shape (H, W, 3).
        keyword:  YOLOE text prompt extracted by the LLM.
        gt_mask:  Ground-truth binary mask (H, W) or None.
                  Used only for IoU computation and as a fallback mask.
        H, W:     Image dimensions.
        conf, conf_fallbacks, use_first_det, bgr_input:
                  Forwarded verbatim to detect_mask() — see its docstring.

    Returns:
        mask:         The mask to pass to Any6D.  YOLOE's prediction if
                      detection succeeded, otherwise gt_mask (or ones).
        yoloe_det:    True if YOLOE found a detection above threshold.
        conf_score:   YOLOE confidence of the chosen detection (0.0 if none).
        iou:          IoU between the YOLOE mask and gt_mask.
                      -1.0 if gt_mask is None or YOLOE did not detect.
    """
    yoloe_mask, conf_score = detect_mask(img_rgb, keyword, H, W,
                                         conf=conf,
                                         conf_fallbacks=conf_fallbacks,
                                         use_first_det=use_first_det,
                                         bgr_input=bgr_input)

    if yoloe_mask is not None:
        iou = compute_iou(yoloe_mask, gt_mask)
        return yoloe_mask, True, conf_score, iou

    # Fallback: use GT mask for pose computation, flag detection as failed.
    fallback = gt_mask if gt_mask is not None else np.ones((H, W), dtype=bool)
    return fallback, False, 0.0, -1.0


def compute_iou(mask_a: np.ndarray, mask_b: Optional[np.ndarray]) -> float:
    """
    Pixel-level Intersection-over-Union between two binary masks.

    Args:
        mask_a: Predicted mask, shape (H, W), dtype bool or 0/1.
        mask_b: Reference mask, shape (H, W), or None.

    Returns:
        IoU in [0.0, 1.0], or -1.0 when mask_b is None.
        A tiny epsilon (1e-6) in the denominator prevents division-by-zero
        when both masks are empty (union = 0).
    """
    if mask_b is None:
        return -1.0
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / (union + 1e-6))
