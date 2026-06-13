"""
LLaMA → YOLOE → Any6D Pipeline
User prompt → object categories → mask → 6D pose
"""
import os
import sys
import json
import ast
import requests
import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser('~/open-vocabulary-6d-pose-yoloe/Any6D/yoloe'))
from ultralytics import YOLOE

# ── Config ────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "pose-extractor"
YOLOE_MODEL  = os.path.expanduser(
    "~/open-vocabulary-6d-pose-yoloe/Any6D/yoloe/yoloe-26l-seg.pt")

# ── Module 1 : LLaMA → prompts ───────────────────────────────────
def extract_prompts(user_instruction: str) -> list[str]:
    """
    Sends the user instruction to LLaMA pose-extractor
    and returns a list of object categories for YOLOE.

    Example:
        "I want to grab the yellow bottle"
        → ["yellow bottle", "bottle", "container", "cylinder"]
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": user_instruction,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    response.raise_for_status()
    raw = response.json()["response"].strip()

    # Parse the Python list returned by LLaMA
    try:
        prompts = ast.literal_eval(raw)
        if isinstance(prompts, list):
            return [str(p) for p in prompts]
    except Exception:
        pass

    # Fallback: extract strings between quotes
    import re
    prompts = re.findall(r'"([^"]+)"', raw)
    return prompts if prompts else [raw]


# ── Module 2 : YOLOE → best mask ─────────────────────────────────
_yoloe_model = None

def detect_object(image_bgr: np.ndarray,
                  prompts: list[str],
                  conf: float = 0.1) -> dict | None:
    """
    Tries each prompt in order and returns the best detection.

    Returns:
        {
          "prompt"  : str,            # prompt that triggered detection
          "score"   : float,          # confidence score
          "mask"    : np.ndarray,     # binary mask (H×W) bool
          "bbox"    : [x1,y1,x2,y2], # bounding box
          "mask_vis": np.ndarray,     # RGB image with green overlay
        }
    """
    global _yoloe_model
    if _yoloe_model is None:
        _yoloe_model = YOLOE(YOLOE_MODEL)

    H, W = image_bgr.shape[:2]
    best = None

    for prompt in prompts:
        _yoloe_model.set_classes([prompt],
                                  _yoloe_model.get_text_pe([prompt]))
        for threshold in [conf, 0.05, 0.03]:
            results = _yoloe_model.predict(image_bgr,
                                           conf=threshold,
                                           verbose=False)
            if len(results[0].boxes) == 0:
                continue

            score = results[0].boxes.conf[0].item()
            if best and score <= best["score"]:
                continue

            mask_raw = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask_raw, (W, H)) > 0.5

            # Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_u = mask.astype(np.uint8) * 255
            mask_u = cv2.morphologyEx(mask_u, cv2.MORPH_CLOSE, kernel)
            mask_u = cv2.morphologyEx(mask_u, cv2.MORPH_OPEN,  kernel)
            mask_u = cv2.erode(mask_u, kernel, iterations=1)
            mask   = mask_u > 127

            # Visualisation overlay
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            overlay = img_rgb.copy()
            overlay[mask] = overlay[mask] * 0.4 + np.array([0, 200, 0]) * 0.6

            best = {
                "prompt"  : prompt,
                "score"   : score,
                "mask"    : mask,
                "bbox"    : results[0].boxes.xyxy[0].cpu().numpy().tolist(),
                "mask_vis": overlay,
            }
            break

    return best


# ── Full pipeline ─────────────────────────────────────────────────
def run_pipeline(user_instruction: str,
                 image_path: str,
                 verbose: bool = True) -> dict | None:
    """
    Full pipeline: instruction → prompts → detection → mask

    Usage:
        result = run_pipeline(
            "I want to grab the yellow bottle",
            "scene.jpg"
        )
        mask  = result["mask"]    # → Any6D
        score = result["score"]
    """
    # Step 1 — LLaMA extracts prompts
    prompts = extract_prompts(user_instruction)
    if verbose:
        print(f"[LLaMA] '{user_instruction}'")
        print(f"        → {prompts}")

    # Step 2 — Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 3 — YOLOE detects
    result = detect_object(image, prompts)
    if result is None:
        print(f"[YOLOE] No detection for: {prompts}")
        return None

    if verbose:
        print(f"[YOLOE] Detected '{result['prompt']}' "
              f"score={result['score']:.3f} "
              f"mask={result['mask'].sum()} px")

    result["prompts"]     = prompts
    result["instruction"] = user_instruction
    result["image_path"]  = image_path
    return result


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    TEST_CASES = [
        ("I want to grab the yellow bottle",
         "Any6D/dataset/ho3d/HO3D_data/evaluation/MPM10/rgb/0000.jpg"),
        ("can you hand me the blue pitcher",
         "Any6D/dataset/ho3d/HO3D_data/evaluation/AP11/rgb/0000.jpg"),
        ("give me the soup can",
         "Any6D/dataset/ho3d/HO3D_data/evaluation/SM1/rgb/0000.jpg"),
    ]

    os.chdir(os.path.expanduser("~/open-vocabulary-6d-pose-yoloe"))

    for instruction, img_path in TEST_CASES:
        print(f"\n{'='*60}")
        result = run_pipeline(instruction, img_path)
        if result is None:
            continue

        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'"{instruction}"', fontsize=11)

        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original image")
        axes[0].axis("off")

        axes[1].imshow(result["mask_vis"])
        axes[1].set_title(
            f"YOLOE: '{result['prompt']}' (score={result['score']:.2f})")
        axes[1].axis("off")

        plt.tight_layout()
        name = instruction.replace(" ", "_")[:30]
        out  = f"pipeline/output_{name}.png"
        os.makedirs("pipeline", exist_ok=True)
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"[VIZ] Saved: {out}")
