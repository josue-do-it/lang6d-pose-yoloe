import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE

HO3D_YOLOE_PROMPTS = {
    "MPM10": "canned food",
    "MPM11": "spam can",
    "MPM12": "spam can",
    "MPM13": "canned food",
    "MPM14": "canned food",
    "AP10":  "blue jug",
    "AP11":  "blue pitcher",
    "AP12":  "blue object",
    "AP13":  "cup with handle",
    "AP14":  "cup with handle",
    "SB11":  "white plastic bottle",
    "SB13":  "white plastic bottle",
    "SM1":   "canned food",
}

HO3D_ROOT = "/dataset/ho3d/HO3D_data"
SAVE_DIR  = "/workspace/results/yoloe_prompt_check"
os.makedirs(SAVE_DIR, exist_ok=True)

os.chdir('/workspace/yoloe')
model = YOLOE("yoloe-26l-seg.pt")
os.chdir('/workspace')

objects = list(HO3D_YOLOE_PROMPTS.keys())
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
axes = axes.flatten()

for i, obj_f in enumerate(objects):
    prompt = HO3D_YOLOE_PROMPTS[obj_f]
    seq_dir = os.path.join(HO3D_ROOT, "evaluation", obj_f, "rgb")
    frames = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg') or f.endswith('.png')])

    # take frame from the middle
    img_path = os.path.join(seq_dir, frames[len(frames)//2])
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    model.set_classes([prompt], model.get_text_pe([prompt]))
    mask = None
    conf_used = None
    for conf in [0.1, 0.05, 0.03]:
        results = model.predict(img_bgr, conf=conf, verbose=False)
        if len(results[0].boxes) > 0:
            m = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(m, (W, H)) > 0.5
            conf_used = conf
            break

    overlay = img_rgb.copy()
    if mask is not None:
        overlay[mask] = (overlay[mask] * 0.4 + np.array([0, 220, 0]) * 0.6).astype(np.uint8)
        status = f"OK (conf={conf_used})"
        color = "green"
    else:
        status = "NO DETECTION"
        color = "red"

    axes[i].imshow(overlay)
    axes[i].set_title(f"{obj_f}\n\"{prompt}\"\n{status}", color=color, fontsize=9)
    axes[i].axis("off")
    print(f"{obj_f:6s} | {prompt:25s} | {status}")

# hide unused axes
for j in range(len(objects), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
out = os.path.join(SAVE_DIR, "prompt_check_all_objects.png")
plt.savefig(out, dpi=120)
print(f"\nSaved to {out}")
