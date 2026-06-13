"""
Post-hoc YOLOE detection stats for objects already processed without detection logging.
Runs YOLOE-only (no pose) on each frame and saves {OBJ}_yoloe_detection.json.
"""
import sys, os, json, argparse
import cv2
import numpy as np

sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE

sys.path.insert(0, '/workspace')
from foundationpose.datareader import Ho3dReader

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--hot3d_data_root', type=str, default='/dataset/ho3d/HO3D_data')
    parser.add_argument('--running_stride', type=int, default=10)
    parser.add_argument('--objects', type=str, nargs='+', default=None,
                        help='Objects to process (default: those missing detection json)')
    args = parser.parse_args()

    os.chdir('/workspace/yoloe')
    model = YOLOE("yoloe-26l-seg.pt")
    os.chdir('/workspace')

    all_objects = list(HO3D_YOLOE_PROMPTS.keys())
    targets = args.objects if args.objects else [
        obj for obj in all_objects
        if not os.path.exists(os.path.join(args.results_dir, f'{obj}_yoloe_detection.json'))
    ]

    print(f"Objects to process: {targets}")

    for obj_f in targets:
        prompt = HO3D_YOLOE_PROMPTS[obj_f]
        video_dir = os.path.join(args.hot3d_data_root, 'evaluation', obj_f)
        reader = Ho3dReader(video_dir, args.hot3d_data_root)
        reader.color_files = reader.color_files[::args.running_stride]

        model.set_classes([prompt], model.get_text_pe([prompt]))

        det_total = 0
        det_detected = 0
        iou_list = []

        for i in range(len(reader.color_files)):
            gt_pose_q = reader.get_gt_pose(i)
            if gt_pose_q is None:
                continue

            det_total += 1
            color_file = reader.color_files[i]
            img_bgr = cv2.imread(color_file)
            H, W = img_bgr.shape[:2]

            yoloe_mask = None
            for conf in [0.1, 0.05, 0.03]:
                results = model.predict(img_bgr, conf=conf, verbose=False)
                if len(results[0].boxes) > 0:
                    m = results[0].masks.data[0].cpu().numpy()
                    yoloe_mask = cv2.resize(m, (W, H)) > 0.5
                    break

            if yoloe_mask is not None:
                det_detected += 1
                gt_mask_raw = reader.get_mask(i)
                if gt_mask_raw is not None:
                    gt_m = gt_mask_raw.astype(bool)
                    inter = np.logical_and(yoloe_mask, gt_m).sum()
                    union = np.logical_or(yoloe_mask, gt_m).sum()
                    if union > 0:
                        iou_list.append(float(inter) / float(union))

        stats = {
            'prompt': prompt,
            'total_frames': det_total,
            'detected_frames': det_detected,
            'detection_rate': round(det_detected / det_total * 100, 1) if det_total > 0 else 0.0,
            'mean_iou': round(float(np.mean(iou_list)) * 100, 1) if iou_list else 0.0,
            'fallback_frames': det_total - det_detected,
        }

        out = os.path.join(args.results_dir, f'{obj_f}_yoloe_detection.json')
        with open(out, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"{obj_f}: det_rate={stats['detection_rate']}%  mean_iou={stats['mean_iou']}%  prompt=\"{prompt}\"")

    print("\nDone.")
