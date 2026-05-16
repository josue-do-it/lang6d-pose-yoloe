"""
any6d_yoloe_pipeline.py
Main entry point for the YOLOE -> Any6D 6D pose estimation pipeline.

Usage:
    python any6d_yoloe_pipeline.py
    python any6d_yoloe_pipeline.py --obj 006_mustard_bottle --conf 0.1
"""

import os
import sys
import argparse
import numpy as np
import cv2
import trimesh

BASE_DIR = os.path.expanduser('~/open-vocabulary-6d-pose-yoloe')
sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))

from any6d_utils import (
    mesh_bbox_corners, project_points,
    draw_3d_bbox, draw_pose_axes,
    mask_overlay, save_fig,
    translation_error, rotation_error
)
from pipeline_utils import run_any6d_docker, compute_errors
from yoloe_helpers  import yoloe_text_prompt

ANY6D_DIR  = os.path.join(BASE_DIR, 'Any6D')
ANCHOR_DIR = os.path.join(ANY6D_DIR, 'anchor_results', 'dexycb_reference_view_ours')
RESULTS_DIR = os.path.join(ANY6D_DIR, 'results', 'pipeline')
VIZ_DIR     = os.path.join(BASE_DIR, 'notebooks', 'outputs')

OBJECTS = [
    {'folder': '006_mustard_bottle',  'prompt': 'mustard bottle',
     'mesh': 'center_mesh_006_mustard_bottle.obj'},
    {'folder': '021_bleach_cleanser', 'prompt': ['white bottle', 'detergent bottle', 'cleaning bottle'],
     'mesh': 'center_mesh_021_bleach_cleanser.obj'},
    {'folder': '019_pitcher_base',    'prompt': ['blue pitcher', 'blue jug', 'plastic pitcher'],
     'mesh': 'center_mesh_019_pitcher_base.obj'},
    {'folder': '004_sugar_box',       'prompt': 'sugar box',
     'mesh': 'center_mesh_004_sugar_box.obj'},
    {'folder': '005_tomato_soup_can', 'prompt': 'tomato soup can',
     'mesh': 'center_mesh_005_tomato_soup_can.obj'},
    {'folder': '003_cracker_box',     'prompt': 'cracker box',
     'mesh': 'center_mesh_003_cracker_box.obj'},
    {'folder': '010_potted_meat_can', 'prompt': ['yellow box', 'small box', 'square can', 'yellow can'],
     'mesh': 'center_mesh_010_potted_meat_can.obj'},
]


def load_yoloe_model(model_path):
    """Load YOLOE model onto CUDA."""
    from ultralytics import YOLOE
    model = YOLOE(model_path)
    model.to('cuda')
    return model


def run_single_object(model, obj, conf, iteration):
    """
    Run the full YOLOE -> Any6D pipeline for one object.

    Args:
        model     : loaded YOLOE model
        obj       : dict with folder, prompt, mesh keys
        conf      : YOLOE confidence threshold
        iteration : Any6D refinement iterations
    Returns:
        dict with pred_pose, gt_pose, err_t, err_R, K, success
    """
    folder     = obj['folder']
    obj_path   = os.path.join(ANCHOR_DIR, folder)
    color_path = os.path.join(obj_path, 'color.png')
    depth_path = os.path.join(obj_path, 'depth.png')
    mesh_path  = os.path.join(obj_path, obj['mesh'])
    K          = np.loadtxt(os.path.join(obj_path, 'K.txt'))
    save_path  = os.path.join(RESULTS_DIR, folder)

    print(f'\n[YOLOE] {folder}')
    bbox_xyxy, mask_bool, confidence, _ = yoloe_text_prompt(
        model, color_path, obj['prompt'], conf=conf
    )

    print(f'[Any6D] {folder}  mask={mask_bool.sum()} px')
    pred_pose = run_any6d_docker(
        color_path=color_path,
        depth_path=depth_path,
        mask_bool=mask_bool,
        K=K,
        mesh_path=mesh_path,
        save_path=save_path,
        name=folder,
        iteration=iteration
    )

    gt_pose_path = os.path.join(obj_path, f'{folder}_gt_pose.txt')
    gt_pose      = np.loadtxt(gt_pose_path) if os.path.exists(gt_pose_path) else None

    err_t, err_R = compute_errors(pred_pose, gt_pose) if gt_pose is not None else (0.0, 0.0)
    print(f'  T error: {err_t:.2f} cm   R error: {err_R:.2f} deg')

    return {
        'pred_pose': pred_pose,
        'gt_pose':   gt_pose,
        'mask':      mask_bool,
        'K':         K,
        'err_t':     err_t,
        'err_R':     err_R,
        'success':   True
    }


def print_summary(results):
    """Print final results table."""
    print('\n' + '=' * 55)
    print('YOLOE -> ANY6D PIPELINE — FINAL SUMMARY')
    print('=' * 55)

    success = {k: v for k, v in results.items() if v.get('success')}
    failed  = {k: v for k, v in results.items() if not v.get('success')}

    print(f'\n  Completed: {len(success)}/{len(results)} objects')

    if success:
        err_ts = [v['err_t'] for v in success.values()]
        err_Rs = [v['err_R'] for v in success.values()]
        print(f'\n  {"Object":<30} {"T err (cm)":>12} {"R err (deg)":>12}')
        print(f'  {"-"*56}')
        for folder, res in success.items():
            print(f'  {folder:<30} {res["err_t"]:>12.1f} {res["err_R"]:>12.1f}')
        print(f'  {"-"*56}')
        print(f'  {"MEAN":<30} {np.mean(err_ts):>12.1f} {np.mean(err_Rs):>12.1f}')

    if failed:
        print(f'\n  Failed:')
        for folder, res in failed.items():
            print(f'  {folder}: {res.get("error", "unknown")}')


def main():
    parser = argparse.ArgumentParser(description='YOLOE -> Any6D 6D pose estimation pipeline')
    parser.add_argument('--obj',       type=str,   default=None,  help='Run only this object folder name')
    parser.add_argument('--conf',      type=float, default=0.1,   help='YOLOE confidence threshold')
    parser.add_argument('--iteration', type=int,   default=5,     help='Any6D refinement iterations')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR,     exist_ok=True)

    yoloe_model_path = os.path.join(BASE_DIR, 'notebooks', 'yoloe-26l-seg.pt')
    print('Loading YOLOE model...')
    model = load_yoloe_model(yoloe_model_path)

    objects_to_run = OBJECTS
    if args.obj:
        objects_to_run = [o for o in OBJECTS if o['folder'] == args.obj]
        if not objects_to_run:
            print(f'Object not found: {args.obj}')
            sys.exit(1)

    results = {}
    for obj in objects_to_run:
        try:
            results[obj['folder']] = run_single_object(
                model, obj, args.conf, args.iteration
            )
        except Exception as e:
            print(f'  Failed: {e}')
            results[obj['folder']] = {'success': False, 'error': str(e)}

    print_summary(results)


if __name__ == '__main__':
    main()
