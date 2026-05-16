"""
pipeline_utils.py
Pipeline-specific utilities for the YOLOE -> Any6D pipeline.
Used by: 06_pipeline_yoloe_any6d, any6d_yoloe_pipeline.py

Imports shared geometry and drawing functions from any6d_utils.
"""

import os
import subprocess
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

BASE_DIR         = os.path.expanduser('~/open-vocabulary-6d-pose-yoloe')
ANY6D_DIR        = os.path.join(BASE_DIR, 'Any6D')
YOLOE_MODEL_PATH = os.path.join(BASE_DIR, 'notebooks', 'yoloe-26l-seg.pt')


# ── Any6D Docker ──────────────────────────────────────────────────────────────

def run_any6d_docker(
    color_path, depth_path, mask_bool,
    K, mesh_path, save_path, name='run',
    iteration=5
):
    """
    Run Any6D pose estimation inside the Docker container.

    Saves color, depth, mask, K, and mesh to save_path,
    then calls register_any6d() inside Docker via a generated script.

    Args:
        color_path : path to RGB image
        depth_path : path to depth image (16-bit PNG, mm)
        mask_bool  : (H, W) boolean mask from YOLOE
        K          : (3, 3) camera intrinsic matrix
        mesh_path  : path to .obj mesh file
        save_path  : folder to save inputs and results
        name       : run identifier
        iteration  : number of Any6D refinement iterations
    Returns:
        pred_pose : (4, 4) pose matrix
    """
    os.makedirs(save_path, exist_ok=True)

    cv2.imwrite(os.path.join(save_path, 'color.png'), cv2.imread(color_path))
    cv2.imwrite(os.path.join(save_path, 'depth.png'), cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH))
    cv2.imwrite(os.path.join(save_path, 'mask.png'),  mask_bool.astype(np.uint8) * 255)
    np.savetxt(os.path.join(save_path, 'K.txt'), K)
    shutil.copy2(mesh_path, os.path.join(save_path, 'mesh.obj'))

    rel_path   = os.path.relpath(save_path, ANY6D_DIR)
    run_script = f"""
import sys, os, numpy as np, cv2, trimesh
sys.path.insert(0, '/workspace/foundationpose/mycpp/build')
import nvdiffrast.torch as dr
from estimater import Any6D
from pytorch_lightning import seed_everything

seed_everything(0)
save_path = '/workspace/{rel_path}'

color = cv2.cvtColor(cv2.imread(os.path.join(save_path, 'color.png')), cv2.COLOR_BGR2RGB)
depth = cv2.imread(os.path.join(save_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
mask  = cv2.imread(os.path.join(save_path, 'mask.png'), cv2.IMREAD_GRAYSCALE).astype(bool)
K     = np.loadtxt(os.path.join(save_path, 'K.txt'))
mesh  = trimesh.load(os.path.join(save_path, 'mesh.obj'))

print(f'color: {{color.shape}}')
print(f'depth: {{depth.shape}} range [{{depth.min():.3f}}, {{depth.max():.3f}}] m')
print(f'mask : {{mask.sum()}} pixels')

glctx = dr.RasterizeCudaContext()
est   = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=2, glctx=glctx)
pose  = est.register_any6d(K=K, rgb=color, depth=depth, ob_mask=mask, iteration={iteration}, name='{name}')

np.savetxt(os.path.join(save_path, 'pred_pose.txt'), pose)
print('POSE_SAVED')
print('translation (m):', pose[:3, 3].tolist())
"""
    script_path = os.path.join(ANY6D_DIR, '_run_any6d.py')
    with open(script_path, 'w') as f:
        f.write(run_script)

    skip_kw = [
        'FutureWarning', 'torch.load', 'weights_only', 'ckpt_dir',
        'pytree', 'torch.set_default', 'torch.cuda.amp', 'pickle',
        'allowlist', 'open an issue'
    ]

    process = subprocess.Popen(
        ['docker', 'compose', 'run', '--rm', '--entrypoint', '', 'any6d',
         'bash', '-c', 'cd /workspace && python _run_any6d.py'],
        cwd=ANY6D_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in process.stdout:
        line = line.rstrip()
        if not any(k in line for k in skip_kw):
            print(f'  {line}')
    process.wait()

    os.remove(script_path)
    subprocess.run(['chmod', '-R', '777', save_path], capture_output=True)

    pose_path = os.path.join(save_path, 'pred_pose.txt')
    if not os.path.exists(pose_path):
        raise RuntimeError('Any6D failed — pred_pose.txt not found')

    pred_pose = np.loadtxt(pose_path)
    print(f'  Pose saved: {pose_path}')
    return pred_pose


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_errors(pred_pose, gt_pose):
    """
    Compute translation error (cm) and rotation error (degrees).

    Args:
        pred_pose : (4, 4) predicted pose matrix
        gt_pose   : (4, 4) ground truth pose matrix
    Returns:
        err_t : translation error in cm
        err_R : rotation error in degrees
    """
    err_t  = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3]) * 100
    R_diff = pred_pose[:3, :3] @ gt_pose[:3, :3].T
    trace  = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    err_R  = np.degrees(np.arccos(trace))
    return err_t, err_R


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_detection(scene_rgb, mask_bool, bbox_xyxy, confidence,
                   text_prompt, obj_name, viz_dir, save=True):
    """
    Plot YOLOE detection: original scene, detection bbox, and mask overlay.

    Args:
        scene_rgb  : (H, W, 3) RGB image
        mask_bool  : (H, W) boolean mask
        bbox_xyxy  : [x1, y1, x2, y2]
        confidence : float
        text_prompt: str or list used as prompt
        obj_name   : string identifier for saving
        viz_dir    : output directory
        save       : whether to save the figure
    """
    from any6d_utils import mask_overlay, save_fig

    overlay   = mask_overlay(scene_rgb, mask_bool)
    scene_box = scene_rgb.copy()
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(scene_box, (x1, y1), (x2, y2), (255, 165, 0), 3)
    cv2.putText(scene_box, f'{text_prompt} {confidence:.2f}',
                (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(scene_rgb);  axes[0].set_title('Original scene');                              axes[0].axis('off')
    axes[1].imshow(scene_box);  axes[1].set_title(f'Detection — "{text_prompt}" conf={confidence:.2f}'); axes[1].axis('off')
    axes[2].imshow(overlay);    axes[2].set_title(f'Mask overlay — {mask_bool.sum()} px');        axes[2].axis('off')

    plt.suptitle(f'YOLOE text prompt — {obj_name}')
    plt.tight_layout()

    if save:
        save_fig(fig, os.path.join(viz_dir, f'{obj_name}_yoloe_detection.png'))
    plt.show()


def plot_pose_result(scene_rgb, pred_pose, gt_pose, K,
                     mesh, mask_bool, confidence,
                     err_t, err_R, obj_name, viz_dir, save=True):
    """
    Plot 6D pose result: predicted pose vs ground truth on scene image.

    Args:
        scene_rgb  : (H, W, 3) RGB image
        pred_pose  : (4, 4) predicted pose
        gt_pose    : (4, 4) ground truth pose or None
        K          : (3, 3) camera intrinsics
        mesh       : trimesh mesh object
        mask_bool  : (H, W) boolean mask
        confidence : YOLOE confidence float
        err_t      : translation error in cm
        err_R      : rotation error in degrees
        obj_name   : string identifier
        viz_dir    : output directory
        save       : whether to save the figure
    """
    from any6d_utils import (
        mesh_bbox_corners, project_points,
        draw_3d_bbox, draw_pose_axes,
        mask_overlay, axes_legend, save_fig
    )

    corners_3d = mesh_bbox_corners(mesh)
    c_pred     = project_points(corners_3d, pred_pose, K)
    base       = mask_overlay(scene_rgb, mask_bool, alpha=0.3)

    img_pred = draw_pose_axes(draw_3d_bbox(base, c_pred, (30, 120, 255)), pred_pose, K)

    if gt_pose is not None:
        c_gt     = project_points(corners_3d, gt_pose, K)
        img_gt   = draw_pose_axes(draw_3d_bbox(base, c_gt, (30, 200, 80)), gt_pose, K)
        img_both = draw_pose_axes(
            draw_3d_bbox(draw_3d_bbox(base.copy(), c_gt, (30, 200, 80)), c_pred, (30, 120, 255)),
            pred_pose, K
        )
        n_cols = 3
    else:
        n_cols = 1

    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img_pred)
    axes[0].set_title(f'Predicted pose — T err={err_t:.1f} cm  R err={err_R:.1f} deg')
    axes[0].axis('off')
    axes_legend(axes[0], [('#1E78FF','Predicted bbox'),('#DC3232','X'),('#32DC32','Y'),('#3264E6','Z')])

    if gt_pose is not None:
        axes[1].imshow(img_gt)
        axes[1].set_title('Ground truth pose')
        axes[1].axis('off')
        axes_legend(axes[1], [('#1EC850','GT bbox')])

        axes[2].imshow(img_both)
        axes[2].set_title('Overlay — Blue=Predicted  Green=GT')
        axes[2].axis('off')
        axes_legend(axes[2], [('#1E78FF','Predicted'),('#1EC850','GT')])

    plt.suptitle(f'Any6D — 6D pose: {obj_name}  |  Axes: Red=X  Green=Y  Blue=Z')
    plt.tight_layout()

    if save:
        save_fig(fig, os.path.join(viz_dir, f'{obj_name}_pose_result.png'))
    plt.show()


def plot_all_results_summary(all_results, viz_dir, save=True):
    """
    Plot summary of all objects: translation errors, rotation errors,
    and pose thumbnails.

    Args:
        all_results : list of dicts with keys:
                      obj_name, err_t, err_R, t_pred, t_gt, img_pose
        viz_dir     : output directory
        save        : whether to save the figure
    """
    from any6d_utils import save_fig

    if not all_results:
        print('No results to plot')
        return

    names  = [r['obj_name'].replace('_', '\n') for r in all_results]
    err_ts = [r['err_t'] for r in all_results]
    err_Rs = [r['err_R'] for r in all_results]

    fig = plt.figure(figsize=(20, 14))
    gs  = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    mean_t   = np.mean(err_ts)
    colors_t = ['#4CAF50' if e < mean_t else '#FF5722' for e in err_ts]
    ax1      = fig.add_subplot(gs[0, :2])
    bars     = ax1.bar(range(len(names)), err_ts, color=colors_t, alpha=0.85)
    ax1.axhline(mean_t, color='black', linestyle='--', linewidth=2, label=f'Mean = {mean_t:.1f} cm')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel('Translation error (cm)')
    ax1.set_title('Translation error per object (green = below mean)')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, err_ts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    mean_R   = np.mean(err_Rs)
    colors_R = ['#4CAF50' if e < mean_R else '#FF5722' for e in err_Rs]
    ax2      = fig.add_subplot(gs[0, 2:])
    bars2    = ax2.bar(range(len(names)), err_Rs, color=colors_R, alpha=0.85)
    ax2.axhline(mean_R, color='black', linestyle='--', linewidth=2, label=f'Mean = {mean_R:.1f} deg')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel('Rotation error (deg)')
    ax2.set_title('Rotation error per object (green = below mean)')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, err_Rs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    for idx, result in enumerate(all_results):
        row = 1 + idx // 4
        col = idx % 4
        if row > 2:
            break
        ax = fig.add_subplot(gs[row, col])
        if result.get('img_pose') is not None:
            ax.imshow(result['img_pose'])
        ax.set_title(
            f"{result['obj_name'].split('_')[0]}\n"
            f"T={result['err_t']:.1f}cm  R={result['err_R']:.0f}deg",
            fontsize=7
        )
        ax.axis('off')

    plt.suptitle(
        f'YOLOE -> Any6D Pipeline — All {len(all_results)} objects\n'
        f'Mean T error: {mean_t:.1f} cm  |  Mean R error: {mean_R:.1f} deg'
    )

    if save:
        save_fig(fig, os.path.join(viz_dir, 'pipeline_all_results_summary.png'))
    plt.show()
