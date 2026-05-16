"""
any6d_utils.py
Shared utilities for the Any6D evaluation pipeline.
Used by: 03_any6d_demo_run, 04_ho3d_anchor_pipeline, 05_ho3d_query_pipeline
"""

import os
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ── Docker ────────────────────────────────────────────────────────────────────

def docker_run(any6d_dir, cmd_inside, timeout=600):
    """
    Run a bash command inside the any6d Docker container.
    --entrypoint "" is required to bypass the interactive shell entrypoint.
    """
    full_cmd = [
        'docker', 'compose', 'run', '--rm',
        '--entrypoint', '',
        'any6d',
        'bash', '-c', cmd_inside
    ]
    return subprocess.run(
        full_cmd,
        cwd=any6d_dir,
        capture_output=True,
        text=True,
        timeout=timeout
    )


def check_docker(any6d_dir, timeout=60):
    """
    Verify that the any6d Docker container is operational.
    Checks: nvdiffrast, Any6D estimator, CUDA GPU.
    Returns True if all checks pass.
    """
    check_script = """
import sys
sys.path.insert(0, '/workspace/foundationpose/mycpp/build')
import nvdiffrast.torch as dr
from estimater import Any6D
import torch
print('GPU:', torch.cuda.get_device_name(0))
print('DOCKER_OK')
"""
    script_path = os.path.join(any6d_dir, '_check.py')
    with open(script_path, 'w') as f:
        f.write(check_script)

    r = docker_run(any6d_dir, 'cd /workspace && python _check.py', timeout=timeout)
    os.remove(script_path)

    if r.returncode == 0 and 'DOCKER_OK' in r.stdout:
        for line in r.stdout.split('\n'):
            if any(k in line for k in ['GPU:', 'DOCKER_OK']):
                print(f'  {line.strip()}')
        return True
    else:
        print('  Docker check failed')
        print(r.stderr[-300:])
        return False


def check_docker_volumes(any6d_dir):
    """
    Verify that required Docker volume mounts are accessible.
    Returns dict with mount status.
    """
    mounts = {
        'anchor': ('ls /workspace/anchor_results/dexycb_reference_view_ours/ && echo ANCHOR_OK', 'ANCHOR_OK'),
        'dataset': ('ls /workspace/dataset/ho3d/ && echo DATASET_OK', 'DATASET_OK'),
        'evaluation': ('ls /workspace/dataset/ho3d/evaluation/ && echo EVAL_OK', 'EVAL_OK'),
    }
    results = {}
    for name, (cmd, token) in mounts.items():
        r = docker_run(any6d_dir, cmd, timeout=30)
        results[name] = token in r.stdout
        status = 'OK' if results[name] else 'FAIL'
        print(f'  [{status}] /workspace/{name}')
    return results


# ── Geometry ──────────────────────────────────────────────────────────────────

def project_points(pts_3d, pose, K):
    """
    Project (N, 3) 3D points onto 2D image coordinates.

    Args:
        pts_3d : (N, 3) array of 3D points in object frame
        pose   : (4, 4) transformation matrix [R | t]
        K      : (3, 3) camera intrinsic matrix
    Returns:
        (N, 2) integer pixel coordinates
    """
    R, t    = pose[:3, :3], pose[:3, 3]
    pts_cam = (R @ pts_3d.T).T + t
    pts_h   = (K @ pts_cam.T).T
    return (pts_h[:, :2] / pts_h[:, 2:]).astype(int)


def mesh_bbox_corners(mesh):
    """
    Return the 8 corners of the axis-aligned bounding box of a trimesh mesh.

    Returns:
        (8, 3) array of 3D corner coordinates
    """
    b = mesh.bounds
    return np.array([
        [b[0,0], b[0,1], b[0,2]],
        [b[1,0], b[0,1], b[0,2]],
        [b[1,0], b[1,1], b[0,2]],
        [b[0,0], b[1,1], b[0,2]],
        [b[0,0], b[0,1], b[1,2]],
        [b[1,0], b[0,1], b[1,2]],
        [b[1,0], b[1,1], b[1,2]],
        [b[0,0], b[1,1], b[1,2]],
    ])


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_3d_bbox(img, corners_2d, color, thickness=2):
    """
    Draw a 3D bounding box on an image from 8 projected corners.
    Bottom face: corners 0-3. Top face: corners 4-7.

    Args:
        img        : (H, W, 3) RGB image
        corners_2d : (8, 2) projected corner coordinates
        color      : BGR color tuple
        thickness  : line thickness in pixels
    Returns:
        Copy of img with bbox drawn.
    """
    img = img.copy()
    H, W = img.shape[:2]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i, j in edges:
        p1 = tuple(np.clip(corners_2d[i], [0,0], [W-1,H-1]))
        p2 = tuple(np.clip(corners_2d[j], [0,0], [W-1,H-1]))
        cv2.line(img, p1, p2, color, thickness)
    return img


def draw_pose_axes(img, pose, K, length=0.05):
    """
    Draw 3D coordinate axes on an image to visualize object orientation.
    Red = X  |  Green = Y  |  Blue = Z

    Args:
        img    : (H, W, 3) RGB image
        pose   : (4, 4) transformation matrix
        K      : (3, 3) camera intrinsic matrix
        length : axis length in metres
    Returns:
        Copy of img with axes drawn.
    """
    img = img.copy()
    H, W = img.shape[:2]
    R, t = pose[:3, :3], pose[:3, 3]

    def proj(pt):
        p = K @ (R @ np.array(pt) + t)
        return tuple((p[:2] / p[2]).astype(int))

    def clip(p):
        return (int(np.clip(p[0], 0, W-1)), int(np.clip(p[1], 0, H-1)))

    try:
        origin = proj([0, 0, 0])
        cv2.arrowedLine(img, clip(origin), clip(proj([length, 0, 0])), (220, 50,  50),  3, tipLength=0.25)
        cv2.arrowedLine(img, clip(origin), clip(proj([0, length, 0])), (50,  210, 50),  3, tipLength=0.25)
        cv2.arrowedLine(img, clip(origin), clip(proj([0, 0, length])), (50,  100, 230), 3, tipLength=0.25)
        cv2.putText(img, 'X', clip(proj([length,0,0])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,50,50),  2)
        cv2.putText(img, 'Y', clip(proj([0,length,0])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50,210,50),  2)
        cv2.putText(img, 'Z', clip(proj([0,0,length])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50,100,230), 2)
    except Exception:
        pass
    return img


def mask_overlay(color, mask_bool, alpha=0.6):
    """
    Apply a green overlay on pixels where mask_bool is True.

    Args:
        color     : (H, W, 3) RGB image
        mask_bool : (H, W) boolean mask
        alpha     : overlay opacity (0 = transparent, 1 = full green)
    Returns:
        (H, W, 3) RGB image with overlay applied.
    """
    overlay = color.copy()
    green   = np.zeros_like(color)
    green[:, :, 1] = 255
    overlay[mask_bool] = (
        color[mask_bool] * (1 - alpha) + green[mask_bool] * alpha
    ).astype(np.uint8)
    return overlay


def colormap_depth(depth):
    """
    Convert a float depth image to a JET colormap RGB image.

    Args:
        depth : (H, W) float array in metres
    Returns:
        (H, W, 3) RGB image
    """
    depth_norm  = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)


# ── Pose metrics ──────────────────────────────────────────────────────────────

def translation_error(pose_pred, pose_gt):
    """Euclidean distance between predicted and ground truth translations, in cm."""
    return np.linalg.norm(pose_pred[:3, 3] - pose_gt[:3, 3]) * 100


def rotation_error(pose_pred, pose_gt):
    """Geodesic rotation error on SO(3), in degrees."""
    R_diff = pose_pred[:3, :3] @ pose_gt[:3, :3].T
    trace  = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(trace))


# ── Plot helpers ──────────────────────────────────────────────────────────────

def axes_legend(ax, items):
    """
    Add a legend for pose axes colour convention.

    Args:
        ax    : matplotlib Axes
        items : list of (color_hex, label) tuples
    """
    ax.legend(
        handles=[patches.Patch(color=c, label=l) for c, l in items],
        loc='lower left',
        fontsize=8,
        framealpha=0.7
    )


def save_fig(fig, path):
    """Save figure to path at 150 dpi and print confirmation."""
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
