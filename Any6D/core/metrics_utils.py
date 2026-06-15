"""
BOP standard pose metrics: ADD, ADD-S, MSSD, MSPD, AR.

Metric overview
---------------
ADD:    Average Distance of Model Points.
        Transform the mesh point cloud with pred_pose and gt_pose, then take
        the mean point-to-point distance.  Success if < 10% of object diameter.
        Used for asymmetric objects.

ADD-S:  Symmetric variant of ADD.
        For each predicted point, find its NEAREST ground-truth point (not
        point-to-point).  This correctly handles objects like bowls or cylinders
        where 180° rotation looks identical.

MSSD:   Maximum Symmetry-aware Surface Distance (BOP 2022+).
        Like ADD-S but uses the maximum instead of mean, making it more
        sensitive to outlier regions.

MSPD:   Maximum Symmetry-aware Projection Distance.
        Projects the mesh into 2D and measures the maximum pixel displacement.
        Captures cases where the 3D pose is wrong but the 2D projection looks fine.

AR:     Average Recall (BOP benchmark aggregate).
        Binary mean of (MSSD < 0.2 * diameter) and (MSPD < 0.1 * max(H, W)).
        NaN when bop_toolkit_lib is unavailable (e.g. outside Docker).

R_error / T_error:
        Diagnostic values, not BOP metrics.  Useful for understanding whether
        failures come from rotation or translation errors.
"""
import numpy as np
from .constants import ADD_THRESH_RATIO
from .pose_utils import rotation_error_deg, translation_error_cm

try:
    from bop_toolkit_lib.pose_error_custom import mssd, mspd
    _BOP_METRICS = True
except Exception:
    # bop_toolkit_lib is only available inside Docker; outside, MSSD/MSPD/AR
    # will be NaN but ADD and ADD-S still work.
    _BOP_METRICS = False


def nanmean(values: list) -> float:
    """
    Arithmetic mean that silently ignores NaN and None entries.

    Used instead of np.nanmean() because metric lists may contain Python None
    (not numpy NaN) when a frame was skipped or YOLOE failed and we chose not
    to compute pose metrics for it.

    Args:
        values: List of floats, possibly containing float('nan') or None.

    Returns:
        Mean of valid entries, or float('nan') if no valid entries exist.
    """
    valid = [v for v in values
             if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(valid)) if valid else float('nan')


def compute_frame_metrics(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
    mesh_vertices: np.ndarray,
    diameter_m: float,
    K: np.ndarray,
    image_wh: tuple = (640, 480),
) -> dict:
    """
    Compute all BOP pose metrics for a single frame.

    The ADD threshold is 10% of the object's bounding-sphere diameter (BOP
    standard).  Changing ADD_THRESH_RATIO in constants.py adjusts this for
    all datasets simultaneously.

    Args:
        pred_pose:     4×4 SE(3) predicted pose, metres.
        gt_pose:       4×4 SE(3) ground-truth pose, metres.
        mesh_vertices: (N, 3) float32 point cloud sampled from the mesh, metres.
                       Used for ADD/ADD-S and MSSD computation.
        diameter_m:    Object bounding-sphere diameter in metres.
                       Sourced from models_info.json in the BOP dataset.
        K:             3×3 camera intrinsic matrix.  Used for MSPD (2D projection).
        image_wh:      (width, height) of the image.  Used to normalise MSPD
                       threshold: success if MSPD < 10% of max(W, H).

    Returns:
        Dictionary with keys:
            "ADD"     — 1.0 if ADD distance < 10% diameter, else 0.0
            "ADD-S"   — 1.0 if ADD-S distance < 10% diameter, else 0.0
            "AR"      — mean of MSSD and MSPD binary scores (NaN if unavailable)
            "MSSD"    — raw MSSD value (NaN if bop_toolkit unavailable)
            "MSPD"    — raw MSPD value in pixels (NaN if bop_toolkit unavailable)
            "R_error" — rotation error in degrees
            "T_error" — translation error in centimetres
    """
    from metrics import compute_add, compute_adds  # Any6D author module

    add_thresh = ADD_THRESH_RATIO * diameter_m

    err_R = rotation_error_deg(pred_pose[:3, :3], gt_pose[:3, :3])
    err_T = translation_error_cm(pred_pose[:3, 3], gt_pose[:3, 3])

    add_ok  = float(compute_add(mesh_vertices, pred_pose, gt_pose)  < add_thresh)
    adds_ok = float(compute_adds(mesh_vertices, pred_pose, gt_pose) < add_thresh)

    mssd_err = mspd_err = float('nan')
    if _BOP_METRICS:
        try:
            # syms=[identity] means no additional symmetry transformations
            # beyond what ADD-S already handles.  For fully symmetric objects,
            # the dataset's model_info would supply the actual symmetry set.
            syms = [{'R': np.eye(3), 't': np.zeros(3)}]
            mssd_err = float(mssd(pose_est=pred_pose, pose_gt=gt_pose,
                                  pts=mesh_vertices, syms=syms))
            mspd_err = float(mspd(pose_est=pred_pose, pose_gt=gt_pose,
                                  pts=mesh_vertices, K=K, syms=syms))
        except Exception:
            pass

    W, H = image_wh
    mean_ar = (float(mssd_err < 0.2 * diameter_m) +
               float(mspd_err < 0.1 * max(W, H))) / 2 \
              if _BOP_METRICS and not np.isnan(mssd_err) else float('nan')

    return {
        'ADD': add_ok, 'ADD-S': adds_ok, 'AR': mean_ar,
        'MSSD': mssd_err, 'MSPD': mspd_err,
        'R_error': err_R, 'T_error': err_T,
    }
