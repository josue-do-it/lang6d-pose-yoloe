"""
Pose geometry utilities: anchor correction, rotation error, translation error.

Anchor correction rationale
---------------------------
Any6D estimates poses relative to its internal coordinate frame, which is
set by the anchor registration.  When the anchor and query views differ
significantly, systematic drift accumulates.  The correction formula:

    corrected = (pred_query @ inv(pred_anchor)) @ gt_anchor

cancels this drift by expressing the query pose relative to the anchor's
predicted frame, then mapping into the anchor's known GT frame.

This formula holds under the assumption that the relative transformation
between anchor and query is estimated more accurately than the absolute
transformation.  It degrades when the anchor registration itself is poor
(e.g. heavily occluded anchor or very different lighting).
"""
import numpy as np
from .constants import ANY6D_ITERS


def estimate_corrected_pose(
    est,
    K: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    pred_pose_anchor: np.ndarray,
    gt_pose_anchor: np.ndarray,
    name: str = "query",
) -> np.ndarray:
    """
    Register the query frame and apply anchor-relative pose correction.

    Why this function exists: Any6D's est.register() returns a pose in an
    internal coordinate system that shifts slightly between calls.  Running
    register() on both the anchor frame (with known GT) and the query frame,
    then applying the correction formula, removes this systematic offset.

    Args:
        est:              Any6D estimator instance (mesh already loaded).
        K:                3×3 camera intrinsic matrix.
        rgb:              Query RGB image, shape (H, W, 3) uint8.
        depth:            Query depth map, shape (H, W) float32, metres.
        mask:             Query object mask, shape (H, W) bool.
        pred_pose_anchor: 4×4 Any6D output for the anchor frame.
        gt_pose_anchor:   4×4 ground-truth pose for the anchor frame.
        name:             Debug label passed to est.register().

    Returns:
        corrected_pose: 4×4 SE(3) numpy array in metres.
    """
    pred_pose_q        = est.register(K=K, rgb=rgb, depth=depth,
                                      ob_mask=mask, iteration=ANY6D_ITERS, name=name)
    relative_transform = pred_pose_q @ np.linalg.inv(pred_pose_anchor)
    return relative_transform @ gt_pose_anchor


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Geodesic (great-circle) angle between two rotation matrices, in degrees.

    Uses the formula: angle = arccos( (trace(R_pred @ R_gt^T) - 1) / 2 )
    which gives the minimum rotation angle needed to align R_pred with R_gt.

    The result is clipped to [-1, 1] before arccos to guard against floating-
    point values slightly outside this range due to numerical noise.

    Args:
        R_pred: Predicted 3×3 rotation matrix (SO(3)).
        R_gt:   Ground-truth 3×3 rotation matrix (SO(3)).

    Returns:
        Angular error in degrees, in [0°, 180°].
    """
    cos_val = np.clip((np.trace(R_pred @ R_gt.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_val)))


def translation_error_cm(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    Euclidean distance between two translation vectors, converted to centimetres.

    Args:
        t_pred: Predicted translation [x, y, z] in metres.
        t_gt:   Ground-truth translation [x, y, z] in metres.

    Returns:
        L2 distance in centimetres (multiply metres by 100).
    """
    return float(np.linalg.norm(t_pred - t_gt) * 100)
