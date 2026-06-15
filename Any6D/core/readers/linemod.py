"""
BOP-format LINEMOD dataset reader for a single object.
"""
import os
import json
import numpy as np
import cv2
import trimesh

from ..constants import MM_TO_M

LM_ROOT_DEFAULT = "/dataset/lm"


class LineMODReader:
    """Reads BOP-format LINEMOD dataset for a given object ID."""

    def __init__(self, obj_id: int, lm_root: str = LM_ROOT_DEFAULT):
        self.obj_id    = obj_id
        self.lm_root   = lm_root
        self.scene_dir = os.path.join(lm_root, "test", f"{obj_id:06d}")

        cam_path = os.path.join(lm_root, "lm", "camera.json")
        with open(cam_path) as f:
            c = json.load(f)
        self.K = np.array([[c['fx'], 0, c['cx']],
                           [0, c['fy'], c['cy']],
                           [0, 0, 1]], dtype=np.float64)
        self.depth_scale = c.get('depth_scale', 1.0)

        with open(os.path.join(self.scene_dir, "scene_gt.json")) as f:
            self._scene_gt = json.load(f)

        targets_path = os.path.join(lm_root, "lm", "test_targets_bop19.json")
        with open(targets_path) as f:
            all_targets = json.load(f)
        self.test_im_ids = sorted(
            t['im_id'] for t in all_targets
            if t['obj_id'] == obj_id and t['scene_id'] == obj_id)

    def get_rgb(self, im_id: int) -> np.ndarray:
        path = os.path.join(self.scene_dir, "rgb", f"{im_id:06d}.png")
        img  = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"RGB not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_depth(self, im_id: int) -> np.ndarray:
        path  = os.path.join(self.scene_dir, "depth", f"{im_id:06d}.png")
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        return depth * self.depth_scale * MM_TO_M

    def get_mask_visib(self, im_id: int):
        path = os.path.join(self.scene_dir, "mask_visib",
                            f"{im_id:06d}_000000.png")
        if not os.path.exists(path):
            return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return m > 0 if m is not None else None

    def get_gt_pose(self, im_id: int) -> np.ndarray:
        ann  = self._scene_gt[str(im_id)][0]
        R    = np.array(ann['cam_R_m2c']).reshape(3, 3)
        t    = np.array(ann['cam_t_m2c']) * MM_TO_M
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3,  3] = t
        return pose

    def load_gt_mesh(self, scale_to_metres: bool = True) -> trimesh.Trimesh:
        path = os.path.join(self.lm_root, "models", f"obj_{self.obj_id:06d}.ply")
        mesh = trimesh.load(path, force='mesh')
        if scale_to_metres:
            mesh.apply_scale(MM_TO_M)
        return mesh
