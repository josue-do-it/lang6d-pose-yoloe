"""
BOP-format YCB-Video dataset reader for a given (scene_id, obj_id).
"""
import os
import json
import numpy as np
import cv2

from ..constants import MM_TO_M

YCBV_ROOT_DEFAULT = "/dataset/ycbv"


class YCBVReader:
    """Reads BOP-format YCB-Video dataset for a given (scene_id, obj_id)."""

    def __init__(self, scene_id: int, obj_id: int,
                 ycbv_root: str = YCBV_ROOT_DEFAULT):
        self.scene_id  = scene_id
        self.obj_id    = obj_id
        self.scene_dir = os.path.join(ycbv_root, "test", f"{scene_id:06d}")

        with open(os.path.join(self.scene_dir, "scene_camera.json")) as f:
            self._scene_cam = json.load(f)
        with open(os.path.join(self.scene_dir, "scene_gt.json")) as f:
            self._scene_gt = json.load(f)

        first_key = list(self._scene_cam.keys())[0]
        km = self._scene_cam[first_key]['cam_K']
        self.K = np.array([[km[0], km[1], km[2]],
                           [km[3], km[4], km[5]],
                           [km[6], km[7], km[8]]], dtype=np.float64)
        self.depth_scale = self._scene_cam[first_key].get('depth_scale', 0.1)

        self._ann_idx = {}
        for im_id_str, anns in self._scene_gt.items():
            im_id = int(im_id_str)
            for idx, ann in enumerate(anns):
                if ann['obj_id'] == obj_id:
                    self._ann_idx[im_id] = idx
                    break

    def im_ids_with_obj(self) -> list:
        return sorted(self._ann_idx.keys())

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
        ann_idx = self._ann_idx.get(im_id, 0)
        path = os.path.join(self.scene_dir, "mask_visib",
                            f"{im_id:06d}_{ann_idx:06d}.png")
        if not os.path.exists(path):
            return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return (m > 0) if m is not None else None

    def get_gt_pose(self, im_id: int) -> np.ndarray:
        ann_idx = self._ann_idx.get(im_id, 0)
        ann = self._scene_gt[str(im_id)][ann_idx]
        R   = np.array(ann['cam_R_m2c']).reshape(3, 3)
        t   = np.array(ann['cam_t_m2c']) * MM_TO_M
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3,  3] = t
        return pose
