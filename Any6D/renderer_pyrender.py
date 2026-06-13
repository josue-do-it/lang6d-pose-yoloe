import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np
import pyrender
import trimesh

class RendererVispy:
    def __init__(self, W, H, mode="depth"):
        self.W = W
        self.H = H
        self.renderer = pyrender.OffscreenRenderer(W, H)
        self.meshes = {}

    def my_add_object(self, mesh_dict, ob_id):
        vertices = mesh_dict["pts"] / 1000.0
        faces = mesh_dict["faces"]
        tri = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.meshes[ob_id] = pyrender.Mesh.from_trimesh(tri)

    def render_object(self, obj_id, R, t, fx, fy, cx, cy):
        scene = pyrender.Scene()
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=10.0)
        scene.add(camera, pose=np.eye(4))
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten() / 1000.0
        pose = np.diag([1, -1, -1, 1]).astype(float) @ pose
        scene.add(self.meshes[obj_id], pose=pose)
        color, depth = self.renderer.render(scene)
        return {"depth": depth * 1000.0}
