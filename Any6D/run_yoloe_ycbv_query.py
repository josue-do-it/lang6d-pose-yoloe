import copy
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/yoloe')
from ultralytics import YOLOE as _YOLOE

HO3D_YOLOE_PROMPTS = {
    "MPM10": "canned food",          "MPM11": "spam can",
    "MPM12": "spam can",             "MPM13": "canned food",
    "MPM14": "canned food",          "AP10":  "blue jug",
    "AP11":  "blue pitcher",         "AP12":  "blue object",
    "AP13":  "cup with handle",      "AP14":  "cup with handle",
    "SB11":  "white plastic bottle", "SB13":  "white plastic bottle",
    "SM1":   "canned food",
}

YCBV_YOLOE_PROMPTS = {
    2:  "cracker box",         3:  "domino sugar box",
    4:  "small red can",       5:  "yellow mustard bottle",
    9:  "spam box",            13: "large red bowl",
    14: "red coffee mug",
}
YCBV_NAMES = {
    2: "003_cracker_box",     3: "004_sugar_box",
    4: "005_tomato_soup_can", 5: "006_mustard_bottle",
    9: "010_potted_meat_can", 13: "019_pitcher_base",
    14: "021_bleach_cleanser",
}

# YCB-V BOP prompts (calibrated visually)
YCBV_YOLOE_PROMPTS = {
    2:  "cracker box",
    3:  "domino sugar box",
    4:  "small red can",
    5:  "yellow mustard bottle",
    9:  "spam box",
    13: "large red bowl",
    14: "red coffee mug",
}

YCBV_NAMES = {
    2: "003_cracker_box",     3: "004_sugar_box",
    4: "005_tomato_soup_can", 5: "006_mustard_bottle",
    9: "010_potted_meat_can", 13: "019_pitcher_base",
    14: "021_bleach_cleanser",
}
_yoloe_model = None

def get_yoloe_mask(img_rgb, obj_f, H, W):
    global _yoloe_model
    import os, cv2 as _cv2
    _orig_dir = os.getcwd()
    if _yoloe_model is None:
        os.chdir('/workspace/yoloe')
        _yoloe_model = _YOLOE("yoloe-26l-seg.pt")
        os.chdir(_orig_dir)
    if isinstance(obj_f, int):
        prompt = YCBV_YOLOE_PROMPTS.get(obj_f, "object")
    else:
        prompt = HO3D_YOLOE_PROMPTS.get(obj_f, "object")
    _yoloe_model.set_classes([prompt], _yoloe_model.get_text_pe([prompt]))
    img_bgr = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)
    for conf in [0.1, 0.05, 0.03]:
        results = _yoloe_model.predict(img_bgr, conf=conf, verbose=False)
        if len(results[0].boxes) > 0:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = _cv2.resize(mask, (W, H))
            os.chdir(_orig_dir)
            return (mask > 0.5)
    os.chdir(_orig_dir)
    return None

from foundationpose.datareader import Ho3dReader

def plot_yoloe_frame(rgb, depth, mask, pose_4x4, K, obj_f, frame_idx, save_path):
    """Plot RGB | Depth | YOLOE mask | Pose axes"""
    import numpy as np, cv2 as _cv2
    H, W = rgb.shape[:2]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{obj_f} — frame {frame_idx}", fontsize=12)

    # 1. RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    # 2. Depth
    d = depth.copy().astype(np.float32)
    d[d < 0.001] = np.nan
    if not np.all(np.isnan(d)):
        im = axes[1].imshow(d, cmap="jet",
                            vmin=np.nanmin(d), vmax=np.nanmax(d))
        plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title("Depth")
    axes[1].axis("off")

    # 3. YOLOE mask overlay
    overlay = rgb.copy()
    if mask is not None:
        overlay[mask] = overlay[mask] * 0.4 + np.array([0, 200, 0]) * 0.6
    axes[2].imshow(overlay)
    axes[2].set_title(f"YOLOE mask")
    axes[2].axis("off")

    # 4. Pose axes projetés
    pose_img = rgb.copy()
    if pose_4x4 is not None and K is not None:
        R = pose_4x4[:3, :3]
        t = pose_4x4[:3, 3]
        rvec, _ = _cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        axes_3d = np.float32([[0,0,0],[0.05,0,0],[0,0.05,0],[0,0,0.05]])
        try:
            pts2d, _ = _cv2.projectPoints(axes_3d, rvec, tvec,
                                           K.reshape(3,3), np.zeros(5))
            pts2d = pts2d.astype(int).reshape(-1, 2)
            o = tuple(pts2d[0])
            _cv2.arrowedLine(pose_img, o, tuple(pts2d[1]), (255,0,0), 3)
            _cv2.arrowedLine(pose_img, o, tuple(pts2d[2]), (0,255,0), 3)
            _cv2.arrowedLine(pose_img, o, tuple(pts2d[3]), (0,0,255), 3)
        except Exception:
            pass
    axes[3].imshow(pose_img)
    axes[3].set_title("Pose estimée")
    axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out = f"{save_path}/viz_{obj_f}_frame{frame_idx:04d}.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close()
from estimater import *
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd

from metrics import *
import json
from renderer_pyrender import RendererVispy
from pytorch_lightning import seed_everything
from datetime import datetime

if __name__ == '__main__':
    seed_everything(0)

    YCBV_DIR    = '/workspace/dataset/ycbv/test'
    MODELS_DIR  = '/workspace/dataset/ycbv/models'
    ANCHOR_DIR  = '/workspace/anchor_results/dexycb_reference_view_ours'
    RESULTS_DIR = '/workspace/results/ycbv_yoloe_eval'
    MODELS_INFO = '/workspace/dataset/ycbv/models/models_info.json'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    date_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S}'
    save_results_est_path = f'{RESULTS_DIR}/{date_str}'
    os.makedirs(save_results_est_path, exist_ok=True)

    selected = json.load(open('/workspace/results/ycbv_eval_set_v2.json'))

    with open(MODELS_INFO, 'r') as f:
        model_info = json.load(f)

    all_frame_data = {
        'Frame_ID': [], 'Class': [], 'ADD-S': [], 'ADD': [],
        'pred_R': [], 'pred_t': [], 'gt_R_m': [], 'gt_t_m': [],
        'AR': [], 'MSSD': [], 'MSPD': [], 'VSD': [],
        'R_error': [], 'T_error': [],
    }

    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)))
    mesh_init = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(), faces=mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh_init, scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                debug_dir=save_results_est_path, debug=0, glctx=glctx)

    renderer = RendererVispy(640, 480, mode='depth')
    obj_count = 0

    for s in tqdm(selected, desc="Evaluating Frames"):
        obj_id   = s['obj_id']
        obj_name = s['obj_name']

        img_bgr = cv2.imread(s['img_path'])
        color   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = color.shape[:2]

        dep   = cv2.imread(s['dep_path'], cv2.IMREAD_UNCHANGED)
        depth = dep.astype(np.float32) * s['depth_scale'] / 1000.0 if dep is not None else None

        K = np.array(s['K_flat']).reshape(3, 3)

        T_gt_m = np.eye(4)
        T_gt_m[:3,:3] = np.array(s['gt_R']).reshape(3,3)
        T_gt_m[:3, 3] = np.array(s['gt_t']) / 1000.0

        mg      = cv2.imread(s['mask_gt_path'], cv2.IMREAD_GRAYSCALE)
        mask_gt = mg > 127 if mg is not None else None

        yoloe_mask = get_yoloe_mask(color, obj_id, H, W)
        mask = yoloe_mask if yoloe_mask is not None else mask_gt
        if mask is None:
            print(f'  [{obj_name}] No mask')
            continue

        mesh_path = f"{ANCHOR_DIR}/{obj_name}/center_mesh_{obj_name}.obj"
        if not os.path.exists(mesh_path):
            mesh_path = f"{MODELS_DIR}/obj_{obj_id:06d}.ply"
        if not os.path.exists(mesh_path):
            print(f'  [{obj_name}] Mesh not found')
            continue

        mesh = trimesh.load(mesh_path)
        est.reset_object(mesh=mesh, symmetry_tfs=None)

        pred_pose = est.register(K=K, rgb=color, depth=depth,
                                  ob_mask=mask, iteration=5, name=obj_name)

        if obj_count % 2 == 0:
            plot_yoloe_frame(color, depth,
                             yoloe_mask if yoloe_mask is not None else mask,
                             pred_pose, K, obj_name, obj_count,
                             save_results_est_path)

        gt_diameter = float(np.linalg.norm(
            np.array(mesh.vertices).max(0) - np.array(mesh.vertices).min(0)))

        trans_disc = [{"R": np.eye(3), "t": np.array([[0,0,0]]).T}]
        str_id = str(obj_id)
        if str_id in model_info and "symmetries_discrete" in model_info[str_id]:
            for sym in model_info[str_id]["symmetries_discrete"]:
                sym_4x4 = np.reshape(sym, (4,4))
                trans_disc.append({"R": sym_4x4[:3,:3], "t": sym_4x4[:3,3].reshape(3,1)})

        add  = compute_add(np.array(mesh.vertices), pred_pose, T_gt_m)
        adds = compute_adds(np.array(mesh.vertices), pred_pose, T_gt_m)
        add_thres  = float(add  <= gt_diameter * 0.1)
        adds_thres = float(adds <= gt_diameter * 0.1)

        err_R, err_T = compute_RT_distances(pred_pose, T_gt_m)

        pred_q = pred_pose.astype(np.float16)
        gt_q   = T_gt_m.astype(np.float16)
        pred_r = pred_q[:3,:3]
        pred_t = np.expand_dims(pred_q[:3,3], axis=1) * 1e3
        gt_r   = gt_q[:3,:3]
        gt_t   = np.expand_dims(gt_q[:3,3], axis=1) * 1e3

        gt_mesh_dict = {
            'pts': np.array(mesh.vertices) * 1e3,
            'normals': np.array(mesh.face_normals),
            'faces': np.array(mesh.faces),
        }
        renderer.my_add_object(gt_mesh_dict, obj_id)

        mssd_err = mssd(pose_est=pred_q, pose_gt=gt_q,
                        pts=np.array(mesh.vertices), syms=trans_disc) * 1e3
        mspd_err = mspd(pose_est=pred_q, pose_gt=gt_q,
                        pts=np.array(mesh.vertices), K=K, syms=trans_disc)

        mssd_rec  = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
        mspd_rec  = np.array([5,10,15,20,25,30,35,40,45,50])
        vsd_delta = 15.0
        vsd_taus  = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        vsd_rec   = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])

        vsd_errs = vsd(pred_r, pred_t, gt_r, gt_t,
                       depth * 1e3, K.reshape(3,3),
                       vsd_delta, vsd_taus, True,
                       gt_diameter * 1e3, renderer, obj_id)
        vsd_errs  = np.asarray(vsd_errs)
        mean_vsd  = np.stack([vsd_errs < r for r in vsd_rec], axis=1).mean()
        mean_mssd = (mssd_err < mssd_rec * gt_diameter * 1e3).mean()
        mean_mspd = (mspd_err < mspd_rec).mean()
        mean_ar   = (mean_mssd + mean_mspd + mean_vsd) / 3.0

        all_frame_data['Frame_ID'].append(obj_count)
        all_frame_data['Class'].append(obj_name)
        all_frame_data['ADD-S'].append(adds_thres)
        all_frame_data['ADD'].append(add_thres)
        all_frame_data['AR'].append(mean_ar)
        all_frame_data['MSSD'].append(mean_mssd)
        all_frame_data['MSPD'].append(mean_mspd)
        all_frame_data['VSD'].append(mean_vsd)
        all_frame_data['pred_R'].append(pred_pose[:3,:3].tolist())
        all_frame_data['pred_t'].append(pred_pose[:3,3].tolist())
        all_frame_data['gt_R_m'].append(T_gt_m[:3,:3].tolist())
        all_frame_data['gt_t_m'].append(T_gt_m[:3,3].tolist())
        all_frame_data['R_error'].append(float(err_R))
        all_frame_data['T_error'].append(float(err_T))
        obj_count += 1

    df_all = pd.DataFrame(all_frame_data)
    means_row = {
        'Frame_ID': 'MEAN', 'Class': 'ALL',
        'ADD-S': f"{df_all['ADD-S'].mean()*100:.1f}",
        'ADD':   f"{df_all['ADD'].mean()*100:.1f}",
        'AR':    f"{df_all['AR'].mean()*100:.1f}",
        'MSSD':  f"{df_all['MSSD'].mean()*100:.1f}",
        'MSPD':  f"{df_all['MSPD'].mean()*100:.1f}",
        'VSD':   f"{df_all['VSD'].mean()*100:.1f}",
        'R_error': f"{df_all['R_error'].mean():.1f}",
        'T_error': f"{df_all['T_error'].mean():.1f}",
    }
    df_all = pd.concat([df_all, pd.DataFrame([means_row])], ignore_index=True)
    df_all.to_excel(f'{save_results_est_path}/ycbv_all_frames_metrics.xlsx', index=False)

    print(f"\nMEAN ADD={df_all['ADD'][:-1].astype(float).mean()*100:.1f}% ADD-S={df_all['ADD-S'][:-1].astype(float).mean()*100:.1f}% AR={df_all['AR'][:-1].astype(float).mean()*100:.1f}%")
    print(f"Saved: {save_results_est_path}/ycbv_all_frames_metrics.xlsx")

    final_json = {
        'dataset': 'YCB-V BOP',
        'n_frames': obj_count,
        'mean_ADD':   round(float(df_all['ADD'][:-1].astype(float).mean()*100), 1),
        'mean_ADD-S': round(float(df_all['ADD-S'][:-1].astype(float).mean()*100), 1),
        'mean_AR':    round(float(df_all['AR'][:-1].astype(float).mean()*100), 1),
        'mean_R_error': round(float(df_all['R_error'][:-1].astype(float).mean()), 1),
        'mean_T_error': round(float(df_all['T_error'][:-1].astype(float).mean()), 2),
    }
    with open(f'{save_results_est_path}/ycbv_final_metrics.json', 'w') as f:
        json.dump(final_json, f, indent=2)
    print(f"Saved: {save_results_est_path}/ycbv_final_metrics.json")
