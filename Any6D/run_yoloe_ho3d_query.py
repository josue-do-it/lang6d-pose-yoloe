import copy
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
_yoloe_model = None

def get_yoloe_mask(img_rgb, obj_f, H, W):
    global _yoloe_model
    import os, cv2 as _cv2
    _orig_dir = os.getcwd()
    if _yoloe_model is None:
        os.chdir('/workspace/yoloe')
        _yoloe_model = _YOLOE("yoloe-26l-seg.pt")
        os.chdir(_orig_dir)
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

    parser = argparse.ArgumentParser(description="Set experiment name and paths")

    parser.add_argument("--name", type=str, default="any6d", help="Experiment name")
    parser.add_argument("--anchor_path", type=str, default="/home/miruware/ssd_4tb/cvpr_2025_results/anchor_results/dexycb_reference_view_ours", help="Path to the YCB-V model info JSON")
    parser.add_argument("--hot3d_data_root", type=str, default="/home/miruware/ssd_4tb/dataset/ho3d", help="Path to the HO3D dataset root")
    parser.add_argument("--ycb_model_path", type=str, default="/home/miruware/ssd_4tb/dataset/ho3d/YCB_Video_Models", help="Path to the YCB Video Models")
    parser.add_argument("--ycbv_modesl_info_path", type=str, default="./models_info.json", help="Path to the YCB-V model info JSON")
    parser.add_argument("--start_idx", type=int, default=0, help="Resume from object index")
    parser.add_argument("--end_idx", type=int, default=13, help="Stop before this object index (exclusive)")
    parser.add_argument("--running_stride", type=int, default=10, help="Running stride")
    parser.add_argument("--save_dir", type=str, default=None, help="Fixed save directory (for multi-run aggregation)")

    args = parser.parse_args()

    name = args.name
    hot3d_data_root = args.hot3d_data_root
    ycbv_modesl_info_path = args.ycbv_modesl_info_path
    running_stride = args.running_stride
    start_idx = args.start_idx
    end_idx = args.end_idx
    anchor_path = args.anchor_path
    ycb_model_path = args.ycb_model_path

    if args.save_dir:
        save_results_est_path = args.save_dir
    else:
        date_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S}'
        save_results_est_path = f"./results/ho3d_results/{name}/{date_str}"

    os.makedirs(save_results_est_path, exist_ok=True)

    obj_folder =[
        'MPM10',
        'MPM11',
        'MPM12',
        'MPM13',
        'MPM14',
        'AP10',
        'AP11',
        'AP12',
        'AP13',
        'AP14',
        'SB11',
        'SB13',
        'SM1',
        ]

    object_metrics = {obj: {
        'ADD': [], 'ADD-S': [], 'AR': [], 'VSD': [], 'MSSD': [], 'MSPD': [],
        'R error': [], 'T error': [], 'cls_id': [], 'instance_id': []
        } for obj in obj_folder}
    all_frame_data = {
        'Frame_ID': [],
        'Class': [],
        'ADD-S': [],
        'ADD': [],
        'AR': [],
        'MSSD': [],
        'MSPD': [],
        'VSD': [],
        'R_error': [],
        'T_error': [],
        }

    excel_files = []
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)))
    mesh = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(), faces= mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir=save_results_est_path, debug=0, glctx=glctx)

    renderer = RendererVispy(640, 480, mode='depth')
    obj_count = 0

    data = []

    obj_folder = obj_folder[start_idx:end_idx]
    for obj_f in tqdm(obj_folder, desc="Evaluating Object"):

        video_dir = os.path.join(f"{hot3d_data_root}/evaluation", obj_f)
        reader = Ho3dReader(video_dir, hot3d_data_root)
        reader.color_files = reader.color_files[::running_stride]

        ob_id = reader.get_obj_id()

        # get bop information
        with open(ycbv_modesl_info_path, 'r') as f:
            model_info = json.load(f)
        trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
        if "symmetries_discrete" in model_info[f"{ob_id}"]:
            for sym in model_info[f"{ob_id}"]["symmetries_discrete"]:
                sym_4x4 = np.reshape(sym, (4, 4))
                R = sym_4x4[:3, :3]
                t = sym_4x4[:3, 3].reshape((3, 1))
                trans_disc.append({"R": R, "t": t})

        K_anchor = np.loadtxt(reader.get_reference_K(anchor_path))


        gt_mesh = reader.get_gt_mesh(ycb_model_path)
        gt_diameter = reader.get_gt_mesh_diamter(ycb_model_path)
        mesh = trimesh.load(reader.get_reference_view_1_mesh(anchor_path))

        gt_mesh_dict = {
            'pts': np.asarray(gt_mesh.vertices) * 1e3,
            'normals': np.asarray(gt_mesh.face_normals),
            'faces': np.asarray(gt_mesh.faces),
            }
        renderer.my_add_object(gt_mesh_dict, ob_id)

        pred_pose_a = np.loadtxt(reader.get_reference_view_1_pose(anchor_path))
        gt_pose_a = np.loadtxt(reader.get_reference_view_1_pose(anchor_path).replace('initial','gt'))

        est.reset_object(mesh=mesh, symmetry_tfs=None)

        det_total = 0
        det_detected = 0
        det_iou_list = []

        for i in tqdm(range(0, len(reader.color_files), 1), desc=f"{obj_f} - Frames"):
            gt_pose_q = reader.get_gt_pose(i)

            if gt_pose_q is None:
                continue

            det_total += 1
            color_file = reader.color_files[i]
            color = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
            H, W = color.shape[:2]
            depth = reader.get_depth(i)
            yoloe_mask = get_yoloe_mask(color, obj_f, H, W)
            gt_mask_raw = reader.get_mask(i)
            if yoloe_mask is not None:
                det_detected += 1
                mask = yoloe_mask
                if gt_mask_raw is not None:
                    gt_m = gt_mask_raw.astype(bool)
                    intersection = np.logical_and(yoloe_mask, gt_m).sum()
                    union = np.logical_or(yoloe_mask, gt_m).sum()
                    if union > 0:
                        det_iou_list.append(float(intersection) / float(union))
            else:
                if gt_mask_raw is None:
                    continue
                mask = gt_mask_raw.astype(np.bool_)
            pred_pose_q = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=obj_f)

            # Plot pour la première frame de chaque objet
            if i == 0:
                plot_yoloe_frame(color, depth, yoloe_mask if yoloe_mask is not None else mask,
                                 pred_pose_q, reader.K, obj_f, i, save_results_est_path)

            pose_aq = pred_pose_q @ np.linalg.inv(pred_pose_a)  # obtained pose A->Q
            pred_q = pose_aq @ gt_pose_a


            err_R, err_T = compute_RT_distances(pred_q, gt_pose_q)

            pose_recall_th = [(5, 5), (5, 10), (10, 10)]

            for r_th, t_th in pose_recall_th:
                succ_r, succ_t = err_R <= r_th, err_T <= t_th
                succ_pose = np.logical_and(succ_r, succ_t).astype(float)

            add = compute_add(gt_mesh.vertices, pred_q, gt_pose_q)
            adds = compute_adds(gt_mesh.vertices, pred_q, gt_pose_q)


            add_thres = float(add <= gt_diameter * 0.1)
            adds_thres = float(adds <= gt_diameter * 0.1)


            pred_q, gt_pose_q = pred_q.astype(np.float16), gt_pose_q.astype(np.float16)

            pred_r, pred_t = pred_q[:3, :3], np.expand_dims(pred_q[:3, 3], axis=1) * 1e3
            gt_r, gt_t = gt_pose_q[:3, :3], np.expand_dims(gt_pose_q[:3, 3], axis=1) * 1e3

            mssd_err = mssd(pose_est=pred_q, pose_gt=gt_pose_q, pts=gt_mesh.vertices, syms=trans_disc) * 1e3
            mspd_err = mspd(pose_est=pred_q, pose_gt=gt_pose_q, pts=gt_mesh.vertices, K=reader.K, syms=trans_disc)

            mssd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

            vsd_delta = 15.0
            vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            vsd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

            vsd_errs = vsd(pred_r, pred_t, gt_r, gt_t, (depth *1e3), reader.K.reshape(3, 3), vsd_delta, vsd_taus, True, (gt_diameter*1e3), renderer, ob_id)
            vsd_errs = np.asarray(vsd_errs)
            all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
            mean_vsd = all_vsd_recs.mean()

            mssd_cur_rec = mssd_rec * (gt_diameter * 1e3)
            mean_mssd = (mssd_err < mssd_cur_rec).mean()
            mean_mspd = (mspd_err < mspd_rec).mean()

            mean_ar = (mean_mssd + mean_mspd + mean_vsd) / 3.


            object_metrics[obj_f]['ADD'].append(add_thres)
            object_metrics[obj_f]['ADD-S'].append(adds_thres)
            object_metrics[obj_f]['AR'].append(mean_ar)
            object_metrics[obj_f]['VSD'].append(mean_vsd)
            object_metrics[obj_f]['MSSD'].append(mean_mssd)
            object_metrics[obj_f]['MSPD'].append(mean_mspd)
            object_metrics[obj_f]['R error'].append(err_R.tolist()[0])
            object_metrics[obj_f]['T error'].append(err_T.tolist()[0])
            object_metrics[obj_f]['cls_id'].append(obj_f)
            object_metrics[obj_f]['instance_id'].append(obj_count)

            try:
                visualize_frame_results_gt(color=color, gt_mesh=gt_mesh, K=reader.K, gt_pose=gt_pose_q, pred_pose=pred_pose_q, metric=object_metrics[obj_f], obj_f=f"{obj_f}", frame_idx=i, save_path=save_results_est_path, glctx=glctx, name=f"{len(reader.color_files)}_{name}",nocs_metric=True, est_mesh=est.mesh)
            except:
                pass
            obj_count+=1


        df_obj = pd.DataFrame({
            'Frame_ID': object_metrics[obj_f]['instance_id'],
            'Class': object_metrics[obj_f]['cls_id'],
            'ADD-S': object_metrics[obj_f]['ADD-S'],
            'ADD': object_metrics[obj_f]['ADD'],
            'AR': object_metrics[obj_f]['AR'],
            'MSSD': object_metrics[obj_f]['MSSD'],
            'MSPD': object_metrics[obj_f]['MSPD'],
            'VSD': object_metrics[obj_f]['VSD'],
            'R_error': object_metrics[obj_f]['R error'],
            'T_error': object_metrics[obj_f]['T error'],
            })

        means_all = {
            'ADD-S': np.mean(object_metrics[obj_f]['ADD-S']) * 100,
            'ADD': np.mean(object_metrics[obj_f]['ADD']) * 100,
            'AR': np.mean(object_metrics[obj_f]['AR']) * 100,
            'MSSD': np.mean(object_metrics[obj_f]['MSSD']) * 100,
            'MSPD': np.mean(object_metrics[obj_f]['MSPD']) * 100,
            'VSD': np.mean(object_metrics[obj_f]['VSD']) * 100,
            'R_error': np.mean(object_metrics[obj_f]['R error']),
            'T_error': np.mean(object_metrics[obj_f]['T error'])
            }

        mean_row_df = pd.DataFrame({
            'Frame_ID': ['MEAN'],
            'Class': [obj_f],
            'ADD-S': [f"{means_all['ADD-S']:.1f}"],
            'ADD': [f"{means_all['ADD']:.1f}"],
            'AR': [f"{means_all['AR']:.1f}"],
            'MSSD': [f"{means_all['MSSD']:.1f}"],
            'MSPD': [f"{means_all['MSPD']:.1f}"],
            'VSD': [f"{means_all['VSD']:.1f}"],
            'R_error': [f"{means_all['R_error']:.1f}"],
            'T_error': [f"{means_all['T_error']:.1f}"]
            })

        df_obj = pd.concat([df_obj, mean_row_df], ignore_index=True)

        row_data = {
            'Class_ID': obj_f,
            'ADD-S': f"{means_all['ADD-S']:.1f}",
            'ADD': f"{means_all['ADD']:.1f}",
            'AR': f"{means_all['AR']:.1f}",
            'MSSD': f"{means_all['MSSD']:.1f}",
            'MSPD': f"{means_all['MSPD']:.1f}",
            'VSD': f"{means_all['VSD']:.1f}",
            }

        data.append(row_data)

        df_obj.to_excel(f'{save_results_est_path}/{obj_f}_metrics_results.xlsx', index=False)
        import json as _json
        _det_stats = {
            'prompt': HO3D_YOLOE_PROMPTS.get(obj_f, 'object'),
            'total_frames': det_total,
            'detected_frames': det_detected,
            'detection_rate': round(det_detected / det_total * 100, 1) if det_total > 0 else 0.0,
            'mean_iou': round(float(np.mean(det_iou_list)) * 100, 1) if det_iou_list else 0.0,
            'fallback_frames': det_total - det_detected,
        }
        with open(f'{save_results_est_path}/{obj_f}_yoloe_detection.json', 'w') as _jf:
            _json.dump(_det_stats, _jf, indent=2)
        _obj_summary = {
            'ADD':  round(float(np.mean(object_metrics[obj_f]['ADD'])) * 100, 2),
            'ADD-S': round(float(np.mean(object_metrics[obj_f]['ADD-S'])) * 100, 2),
            'MSSD': round(float(np.mean(object_metrics[obj_f]['MSSD'])) * 100, 2),
            'MSPD': round(float(np.mean(object_metrics[obj_f]['MSPD'])) * 100, 2),
            'VSD':  round(float(np.mean(object_metrics[obj_f]['VSD'])) * 100, 2),
            'AR':   round(float(np.mean(object_metrics[obj_f]['AR'])) * 100, 2),
            'R_error': round(float(np.mean(object_metrics[obj_f]['R error'])), 2),
            'T_error': round(float(np.mean(object_metrics[obj_f]['T error'])), 2),
        }
        with open(f'{save_results_est_path}/{obj_f}_metrics.json', 'w') as _jf:
            _json.dump(_obj_summary, _jf, indent=2)
        all_frame_data['Frame_ID'].extend(object_metrics[obj_f]['instance_id'])
        all_frame_data['Class'].extend(object_metrics[obj_f]['cls_id'])
        all_frame_data['ADD-S'].extend(object_metrics[obj_f]['ADD-S'])
        all_frame_data['ADD'].extend(object_metrics[obj_f]['ADD'])
        all_frame_data['AR'].extend(object_metrics[obj_f]['AR'])
        all_frame_data['MSSD'].extend(object_metrics[obj_f]['MSSD'])
        all_frame_data['MSPD'].extend(object_metrics[obj_f]['MSPD'])
        all_frame_data['VSD'].extend(object_metrics[obj_f]['VSD'])
        all_frame_data['R_error'].extend(object_metrics[obj_f]['R error'])
        all_frame_data['T_error'].extend(object_metrics[obj_f]['T error'])

    overall_means = {
        'ADD-S': np.mean([np.mean(object_metrics[obj]['ADD-S']) for obj in obj_folder]) * 100,
        'ADD': np.mean([np.mean(object_metrics[obj]['ADD']) for obj in obj_folder]) * 100,
        'AR': np.mean([np.mean(object_metrics[obj]['AR']) for obj in obj_folder]) * 100,
        'MSSD': np.mean([np.mean(object_metrics[obj]['MSSD']) for obj in obj_folder]) * 100,
        'MSPD': np.mean([np.mean(object_metrics[obj]['MSPD']) for obj in obj_folder]) * 100,
        'VSD': np.mean([np.mean(object_metrics[obj]['VSD']) for obj in obj_folder]) * 100,
        }

    mean_row = {
        'Class_ID': 'MEAN',
        'ADD-S': f"{overall_means['ADD-S']:.1f}",
        'ADD': f"{overall_means['ADD']:.1f}",
        'AR': f"{overall_means['AR']:.1f}",
        'MSSD': f"{overall_means['MSSD']:.1f}",
        'MSPD': f"{overall_means['MSPD']:.1f}",
        'VSD': f"{overall_means['VSD']:.1f}",
        }
    data.append(mean_row)

    latex_str = f"MEAN & {means_all['AR']:.1f} & {means_all['VSD']:.1f} & {means_all['MSSD']:.1f} & {means_all['MSPD']:.1f} & {means_all['ADD-S']:.1f} & - \\\\"
    print("\n" + latex_str)

    df = pd.DataFrame(data)
    df.to_excel(f'{save_results_est_path}/0_mean_all_metrics_classes_results.xlsx', index=False)

    df_all_frames = pd.DataFrame(all_frame_data)

    means_all = {
        'Frame_ID': 'MEAN',
        'Class': 'ALL',
        'ADD-S': f"{df_all_frames['ADD-S'].mean() * 100:.1f}",
        'ADD': f"{df_all_frames['ADD'].mean() * 100:.1f}",
        'AR': f"{df_all_frames['AR'].mean() * 100:.1f}",
        'MSSD': f"{df_all_frames['MSSD'].mean() * 100:.1f}",
        'MSPD': f"{df_all_frames['MSPD'].mean() * 100:.1f}",
        'VSD': f"{df_all_frames['VSD'].mean() * 100:.1f}",
        'R_error': f"{df_all_frames['R_error'].mean():.1f}",
        'T_error': f"{df_all_frames['T_error'].mean():.1f}",
        }

    df_all_frames = pd.concat([df_all_frames, pd.DataFrame([means_all])], ignore_index=True)

    output_path = f'{save_results_est_path}/0_all_frames_metrics_results.xlsx'
    df_all_frames.to_excel(output_path, index=False)
    print(f"\nAll frames metrics saved to {output_path}")

    print("\nSaved data preview:")
    print(df_all_frames)




