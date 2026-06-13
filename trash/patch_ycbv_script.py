with open('/workspace/run_yoloe_ycbv_query.py', 'r') as f:
    content = f.read()

old_prompts = 'HO3D_YOLOE_PROMPTS = {\n    "MPM10": "canned food",          "MPM11": "spam can",\n    "MPM12": "spam can",             "MPM13": "canned food",\n    "MPM14": "canned food",          "AP10":  "blue jug",\n    "AP11":  "blue pitcher",         "AP12":  "blue object",\n    "AP13":  "cup with handle",      "AP14":  "cup with handle",\n    "SB11":  "white plastic bottle", "SB13":  "white plastic bottle",\n    "SM1":   "canned food",\n}'

new_prompts = old_prompts + '''

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
}'''

assert old_prompts in content, "prompts not found"
content = content.replace(old_prompts, new_prompts)

old_mask_line = '    prompt = HO3D_YOLOE_PROMPTS.get(obj_f, "object")'
new_mask_line = '    if isinstance(obj_f, int):\n        prompt = YCBV_YOLOE_PROMPTS.get(obj_f, "object")\n    else:\n        prompt = HO3D_YOLOE_PROMPTS.get(obj_f, "object")'
assert old_mask_line in content, "mask line not found"
content = content.replace(old_mask_line, new_mask_line)

main_start = content.index("if __name__ == '__main__':")
header = content[:main_start]

new_main = """if __name__ == '__main__':
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

    print(f"\\nMEAN ADD={df_all['ADD'][:-1].astype(float).mean()*100:.1f}% ADD-S={df_all['ADD-S'][:-1].astype(float).mean()*100:.1f}% AR={df_all['AR'][:-1].astype(float).mean()*100:.1f}%")
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
"""

new_content = header + new_main
with open('/workspace/run_yoloe_ycbv_query.py', 'w') as f:
    f.write(new_content)
print("Script patched OK")
