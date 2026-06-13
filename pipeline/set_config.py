"""
set_config.py
Central configuration for the YOLOE + Any6D pipeline.
Edit this file to add new datasets, objects, or change parameters.
"""

import os

# ── Root paths ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.expanduser('~/open-vocabulary-6d-pose-yoloe')
ANY6D_DIR    = os.path.join(BASE_DIR, 'Any6D')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
EVAL_DIR     = os.path.join(BASE_DIR, 'evaluation', 'pipeline_results')

# ── Model paths ───────────────────────────────────────────────────────────────
YOLOE_MODEL_PATH = os.path.join(BASE_DIR, 'notebooks', 'yoloe-26l-seg.pt')

# ── YOLOE parameters ──────────────────────────────────────────────────────────
YOLOE_CONF = 0.1      # detection confidence threshold
YOLOE_IOU  = 0.45     # NMS IoU threshold

# ── Any6D parameters ──────────────────────────────────────────────────────────
ANY6D_ITER = 5        # refinement iterations inside Docker

# ── BOP metric thresholds (standard BOP 2019) ────────────────────────────────
THRESH_ADD_RATIO  = 0.10   # ADD / ADD-S < 10% of object diameter
THRESH_MSSD_RATIO = 0.20   # MSSD < 20% of object diameter
THRESH_MSPD_PX    = 20.0   # MSPD < 20 pixels

# ── Dataset definitions ───────────────────────────────────────────────────────
# anchor_dir : where the anchor frames live (color, depth, mask, K, mesh, gt_pose)
# objects    : list of objects in this dataset, each with:
#   folder   : subfolder name inside anchor_dir
#   prompt   : text prompt(s) for YOLOE  (str or list of str)
#   mesh     : mesh filename inside the folder
#   symmetric: True → use ADD-S + 36-step Z-axis symmetry for MSSD/MSPD

DATASETS = {

    # ── DexYCB ────────────────────────────────────────────────────────────────
    'dexycb': {
        'name'      : 'DexYCB',
        'use_rfix'  : True,   # DexYCB GT uses YCB-canonical frame (90-176° diff vs Any6D)
        'anchor_dir': os.path.join(ANY6D_DIR, 'anchor_results',
                                   'dexycb_reference_view_ours'),
        'objects': [
            {'folder': '003_cracker_box',
             'prompt': 'cracker box',
             'mesh'  : 'center_mesh_003_cracker_box.obj',
             'symmetric': False},
            {'folder': '004_sugar_box',
             'prompt': 'sugar box',
             'mesh'  : 'center_mesh_004_sugar_box.obj',
             'symmetric': False},
            {'folder': '005_tomato_soup_can',
             'prompt': 'tomato soup can',
             'mesh'  : 'center_mesh_005_tomato_soup_can.obj',
             'symmetric': True},
            {'folder': '006_mustard_bottle',
             'prompt': 'mustard bottle',
             'mesh'  : 'center_mesh_006_mustard_bottle.obj',
             'symmetric': False},
            {'folder': '010_potted_meat_can',
             'prompt': ['yellow box', 'small box', 'square can', 'yellow can'],
             'mesh'  : 'center_mesh_010_potted_meat_can.obj',
             'symmetric': False},
            {'folder': '019_pitcher_base',
             'prompt': ['blue pitcher', 'blue jug', 'blue water pitcher',
                        'blue plastic pitcher', 'blue container', 'blue cup'], 
             'mesh'  : 'center_mesh_019_pitcher_base.obj',
             'symmetric': False},
            {'folder': '021_bleach_cleanser',
             'prompt': ['white bottle', 'detergent bottle', 'cleaning bottle'],
             'mesh'  : 'center_mesh_021_bleach_cleanser.obj',
             'symmetric': True},
        ],
    },

    # ── HO3D ─────────────────────────────────────────────────────────────────
    'ho3d': {
        'name'      : 'HO3D',
        'use_rfix'  : False,   # HO3D GT poses are in camera frame — no fix needed
        # Anchor frames come from DexYCB (same objects, confirmed in Any6D paper)
        'anchor_dir': os.path.join(ANY6D_DIR, 'anchor_results',
                                   'dexycb_reference_view_ours'),
        # HO3D dataset root (evaluation sequences live at {ho3d_root}/evaluation/)
        'ho3d_root' : os.path.join(os.path.expanduser('~'), 'ssd_4tb',
                                   'dataset', 'ho3d'),
        # YCB mesh models for GT diameter / GT comparison
        'ycb_models': os.path.join(os.path.expanduser('~'), 'ssd_4tb',
                                   'dataset', 'ho3d', 'YCB_Video_Models'),
        # Evaluation sequences and their objects (from Any6D paper Table 1)
        'sequences' : {
            'AP10': '019_pitcher_base',
            'AP11': '019_pitcher_base',
            'AP12': '019_pitcher_base',
            'AP13': '019_pitcher_base',
            'AP14': '019_pitcher_base',
            'SM1' : '006_mustard_bottle',
            'SB11': '021_bleach_cleanser',
            'SB13': '021_bleach_cleanser',
            'MPM10': '010_potted_meat_can',
            'MPM11': '010_potted_meat_can',
            'MPM12': '010_potted_meat_can',
            'MPM13': '010_potted_meat_can',
            'MPM14': '010_potted_meat_can',
        },
        'objects': [
            {'folder': '019_pitcher_base',
             'prompt': ['blue pitcher', 'blue jug', 'blue water pitcher',
                        'blue plastic pitcher', 'blue container', 'blue cup'],
             'mesh'  : 'final_mesh_019_pitcher_base.obj',
             'symmetric': False},
            {'folder': '006_mustard_bottle',
             'prompt': 'mustard bottle',
             'mesh'  : 'final_mesh_006_mustard_bottle.obj',
             'symmetric': False},
            {'folder': '021_bleach_cleanser',
             'prompt': ['white bottle', 'detergent bottle', 'cleaning bottle'],
             'mesh'  : 'final_mesh_021_bleach_cleanser.obj',
             'symmetric': True},
            {'folder': '010_potted_meat_can',
             'prompt': ['yellow box', 'small box', 'square can', 'yellow can'],
             'mesh'  : 'final_mesh_010_potted_meat_can.obj',
             'symmetric': False},
        ],
    },

    # ── REAL275 ───────────────────────────────────────────────────────────────
    'real275': {
        'name'      : 'REAL275',
        'anchor_dir': os.path.join(DATASETS_DIR, 'real275', 'anchor_frames'),
        'objects'   : [
            # {'folder': 'bottle', 'prompt': 'bottle',
            #  'mesh': 'center_mesh_bottle.obj', 'symmetric': True},
            # {'folder': 'bowl',   'prompt': 'bowl',
            #  'mesh': 'center_mesh_bowl.obj',   'symmetric': True},
        ],
    },

    # ── Toyota-Light ─────────────────────────────────────────────────────────
    'toyota_light': {
        'name'      : 'Toyota-Light',
        'anchor_dir': os.path.join(DATASETS_DIR, 'toyota_light', 'anchor_frames'),
        'objects'   : [],
    },

    # ── LM-O (LINEMOD-Occlusion) ──────────────────────────────────────────────
    'lmo': {
        'name'      : 'LM-O',
        'anchor_dir': os.path.join(DATASETS_DIR, 'lmo', 'anchor_frames'),
        'objects'   : [],
    },
}

# ── Any6D paper results (for comparison in Excel) ────────────────────────────
# Source: Lee et al. "Any6D: Model-free 6D Pose Estimation" CVPR 2025
# Table 1: HO3D results — "Ours" column (SAM2 + Any6D, RGB-D)
# Metrics: ADD-S %, ADD %, AR %  (pass-rate at 10% diameter; AR = VSD+MSSD+MSPD mean)
PAPER_RESULTS = {
    'real275'     : {'ADD-S': None, 'AR': None},
    'toyota_light': {'ADD-S': None, 'AR': None},
    'lmo'         : {'ADD-S': None, 'AR': None},
    'dexycb'      : {'ADD-S': None, 'AR': None},   # not evaluated in paper
    # HO3D: per-sequence paper results from Table 1
    'ho3d': {
        'sequences': {
            'AP10' : {'ADD-S': 100.0, 'ADD':  16.2, 'AR': 22.2},
            'AP11' : {'ADD-S': 100.0, 'ADD':  73.8, 'AR': 59.0},
            'AP12' : {'ADD-S': 100.0, 'ADD':  48.8, 'AR': 28.3},
            'AP13' : {'ADD-S': 100.0, 'ADD':  74.4, 'AR': 45.0},
            'AP14' : {'ADD-S': 100.0, 'ADD':  35.6, 'AR': 29.7},
            'SM1'  : {'ADD-S':  86.5, 'ADD':  34.8, 'AR': 27.8},
            'SB11' : {'ADD-S': 100.0, 'ADD':  86.8, 'AR': 68.9},
            'SB13' : {'ADD-S':  99.4, 'ADD':  64.1, 'AR': 54.6},
            'MPM10': {'ADD-S':  98.7, 'ADD':  26.8, 'AR': 31.3},
            'MPM11': {'ADD-S': 100.0, 'ADD':   3.2, 'AR': 32.4},
            'MPM12': {'ADD-S': 100.0, 'ADD':   1.3, 'AR': 23.5},
            'MPM13': {'ADD-S': 100.0, 'ADD':  15.9, 'AR': 30.9},
            'MPM14': {'ADD-S':  98.7, 'ADD':  43.9, 'AR': 44.0},
        },
        'mean': {'ADD-S': 98.7, 'ADD': 40.4, 'AR': 38.3},
    },
}
