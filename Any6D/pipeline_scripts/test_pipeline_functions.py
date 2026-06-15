"""
Unit tests for core/ shared utilities.
Tests only pure Python logic — no Docker, no GPU, no dataset required.

Run:
    python -m pytest Any6D/pipeline_scripts/test_pipeline_functions.py -v
    # or inside Docker:
    /opt/conda/envs/Any6D/bin/python3 -m pytest /workspace/pipeline_scripts/test_pipeline_functions.py -v
"""

import sys
import os
import numpy as np
import pytest

# Allow imports from /workspace/core inside Docker, or from Any6D/core locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.llm import parse_llm
from core.pose_utils import rotation_error_deg, translation_error_cm
from core.metrics_utils import nanmean
from core.detection import compute_iou
from core.constants import ADD_THRESH_RATIO, MM_TO_M

# LM-specific metadata stays in the pipeline script
sys.path.insert(0, os.path.dirname(__file__))
from run_full_pipeline_linemod import LM_NAMES, LM_SYMMETRIC, LM_INSTRUCTIONS, CALIBRATED_SYSTEM

_SYSTEM_TXT = os.path.join(os.path.dirname(__file__), 'calibrated_system_lm.txt')


# ── parse_llm ─────────────────────────────────────────────────────────────────

class TestParseLlm:

    def test_arrow_format(self):
        assert parse_llm("→ rubber duck") == "rubber duck"

    def test_quoted_keyword(self):
        assert parse_llm('The keyword is "yellow driller"') == "yellow driller"

    def test_plain_single_word(self):
        assert parse_llm("driller") == "driller"

    def test_strips_prefix(self):
        assert parse_llm("keyword: benchvise") == "benchvise"

    def test_multiline_takes_first_valid(self):
        raw = "keyword: rubber duck\nSome explanation after."
        assert parse_llm(raw) == "rubber duck"

    def test_empty_string_returns_empty(self):
        assert parse_llm("") == ""

    def test_too_many_words_skips_line(self):
        raw = "this has five words here\nduck"
        assert parse_llm(raw) == "duck"

    def test_max_3_words_accepted(self):
        assert parse_llm("orange glue bottle") == "orange glue bottle"


# ── rotation_error_deg ────────────────────────────────────────────────────────

class TestRotationError:

    def test_identity_gives_zero(self):
        R = np.eye(3)
        assert rotation_error_deg(R, R) == pytest.approx(0.0, abs=1e-5)

    def test_90_degree_rotation(self):
        R_pred = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        R_gt   = np.eye(3)
        assert rotation_error_deg(R_pred, R_gt) == pytest.approx(90.0, abs=1e-4)

    def test_180_degree_rotation(self):
        R_pred = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        R_gt   = np.eye(3)
        assert rotation_error_deg(R_pred, R_gt) == pytest.approx(180.0, abs=1e-4)

    def test_same_rotation_gives_zero(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        assert rotation_error_deg(R, R) == pytest.approx(0.0, abs=1e-5)

    def test_returns_float(self):
        assert isinstance(rotation_error_deg(np.eye(3), np.eye(3)), float)


# ── translation_error_cm ──────────────────────────────────────────────────────

class TestTranslationError:

    def test_zero_error(self):
        t = np.array([0.1, 0.2, 0.5])
        assert translation_error_cm(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_10cm_offset(self):
        t_pred = np.array([0.10, 0.0, 0.0])
        t_gt   = np.array([0.00, 0.0, 0.0])
        assert translation_error_cm(t_pred, t_gt) == pytest.approx(10.0, abs=1e-4)

    def test_diagonal_offset(self):
        t_pred = np.array([0.01, 0.01, 0.01])
        t_gt   = np.zeros(3)
        assert translation_error_cm(t_pred, t_gt) == pytest.approx(
            np.sqrt(3) * 1.0, abs=1e-4)

    def test_returns_float(self):
        assert isinstance(translation_error_cm(np.zeros(3), np.zeros(3)), float)


# ── nanmean ───────────────────────────────────────────────────────────────────

class TestNanmean:

    def test_all_valid(self):
        assert nanmean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_with_nans(self):
        assert nanmean([1.0, float('nan'), 3.0]) == pytest.approx(2.0)

    def test_all_nan_returns_nan(self):
        assert np.isnan(nanmean([float('nan'), float('nan')]))

    def test_empty_returns_nan(self):
        assert np.isnan(nanmean([]))

    def test_single_value(self):
        assert nanmean([5.0]) == pytest.approx(5.0)


# ── compute_iou ───────────────────────────────────────────────────────────────

class TestComputeIoU:

    def test_perfect_overlap(self):
        mask = np.ones((10, 10), dtype=bool)
        assert compute_iou(mask, mask) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:5, :] = True
        b[5:, :] = True
        assert compute_iou(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_half_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:, :5]  = True
        b[:, 2:7] = True
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_none_gt_returns_minus_one(self):
        mask = np.ones((10, 10), dtype=bool)
        assert compute_iou(mask, None) == -1.0


# ── Metadata consistency ──────────────────────────────────────────────────────

class TestMetadata:

    def test_all_15_objects_have_names(self):
        assert set(LM_NAMES.keys()) == set(range(1, 16))

    def test_all_15_objects_have_instructions(self):
        assert set(LM_INSTRUCTIONS.keys()) == set(range(1, 16))

    def test_symmetric_objects_are_valid_ids(self):
        assert LM_SYMMETRIC.issubset(set(LM_NAMES.keys()))

    def test_add_thresh_ratio_is_10_percent(self):
        assert ADD_THRESH_RATIO == pytest.approx(0.10)

    def test_mm_to_m_conversion(self):
        assert MM_TO_M == pytest.approx(0.001)

    def test_instructions_are_non_empty_strings(self):
        for obj_id, instr in LM_INSTRUCTIONS.items():
            assert isinstance(instr, str) and len(instr) > 5, \
                f"Instruction for obj {obj_id} is too short: '{instr}'"


# ── CALIBRATED_SYSTEM ─────────────────────────────────────────────────────────

class TestCalibratedSystem:

    def test_system_is_non_empty(self):
        assert len(CALIBRATED_SYSTEM.strip()) > 100

    def test_system_has_many_examples(self):
        count = CALIBRATED_SYSTEM.count('→')
        assert count >= 600, f"Expected >=600 examples, got {count}"

    def test_system_covers_all_lm_objects(self):
        lm_keywords = ['toy monkey', 'benchvise', 'red bowl', 'black camera',
                       'steel can', 'orange cat', 'blue cup', 'yellow drill',
                       'rubber duck', 'egg box', 'glue bottle', 'hole puncher',
                       'toy iron', 'desk lamp', 'mobile phone']
        for kw in lm_keywords:
            assert kw in CALIBRATED_SYSTEM, f"Missing keyword '{kw}' in CALIBRATED_SYSTEM"

    def test_system_covers_ycb_objects(self):
        ycb_keywords = ['tomato soup can', 'yellow mustard', 'sugar box',
                        'tuna fish can', 'blue pudding box', 'yellow banana',
                        'bleach cleanser', 'red mug', 'yellow drill', 'red scissors']
        for kw in ycb_keywords:
            assert kw in CALIBRATED_SYSTEM, f"Missing YCB keyword '{kw}'"

    def test_system_covers_hope_objects(self):
        hope_keywords = ['alphabet soup can', 'BBQ sauce bottle', 'butter box',
                         'granola bars box', 'honey bottle', 'mayonnaise jar',
                         'popcorn bag', 'spaghetti box', 'tomato sauce can']
        for kw in hope_keywords:
            assert kw in CALIBRATED_SYSTEM, f"Missing HOPE keyword '{kw}'"

    def test_system_covers_industrial_objects(self):
        industrial = ['gray plastic cylinder', 'gray flat disc', 'gray L bracket',
                      'shiny metal bracket', 'silver metal ring', 'metal elbow connector']
        for kw in industrial:
            assert kw in CALIBRATED_SYSTEM, f"Missing T-LESS/ITODD keyword '{kw}'"

    def test_txt_file_exists(self):
        assert os.path.isfile(_SYSTEM_TXT), f"Backup txt not found: {_SYSTEM_TXT}"

    def test_txt_file_matches_code(self):
        with open(_SYSTEM_TXT, 'r') as f:
            txt_content = f.read().strip()
        assert txt_content in CALIBRATED_SYSTEM, \
            "calibrated_system_lm.txt content does not match CALIBRATED_SYSTEM in code"

    def test_no_french_text_in_examples(self):
        french_words = ['trouve', 'donne', 'prend', 'passe', 'récupère']
        for word in french_words:
            assert word not in CALIBRATED_SYSTEM.lower(), \
                f"French word '{word}' found in CALIBRATED_SYSTEM"

    def test_all_examples_have_arrow_and_keyword(self):
        for i, line in enumerate(CALIBRATED_SYSTEM.splitlines(), 1):
            if '→' in line:
                parts = line.split('→')
                assert len(parts) == 2, f"Line {i} malformed: {line!r}"
                keyword = parts[1].strip()
                assert 1 <= len(keyword.split()) <= 4, \
                    f"Line {i} keyword too long or empty: '{keyword}'"


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', __file__, '-v', '--tb=short'],
        capture_output=False)
    sys.exit(result.returncode)
