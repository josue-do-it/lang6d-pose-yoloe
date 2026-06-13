"""
Unit tests for run_full_pipeline_linemod.py utility functions.
Tests only the pure Python logic — no Docker, no GPU, no dataset required.

Run:
    python -m pytest Any6D/pipeline_scripts/test_pipeline_functions.py -v
    # or directly:
    python Any6D/pipeline_scripts/test_pipeline_functions.py
"""

import sys
import os
import json
import tempfile

import numpy as np
import pytest

# Add pipeline_scripts to path so we can import without Docker
sys.path.insert(0, os.path.dirname(__file__))

# Import only the pure functions (no GPU/Docker deps)
from run_full_pipeline_linemod import (
    _parse_llm_response,
    _rotation_error_deg,
    _translation_error_cm,
    _nanmean,
    _compute_iou,
    LM_NAMES,
    LM_SYMMETRIC,
    LM_INSTRUCTIONS,
    ADD_THRESH_RATIO,
    MM_TO_M,
)


# ── _parse_llm_response ───────────────────────────────────────────────────────

class TestParseLlmResponse:

    def test_arrow_format(self):
        assert _parse_llm_response("→ rubber duck") == "rubber duck"

    def test_quoted_keyword(self):
        assert _parse_llm_response('The keyword is "yellow driller"') == "yellow driller"

    def test_plain_single_word(self):
        assert _parse_llm_response("driller") == "driller"

    def test_strips_prefix(self):
        assert _parse_llm_response("keyword: benchvise") == "benchvise"

    def test_multiline_takes_first_valid(self):
        # "keyword: " prefix is stripped → "rubber duck" is the first valid line
        raw = "keyword: rubber duck\nSome explanation after."
        assert _parse_llm_response(raw) == "rubber duck"

    def test_empty_string_returns_empty(self):
        assert _parse_llm_response("") == ""

    def test_too_many_words_skips_line(self):
        # 5 words → should skip to next valid line
        raw = "this has five words here\nduck"
        assert _parse_llm_response(raw) == "duck"

    def test_max_3_words_accepted(self):
        assert _parse_llm_response("orange glue bottle") == "orange glue bottle"


# ── _rotation_error_deg ───────────────────────────────────────────────────────

class TestRotationError:

    def test_identity_gives_zero(self):
        R = np.eye(3)
        assert _rotation_error_deg(R, R) == pytest.approx(0.0, abs=1e-5)

    def test_90_degree_rotation(self):
        R_pred = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        R_gt   = np.eye(3)
        assert _rotation_error_deg(R_pred, R_gt) == pytest.approx(90.0, abs=1e-4)

    def test_180_degree_rotation(self):
        R_pred = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        R_gt   = np.eye(3)
        assert _rotation_error_deg(R_pred, R_gt) == pytest.approx(180.0, abs=1e-4)

    def test_same_rotation_gives_zero(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        assert _rotation_error_deg(R, R) == pytest.approx(0.0, abs=1e-5)

    def test_returns_float(self):
        assert isinstance(_rotation_error_deg(np.eye(3), np.eye(3)), float)


# ── _translation_error_cm ─────────────────────────────────────────────────────

class TestTranslationError:

    def test_zero_error(self):
        t = np.array([0.1, 0.2, 0.5])
        assert _translation_error_cm(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_10cm_offset(self):
        t_pred = np.array([0.10, 0.0, 0.0])
        t_gt   = np.array([0.00, 0.0, 0.0])
        assert _translation_error_cm(t_pred, t_gt) == pytest.approx(10.0, abs=1e-4)

    def test_diagonal_offset(self):
        # sqrt(3) * 1cm = 1.732 cm
        t_pred = np.array([0.01, 0.01, 0.01])
        t_gt   = np.zeros(3)
        expected = np.sqrt(3) * 1.0
        assert _translation_error_cm(t_pred, t_gt) == pytest.approx(expected, abs=1e-4)

    def test_returns_float(self):
        assert isinstance(_translation_error_cm(np.zeros(3), np.zeros(3)), float)


# ── _nanmean ──────────────────────────────────────────────────────────────────

class TestNanmean:

    def test_all_valid(self):
        assert _nanmean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_with_nans(self):
        assert _nanmean([1.0, float('nan'), 3.0]) == pytest.approx(2.0)

    def test_all_nan_returns_nan(self):
        assert np.isnan(_nanmean([float('nan'), float('nan')]))

    def test_empty_returns_nan(self):
        assert np.isnan(_nanmean([]))

    def test_single_value(self):
        assert _nanmean([5.0]) == pytest.approx(5.0)


# ── _compute_iou ──────────────────────────────────────────────────────────────

class TestComputeIoU:

    def test_perfect_overlap(self):
        mask = np.ones((10, 10), dtype=bool)
        assert _compute_iou(mask, mask) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:5, :] = True
        b[5:, :] = True
        assert _compute_iou(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_half_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:, :5]  = True   # left half
        b[:, 2:7] = True   # shifted
        iou = _compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_none_gt_returns_minus_one(self):
        mask = np.ones((10, 10), dtype=bool)
        assert _compute_iou(mask, None) == -1.0


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


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', __file__, '-v', '--tb=short'],
        capture_output=False)
    sys.exit(result.returncode)
