"""
tests/test_search_algorithms_and_device.py
============================================
Unit tests cho:
1. Cross-platform device resolution (CUDA, Apple Silicon MPS, CPU fallback).
2. Strict tiered scoring & exact phrase matching in qdrant_repo.
3. Temporal NMS / Slide diversity filtering in retrieval_pipeline.
"""

import pytest
from unittest.mock import patch
import torch

from shared.utils.device_utils import get_optimal_device, resolve_torch_device
from serving.api.repo.qdrant_repo import _calculate_match_score, _normalize_text_for_scoring
from serving.api.service.retrieval.retrieval_pipeline import _apply_temporal_diversity_filter
from serving.api.schema.retrieval import CandidateResult


# =============================================================================
# 1. Cross-Platform Device Resolution Tests
# =============================================================================

def test_resolve_torch_device_cpu():
    assert resolve_torch_device("cpu") == "cpu"
    assert resolve_torch_device("") == "cpu"
    assert resolve_torch_device(None) == "cpu"


def test_resolve_torch_device_mps_fallback_when_unavailable():
    with patch.object(torch.backends, "mps", create=True) as mock_mps:
        mock_mps.is_available.return_value = False
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert resolve_torch_device("mps") == "cpu"


def test_get_optimal_device_mps():
    with patch.object(torch.cuda, "is_available", return_value=False):
        with patch.object(torch.backends, "mps", create=True) as mock_mps:
            mock_mps.is_available.return_value = True
            assert get_optimal_device() == "mps"


def test_get_optimal_device_cpu():
    with patch.object(torch.cuda, "is_available", return_value=False):
        with patch.object(torch.backends, "mps", create=True) as mock_mps:
            mock_mps.is_available.return_value = False
            assert get_optimal_device() == "cpu"


# =============================================================================
# 2. Exact Phrase Tiered Scoring Tests
# =============================================================================

def test_exact_full_phrase_match_scores_dominant():
    query = "PHÂN BỐ DÂN CƯ CHƯA HỢP LÝ"
    doc_exact = "II. BÀI GIẢNG: PHÂN BỐ DÂN CƯ CHƯA HỢP LÝ & ĐÔ THỊ HÓA TRONG NƯỚC"
    doc_partial = "I. ĐẶC ĐIỂM: PHÂN BỐ DÂN CƯ KHÔNG ĐỀU GIỮA CÁC VÙNG"

    score_exact = _calculate_match_score(query, doc_exact)
    score_partial = _calculate_match_score(query, doc_partial)

    # Exact full phrase match must get dominant tiered score >= 2.0
    assert score_exact >= 2.0
    # Partial match with missing word ('chua', 'hop', 'ly') must be penalized
    assert score_partial < 1.0
    assert score_exact > 2 * score_partial


def test_math_coordinates_exact_match_score():
    query = "cho điểm A(4; -2; 2)"
    doc_exact = "Câu 4: Trong không gian Oxyz, cho điểm A(4; -2; 2) và mặt cầu (S)..."
    doc_other = "Câu 1: Trong không gian Oxyz, cho mặt phẳng (P) đi qua điểm A(1; 2; 3)..."

    score_exact = _calculate_match_score(query, doc_exact)
    score_other = _calculate_match_score(query, doc_other)

    assert score_exact >= 2.0
    assert score_other <= 1.0
    assert score_exact > 2 * score_other


# =============================================================================
# 3. Temporal NMS / Slide Diversity Filter Tests
# =============================================================================

def _mk_candidate(vid: str, fid: int, t_mid: float, score: float) -> CandidateResult:
    return CandidateResult(
        video_id=vid,
        scene_id=f"{vid}_S{fid:03d}",
        frame_id=fid,
        timestamp_start=t_mid - 1.0,
        timestamp_end=t_mid + 1.0,
        timestamp_mid=t_mid,
        keyframe_path=f"{vid}/{fid}.webp",
        ocr_text="Sample Slide OCR",
        asr_text="",
        score=score,
    )


def test_temporal_diversity_filter_suppresses_duplicate_slide_bursts():
    # 5 consecutive frames from the same video within 4 seconds (same slide)
    burst = [
        _mk_candidate("L25_V060", 100, 10.0, 0.95),
        _mk_candidate("L25_V060", 101, 11.0, 0.94),
        _mk_candidate("L25_V060", 102, 12.0, 0.93),
        _mk_candidate("L25_V060", 103, 13.0, 0.92),
        _mk_candidate("L25_V060", 104, 14.0, 0.91),
        # Another frame 60 seconds later (different slide/exercise)
        _mk_candidate("L25_V060", 200, 75.0, 0.90),
        # Frame from another video
        _mk_candidate("L25_V017", 300, 10.0, 0.89),
    ]

    filtered = _apply_temporal_diversity_filter(burst, max_frames_per_cluster=2, time_window_sec=6.0)

    # In top 4 results:
    # 2 top burst frames (100, 101) + frame 200 (75s) + frame 300 (L25_V017)
    top_4_ids = [(c.video_id, c.frame_id) for c in filtered[:4]]
    assert ("L25_V060", 100) in top_4_ids
    assert ("L25_V060", 101) in top_4_ids
    assert ("L25_V060", 200) in top_4_ids
    assert ("L25_V017", 300) in top_4_ids

    # Overflow duplicate frames (102, 103, 104) are pushed to the end
    tail_3_fids = [c.frame_id for c in filtered[4:]]
    assert tail_3_fids == [102, 103, 104]
