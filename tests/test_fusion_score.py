"""
tests/test_fusion_score.py
===========================
Unit tests cho TASK-TST-01: Reranker & Fusion Score (viranker, jina_reranker_m0, fusion_score).
"""

from __future__ import annotations

import pytest
from serving.api.schema.retrieval import CandidateResult
from serving.api.service.reranker.skeletons.text_branch import viranker
from serving.api.service.reranker.skeletons.crossmodal_branch import jina_reranker_m0
from serving.api.service.reranker.skeletons import fusion_score


@pytest.fixture
def sample_candidates():
    return [
        CandidateResult(
            video_id="V001",
            scene_id="V001_S001",
            frame_id=10,
            timestamp_start=0.0,
            timestamp_end=5.0,
            timestamp_mid=2.5,
            keyframe_path="V001/V001_S001.jpg",
            ocr_text="Cửa hàng tạp hóa",
            asr_text="người đàn ông đi vào cửa hàng",
            score=0.8,
        ),
        CandidateResult(
            video_id="V001",
            scene_id="V001_S002",
            frame_id=20,
            timestamp_start=5.0,
            timestamp_end=10.0,
            timestamp_mid=7.5,
            keyframe_path="V001/V001_S002.jpg",
            ocr_text="Khuyến mãi lớn",
            asr_text="mua sản phẩm giảm giá",
            score=0.6,
        ),
        CandidateResult(
            video_id="V002",
            scene_id="V002_S001",
            frame_id=5,
            timestamp_start=0.0,
            timestamp_end=3.0,
            timestamp_mid=1.5,
            keyframe_path="V002/V002_S001.jpg",
            ocr_text="",
            asr_text="xe máy chạy trên đường",
            score=0.9,
        ),
    ]


def test_viranker_score(sample_candidates):
    scores = viranker.score("cửa hàng tạp hóa", sample_candidates)
    assert len(scores) == len(sample_candidates)
    assert all(isinstance(s, float) for s in scores)


def test_jina_reranker_m0_score(sample_candidates):
    scores = jina_reranker_m0.score("a man entering store", sample_candidates, keyframes_root_dir=".")
    assert len(scores) == len(sample_candidates)
    assert all(isinstance(s, float) for s in scores)


def test_fusion_score_linear(sample_candidates):
    text_scores = [0.9, 0.4, 0.2]
    crossmodal_scores = [0.5, 0.8, 0.1]

    fused = fusion_score.fuse(text_scores, crossmodal_scores, sample_candidates, alpha=0.6)
    assert len(fused) == len(sample_candidates)
    # Verify scores are sorted descending
    for i in range(len(fused) - 1):
        assert fused[i].score >= fused[i + 1].score


def test_fusion_score_empty():
    assert fusion_score.fuse([], [], []) == []


def test_viranker_empty():
    assert viranker.score("query", []) == []


def test_jina_reranker_empty():
    assert jina_reranker_m0.score("query", []) == []
