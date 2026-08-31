import pytest
import numpy as np

from serving.api.schema.retrieval import CandidateResult
from serving.api.service.retrieval.hybrid_search import (
    extract_entity_keywords,
    extract_sparse_tokens,
)
from serving.api.service.retrieval.text_query import extract_negative_keywords
from serving.api.service.reranker.skeletons.text_branch.viranker import (
    _slice_context_around_keywords,
    _build_doc_text,
    _extract_keywords,
)
from serving.api.service.retrieval.retrieval_pipeline import _apply_temporal_scene_smoothing


def _create_cand(video_id: str, frame_id: int, timestamp_mid: float, score: float, ocr: str = "", asr: str = "", desc: str = "") -> CandidateResult:
    return CandidateResult(
        video_id=video_id,
        frame_id=frame_id,
        scene_id=f"{video_id}_{frame_id}",
        timestamp_start=timestamp_mid,
        timestamp_end=timestamp_mid,
        timestamp_mid=timestamp_mid,
        keyframe_path=f"data/keyframes/{video_id}_{frame_id}.jpg",
        ocr_text=ocr,
        asr_text=asr,
        description=desc,
        score=score,
    )


def test_extract_entity_keywords():
    query_1 = 'Tìm xe ô tô có biển số "59F1-12345" tại ngã tư'
    entities_1 = extract_entity_keywords(query_1)
    assert any("59f1-12345" in e or "12345" in e for e in entities_1)
    assert any("59f1-12345" in e for e in entities_1)

    query_2 = 'Cảnh quay mã số L21_V005 có người mặc áo "cảnh sát giao thông"'
    entities_2 = extract_entity_keywords(query_2)
    assert "cảnh sát giao thông" in entities_2

    query_3 = 'Người đàn ông cầm biển số 51A 99999'
    entities_3 = extract_entity_keywords(query_3)
    assert any("51a 99999" in e or "99999" in e for e in entities_3)


def test_extract_negative_keywords():
    q1 = "Người đi xe máy màu đỏ không đội mũ bảo hiểm"
    neg1 = extract_negative_keywords(q1)
    assert len(neg1) > 0
    assert any("mũ bảo hiểm" in n or "đội mũ" in n for n in neg1)

    q2 = "Cảnh ngoài trời không có xe hơi và chưa có người đi bộ"
    neg2 = extract_negative_keywords(q2)
    assert len(neg2) >= 1
    assert any("xe hơi" in n for n in neg2)

    q3 = "Đoạn đường đông đúc"
    neg3 = extract_negative_keywords(q3)
    assert len(neg3) == 0


def test_viranker_context_slicing():
    long_desc = " ".join([f"word{i}" for i in range(100)]) + " biển số 59F1 xuất hiện ở góc trái " + " ".join([f"tail{i}" for i in range(100)])
    keywords = {"59f1", "biển"}
    sliced = _slice_context_around_keywords(long_desc, keywords, window_words=30)
    assert "59f1" in sliced.lower() or "biển" in sliced.lower()
    assert len(sliced.split()) <= 35

    # Test candidate build doc text
    cand = _create_cand("L01_V001", 100, 10.0, 0.8, ocr="Biển số 59F1", desc=long_desc)
    doc = _build_doc_text(cand, keywords=keywords)
    assert "OCR:" in doc
    assert "59F1" in doc


def test_temporal_scene_smoothing():
    # Video 1: 3 frames gần nhau trong cùng phân cảnh (t=10s, 12s, 14s) có điểm tốt -> được boost cộng hưởng
    c1 = _create_cand("L01_V001", 1, 10.0, 0.70)
    c2 = _create_cand("L01_V001", 2, 12.0, 0.75)
    c3 = _create_cand("L01_V001", 3, 14.0, 0.72)

    # Video 2: 1 frame đơn độc cách xa hàng trăm giây (nhiễu đơn lẻ)
    c4 = _create_cand("L02_V002", 10, 500.0, 0.76)

    candidates = [c4, c2, c3, c1]  # c4 ban đầu đứng đầu với 0.76
    smoothed = _apply_temporal_scene_smoothing(candidates, sigma_sec=8.0, weight=0.20)

    # Sau khi làm mịn theo ngữ cảnh, cụm c2 (được cộng hưởng từ c1 và c3) sẽ vượt lên trên c4
    res_c2 = next(c for c in smoothed if c.video_id == "L01_V001" and c.frame_id == 2)
    res_c4 = next(c for c in smoothed if c.video_id == "L02_V002")

    assert res_c2.score > c2.score  # được tăng điểm cộng hưởng
    assert res_c2.score > res_c4.score  # vượt lên trên frame nhiễu đơn lẻ
    assert smoothed[0].video_id == "L01_V001"


def test_negation_filter_penalization():
    cand_match_forbidden = _create_cand("L01_V001", 1, 10.0, 0.90, desc="Người đi xe máy có đội mũ bảo hiểm màu xanh")
    cand_valid = _create_cand("L01_V001", 2, 12.0, 0.85, desc="Người đi xe máy tóc dài không đội gì cả")

    query = "Người đi xe máy không đội mũ bảo hiểm"
    neg_terms = extract_negative_keywords(query)
    assert len(neg_terms) > 0

    cands = [cand_match_forbidden, cand_valid]
    penalty = 0.50
    for c in cands:
        doc_lower = f"{c.ocr_text} {c.asr_text} {c.description}".lower()
        if any(nt in doc_lower for nt in neg_terms):
            c.score = round(c.score * penalty, 4)

    # cand_match_forbidden bị phạt từ 0.90 -> 0.45, cand_valid giữ nguyên 0.85
    assert cand_match_forbidden.score == 0.45
    assert cand_valid.score == 0.85
    assert cand_valid.score > cand_match_forbidden.score
