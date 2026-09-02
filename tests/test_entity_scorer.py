"""
tests/test_entity_scorer.py
===========================
Unit tests for Generic Multi-Entity Coverage Scorer.
"""

import pytest
from serving.api.schema.retrieval import CandidateResult
from serving.api.service.reranker.skeletons.text_branch.entity_scorer import (
    extract_entity_clauses,
    compute_entity_coverage_scores,
)


def test_extract_entity_clauses_basic():
    query_vi = "Cảnh quay liệt kê các nguyên liệu để nấu một món ăn. Ảnh nền bao gồm dĩa thịt, bó lá tươi xanh, gói hạt nêm, hũ nước cốt dừa, bột cà ri, nấm mèo khô, sả cây và ớt hiểm đỏ."
    query_en = "The scene lists ingredients. Background includes plate of meat, green leaves, seasoning packet, coconut milk, curry powder, dried wood ear mushrooms, lemongrass and red chili."
    
    clauses = extract_entity_clauses(query_vi, query_en)
    assert len(clauses) >= 5
    clauses_str = " ".join(clauses)
    assert "thịt" in clauses_str or "meat" in clauses_str
    assert "lá tươi" in clauses_str or "leaves" in clauses_str
    assert "sả" in clauses_str or "lemongrass" in clauses_str


def _make_candidate(vid: str, fid: int, desc: str, ocr: str = "") -> CandidateResult:
    return CandidateResult(
        video_id=vid,
        frame_id=fid,
        scene_id=f"{vid}_S1",
        timestamp_start=0.0,
        timestamp_end=5.0,
        timestamp_mid=2.5,
        keyframe_path=f"data/{vid}/{fid}.jpg",
        score=0.9,
        description=desc,
        ocr_text=ocr,
    )


def test_compute_entity_coverage_scores_multi_entity():
    query_vi = "Dĩa thịt, bó lá tươi xanh, một gói hạt nêm, hũ nước cốt dừa, bột cà ri, nấm mèo, sả cây và ớt hiểm đỏ."
    query_en = "Plate of meat, fresh green leaves, seasoning packet, coconut milk, curry powder, wood ear mushrooms, lemongrass, red chili."

    cand_full = _make_candidate(
        vid="L26_V439",
        fid=1383,
        desc="- ACTION: None - OBJECTS: mushrooms, milk, Ajinomoto seasoning packet, ground spices, green leaves, red chili peppers, lemongrass - SCENE: Blue wooden surface",
        ocr="AJi-ngon Giá là chính hiệu",
    )

    cand_partial = _make_candidate(
        vid="L26_V034",
        fid=4021,
        desc="- ACTION: static - OBJECTS: red chili peppers, lemongrass stalks, white plate",
        ocr="",
    )

    scores = compute_entity_coverage_scores([cand_full, cand_partial], query_vi=query_vi, query_en=query_en)
    assert len(scores) == 2
    # cand_full must have significantly higher entity coverage than cand_partial
    assert scores[0] > scores[1]
    assert scores[0] >= 0.70
    assert scores[1] <= 0.35


def test_compute_entity_coverage_scores_short_query():
    # Query ngắn (< 3 thực thể) không bị thiên lệch
    query_vi = "Người phụ nữ áo đỏ"
    query_en = "A woman in red dress"
    
    cands = [
        _make_candidate(vid="V1", fid=10, desc="woman in red"),
        _make_candidate(vid="V2", fid=20, desc="car on street"),
    ]
    
    scores = compute_entity_coverage_scores(cands, query_vi=query_vi, query_en=query_en)
    assert scores == [1.0, 1.0]
