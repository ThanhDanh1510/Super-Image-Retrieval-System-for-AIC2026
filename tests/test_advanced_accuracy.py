"""
tests/test_advanced_accuracy.py
================================
Unit tests for Advanced Accuracy Optimization:
1. Fuzzy Unaccented Matching & Levenshtein Edit Distance (OCR / ASR).
2. Intent Analysis & Dynamic Adaptive Fusion Weights (QueryAnalyzer).
3. Entity & Quantity / Unit Extraction (300g, 7 cái, 15‰, 59F1).
"""

import pytest
from serving.api.service.retrieval.hybrid_search import (
    remove_accents,
    compact_text,
    levenshtein_dist,
    match_entity_fuzzy,
    extract_entity_keywords,
)
from serving.api.service.retrieval.query_analyzer import QueryAnalyzer


def test_remove_accents_and_compact_text():
    """Kiểm tra loại bỏ dấu tiếng Việt và chuẩn hóa compact chuỗi."""
    assert remove_accents("Cá sòng nướng") == "ca song nuong"
    assert remove_accents("Nón lá Việt Nam") == "non la viet nam"
    assert remove_accents("Đường Lý Thường Kiệt") == "duong ly thuong kiet"

    assert compact_text("59-F1 123.45") == "59f112345"
    assert compact_text("51A - 9999") == "51a9999"


def test_levenshtein_dist():
    """Kiểm tra tính khoảng cách Levenshtein giữa các từ."""
    assert levenshtein_dist("song", "song") == 0
    assert levenshtein_dist("song", "sog") == 1
    assert levenshtein_dist("ngheu", "nghe") == 1
    assert levenshtein_dist("59f1", "59fi") == 1


def test_match_entity_fuzzy_plate_and_vietnamese():
    """Kiểm tra so khớp mờ thực thể với OCR và ASR text."""
    # 1. Biển số xe có dấu gạch nối và khoảng trắng
    assert match_entity_fuzzy("59F1", "Xe máy biển kiểm soát 59-F1 9999") is True
    assert match_entity_fuzzy("59-F1-12345", "59F1 12345") is True

    # 2. Tiếng Việt có dấu vs không dấu
    assert match_entity_fuzzy("cá sòng", "Hôm nay tôi mua ca song rất tươi") is True
    assert match_entity_fuzzy("nghêu", "Món canh NGHEU hap sa") is True
    assert match_entity_fuzzy("nón lá", "Người phụ nữ đội non la đi chợ") is True

    # 3. Đơn vị đo lường và số lượng
    assert match_entity_fuzzy("300g", "Gói bột 300g đóng hộp") is True
    assert match_entity_fuzzy("1204", "Phòng số 1204 khách sạn") is True


def test_extract_entity_keywords_quantities():
    """Kiểm tra trích xuất thực thể biển số, số lượng, đơn vị đo lường."""
    q1 = "Người đàn ông đi xe máy 59F1-12345 cầm gói quà 300g"
    ents1 = extract_entity_keywords(q1)
    assert any("59f1" in e for e in ents1)
    assert "300g" in ents1

    q2 = "Có bao nhiêu cái bánh trên đĩa 'bánh chưng'?"
    ents2 = extract_entity_keywords(q2)
    assert "bánh chưng" in ents2

    q3 = "Độ mặn của nước là 15‰ đến 20‰"
    ents3 = extract_entity_keywords(q3)
    assert any("15‰" in e or "20‰" in e for e in ents3)


def test_query_analyzer_adaptive_fusion_weights():
    """Kiểm tra phân tích ý đồ và sinh trọng số thích ứng theo Intent."""
    # OCR Intent
    intent_ocr = QueryAnalyzer.analyze("Tìm biển số xe 59F1 trên đường")
    assert intent_ocr.has_ocr_intent is True
    assert intent_ocr.primary_intent == "ocr"
    assert intent_ocr.custom_weights.get("ocr_dense", 0.0) >= 3.0

    # ASR Intent
    intent_asr = QueryAnalyzer.analyze("Người phụ nữ phát biểu và nói rằng xin chào")
    assert intent_asr.has_asr_intent is True
    assert intent_asr.primary_intent == "asr"
    assert intent_asr.custom_weights.get("asr_dense", 0.0) >= 2.5

    # QA Intent
    intent_qa = QueryAnalyzer.analyze("Có bao nhiêu người trong phòng?", task_type="qa")
    assert intent_qa.has_qa_intent is True
    assert intent_qa.primary_intent == "qa"
    assert intent_qa.custom_weights.get("description_dense", 0.0) >= 2.5

    # Visual Intent
    intent_vis = QueryAnalyzer.analyze("Toàn cảnh hoàng hôn trên bãi biển từ trên cao flycam")
    assert intent_vis.has_visual_intent is True
    assert intent_vis.primary_intent == "visual"
    assert intent_vis.custom_weights.get("image_dense", 0.0) >= 2.8
