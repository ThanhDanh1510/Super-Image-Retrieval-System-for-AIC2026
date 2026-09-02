"""
tests/test_query_analyzer.py
============================
Unit tests cho QueryAnalyzer (nhận diện ý đồ OCR, ASR, Visual và sinh trọng số động).
"""

from serving.api.service.retrieval.query_analyzer import QueryAnalyzer


def test_ocr_intent_detection():
    # Ngoặc kép hoặc từ khóa chỉ chữ viết
    q1 = 'người đứng cạnh biển hiệu "Highland Coffee"'
    res1 = QueryAnalyzer.analyze(q1)
    assert res1.has_ocr_intent is True
    assert "Highland Coffee" in res1.extracted_ocr_terms
    assert res1.primary_intent == "ocr"
    assert res1.custom_weights["ocr_dense"] >= 2.5

    q2 = "chiếc xe tải có in chữ Cứu Hộ Giao Thông"
    res2 = QueryAnalyzer.analyze(q2)
    assert res2.has_ocr_intent is True
    assert res2.primary_intent == "ocr"


def test_asr_intent_detection():
    # Lời nói, phát biểu, giới thiệu
    q = "người dẫn chương trình nói lời chào khán giả"
    res = QueryAnalyzer.analyze(q)
    assert res.has_asr_intent is True
    assert res.primary_intent == "asr"
    assert res.custom_weights["asr_dense"] >= 2.0


def test_visual_intent_detection():
    # Thuần thị giác / góc quay
    q = "cận cảnh con lân đang nhảy múa trên cột"
    res = QueryAnalyzer.analyze(q)
    assert res.has_visual_intent is True
    assert res.primary_intent == "visual"
    assert res.custom_weights["image_dense"] >= 2.5
    assert res.custom_weights["ocr_dense"] <= 0.5


def test_empty_or_balanced_intent():
    q = "bầu trời xanh"
    res = QueryAnalyzer.analyze(q)
    assert "image_dense" in res.custom_weights
    assert "description_dense" in res.custom_weights
