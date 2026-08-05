"""
tests/test_search_services.py
==============================
Systematic automated test suite for all retrieval search services:
- KIS (Known-Item Search)
- VQA (Visual Question Answering)
- TRAKE (Temporal Retrieval / Multi-event alignment)
- Similar Image Search
"""

import httpx
import pytest

BASE_URL = "http://localhost:8000"


# =============================================================================
# 1. KIS (Known-Item Search) Endpoint Tests
# =============================================================================

def test_kis_text_search():
    """Test standard KIS text query retrieval."""
    payload = {
        "query": "người phụ nữ đi xe máy trên đường phố",
        "top_k": 5
    }
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json=payload, timeout=120.0)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert "total" in data
    assert data["total"] <= 5
    assert len(data["results"]) > 0
    first_res = data["results"][0]
    assert "video_id" in first_res
    assert "scene_id" in first_res
    assert "frame_id" in first_res
    assert "score" in first_res


def test_kis_user_choices_prioritization():
    """Test user manual frame selection override in KIS."""
    user_choice = {
        "video_id": "CUSTOM_VID_001",
        "frame_id": 999,
        "scene_id": "CUSTOM_VID_001_S001",
        "timestamp_start": 10.0,
        "timestamp_end": 15.0,
        "timestamp_mid": 12.5,
        "keyframe_path": "CUSTOM_VID_001/frame_0999.webp",
        "ocr_text": "Sample OCR",
        "asr_text": "Sample ASR",
    }
    payload = {
        "query": "phong cảnh thiên nhiên núi rừng",
        "top_k": 5,
        "user_choices": [user_choice]
    }
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json=payload, timeout=120.0)
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) > 0
    # First result must be the user choice override
    first = data["results"][0]
    assert first["video_id"] == "CUSTOM_VID_001"
    assert first["frame_id"] == 999
    assert first["score"] == 1.0


# =============================================================================
# 2. VQA (Visual Question Answering) Endpoint Tests
# =============================================================================

def test_vqa_search():
    """Test Visual Question Answering retrieval service."""
    payload = {
        "query": "cảnh gian hàng bán trái cây trong chợ",
        "question": "Có những loại quả nào trên bàn?",
        "top_k": 5
    }
    res = httpx.post(f"{BASE_URL}/api/v1/vqa", json=payload, timeout=120.0)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert "answers" in data
    assert len(data["results"]) == len(data["answers"])
    assert data["total"] <= 5


def test_vqa_user_choices_with_answer():
    """Test VQA manual selection carrying user-typed answer."""
    user_choice = {
        "video_id": "L10_V005",
        "frame_id": 120,
        "scene_id": "L10_V005_S002",
        "keyframe_path": "L10_V005/L10_V005_S002.webp",
        "answer": "Quả dưa hấu và quả chuối"
    }
    payload = {
        "query": "cảnh mua bán trái cây",
        "question": "Loại quả gì?",
        "top_k": 3,
        "user_choices": [user_choice]
    }
    res = httpx.post(f"{BASE_URL}/api/v1/vqa", json=payload, timeout=120.0)
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) > 0
    assert data["results"][0]["video_id"] == "L10_V005"
    assert data["answers"][0] == "Quả dưa hấu và quả chuối"


# =============================================================================
# 3. TRAKE (Temporal Retrieval / Multi-event) Endpoint Tests
# =============================================================================

def test_trake_temporal_search():
    """Test Temporal Retrieval multi-event chain matching."""
    payload = {
        "query": "Đầu tiên người đàn ông mở cửa đi vào nhà. Sau đó ông ta ngồi xuống ghế sofa mở tivi.",
        "num_events": 2,
        "top_k": 3
    }
    res = httpx.post(f"{BASE_URL}/api/v1/trake", json=payload, timeout=30.0)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert "num_events" in data
    assert data["num_events"] == 2
    # Each item in results should be a list of N CandidateResult items from the SAME video
    for chain in data["results"]:
        assert isinstance(chain, list)
        assert len(chain) == 2
        # CRITICAL CONSTRAINTS: All candidates in chain must belong to the exact same video_id
        video_ids = {cand["video_id"] for cand in chain}
        assert len(video_ids) == 1, f"TRAKE chain contains candidates from multiple videos: {video_ids}"
        # CRITICAL CONSTRAINTS: Timestamps within a chain must be strictly increasing (NO exact duplicates)
        timestamps = [round(cand["timestamp_mid"], 2) for cand in chain]
        assert len(set(timestamps)) == len(timestamps), f"TRAKE chain contains duplicate exact timestamps: {timestamps}"
        assert timestamps == sorted(timestamps), f"TRAKE chain timestamps are not sequential: {timestamps}"


# =============================================================================
# 4. Similar Image Search Endpoint Tests
# =============================================================================

def test_similar_image_search():
    """Test Image-to-Image visual similarity search."""
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00"
        b"\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd"
        b"\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    files = {"file": ("query_image.png", png_bytes, "image/png")}
    res = httpx.post(f"{BASE_URL}/api/v1/similar-image?top_k=5", files=files, timeout=30.0)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert "total" in data
    assert data["total"] <= 5
    assert len(data["results"]) > 0
