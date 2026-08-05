"""
tests/test_edge_cases.py
========================
Automated edge-case test suite for FastAPI serving layer.
Sends requests to live FastAPI server (or via TestClient with lifespan).
"""

import httpx
import pytest

BASE_URL = "http://localhost:8000"


# =============================================================================
# 1. Healthcheck Test
# =============================================================================

def test_healthcheck():
    res = httpx.get(f"{BASE_URL}/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


# =============================================================================
# 2. Text KIS Query Edge Cases
# =============================================================================

def test_kis_empty_query():
    """Empty query should fail validation with 422 Unprocessable Entity."""
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json={"query": "", "top_k": 10})
    assert res.status_code == 422


def test_kis_extremely_long_query():
    """Query exceeding 2000 characters should trigger validation error 422."""
    long_text = "mèo " * 600  # > 2400 chars
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json={"query": long_text, "top_k": 10})
    assert res.status_code == 422


def test_kis_emojis_and_unicode():
    """Emojis and non-BMP characters should be processed without crashing."""
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json={"query": "🏎️🔥 Cảnh xe đua 🎉✨", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert "results" in data


def test_kis_injection_payload():
    """SQL/HTML injection strings should be treated as raw text without errors."""
    payload = "<script>alert(1)</script>' OR '1'='1"
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json={"query": payload, "top_k": 3})
    assert res.status_code == 200
    assert "results" in res.json()


def test_kis_pure_numbers_symbols():
    """Numbers and math symbols should not break text embedding."""
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json={"query": "123456789 +-*/ = 0", "top_k": 2})
    assert res.status_code == 200


# =============================================================================
# 3. Top-K Parameter Boundary Tests
# =============================================================================

def test_top_k_zero():
    """top_k = 0 violates ge=1 constraint."""
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json={"query": "con mèo", "top_k": 0})
    assert res.status_code == 422


def test_top_k_negative():
    """Negative top_k violates ge=1 constraint."""
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json={"query": "con mèo", "top_k": -10})
    assert res.status_code == 422


def test_top_k_exceeds_maximum():
    """top_k > 200 violates le=200 constraint."""
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json={"query": "con mèo", "top_k": 250})
    assert res.status_code == 422


def test_top_k_minimum_valid():
    """top_k = 1 should return exactly 1 candidate."""
    res = httpx.post(f"{BASE_URL}/api/v1/kis", json={"query": "con mèo", "top_k": 1})
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 1


# =============================================================================
# 4. Similar Image Search Edge Cases (/similar-image)
# =============================================================================

def test_similar_image_unsupported_format():
    """Uploading text/pdf file should be rejected with 400 Bad Request."""
    files = {"file": ("test.txt", b"Hello world text content", "text/plain")}
    res = httpx.post(f"{BASE_URL}/api/v1/similar-image?top_k=5", files=files)
    assert res.status_code == 400
    assert "Định dạng file không được hỗ trợ" in res.json()["detail"]


def test_similar_image_exceeds_max_size():
    """Uploading file > 10MB should be rejected with 400 Bad Request."""
    large_dummy_bytes = b"0" * (11 * 1024 * 1024)  # 11MB
    files = {"file": ("large_image.jpg", large_dummy_bytes, "image/jpeg")}
    res = httpx.post(f"{BASE_URL}/api/v1/similar-image?top_k=5", files=files)
    assert res.status_code == 400
    assert "giới hạn cho phép" in res.json()["detail"]


def test_similar_image_corrupt_bytes():
    """Uploading invalid image bytes with image/jpeg header should return error."""
    files = {"file": ("corrupt.jpg", b"NOT_AN_IMAGE_BYTE_DATA", "image/jpeg")}
    res = httpx.post(f"{BASE_URL}/api/v1/similar-image?top_k=5", files=files)
    assert res.status_code in (400, 500)


def test_similar_image_valid_png():
    """Uploading valid minimal PNG image should succeed."""
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00"
        b"\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd"
        b"\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    files = {"file": ("sample.png", png_bytes, "image/png")}
    res = httpx.post(f"{BASE_URL}/api/v1/similar-image?top_k=3", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) <= 3
