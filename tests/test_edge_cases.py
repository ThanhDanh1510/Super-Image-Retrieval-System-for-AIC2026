"""
tests/test_edge_cases.py
========================
Automated edge-case test suite for FastAPI serving layer.
Sends requests via TestClient (or to live server if active).
"""

import httpx
import pytest
from starlette.testclient import TestClient
from serving.api.main import app


@pytest.fixture(scope="module")
def client():
    """Tự động sử dụng Live Server nếu đang chạy, hoặc TestClient in-process."""
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=0.5)
        if r.status_code == 200:
            c = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)
            yield c
            c.close()
            return
    except Exception:
        pass
    with TestClient(app) as tc:
        yield tc


# =============================================================================
# 1. Healthcheck Test
# =============================================================================

def test_healthcheck(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


# =============================================================================
# 2. Text KIS Query Edge Cases
# =============================================================================

def test_kis_empty_query(client):
    """Empty query should fail validation with 422 Unprocessable Entity."""
    res = client.post("/api/v1/kis", json={"query": "", "top_k": 10})
    assert res.status_code == 422


def test_kis_extremely_long_query(client):
    """Query exceeding 2000 characters should trigger validation error 422."""
    long_text = "mèo " * 600  # > 2400 chars
    res = client.post("/api/v1/kis", json={"query": long_text, "top_k": 10})
    assert res.status_code == 422


def test_kis_emojis_and_unicode(client):
    """Emojis and non-BMP characters should be processed without crashing."""
    res = client.post("/api/v1/kis", json={"query": "🏎️🔥 Cảnh xe đua 🎉✨", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert "results" in data


def test_kis_injection_payload(client):
    """SQL/HTML injection strings should be treated as raw text without errors."""
    payload = "<script>alert(1)</script>' OR '1'='1"
    res = client.post("/api/v1/kis", json={"query": payload, "top_k": 3})
    assert res.status_code == 200
    assert "results" in res.json()


def test_kis_pure_numbers_symbols(client):
    """Numbers and math symbols should not break text embedding."""
    res = client.post("/api/v1/kis", json={"query": "123456789 +-*/ = 0", "top_k": 2})
    assert res.status_code == 200


# =============================================================================
# 3. Top-K Parameter Boundary Tests
# =============================================================================

def test_top_k_zero(client):
    """top_k = 0 violates ge=1 constraint."""
    res = client.post("/api/v1/kis", json={"query": "con mèo", "top_k": 0})
    assert res.status_code == 422


def test_top_k_negative(client):
    """Negative top_k violates ge=1 constraint."""
    res = client.post("/api/v1/kis", json={"query": "con mèo", "top_k": -10})
    assert res.status_code == 422


def test_top_k_exceeds_maximum(client):
    """top_k > 500 violates le=500 constraint."""
    res = client.post("/api/v1/kis", json={"query": "con mèo", "top_k": 600})
    assert res.status_code == 422


def test_top_k_minimum_valid(client):
    """top_k = 1 should return exactly 1 candidate."""
    res = client.post("/api/v1/kis", json={"query": "con mèo", "top_k": 1})
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 1


# =============================================================================
# 4. Similar Image Search Edge Cases (/similar-image)
# =============================================================================

def test_similar_image_unsupported_format(client):
    """Uploading text/pdf file should be rejected with 400 Bad Request."""
    files = {"file": ("test.txt", b"Hello world text content", "text/plain")}
    res = client.post("/api/v1/similar-image?top_k=5", files=files)
    assert res.status_code == 400
    assert "Định dạng file không được hỗ trợ" in res.json()["detail"]


def test_similar_image_exceeds_max_size(client):
    """Uploading file > 15MB should be rejected with 400 Bad Request."""
    large_dummy_bytes = b"0" * (16 * 1024 * 1024)  # 16MB
    files = {"file": ("large_image.jpg", large_dummy_bytes, "image/jpeg")}
    res = client.post("/api/v1/similar-image?top_k=5", files=files)
    assert res.status_code == 400
    assert "giới hạn cho phép" in res.json()["detail"]


def test_similar_image_corrupt_bytes(client):
    """Uploading invalid image bytes with image/jpeg header should return error."""
    files = {"file": ("corrupt.jpg", b"NOT_AN_IMAGE_BYTE_DATA", "image/jpeg")}
    res = client.post("/api/v1/similar-image?top_k=5", files=files)
    assert res.status_code in (400, 500)


def test_similar_image_valid_png(client):
    """Uploading valid minimal PNG image should succeed."""
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00"
        b"\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd"
        b"\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    files = {"file": ("sample.png", png_bytes, "image/png")}
    res = client.post("/api/v1/similar-image?top_k=3", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) <= 3
