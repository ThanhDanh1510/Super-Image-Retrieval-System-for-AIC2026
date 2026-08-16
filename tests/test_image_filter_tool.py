"""
tests/test_image_filter_tool.py
===============================
Unit tests cho Hệ thống Lọc Keyframe & Duplicate Search.
"""

import json
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import pytest

from tools.image_filter.server import app, state


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_get_batches(client):
    res = client.get("/api/batches")
    assert res.status_code == 200
    data = res.json()
    assert "batches" in data
    assert "total_filtered" in data
    assert len(data["batches"]) > 0
    assert data["batches"][0]["batch"].startswith("L")


def test_get_videos_and_images(client):
    # Lấy video của batch L21
    res_vids = client.get("/api/videos/L21")
    assert res_vids.status_code == 200
    vids = res_vids.json()["videos"]
    assert len(vids) > 0
    first_vid = vids[0]["video_id"]

    # Lấy images của video đầu tiên
    res_imgs = client.get(f"/api/images/L21/{first_vid}")
    assert res_imgs.status_code == 200
    img_data = res_imgs.json()
    assert img_data["batch"] == "L21"
    assert img_data["video_id"] == first_vid
    assert len(img_data["images"]) > 0


def test_filter_add_and_remove(client):
    test_item = {
        "batch": "L21",
        "video_id": "V001",
        "filename": "99999_test.webp",
        "frame_id": 99999,
        "keyframe_path": "L21/V001/99999_test.webp",
        "reason": "unit_test",
    }

    # Add item
    res_add = client.post("/api/filter", json={"action": "add", "items": [test_item]})
    assert res_add.status_code == 200
    assert res_add.json()["status"] == "ok"
    assert res_add.json()["modified_count"] == 1

    # Check stats
    res_stats = client.get("/api/filter/stats")
    assert res_stats.status_code == 200
    assert "L21" in res_stats.json()["by_batch"]

    # Remove item
    res_rem = client.post("/api/filter", json={"action": "remove", "items": [test_item]})
    assert res_rem.status_code == 200
    assert res_rem.json()["status"] == "ok"
    assert res_rem.json()["modified_count"] == 1


def test_similar_search(client):
    # Tìm kiếm tương đồng cho frame 411 của L21_V001
    res = client.post(
        "/api/similar",
        json={
            "batch": "L21",
            "video_id": "V001",
            "frame_id": 411,
            "min_similarity": 0.6,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert data["total_found"] > 0
    # Frame 411 chính nó phải là top 1 với similarity = 1.0 (hoặc 100%)
    top_res = data["results"][0]
    assert top_res["is_query_image"] is True
    assert top_res["similarity_percent"] >= 99.0
