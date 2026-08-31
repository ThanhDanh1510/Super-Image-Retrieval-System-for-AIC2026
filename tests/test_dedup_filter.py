"""
tests/test_dedup_filter.py
==========================
Unit tests cho module dedup_filter (Blacklist, Temporal NMS, Diversity Capping).
"""

import pytest
from pathlib import Path
from serving.api.schema.retrieval import CandidateResult
from serving.api.service.retrieval.dedup_filter import (
    deduplicate_candidates,
    is_blacklisted,
)


def _make_cand(video_id: str, frame_id: int, timestamp_mid: float, score: float, keyframe_path: str = "") -> CandidateResult:
    return CandidateResult(
        video_id=video_id,
        frame_id=frame_id,
        scene_id=f"{video_id}_S1",
        timestamp_start=timestamp_mid - 1.0,
        timestamp_end=timestamp_mid + 1.0,
        timestamp_mid=timestamp_mid,
        keyframe_path=keyframe_path or f"{video_id}/{frame_id}.webp",
        score=score,
    )


def test_blacklist_filtering():
    """Kiểm tra frame trong blacklist bị loại bỏ hoàn toàn."""
    cands = [
        _make_cand("L21_V006", 0, 0.0, 0.99, "L21/V006/0.webp"), # in filtered_keyframes.json
        _make_cand("L21_V006", 150, 6.0, 0.85, "L21/V006/150.webp"),
        _make_cand("L21_V099", 500, 20.0, 0.80, "L21/V099/500.webp"),
    ]

    res = deduplicate_candidates(cands, top_k=10)
    # L21_V006 frame 0 là monochrome bị lọc
    assert not any(c.video_id == "L21_V006" and c.frame_id == 0 for c in res)
    assert len(res) == 2


def test_temporal_nms_within_same_video():
    """Kiểm tra 2 frame trong cùng video cách nhau <= 5s thì chỉ giữ 1 frame điểm cao hơn trong top."""
    cands = [
        _make_cand("V001", 100, 10.0, 0.90),
        _make_cand("V001", 125, 11.0, 0.80), # Cách frame 100 chỉ 1.0s -> overflow
        _make_cand("V002", 200, 10.5, 0.88),
    ]

    res = deduplicate_candidates(cands, top_k=2, time_window_sec=5.0)
    assert len(res) == 2
    # Top 2 phải là V001 (10s, score 0.90) và V002 (10.5s, score 0.88)
    assert res[0].video_id == "V001" and res[0].frame_id == 100
    assert res[1].video_id == "V002" and res[1].frame_id == 200


def test_diversity_capping_max_per_video():
    """Kiểm tra một video không chiếm quá max_per_video trong nhóm top."""
    cands = [
        _make_cand("V001", 100, 10.0, 0.95),
        _make_cand("V001", 300, 30.0, 0.94),
        _make_cand("V001", 500, 50.0, 0.93),
        _make_cand("V001", 700, 70.0, 0.92), # Frame thứ 4 của V001 -> overflow
        _make_cand("V002", 100, 10.0, 0.85),
    ]

    res = deduplicate_candidates(cands, top_k=4, max_per_video=3, time_window_sec=5.0)
    # Trong top 4: 3 frame đầu của V001 và 1 frame của V002
    assert len(res) == 4
    v1_count = sum(1 for c in res if c.video_id == "V001")
    assert v1_count == 3
    assert any(c.video_id == "V002" for c in res)


def test_is_trake_bypasses_diversity_capping():
    """Kiểm tra chế độ TRAKE không áp dụng diversity capping để bảo tồn chuỗi timeline."""
    cands = [
        _make_cand("V001", 100, 10.0, 0.95),
        _make_cand("V001", 300, 30.0, 0.94),
        _make_cand("V001", 500, 50.0, 0.93),
        _make_cand("V001", 700, 70.0, 0.92),
    ]

    res = deduplicate_candidates(cands, top_k=4, max_per_video=2, is_trake=True)
    # Ở is_trake=True, toàn bộ 4 frame của V001 vẫn được giữ nguyên
    assert len(res) == 4
    assert all(c.video_id == "V001" for c in res)
