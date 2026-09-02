"""
tests/test_feedback_halo.py
===========================
Unit tests cho tính năng User Feedback Temporal Halo Boost.
"""

from serving.api.schema.retrieval import CandidateResult, UserChoice
from serving.api.service.retrieval.retrieval_pipeline import _apply_temporal_halo_boost


def _make_cand(video_id: str, frame_id: int, timestamp_mid: float, score: float) -> CandidateResult:
    return CandidateResult(
        video_id=video_id,
        frame_id=frame_id,
        scene_id=f"{video_id}_S1",
        timestamp_start=timestamp_mid - 1.0,
        timestamp_end=timestamp_mid + 1.0,
        timestamp_mid=timestamp_mid,
        keyframe_path=f"{video_id}/{frame_id}.webp",
        score=score,
    )


def test_temporal_halo_boost():
    # User đã chọn frame tại giây thứ 50.0 của video V001
    choices = [
        UserChoice(video_id="V001", frame_id=1250, scene_id="V001_S1", timestamp_start=49.0, timestamp_end=51.0, timestamp_mid=50.0, keyframe_path="V001/1250.webp")
    ]

    cands = [
        _make_cand("V001", 1125, 45.0, 0.80), # Cách 5s -> Nằm trong halo (50 +/- 10s) -> score tăng 1.25x -> 1.0
        _make_cand("V001", 2000, 80.0, 0.82), # Cách 30s -> Ngoài halo -> score giữ nguyên 0.82
        _make_cand("V002", 500, 50.0, 0.85),  # Khác video -> Không boost -> score giữ 0.85
    ]

    boosted = _apply_temporal_halo_boost(choices, cands, halo_radius_sec=10.0, boost_factor=1.25)
    
    # Candidate tại 45s của V001 phải được boost lên top (0.80 * 1.25 = 1.0 > 0.85 > 0.82)
    assert boosted[0].video_id == "V001" and boosted[0].frame_id == 1125
    assert abs(boosted[0].score - 1.0) < 1e-4
    assert boosted[1].video_id == "V002"
    assert boosted[2].video_id == "V001" and boosted[2].frame_id == 2000
