"""
tests/test_query_parsing.py
============================
Unit tests cho TASK-TST-02: Query Parsing & DANTE Temporal Alignment.
"""

from __future__ import annotations

import pytest
from serving.api.schema.retrieval import CandidateResult
from serving.api.service import temporal_aligner


@pytest.fixture
def multi_event_candidates():
    # Event 0 candidates
    event_0 = [
        CandidateResult(
            video_id="V001",
            scene_id="V001_S001",
            frame_id=10,
            timestamp_start=0.0,
            timestamp_end=4.0,
            timestamp_mid=2.0,
            keyframe_path="V001/V001_S001.jpg",
            score=0.9,
        ),
        CandidateResult(
            video_id="V002",
            scene_id="V002_S001",
            frame_id=5,
            timestamp_start=0.0,
            timestamp_end=3.0,
            timestamp_mid=1.5,
            keyframe_path="V002/V002_S001.jpg",
            score=0.85,
        ),
    ]

    # Event 1 candidates
    event_1 = [
        CandidateResult(
            video_id="V001",
            scene_id="V001_S003",
            frame_id=30,
            timestamp_start=10.0,
            timestamp_end=15.0,
            timestamp_mid=12.5,
            keyframe_path="V001/V001_S003.jpg",
            score=0.95,
        ),
        CandidateResult(
            video_id="V002",
            scene_id="V002_S002",
            frame_id=2, # frame_id = 2 < 5 (Out of order for V002!)
            timestamp_start=0.5,
            timestamp_end=2.0,
            timestamp_mid=1.0, # timestamp_mid = 1.0 < 1.5 (Out of order!)
            keyframe_path="V002/V002_S002.jpg",
            score=0.99,
        ),
    ]

    return [event_0, event_1]


def test_dante_align_selects_valid_ascending_chain(multi_event_candidates):
    chains = temporal_aligner.align(multi_event_candidates, max_results=10)
    assert len(chains) >= 1
    
    # V001 có chuỗi timestamp hợp lệ (2.0 -> 12.5)
    best_chain = chains[0]
    assert len(best_chain) == 2
    assert best_chain[0].video_id == "V001"
    assert best_chain[1].video_id == "V001"
    assert best_chain[0].timestamp_mid < best_chain[1].timestamp_mid


def test_dante_align_never_mixes_videos():
    """TRAKE chain MUST stay inside one video folder (e.g. all L001_V001),
    never mixing keyframes from two different videos (L001_V001 + L001_V002)."""
    def _mk(video_id, frame_id, ts_mid, score):
        return CandidateResult(
            video_id=video_id,
            scene_id=f"{video_id}_S{frame_id:03d}",
            frame_id=frame_id,
            timestamp_start=ts_mid - 0.5,
            timestamp_end=ts_mid + 0.5,
            timestamp_mid=ts_mid,
            keyframe_path=f"{video_id}/{video_id}_S{frame_id:03d}.jpg",
            score=score,
        )

    # Two videos, each with candidates for 3 events.
    # Currently none of the events has a clear per-video winner, so the
    # worst case is a chain mixing L001_V001 and L001_V002 — which must NOT happen.
    ev_candidates = [
        [
            _mk("L001_V001", 10, 1.0, 0.9),
            _mk("L001_V002", 110, 1.0, 0.7),
        ],
        [
            _mk("L001_V001", 20, 2.0, 0.85),
            _mk("L001_V002", 120, 2.0, 0.8),
        ],
        [
            _mk("L001_V001", 30, 4.0, 0.95),
            _mk("L001_V002", 130, 4.0, 0.9),
        ],
    ]

    chains = temporal_aligner.align(ev_candidates, max_results=10)
    assert len(chains) >= 1

    for chain in chains:
        assert len(chain) == 3
        # 1. All keyframes must come from the SAME video folder
        video_ids = {c.video_id for c in chain}
        assert len(video_ids) == 1, (
            f"TRAKE chain mixed videos: {video_ids}. "
            f"Every chain must live in a single video folder like L001_V001."
        )
        # 2. Timestamps must be strictly ascending and distinct
        timestamps = [round(c.timestamp_mid, 2) for c in chain]
        assert timestamps == sorted(timestamps), f"TRAKE timestamps not ascending: {timestamps}"
        assert len(set(timestamps)) == len(timestamps), f"TRAKE timestamps duplicated: {timestamps}"


def test_dante_align_empty_input():
    assert temporal_aligner.align([]) == []
    assert temporal_aligner.align([[]]) == []


def test_dante_align_single_event():
    cand = CandidateResult(
        video_id="V001",
        scene_id="V001_S001",
        frame_id=1,
        timestamp_start=0.0,
        timestamp_end=2.0,
        timestamp_mid=1.0,
        keyframe_path="V001/V001_S001.jpg",
        score=0.9,
    )
    chains = temporal_aligner.align([[cand]], max_results=5)
    assert len(chains) == 1
    assert chains[0] == [cand]
