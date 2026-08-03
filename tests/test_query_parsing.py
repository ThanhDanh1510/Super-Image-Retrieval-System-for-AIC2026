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
