"""
tests/test_dante_dp_alignment.py
================================
Unit tests cho Dynamic Programming Monotonic Sequence Alignment (DANTE)
va Sub-Event Parser cho bai toan TRAKE.
"""

import pytest
from serving.api.schema.retrieval import CandidateResult
from serving.api.service.retrieval.sub_event_parser import parse_sub_events
from serving.api.service.temporal_aligner import align


def _create_cand(video_id: str, frame_id: int, timestamp_mid: float, score: float = 1.0) -> CandidateResult:
    return CandidateResult(
        video_id=video_id,
        frame_id=frame_id,
        scene_id=f"{video_id}_S{frame_id:05d}",
        timestamp_start=max(0.0, timestamp_mid - 0.5),
        timestamp_end=timestamp_mid + 0.5,
        timestamp_mid=timestamp_mid,
        keyframe_path=f"keyframes/{video_id}/{frame_id:06d}.jpg",
        score=score,
    )


def test_sub_event_parser_numbered():
    query = "1. Xe o to mau trang chay toi 2. Nguoi dan ong buoc xuong 3. Di vao toa nha"
    events = parse_sub_events(query, target_num_events=3)
    assert len(events) == 3
    assert "Xe o to mau trang chay toi" in events[0]
    assert "Nguoi dan ong buoc xuong" in events[1]
    assert "Di vao toa nha" in events[2]


def test_sub_event_parser_arrow():
    query = "xe may dung lai -> rut chia khoa -> mo cua phong"
    events = parse_sub_events(query, target_num_events=3)
    assert len(events) == 3
    assert events[0] == "xe may dung lai"
    assert events[1] == "rut chia khoa"
    assert events[2] == "mo cua phong"


def test_sub_event_parser_temporal_vietnamese():
    query = "Dau tien co gai deo kinh vao, sau do cam coc nuoc uong, cuoi cung dat coc xuong ban"
    events = parse_sub_events(query, target_num_events=3)
    assert len(events) == 3
    assert "co gai deo kinh" in events[0]
    assert "cam coc nuoc uong" in events[1]
    assert "dat coc xuong ban" in events[2]


def test_sub_event_parser_newlines():
    query = "Buoc 1: Chon hang tren ke\nBuoc 2: Di toi quay thu ngan\nBuoc 3: Nhan hoa don"
    events = parse_sub_events(query, target_num_events=3)
    assert len(events) == 3
    assert "Chon hang tren ke" in events[0]


def test_dante_dp_monotonic_alignment_basic():
    # Tao tap candidate cho 3 su kien trong 2 video: L21_V001 (khop tot) va L21_V002 (sai thu tu)
    cands_e1 = [
        _create_cand(video_id="L21_V001", frame_id=100, timestamp_mid=4.0, score=0.9),
        _create_cand(video_id="L21_V002", frame_id=500, timestamp_mid=20.0, score=0.8),
    ]
    cands_e2 = [
        _create_cand(video_id="L21_V001", frame_id=200, timestamp_mid=8.0, score=0.85),
        _create_cand(video_id="L21_V002", frame_id=300, timestamp_mid=12.0, score=0.7), # L21_V002 event 2 co timestamp truoc event 1
    ]
    cands_e3 = [
        _create_cand(video_id="L21_V001", frame_id=350, timestamp_mid=14.0, score=0.88),
        _create_cand(video_id="L21_V002", frame_id=600, timestamp_mid=24.0, score=0.6),
    ]

    event_candidates = [cands_e1, cands_e2, cands_e3]
    aligned_chains = align(event_candidates, max_results=10, min_separation_sec=1.0)

    assert len(aligned_chains) >= 1
    top_chain = aligned_chains[0]
    assert len(top_chain) == 3

    # Kiem tra tat ca frame deu thuoc cung 1 video
    assert top_chain[0].video_id == "L21_V001"
    assert top_chain[1].video_id == "L21_V001"
    assert top_chain[2].video_id == "L21_V001"

    # Kiem tra thu tu thoi gian tang dan nghiem ngat
    assert top_chain[0].timestamp_mid < top_chain[1].timestamp_mid < top_chain[2].timestamp_mid
    assert top_chain[0].frame_id == 100
    assert top_chain[1].frame_id == 200
    assert top_chain[2].frame_id == 350


def test_dante_dp_min_separation_window():
    # 2 keyframes qua sat nhau (< 1.0s) se khong duoc xep lien tiep neu co keyframe tiep theo cach >= 1.0s
    cands_e1 = [
        _create_cand(video_id="L22_V010", frame_id=100, timestamp_mid=5.0, score=0.9),
    ]
    cands_e2 = [
        _create_cand(video_id="L22_V010", frame_id=105, timestamp_mid=5.2, score=0.95), # Cach e1 chi 0.2s
        _create_cand(video_id="L22_V010", frame_id=200, timestamp_mid=8.0, score=0.85), # Cach e1 3.0s
    ]

    event_candidates = [cands_e1, cands_e2]
    aligned_chains = align(event_candidates, max_results=5, min_separation_sec=1.0)

    assert len(aligned_chains) >= 1
    top_chain = aligned_chains[0]
    assert top_chain[0].frame_id == 100
    assert top_chain[1].frame_id == 200 # Phai chon frame 200 thoa man min separation 1.0s
