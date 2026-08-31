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


def test_sub_event_parser_complex_preamble_and_e_tags():
    query = (
        "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, bên cạnh lá cờ trắng viền đỏ. "
        "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng. "
        "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ (thời điểm đầu tiên các chân của lân đặt trên trụ sau khi xoay). "
        "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân."
    )
    events = parse_sub_events(query, target_num_events=3)
    assert len(events) == 3
    assert "hai con rồng vàng đang xoay vòng" in events[0]
    assert "con lân hoàn tất cú xoay người" in events[1]
    assert "dùi chạm vào kẻng đồng" in events[2]


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


def test_dense_similarity_computation():
    import numpy as np
    from serving.api.service.dense_similarity import compute_dense_similarity

    # N=3 sub-events, T=4 keyframes, D=8
    queries = np.random.randn(3, 8).astype(np.float32)
    keyframes = np.random.randn(4, 8).astype(np.float32)

    sim = compute_dense_similarity(queries, keyframes)
    assert sim.shape == (3, 4)
    assert np.all(sim >= 0.0)
    assert np.all(sim <= 1.0)


def test_dante_dp_earliest_bias():
    # 2 chuoi cung video co score tuong duong nhau, chuoi xuat hien som hon (t=2s -> t=6s) phai dung tren chuoi muon hon (t=20s -> t=24s)
    cands_e1 = [
        _create_cand(video_id="L23_V001", frame_id=10, timestamp_mid=2.0, score=0.9),
        _create_cand(video_id="L23_V001", frame_id=100, timestamp_mid=20.0, score=0.9),
    ]
    cands_e2 = [
        _create_cand(video_id="L23_V001", frame_id=30, timestamp_mid=6.0, score=0.9),
        _create_cand(video_id="L23_V001", frame_id=120, timestamp_mid=24.0, score=0.9),
    ]

    event_candidates = [cands_e1, cands_e2]
    aligned_chains = align(event_candidates, max_results=5, earliest_bias=0.1)

    assert len(aligned_chains) >= 2
    top_chain = aligned_chains[0]
    assert top_chain[0].frame_id == 10 # Chuoi som hon duoc uu tien
    assert top_chain[1].frame_id == 30


def test_dante_dp_dense_matrices_integration():
    import numpy as np
    # Tao dense matrix S[2, 3] cho video L24_V024
    dense_kfs = [
        _create_cand(video_id="L24_V024", frame_id=9807, timestamp_mid=10.0, score=0.0),
        _create_cand(video_id="L24_V024", frame_id=10158, timestamp_mid=15.0, score=0.0),
        _create_cand(video_id="L24_V024", frame_id=10210, timestamp_mid=20.0, score=0.0),
    ]
    # S matrix: Event 0 match frame 9807 (0.95), Event 1 match frame 10158 (0.92), Event 2 match frame 10210 (0.88)
    S_dense = np.array([
        [0.95, 0.10, 0.05],
        [0.10, 0.92, 0.15],
        [0.05, 0.10, 0.88],
    ], dtype=np.float32)

    dense_matrices = {
        "L24_V024": (dense_kfs, S_dense)
    }

    # Pass empty sparse candidate list, DP will use dense_matrices
    event_candidates = [[dense_kfs[0]], [dense_kfs[1]], [dense_kfs[2]]]
    aligned_chains = align(
        event_candidates=event_candidates,
        max_results=5,
        dense_matrices=dense_matrices,
        min_separation_sec=1.0,
    )

    assert len(aligned_chains) >= 1
    top_chain = aligned_chains[0]
    assert len(top_chain) == 3
    assert top_chain[0].frame_id == 9807
    assert top_chain[1].frame_id == 10158
    assert top_chain[2].frame_id == 10210


def test_candidate_score_cloning_immutability():
    # Kiem tra score cua candidate goc khong bi dot bien (mutate in-place) sau align()
    orig_c1 = _create_cand(video_id="L01_V001", frame_id=10, timestamp_mid=1.0, score=0.8)
    orig_c2 = _create_cand(video_id="L01_V001", frame_id=20, timestamp_mid=5.0, score=0.7)

    event_candidates = [[orig_c1], [orig_c2]]
    aligned_chains = align(event_candidates, max_results=1)

    assert len(aligned_chains) == 1
    # Candidate goc van giu nguyen diem ban dau, khong bi overwrite
    assert orig_c1.score == 0.8
    assert orig_c2.score == 0.7


def test_dante_dp_adaptive_lambda():
    # Video co keyframe thua (cach nhau 12s > 5s nguong adaptive), adaptive lambda tu dong giam nhe penalty
    cands_e1 = [_create_cand(video_id="L02_V002", frame_id=1, timestamp_mid=2.0, score=0.9)]
    cands_e2 = [_create_cand(video_id="L02_V002", frame_id=2, timestamp_mid=14.0, score=0.9)]

    event_candidates = [cands_e1, cands_e2]
    aligned_chains = align(
        event_candidates,
        max_results=1,
        lambda_penalty=0.001,
        enable_adaptive_lambda=True,
    )
    assert len(aligned_chains) == 1
    assert aligned_chains[0][0].frame_id == 1
    assert aligned_chains[0][1].frame_id == 2


def test_qdrant_repo_match_value_filter():
    import inspect
    from serving.api.repo import qdrant_repo

    # Kiem tra ma nguon cua get_keyframes_by_video_id va get_keyframes_with_vectors_by_video_id dung MatchValue
    src_by_id = inspect.getsource(qdrant_repo.get_keyframes_by_video_id)
    src_with_vec = inspect.getsource(qdrant_repo.get_keyframes_with_vectors_by_video_id)

    assert "MatchValue(value=video_id)" in src_by_id
    assert "MatchValue(value=video_id)" in src_with_vec
    assert "MatchText(text=video_id)" not in src_by_id
    assert "MatchText(text=video_id)" not in src_with_vec


def test_multi_chain_temporal_separation():
    # Kiem tra align tao ra 2 chuoi cho cung 1 video khi max_chains_per_video=2
    # va cac chuoi co do lech thoi gian
    cands_e1 = [
        _create_cand(video_id="L25_V001", frame_id=10, timestamp_mid=2.0, score=0.9),
        _create_cand(video_id="L25_V001", frame_id=100, timestamp_mid=20.0, score=0.85),
    ]
    cands_e2 = [
        _create_cand(video_id="L25_V001", frame_id=30, timestamp_mid=6.0, score=0.9),
        _create_cand(video_id="L25_V001", frame_id=120, timestamp_mid=24.0, score=0.85),
    ]
    event_candidates = [cands_e1, cands_e2]
    aligned_chains = align(event_candidates, max_results=5, max_chains_per_video=2)

    assert len(aligned_chains) == 2
    assert aligned_chains[0][0].video_id == "L25_V001"
    assert aligned_chains[1][0].video_id == "L25_V001"
    ts_c1 = [c.timestamp_mid for c in aligned_chains[0]]
    ts_c2 = [c.timestamp_mid for c in aligned_chains[1]]
    diff = max(abs(t1 - t2) for t1, t2 in zip(ts_c1, ts_c2))
    assert diff >= 1.0


def test_sub_event_temporal_cue_extraction():
    # Kiem tra trich xuat cac loai temporal cues tu cau query
    from serving.api.service.retrieval.sub_event_parser import extract_temporal_cues

    events = [
        "Cảnh sát vẫy tay nghi phạm vào phòng",
        "Ngay sau đó cảnh sát chỉ tay vào vị trí đứng",
        "Một lúc sau cảnh sát hoàn toàn khom người xuống quét kiểm tra"
    ]
    query = "Cảnh sát vẫy tay. Ngay sau đó cảnh sát chỉ tay. Một lúc sau cảnh sát khom người."
    cues = extract_temporal_cues(events, query)

    assert len(cues) == 2
    # Transition 0 -> 1 la "ngay sau do" (fast)
    assert cues[0]["cue_type"] == "fast"
    assert cues[0]["expected_gap_sec"] == 2.0
    assert cues[0]["min_separation_sec"] == 0.5

    # Transition 1 -> 2 la "mot luc sau" (long)
    assert cues[1]["cue_type"] == "long"
    assert cues[1]["expected_gap_sec"] == 10.0
    assert cues[1]["min_separation_sec"] == 3.0


def test_dante_dp_c7_interpolation_fallback():
    # Giả lập trường hợp Event 2 bị khuyết điểm (S[1, :] = 0.0)
    # C7 Interpolation Fallback phải tự động nội suy frame E2 nằm giữa E1 và E3
    cands_e1 = [_create_cand(video_id="L10_V001", frame_id=10, timestamp_mid=2.0, score=0.9)]
    # Event 2 hoàn toàn không có candidates trong DB cho video này, nhưng có frame kf_20 ở 5.0s
    # Để giả lập trong align(), ta đưa 1 candidate score=0.001 (dưới ngưỡng 0.08)
    cands_e2 = [_create_cand(video_id="L10_V001", frame_id=20, timestamp_mid=6.0, score=0.001)]
    cands_e3 = [_create_cand(video_id="L10_V001", frame_id=30, timestamp_mid=10.0, score=0.9)]

    event_candidates = [cands_e1, cands_e2, cands_e3]
    aligned_chains = align(
        event_candidates,
        max_results=5,
        enable_interpolation=True,
        interpolation_threshold=0.08,
    )

    assert len(aligned_chains) >= 1
    chain = aligned_chains[0]
    assert len(chain) == 3
    assert chain[0].frame_id == 10
    assert chain[1].frame_id == 20
    assert chain[2].frame_id == 30
    assert chain[0].timestamp_mid < chain[1].timestamp_mid < chain[2].timestamp_mid


def test_dante_dp_top_k_viterbi_distinct_chains():
    # Kiểm tra Top-K Viterbi DP trích xuất được 3 chuỗi runner-up thực sự
    cands_e1 = [
        _create_cand(video_id="L12_V001", frame_id=1, timestamp_mid=1.0, score=0.9),
        _create_cand(video_id="L12_V001", frame_id=2, timestamp_mid=2.0, score=0.8),
    ]
    cands_e2 = [
        _create_cand(video_id="L12_V001", frame_id=5, timestamp_mid=5.0, score=0.9),
        _create_cand(video_id="L12_V001", frame_id=6, timestamp_mid=6.0, score=0.8),
    ]
    cands_e3 = [
        _create_cand(video_id="L12_V001", frame_id=9, timestamp_mid=9.0, score=0.9),
        _create_cand(video_id="L12_V001", frame_id=10, timestamp_mid=10.0, score=0.8),
    ]

    event_candidates = [cands_e1, cands_e2, cands_e3]
    aligned_chains = align(
        event_candidates,
        max_results=10,
        max_chains_per_video=3,
        top_k_beam=3,
    )

    assert len(aligned_chains) >= 2
    for chain in aligned_chains:
        assert len(chain) == 3
        # Đảm bảo tính đơn điệu nghiêm ngặt
        assert chain[0].timestamp_mid < chain[1].timestamp_mid < chain[2].timestamp_mid
    # Điểm chuỗi đầu tiên phải >= chuỗi thứ hai
    assert aligned_chains[0][0].score >= aligned_chains[1][0].score


def test_dante_dp_enforces_15s_max_separation():
    # Kiểm tra ràng buộc cứng: các sự kiện liên tiếp KHÔNG ĐƯỢC cách nhau quá 15s
    # Trường hợp 1: khoảng cách giữa các sự kiện > 15s (ví dụ 20s) -> Phải bị từ chối, không tạo chuỗi
    cands_e1_far = [_create_cand(video_id="L12_V001", frame_id=1, timestamp_mid=10.0, score=0.9)]
    cands_e2_far = [_create_cand(video_id="L12_V001", frame_id=2, timestamp_mid=30.0, score=0.9)] # gap = 20s > 15s
    cands_e3_far = [_create_cand(video_id="L12_V001", frame_id=3, timestamp_mid=50.0, score=0.9)] # gap = 20s > 15s

    chains_far = align([cands_e1_far, cands_e2_far, cands_e3_far], max_separation_sec=15.0)
    assert len(chains_far) == 0, "Không được trả về chuỗi khi các sự kiện cách nhau > 15s!"

    # Trường hợp 2: khoảng cách giữa các sự kiện <= 15s (ví dụ 8s) -> Hợp lệ
    cands_e1_near = [_create_cand(video_id="L12_V001", frame_id=1, timestamp_mid=10.0, score=0.9)]
    cands_e2_near = [_create_cand(video_id="L12_V001", frame_id=2, timestamp_mid=18.0, score=0.9)] # gap = 8s <= 15s
    cands_e3_near = [_create_cand(video_id="L12_V001", frame_id=3, timestamp_mid=26.0, score=0.9)] # gap = 8s <= 15s

    chains_near = align([cands_e1_near, cands_e2_near, cands_e3_near], max_separation_sec=15.0)
    assert len(chains_near) == 1
    assert [c.frame_id for c in chains_near[0]] == [1, 2, 3]
    for k in range(len(chains_near[0]) - 1):
        gap = chains_near[0][k + 1].timestamp_mid - chains_near[0][k].timestamp_mid
        assert gap <= 15.0
