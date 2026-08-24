"""
serving/api/service/temporal_aligner.py
========================================
Thuật toán Quy Hoạch Động (Dynamic Programming) & Monotonic Sequence Alignment (DANTE Engine)
cho bài toán Temporal Retrieval and Alignment of Key Events (TRAKE).

Nguyên lý hoạt động:
  1. Constraint bắt buộc: N keyframe trong mỗi chuỗi (chain) phải cùng thuộc về 1 VIDEO (video_id duy nhất).
  2. Monotonicity Constraint: timestamp_mid(event_1) < timestamp_mid(event_2) < ... < timestamp_mid(event_N).
  3. Min Temporal Separation: Khoảng cách giữa 2 sự kiện liên tiếp phải đạt tối thiểu delta (mặc định 1.0s).
  4. Time-Aware Penalty: Phạt tuyến tính theo độ trễ thời gian thực (seconds) thay vì chỉ đếm số frame.
  5. All-Event Consensus Scoring: Ưu tiên các video có độ bao phủ 100% các sự kiện.
  6. Dense Similarity Matrix: Hỗ trợ ma trận True Cosine Similarity (không bị thưa 0.0).
  7. First-Occurrence Bias: Ưu tiên chuỗi xuất hiện sớm nhất khi prompt yêu cầu "khoảnh khắc đầu tiên".
"""

from __future__ import annotations

import logging
import numpy as np
from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)


def _get_lambda_penalty() -> float:
    """Đọc lambda penalty factor từ fusion_config.yaml (mặc định 0.001)."""
    try:
        from shared.utils.config_loader import load_fusion_config
        cfg = load_fusion_config()
        return float(cfg.get("temporal_aligner", {}).get("lambda_penalty", 0.001))
    except Exception:
        return 0.001


def _get_earliest_bias() -> float:
    """Đọc earliest_bias factor từ fusion_config.yaml (mặc định 0.05)."""
    try:
        from shared.utils.config_loader import load_fusion_config
        cfg = load_fusion_config()
        return float(cfg.get("temporal_aligner", {}).get("earliest_bias", 0.05))
    except Exception:
        return 0.05


def align(
    event_candidates: list[list[CandidateResult]],
    max_results: int = 100,
    lambda_penalty: float | None = None,
    min_separation_sec: float = 1.0,
    max_chains_per_video: int = 5,
    earliest_bias: float | None = None,
    dense_matrices: dict[str, tuple[list[CandidateResult], np.ndarray]] | None = None,
) -> list[list[CandidateResult]]:
    """
    Căn chỉnh chuỗi N sự kiện tuần tự trong video sử dụng Quy hoạch động (DANTE Engine).

    Args:
        event_candidates: Danh sách N danh sách CandidateResult (mỗi danh sách ứng với 1 sub-event).
        max_results: Số lượng chuỗi tối đa cần trả về.
        lambda_penalty: Hệ số phạt khoảng cách thời gian (None -> load từ config).
        min_separation_sec: Khoảng cách thời gian tối thiểu giữa 2 sự kiện liên tiếp (giây).
        max_chains_per_video: Số lượng chuỗi tối đa được trích xuất từ mỗi video.
        earliest_bias: Hệ số ưu tiên chuỗi xuất hiện sớm trong video (First-Occurrence Bias).
        dense_matrices: Optional dict mapping video_id -> (candidates, S_dense[N, T]).

    Returns:
        list[list[CandidateResult]]: Danh sách các N-tuples keyframe được sắp xếp giảm dần theo điểm DP.
    """
    if not event_candidates or any(not c for c in event_candidates):
        return []

    num_events = len(event_candidates)
    if num_events < 2:
        return [[c] for c in event_candidates[0][:max_results]]

    if lambda_penalty is None:
        lambda_penalty = _get_lambda_penalty()

    if earliest_bias is None:
        earliest_bias = _get_earliest_bias()

    # 1. Gom nhóm candidates theo từng video_id
    video_map: dict[str, dict[int, list[CandidateResult]]] = {}

    for event_idx, candidates in enumerate(event_candidates):
        for cand in candidates:
            v_id = cand.video_id
            if v_id not in video_map:
                video_map[v_id] = {ev: [] for ev in range(num_events)}
            video_map[v_id][event_idx].append(cand)

    valid_chains: list[tuple[float, list[CandidateResult]]] = []

    # Danh sách toàn bộ video cần xử lý (gồm cả từ video_map và dense_matrices)
    all_vids = list(video_map.keys())
    if dense_matrices:
        for d_vid in dense_matrices.keys():
            if d_vid not in video_map:
                all_vids.append(d_vid)

    # 2. Xử lý quy hoạch động riêng biệt cho từng video
    for v_id in all_vids:
        # Nếu có Dense Similarity Matrix đã tính sẵn cho video này
        if dense_matrices and v_id in dense_matrices:
            dense_kfs, S_dense = dense_matrices[v_id]
            if not dense_kfs or S_dense.shape[0] != num_events or S_dense.shape[1] < num_events:
                continue

            unique_kfs = dense_kfs
            T = len(unique_kfs)
            kf_timestamps = np.array(
                [float(c.timestamp_mid if c.timestamp_mid is not None else (c.frame_id / 25.0)) for c in unique_kfs],
                dtype=np.float32,
            )
            S = S_dense.astype(np.float32)
        else:
            events_dict = video_map.get(v_id)
            if not events_dict:
                continue

            all_cands_in_video: list[CandidateResult] = []
            for ev_idx in range(num_events):
                all_cands_in_video.extend(events_dict[ev_idx])

            if not all_cands_in_video:
                continue

            # Sắp xếp toàn bộ keyframe trong video theo thời gian thực
            sorted_kfs = sorted(all_cands_in_video, key=lambda c: (c.timestamp_mid, c.frame_id))

            # Loại bỏ trùng lặp gần nhau (< 0.2s)
            unique_kfs = []
            seen_timestamps: list[float] = []
            for kf in sorted_kfs:
                ts = float(kf.timestamp_mid if kf.timestamp_mid is not None else (kf.frame_id / 25.0))
                if not any(abs(ts - prev_ts) < 0.2 for prev_ts in seen_timestamps):
                    seen_timestamps.append(ts)
                    unique_kfs.append(kf)

            T = len(unique_kfs)
            if T < num_events:
                continue

            kf_timestamps = np.array(
                [float(c.timestamp_mid if c.timestamp_mid is not None else (c.frame_id / 25.0)) for c in unique_kfs],
                dtype=np.float32,
            )

            # Ma trận tương đồng S[num_events, T] từ candidate scores
            S = np.zeros((num_events, T), dtype=np.float32)
            for i in range(num_events):
                ev_frame_scores = {c.frame_id: c.score for c in events_dict[i]}
                for t, kf in enumerate(unique_kfs):
                    S[i, t] = ev_frame_scores.get(kf.frame_id, 0.0)

        # Bảng quy hoạch động DP[num_events, T] và parent pointer
        DP = np.full((num_events, T), -np.inf, dtype=np.float32)
        parent = np.full((num_events, T), -1, dtype=np.int32)

        # Base case: event đầu tiên (i = 0)
        DP[0, :] = S[0, :]

        # DP Transitions: DP[i, t] = S[i, t] + max_{tau} (DP[i-1, tau] - lambda * (t_sec - tau_sec))
        for i in range(1, num_events):
            running_max = -np.inf
            best_tau = -1
            tau_ptr = 0

            for t in range(T):
                cur_time = kf_timestamps[t]

                # Cập nhật running_max với các tau thỏa mãn min_separation_sec
                while tau_ptr < t and (cur_time - kf_timestamps[tau_ptr]) >= min_separation_sec:
                    prev_val = DP[i - 1, tau_ptr]
                    if prev_val > -np.inf:
                        cand_val = prev_val + lambda_penalty * kf_timestamps[tau_ptr]
                        if cand_val > running_max:
                            running_max = cand_val
                            best_tau = tau_ptr
                    tau_ptr += 1

                if running_max > -np.inf:
                    DP[i, t] = S[i, t] + running_max - lambda_penalty * cur_time
                    parent[i, t] = best_tau

        # 3. Backtrack tìm top chuỗi tốt nhất từ hàng cuối cùng DP[num_events - 1]
        terminal_scores = DP[num_events - 1]
        sorted_terminals = np.argsort(terminal_scores)[::-1]

        chains_from_this_video = 0
        seen_frame_tuples: set[tuple[int, ...]] = set()

        for t_end in sorted_terminals:
            if chains_from_this_video >= max_chains_per_video:
                break

            if terminal_scores[t_end] <= -np.inf:
                continue

            # Truy vết ngược (Backtracking)
            curr_t = int(t_end)
            chain_indices = [0] * num_events
            valid = True

            for i in range(num_events - 1, -1, -1):
                chain_indices[i] = curr_t
                if i > 0:
                    curr_t = parent[i, curr_t]
                    if curr_t < 0:
                        valid = False
                        break

            if valid:
                chain_cands = [unique_kfs[idx] for idx in chain_indices]
                chain_ts = [kf_timestamps[idx] for idx in chain_indices]
                frame_tuple = tuple(c.frame_id for c in chain_cands)

                # Kiểm tra đơn điệu nghiêm ngặt & không trùng lặp chuỗi
                is_strictly_monotonic = all(chain_ts[k] < chain_ts[k + 1] for k in range(num_events - 1))
                if is_strictly_monotonic and frame_tuple not in seen_frame_tuples:
                    seen_frame_tuples.add(frame_tuple)

                    # Tính điểm đồng thuận (All-Event Consensus & Temporal Span Penalty)
                    event_sims = [float(S[k, chain_indices[k]]) for k in range(num_events)]
                    matched_count = sum(1 for s in event_sims if s > 0.01)
                    avg_sim = sum(event_sims) / float(num_events)
                    consensus_factor = (float(matched_count) / float(num_events)) ** 2
                    span_seconds = float(chain_ts[-1] - chain_ts[0])
                    span_penalty = lambda_penalty * span_seconds

                    # First-occurrence bias (ưu tiên sự kiện xuất hiện sớm hơn trong video)
                    t_first = float(chain_ts[0])
                    t_max_vid = float(kf_timestamps[-1]) if len(kf_timestamps) > 0 else 1.0
                    earliest_bonus = earliest_bias * max(0.0, 1.0 - (t_first / (t_max_vid + 1e-5)))

                    final_chain_score = (avg_sim * consensus_factor) - span_penalty + earliest_bonus

                    # Gán điểm cho từng frame để UI hiển thị
                    for k, cand in enumerate(chain_cands):
                        cand.score = round(event_sims[k], 4)

                    valid_chains.append((final_chain_score, chain_cands))
                    chains_from_this_video += 1

        # Fallback cho video nếu DP bị miss do threshold
        if chains_from_this_video == 0 and T >= num_events:
            step = (T - 1) / float(num_events - 1)
            fallback_indices = [int(round(k * step)) for k in range(num_events)]
            for k in range(1, num_events):
                if fallback_indices[k] <= fallback_indices[k - 1]:
                    fallback_indices[k] = fallback_indices[k - 1] + 1

            if fallback_indices[-1] < T:
                fallback_cands = [unique_kfs[idx] for idx in fallback_indices]
                fallback_ts = [kf_timestamps[idx] for idx in fallback_indices]
                if all(fallback_ts[k] < fallback_ts[k + 1] for k in range(num_events - 1)):
                    avg_sc = sum(c.score for c in fallback_cands) / num_events
                    valid_chains.append((avg_sc * 0.5, fallback_cands))

    # 4. Sắp xếp tất cả chuỗi trên toàn bộ các video theo điểm giảm dần
    valid_chains.sort(key=lambda x: x[0], reverse=True)
    return [chain for _, chain in valid_chains[:max_results]]
