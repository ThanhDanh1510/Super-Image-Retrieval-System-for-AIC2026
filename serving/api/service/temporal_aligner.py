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


def _get_interpolation_fallback_enabled() -> bool:
    """Kiểm tra có bật C7 interpolation fallback từ fusion_config.yaml không."""
    try:
        from shared.utils.config_loader import load_fusion_config
        cfg = load_fusion_config()
        return bool(cfg.get("temporal_aligner", {}).get("enable_interpolation_fallback", True))
    except Exception:
        return True


def _get_interpolation_conf_threshold() -> float:
    """Đọc ngưỡng tin cậy để kích hoạt interpolation fallback (mặc định 0.08)."""
    try:
        from shared.utils.config_loader import load_fusion_config
        cfg = load_fusion_config()
        return float(cfg.get("temporal_aligner", {}).get("interpolation_conf_threshold", 0.08))
    except Exception:
        return 0.08


def _get_top_k_beam() -> int:
    """Đọc beam size cho Top-K Viterbi DP (mặc định 3)."""
    try:
        from shared.utils.config_loader import load_fusion_config
        cfg = load_fusion_config()
        return int(cfg.get("temporal_aligner", {}).get("top_k_viterbi_beam", 3))
    except Exception:
        return 3


def _get_adaptive_lambda_enabled() -> bool:
    """Kiểm tra có bật adaptive lambda penalty từ fusion_config.yaml không."""
    try:
        from shared.utils.config_loader import load_fusion_config
        cfg = load_fusion_config()
        return bool(cfg.get("temporal_aligner", {}).get("enable_adaptive_lambda", True))
    except Exception:
        return True


def _get_max_separation_sec() -> float:
    """Đọc khoảng cách tối đa giữa 2 sự kiện liên tiếp từ fusion_config.yaml (mặc định 15.0s)."""
    try:
        from shared.utils.config_loader import load_fusion_config
        cfg = load_fusion_config()
        return float(cfg.get("temporal_aligner", {}).get("max_separation_sec", 15.0))
    except Exception:
        return 15.0


def align(
    event_candidates: list[list[CandidateResult]],
    max_results: int = 100,
    lambda_penalty: float | None = None,
    min_separation_sec: float = 1.0,
    max_separation_sec: float | None = None,
    max_chains_per_video: int = 3,
    earliest_bias: float | None = None,
    dense_matrices: dict[str, tuple[list[CandidateResult], np.ndarray]] | None = None,
    enable_adaptive_lambda: bool | None = None,
    temporal_cues: list[dict[str, any]] | None = None,
    enable_interpolation: bool | None = None,
    interpolation_threshold: float | None = None,
    top_k_beam: int | None = None,
) -> list[list[CandidateResult]]:
    """
    Căn chỉnh chuỗi N sự kiện tuần tự trong video sử dụng Quy hoạch động Top-K Viterbi (DANTE Engine)
    kết hợp C7 Interpolation Fallback và Temporal Cues.
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

    if enable_adaptive_lambda is None:
        enable_adaptive_lambda = _get_adaptive_lambda_enabled()

    if enable_interpolation is None:
        enable_interpolation = _get_interpolation_fallback_enabled()

    if interpolation_threshold is None:
        interpolation_threshold = _get_interpolation_conf_threshold()

    if top_k_beam is None:
        top_k_beam = _get_top_k_beam()

    if max_separation_sec is None:
        max_separation_sec = _get_max_separation_sec()

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

    K = max(1, top_k_beam)

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

        # Tính lambda_penalty thích ứng (adaptive lambda) theo mật độ keyframe của video
        curr_lambda = lambda_penalty
        if enable_adaptive_lambda and T >= 2:
            t_span = float(kf_timestamps[-1] - kf_timestamps[0])
            avg_gap = t_span / max(float(T - 1), 1.0)
            if avg_gap > 5.0:
                curr_lambda = lambda_penalty * (5.0 / avg_gap)

        # Kiểm tra C7: Interpolation Fallback cho các sự kiện bị miss
        missing_events: set[int] = set()
        S_work = S.copy()
        if enable_interpolation and num_events >= 2:
            for ev_idx in range(num_events):
                max_sc = float(np.max(S_work[ev_idx])) if T > 0 else 0.0
                if max_sc < interpolation_threshold:
                    missing_events.add(ev_idx)

            # Chỉ cho phép nội suy nếu không bị miss toàn bộ (miss tối đa N-1 sự kiện)
            if 0 < len(missing_events) < num_events:
                for m_ev in missing_events:
                    S_work[m_ev, :] = 0.12

        # Bảng quy hoạch động Top-K Viterbi DP[num_events, T, K] và parent pointer
        DP = np.full((num_events, T, K), -np.inf, dtype=np.float32)
        parent_tau = np.full((num_events, T, K), -1, dtype=np.int32)
        parent_k = np.full((num_events, T, K), -1, dtype=np.int32)

        # Base case: event đầu tiên (i = 0)
        DP[0, :, 0] = S_work[0, :]

        # DP Transitions với Top-K Viterbi
        for i in range(1, num_events):
            step_min_sep = min_separation_sec
            if temporal_cues and i - 1 < len(temporal_cues):
                step_min_sep = float(temporal_cues[i - 1].get("min_separation_sec", min_separation_sec))

            for t in range(T):
                cur_time = kf_timestamps[t]
                curr_s = S_work[i, t]

                cand_transitions: list[tuple[float, int, int]] = []
                for tau in range(t):
                    dt = cur_time - kf_timestamps[tau]
                    if dt < step_min_sep:
                        continue
                    if dt > max_separation_sec:
                        continue

                    penalty = curr_lambda * dt
                    for k_prev in range(K):
                        prev_sc = DP[i - 1, tau, k_prev]
                        if prev_sc > -np.inf:
                            tot_sc = curr_s + prev_sc - penalty
                            cand_transitions.append((tot_sc, tau, k_prev))

                if cand_transitions:
                    cand_transitions.sort(key=lambda x: x[0], reverse=True)
                    for k_idx, (sc, tau_p, kp) in enumerate(cand_transitions[:K]):
                        DP[i, t, k_idx] = sc
                        parent_tau[i, t, k_idx] = tau_p
                        parent_k[i, t, k_idx] = kp

        # 3. Backtrack tìm top chuỗi tốt nhất từ hàng cuối cùng DP[num_events - 1]
        terminal_candidates: list[tuple[float, int, int]] = []
        for t_end in range(T):
            for k_end in range(K):
                sc = DP[num_events - 1, t_end, k_end]
                if sc > -np.inf:
                    terminal_candidates.append((sc, t_end, k_end))

        terminal_candidates.sort(key=lambda x: x[0], reverse=True)

        chains_from_this_video = 0
        seen_frame_tuples: set[tuple[int, ...]] = set()

        for term_sc, t_end, k_end in terminal_candidates:
            if chains_from_this_video >= max_chains_per_video:
                break

            curr_t = int(t_end)
            curr_k = int(k_end)
            chain_indices = [0] * num_events
            valid = True

            for i in range(num_events - 1, -1, -1):
                chain_indices[i] = curr_t
                if i > 0:
                    next_t = parent_tau[i, curr_t, curr_k]
                    next_k = parent_k[i, curr_t, curr_k]
                    if next_t < 0:
                        valid = False
                        break
                    curr_t = next_t
                    curr_k = next_k

            if valid:
                # Xử lý C7 Interpolation: nếu có sự kiện bị miss, nội suy frame hợp lý nhất
                if missing_events:
                    for m_ev in missing_events:
                        exp_gap = 5.0
                        if temporal_cues and m_ev - 1 >= 0 and m_ev - 1 < len(temporal_cues):
                            exp_gap = float(temporal_cues[m_ev - 1].get("expected_gap_sec", 5.0))

                        if 0 < m_ev < num_events - 1:
                            t_prev = kf_timestamps[chain_indices[m_ev - 1]]
                            t_next = kf_timestamps[chain_indices[m_ev + 1]]
                            t_target = min(t_prev + exp_gap, (t_prev + t_next) / 2.0)
                            best_mid_idx = chain_indices[m_ev]
                            best_diff = float("inf")
                            for c_idx in range(chain_indices[m_ev - 1] + 1, chain_indices[m_ev + 1]):
                                diff = abs(kf_timestamps[c_idx] - t_target)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_mid_idx = c_idx
                            chain_indices[m_ev] = best_mid_idx

                chain_cands = [unique_kfs[idx] for idx in chain_indices]
                chain_ts = [kf_timestamps[idx] for idx in chain_indices]
                frame_tuple = tuple(c.frame_id for c in chain_cands)

                is_strictly_monotonic = True
                for k in range(num_events - 1):
                    gap = chain_ts[k + 1] - chain_ts[k]
                    if gap < 0.2 or gap > max_separation_sec:
                        is_strictly_monotonic = False
                        break

                if is_strictly_monotonic and frame_tuple not in seen_frame_tuples:
                    seen_frame_tuples.add(frame_tuple)

                    event_sims = [
                        0.10 if k in missing_events else float(S[k, chain_indices[k]])
                        for k in range(num_events)
                    ]
                    matched_count = sum(1 for k in range(num_events) if k not in missing_events and event_sims[k] > 0.01)
                    avg_sim = sum(event_sims) / float(num_events)
                    consensus_factor = (float(matched_count) / float(num_events)) ** 2
                    span_seconds = float(chain_ts[-1] - chain_ts[0])
                    span_penalty = curr_lambda * span_seconds

                    t_first = float(chain_ts[0])
                    t_max_vid = float(kf_timestamps[-1]) if len(kf_timestamps) > 0 else 1.0
                    earliest_bonus = earliest_bias * max(0.0, 1.0 - (t_first / (t_max_vid + 1e-5)))

                    final_chain_score = (avg_sim * consensus_factor) - span_penalty + earliest_bonus

                    chain_cands_cloned = [
                        cand.model_copy(update={"score": round(event_sims[k], 4)})
                        for k, cand in enumerate(chain_cands)
                    ]

                    valid_chains.append((final_chain_score, chain_cands_cloned))
                    chains_from_this_video += 1

        # Fallback cho video nếu DP bị miss do threshold: CHỈ chấp nhận nếu các frame liên tiếp nằm trong cửa sổ <= max_separation_sec
        if chains_from_this_video == 0 and T >= num_events:
            for start_idx in range(T - num_events + 1):
                win_indices = list(range(start_idx, start_idx + num_events))
                win_ts = [kf_timestamps[idx] for idx in win_indices]
                if all(0.2 <= win_ts[k + 1] - win_ts[k] <= max_separation_sec for k in range(num_events - 1)):
                    fallback_cands = [unique_kfs[idx] for idx in win_indices]
                    avg_sc = sum(c.score for c in fallback_cands) / num_events
                    fallback_cands_cloned = [
                        cand.model_copy(update={"score": round(cand.score, 4)})
                        for cand in fallback_cands
                    ]
                    valid_chains.append((avg_sc * 0.5, fallback_cands_cloned))
                    break

    # 4. Sắp xếp tất cả chuỗi trên toàn bộ các video theo điểm giảm dần
    valid_chains.sort(key=lambda x: x[0], reverse=True)
    return [chain for _, chain in valid_chains[:max_results]]
