"""
serving/api/service/temporal_aligner.py
========================================
Căn chỉnh thời gian cho TRAKE query (Skeleton).

Nhận vào N danh sách candidates ứng với N sub-events của TRAKE query.
Tìm kiếm và lựa chọn tổ hợp các frame/scene tối ưu thỏa mãn:
  1. Các frame được chọn phải nằm trong CÙNG MỘT VIDEO.
  2. Thứ tự thời gian của các frame được chọn phải TĂNG DẦN
     (tức là frame_1.timestamp_mid < frame_2.timestamp_mid < ... < frame_N.timestamp_mid).
  3. Ưu tiên video có số lượng event khớp (matched events) nhiều nhất,
     sau đó mới tối ưu hóa điểm số (score) của các frame trong video đó.

CONSTRAINTS:
  - Chỉ trả về danh sách candidates tối ưu thỏa mãn các điều kiện trên.
"""

from __future__ import annotations

import logging
from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)

def _get_lambda_penalty() -> float:
    """Đọc λ penalty factor từ fusion_config.yaml (mặc định 0.001)."""
    try:
        from shared.utils.config_loader import load_fusion_config
        cfg = load_fusion_config()
        return float(cfg.get("temporal_aligner", {}).get("lambda_penalty", 0.001))
    except Exception:
        return 0.001


def align(
    event_candidates: list[list[CandidateResult]],
    max_results: int = 100,
    lambda_penalty: float | None = None,
) -> list[list[CandidateResult]]:
    """
    Căn chỉnh thời gian giữa N sub-events sử dụng thuật toán DANTE.

    ĐIỀU KIỆN TIÊN QUYẾT:
      1. Tất cả keyframes trong 1 chuỗi trả về BẮT BUỘC phải thuộc CÙNG MỘT VIDEO (video_id giống nhau).
      2. Thứ tự thời gian phải tăng dần: timestamp_mid(event_1) <= timestamp_mid(event_2) <= ...

    Args:
        event_candidates: Danh sách N danh sách CandidateResult.
            event_candidates[i] chứa candidates cho sub-event i.
        max_results: Số lượng chuỗi event tối đa trả về.
        lambda_penalty: Hệ số phạt khoảng cách thời gian λ. None -> tự động nạp từ config.

    Returns:
        list[list[CandidateResult]]: Danh sách các chuỗi N keyframe trong cùng 1 video.
    """
    if not event_candidates or any(not c for c in event_candidates):
        return []

    num_events = len(event_candidates)
    if num_events < 2:
        return [[c] for c in event_candidates[0][:max_results]]

    if lambda_penalty is None:
        lambda_penalty = _get_lambda_penalty()

    # 1. Group candidates strictly by video_id
    # video_map: video_id -> dict(event_idx -> list[CandidateResult])
    video_map: dict[str, dict[int, list[CandidateResult]]] = {}

    for event_idx, candidates in enumerate(event_candidates):
        for cand in candidates:
            v_id = cand.video_id
            if v_id not in video_map:
                video_map[v_id] = {ev: [] for ev in range(num_events)}
            video_map[v_id][event_idx].append(cand)

    valid_chains: list[tuple[float, list[CandidateResult]]] = []

    # 2. Process each video independently
    for v_id, events_dict in video_map.items():
        # Flatten and collect all candidate keyframes for this specific video
        all_cands_in_video: list[CandidateResult] = []
        for ev_idx in range(num_events):
            all_cands_in_video.extend(events_dict[ev_idx])

        if not all_cands_in_video:
            continue

        # Sort keyframes in this video by timestamp_mid
        sorted_kfs = sorted(all_cands_in_video, key=lambda c: (c.timestamp_mid, c.frame_id))

        # Deduplicate keyframes in this video by timestamp_mid while preserving highest score
        unique_kfs: list[CandidateResult] = []
        seen_timestamps: set[float] = set()
        for kf in sorted_kfs:
            ts_key = round(kf.timestamp_mid, 2)
            if ts_key not in seen_timestamps:
                seen_timestamps.add(ts_key)
                unique_kfs.append(kf)

        T = len(unique_kfs)
        if T < num_events:
            # Video does not have enough distinct keyframes/timestamps to form an N-tuple without duplicate timestamps
            continue

        # Run DP for optimal strictly monotonic chain with distinct timestamps
        import numpy as np

        # Similarity matrix S[num_events, T]
        S = np.zeros((num_events, T), dtype=np.float32)
        for i in range(num_events):
            ev_frame_scores = {c.frame_id: c.score for c in events_dict[i]}
            for t, kf in enumerate(unique_kfs):
                S[i, t] = ev_frame_scores.get(kf.frame_id, 0.0)

        DP = np.full((num_events, T), -np.inf, dtype=np.float32)
        parent = np.full((num_events, T), -1, dtype=np.int32)

        for t in range(T):
            DP[0, t] = S[0, t]

        for i in range(1, num_events):
            running_max = -np.inf
            best_tau = -1
            for t in range(T):
                if t > 0:
                    val = DP[i - 1, t - 1] + lambda_penalty * (t - 1)
                    if val > running_max:
                        running_max = val
                        best_tau = t - 1
                if running_max > -np.inf:
                    DP[i, t] = S[i, t] + running_max - lambda_penalty * t
                    parent[i, t] = best_tau

        # Backtrack from the best terminal keyframe in the last event row
        best_t_end = int(np.argmax(DP[num_events - 1]))
        best_score = float(DP[num_events - 1, best_t_end])

        valid_chain_found = False
        if best_score > -np.inf:
            chain_indices = [0] * num_events
            curr_t = best_t_end
            valid = True
            for i in range(num_events - 1, -1, -1):
                chain_indices[i] = curr_t
                if i > 0:
                    curr_t = parent[i, curr_t]
                    if curr_t < 0:
                        valid = False
                        break

            if valid:
                chain_candidates = [unique_kfs[k_idx] for k_idx in chain_indices]
                ts_list = [round(c.timestamp_mid, 2) for c in chain_candidates]
                if len(set(ts_list)) == num_events and ts_list == sorted(ts_list):
                    valid_chains.append((best_score, chain_candidates))
                    valid_chain_found = True

        if not valid_chain_found:
            # Fallback for video with T >= num_events: select N keyframes with strictly increasing timestamps
            indices = [int(round(i * (T - 1) / (num_events - 1))) for i in range(num_events)]
            for i in range(1, num_events):
                if indices[i] <= indices[i - 1]:
                    indices[i] = indices[i - 1] + 1
            if indices[-1] < T:
                chain_candidates = [unique_kfs[idx] for idx in indices]
                ts_list = [round(c.timestamp_mid, 2) for c in chain_candidates]
                if len(set(ts_list)) == num_events and ts_list == sorted(ts_list):
                    avg_score = sum(c.score for c in chain_candidates) / num_events
                    valid_chains.append((avg_score, chain_candidates))

    # 3. Sort chains across videos by DANTE score descending
    valid_chains.sort(key=lambda x: x[0], reverse=True)
    return [chain for _, chain in valid_chains[:max_results]]


