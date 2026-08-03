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
    Căn chỉnh thời gian giữa N sub-events sử dụng thuật toán DANTE Dynamic Programming.

    Thuật toán DANTE giải quyết đồng thời:
      1. Tối ưu độ tương đồng semantic của từng event với keyframe ứng viên.
      2. Ép buộc thứ tự thời gian nghiêm ngặt (event i+1 phải diễn ra sau event i).
      3. Phạt khoảng cách thời gian λ * (t - τ) giữa 2 event liên tiếp.

    Args:
        event_candidates: Danh sách N danh sách CandidateResult.
            event_candidates[i] chứa candidates cho sub-event i.
        max_results: Số lượng chuỗi event tối đa trả về.
        lambda_penalty: Hệ số phạt khoảng cách thời gian λ. None -> tự động nạp từ config.

    Returns:
        list[list[CandidateResult]]: Danh sách các chuỗi N keyframe thỏa mãn thứ tự thời gian,
            mỗi phần tử là 1 chuỗi [frame_1, frame_2, ..., frame_N] trong cùng 1 video.
    """
    if not event_candidates or any(not c for c in event_candidates):
        return []

    num_events = len(event_candidates)
    if num_events < 2:
        # Nếu chỉ có 1 event, wrap từng candidate thành 1 list đơn tử
        return [[c] for c in event_candidates[0][:max_results]]

    if lambda_penalty is None:
        lambda_penalty = _get_lambda_penalty()

    # 1. Group candidates theo video_id
    # video_map: video_id -> dict(event_idx -> dict(frame_id -> CandidateResult))
    video_map: dict[str, dict[int, dict[int, CandidateResult]]] = {}

    for event_idx, candidates in enumerate(event_candidates):
        for cand in candidates:
            v_id = cand.video_id
            if v_id not in video_map:
                video_map[v_id] = {}
            if event_idx not in video_map[v_id]:
                video_map[v_id][event_idx] = {}
            # Nếu 1 keyframe có nhiều candidates (do duplicate), lấy candidate có score lớn nhất
            f_id = cand.frame_id
            if f_id not in video_map[v_id][event_idx] or cand.score > video_map[v_id][event_idx][f_id].score:
                video_map[v_id][event_idx][f_id] = cand

    valid_chains = []

    # 2. Xử lý DP cho từng video
    for v_id, events_dict in video_map.items():
        # Chỉ xét video có ít nhất candidate cho cả N event (hoặc ít nhất event 0 và event N-1)
        # gom tất cả unique keyframe IDs trong video v_id
        all_frames: dict[int, CandidateResult] = {}
        for ev_idx, f_dict in events_dict.items():
            for f_id, cand in f_dict.items():
                if f_id not in all_frames or cand.timestamp_mid < all_frames[f_id].timestamp_mid:
                    all_frames[f_id] = cand

        if not all_frames:
            continue

        # Sắp xếp keyframes theo timestamp_mid (hoặc frame_id) tăng dần
        sorted_keyframes = sorted(all_frames.values(), key=lambda x: (x.timestamp_mid, x.frame_id))
        T = len(sorted_keyframes)

        if T < num_events:
            # Không đủ keyframes để match N events tăng dần
            continue

        # 3. Xây dựng Similarity Matrix S[N, T]
        # S[i, t] = score của candidate cho event i tại keyframe t
        import numpy as np
        S = np.zeros((num_events, T), dtype=np.float32)

        for i in range(num_events):
            if i in events_dict:
                for t, keyframe in enumerate(sorted_keyframes):
                    if keyframe.frame_id in events_dict[i]:
                        S[i, t] = events_dict[i][keyframe.frame_id].score

        # 4. Run DANTE DP với O(N*T) running max
        # DP[i, t] = score tích lũy tốt nhất cho event i tại keyframe t
        DP = np.full((num_events, T), -np.inf, dtype=np.float32)
        parent = np.full((num_events, T), -1, dtype=np.int32)

        # Base case i = 0 (event đầu tiên)
        for t in range(T):
            DP[0, t] = S[0, t]

        # Recurrence i = 1 .. N-1
        for i in range(1, num_events):
            running_max = -np.inf
            best_tau = -1

            for t in range(T):
                # t-1 là vị trí keyframe của event i-1
                if t > 0:
                    val = DP[i - 1, t - 1] + lambda_penalty * (t - 1)
                    if val > running_max:
                        running_max = val
                        best_tau = t - 1

                if running_max > -np.inf:
                    DP[i, t] = S[i, t] + running_max - lambda_penalty * t
                    parent[i, t] = best_tau

        # 5. Lấy điểm DANTE tốt nhất cho video này
        best_t_end = int(np.argmax(DP[num_events - 1]))
        best_score = float(DP[num_events - 1, best_t_end])

        if best_score == -np.inf:
            continue

        # 6. Backtrack tìm chuỗi keyframes
        chain_indices = [0] * num_events
        curr_t = best_t_end
        valid_chain = True

        for i in range(num_events - 1, -1, -1):
            chain_indices[i] = curr_t
            if i > 0:
                curr_t = parent[i, curr_t]
                if curr_t < 0:
                    valid_chain = False
                    break

        if valid_chain:
            chain_candidates = []
            for i, k_idx in enumerate(chain_indices):
                kf = sorted_keyframes[k_idx]
                # Nếu event i có candidate riêng ở keyframe này, dùng candidate đó; nếu không dùng keyframe chung
                cand = events_dict.get(i, {}).get(kf.frame_id, kf)
                chain_candidates.append(cand)

            valid_chains.append((best_score, chain_candidates))

    # 7. Sắp xếp các chuỗi video theo điểm DANTE giảm dần
    valid_chains.sort(key=lambda x: x[0], reverse=True)

    # Trả về top max_results chuỗi
    return [chain for _, chain in valid_chains[:max_results]]

