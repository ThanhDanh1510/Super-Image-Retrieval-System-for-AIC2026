"""
serving/api/service/temporal_aligner.py
========================================
Thuat toan Quy Hoach Dong (Dynamic Programming) & Monotonic Sequence Alignment (DANTE Engine)
cho bai toan Temporal Retrieval and Alignment of Key Events (TRAKE).

Nguyen ly hoat dong:
  1. Constraint bat buoc: N keyframe trong moi chuoi (chain) phai cung thuoc ve 1 VIDEO (video_id duy nhat).
  2. Monotonicity Constraint: timestamp_mid(event_1) < timestamp_mid(event_2) < ... < timestamp_mid(event_N).
  3. Min Temporal Separation: Khoang cach giua 2 su kien lien tiep phai dat toi thieu delta (mac dinh 1.0s).
  4. Time-Aware Penalty: Phat tuyen tinh theo do tre thoi gian thuc (seconds) thay vi chi dem so frame.
  5. All-Event Consensus Scoring: Uu tien cac video co do bao phu 100% cac su kien.
"""

from __future__ import annotations

import logging
import numpy as np
from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)


def _get_lambda_penalty() -> float:
    """Doc lambda penalty factor tu fusion_config.yaml (mac dinh 0.001)."""
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
    min_separation_sec: float = 1.0,
    max_chains_per_video: int = 3,
) -> list[list[CandidateResult]]:
    """
    Can chinh chuoi N su kien tuan tu trong video su dung Quy hoach dong (DANTE).

    Args:
        event_candidates: Danh sach N danh sach CandidateResult (moi danh sach ung voi 1 sub-event).
        max_results: So luong chuoi toi da can tra ve.
        lambda_penalty: He so phat khoang cach thoi gian (None -> load tu config).
        min_separation_sec: Khoang cach thoi gian toi thieu giua 2 su kien lien tiep (giay).
        max_chains_per_video: So luong chuoi toi da duoc trich xuat tu moi video.

    Returns:
        list[list[CandidateResult]]: Danh sach cac N-tuples keyframe duoc sap xep giam dan theo diem DP.
    """
    if not event_candidates or any(not c for c in event_candidates):
        return []

    num_events = len(event_candidates)
    if num_events < 2:
        return [[c] for c in event_candidates[0][:max_results]]

    if lambda_penalty is None:
        lambda_penalty = _get_lambda_penalty()

    # 1. Gom nhom candidates theo tung video_id
    video_map: dict[str, dict[int, list[CandidateResult]]] = {}

    for event_idx, candidates in enumerate(event_candidates):
        for cand in candidates:
            v_id = cand.video_id
            if v_id not in video_map:
                video_map[v_id] = {ev: [] for ev in range(num_events)}
            video_map[v_id][event_idx].append(cand)

    valid_chains: list[tuple[float, list[CandidateResult]]] = []

    # 2. Xu ly quy hoach dong rieng biet cho tung video
    for v_id, events_dict in video_map.items():
        all_cands_in_video: list[CandidateResult] = []
        for ev_idx in range(num_events):
            all_cands_in_video.extend(events_dict[ev_idx])

        if not all_cands_in_video:
            continue

        # Sap xep toan bo keyframe trong video theo thoi gian thuc
        sorted_kfs = sorted(all_cands_in_video, key=lambda c: (c.timestamp_mid, c.frame_id))

        # Loai bo trung lap gan nhau (< 0.2s) va giu candidate diem cao nhat
        unique_kfs: list[CandidateResult] = []
        seen_timestamps: list[float] = []
        for kf in sorted_kfs:
            ts = float(kf.timestamp_mid if kf.timestamp_mid is not None else (kf.frame_id / 25.0))
            if not any(abs(ts - prev_ts) < 0.2 for prev_ts in seen_timestamps):
                seen_timestamps.append(ts)
                unique_kfs.append(kf)

        T = len(unique_kfs)
        if T < num_events:
            continue

        # Timestamps vector in seconds
        kf_timestamps = np.array(
            [float(c.timestamp_mid if c.timestamp_mid is not None else (c.frame_id / 25.0)) for c in unique_kfs],
            dtype=np.float32,
        )

        # Ma tran tuong dong S[num_events, T]
        S = np.zeros((num_events, T), dtype=np.float32)
        for i in range(num_events):
            ev_frame_scores = {c.frame_id: c.score for c in events_dict[i]}
            for t, kf in enumerate(unique_kfs):
                S[i, t] = ev_frame_scores.get(kf.frame_id, 0.0)

        # Bang quy hoach dong DP[num_events, T] va parent pointer
        DP = np.full((num_events, T), -np.inf, dtype=np.float32)
        parent = np.full((num_events, T), -1, dtype=np.int32)

        # Base case: event dau tien (i = 0)
        DP[0, :] = S[0, :]

        # DP Transitions: DP[i, t] = S[i, t] + max_{tau} (DP[i-1, tau] - lambda * (t_sec - tau_sec))
        for i in range(1, num_events):
            running_max = -np.inf
            best_tau = -1
            tau_ptr = 0

            for t in range(T):
                cur_time = kf_timestamps[t]

                # Cap nhat running_max voi cac tau thoa man min_separation_sec
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

        # 3. Backtrack tim top chuoi tot nhat tu hang cuoi cung DP[num_events - 1]
        terminal_scores = DP[num_events - 1]
        sorted_terminals = np.argsort(terminal_scores)[::-1]

        chains_from_this_video = 0
        seen_frame_tuples: set[tuple[int, ...]] = set()

        for t_end in sorted_terminals:
            if chains_from_this_video >= max_chains_per_video:
                break

            if terminal_scores[t_end] <= -np.inf:
                continue

            # Truy vet nguoc (Backtracking)
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

                # Kiem tra don dieu nghiem ngat & khong trung lap chuoi
                is_strictly_monotonic = all(chain_ts[k] < chain_ts[k + 1] for k in range(num_events - 1))
                if is_strictly_monotonic and frame_tuple not in seen_frame_tuples:
                    seen_frame_tuples.add(frame_tuple)

                    # Tinh diem dong thuan (All-Event Consensus & Temporal Span Penalty)
                    event_sims = [float(S[k, chain_indices[k]]) for k in range(num_events)]
                    matched_count = sum(1 for s in event_sims if s > 0.01)
                    avg_sim = sum(event_sims) / float(num_events)
                    consensus_factor = (float(matched_count) / float(num_events)) ** 2
                    span_seconds = float(chain_ts[-1] - chain_ts[0])
                    span_penalty = lambda_penalty * span_seconds

                    final_chain_score = (avg_sim * consensus_factor) - span_penalty

                    # Gan diem cho tung frame de UI hien thi
                    for k, cand in enumerate(chain_cands):
                        cand.score = round(event_sims[k], 4)

                    valid_chains.append((final_chain_score, chain_cands))
                    chains_from_this_video += 1

        # Fallback cho video neu DP bi miss do threshold
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

    # 4. Sap xep tat ca chuoi tren toan bo cac video theo diem giam dan
    valid_chains.sort(key=lambda x: x[0], reverse=True)
    return [chain for _, chain in valid_chains[:max_results]]
