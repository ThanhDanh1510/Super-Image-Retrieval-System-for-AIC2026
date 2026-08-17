"""
serving/api/service/reranker/skeletons/fusion_score.py
======================================================
Hợp nhất kết quả từ nhánh text_branch và crossmodal_branch (Skeleton).

Hợp nhất tuyến tính (Linear Fusion) hoặc Reciprocal Rank Fusion (RRF).
Trọng số alpha và phương pháp fusion được cấu hình trongconfigs/fusion_config.yaml.
"""

from __future__ import annotations

import logging
from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)

def _normalize(scores: list[float]) -> list[float]:
    """Min-Max Normalization về khoảng [0.0, 1.0]."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def fuse(
    text_scores: list[float],
    crossmodal_scores: list[float],
    candidates: list[CandidateResult],
    alpha: float | None = None,
    fusion_cfg: dict | None = None,
) -> list[CandidateResult]:
    """
    Hợp nhất điểm số và sắp xếp lại danh sách candidates.

    Nếu alpha không truyền vào, đọc từ fusion_cfg hoặc configs/fusion_config.yaml.
    Công thức Linear Fusion:
        fused_score = alpha * text_score + (1 - alpha) * crossmodal_score

    Args:
        text_scores: Điểm số từ nhánh văn bản.
        crossmodal_scores: Điểm số từ nhánh đa phương thức.
        candidates: Danh sách CandidateResult ban đầu.
        alpha: Trọng số cho text branch. None -> tự động load.
        fusion_cfg: Cấu hình fusion. None -> tự động load.

    Returns:
        list[CandidateResult]: Danh sách candidates đã được sắp xếp lại theo fused_score giảm dần.
    """
    if not candidates:
        return []

    # 1. Đọc alpha và config nếu chưa truyền
    if alpha is None:
        if fusion_cfg and "reranker" in fusion_cfg:
            alpha = float(fusion_cfg["reranker"].get("alpha", 0.6))
        else:
            try:
                from shared.utils.config_loader import load_fusion_config
                cfg = load_fusion_config()
                alpha = float(cfg.get("reranker", {}).get("alpha", 0.6))
            except Exception:
                alpha = 0.6

    # 2. Xử lý khớp length và điểm số từng nhánh
    n = len(candidates)
    if text_scores is not None and crossmodal_scores is not None:
        t_norm = _normalize(list(text_scores) if len(text_scores) == n else [c.score for c in candidates])
        c_norm = _normalize(list(crossmodal_scores) if len(crossmodal_scores) == n else [c.score for c in candidates])
        fused_candidates = []
        for i, cand in enumerate(candidates):
            fused_val = alpha * t_norm[i] + (1.0 - alpha) * c_norm[i]
            clamped_score = round(max(0.0, min(1.0, float(fused_val))), 6)
            fused_candidates.append(cand.model_copy(update={"score": clamped_score}))
    elif text_scores is not None:
        t_norm = _normalize(list(text_scores) if len(text_scores) == n else [c.score for c in candidates])
        fused_candidates = []
        for i, cand in enumerate(candidates):
            clamped_score = round(max(0.0, min(1.0, float(t_norm[i]))), 6)
            fused_candidates.append(cand.model_copy(update={"score": clamped_score}))
    elif crossmodal_scores is not None:
        c_norm = _normalize(list(crossmodal_scores) if len(crossmodal_scores) == n else [c.score for c in candidates])
        fused_candidates = []
        for i, cand in enumerate(candidates):
            clamped_score = round(max(0.0, min(1.0, float(c_norm[i]))), 6)
            fused_candidates.append(cand.model_copy(update={"score": clamped_score}))
    else:
        return candidates

    # 5. Sắp xếp giảm dần theo fused score
    fused_candidates.sort(key=lambda x: x.score, reverse=True)
    return fused_candidates

