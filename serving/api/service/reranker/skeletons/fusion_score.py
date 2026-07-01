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
    raise NotImplementedError
