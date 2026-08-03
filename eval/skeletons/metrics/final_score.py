"""
eval/skeletons/metrics/final_score.py
======================================
Tính toán Final Score cho một danh sách kết quả được xếp hạng (Ranked List) (Skeleton).

Hệ thống đánh giá dựa trên Recall tại các mức k (R@k).
Metric chính thức:
  final_score = mean(R@1, R@5, R@20, R@50, R@100)
  Với R@k = max(r_score(pred) cho tất cả pred trong ranked_list[:k])
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

from eval.skeletons.metrics.r_score import r_score


def calculate_recall_at_k(
    ranked_list: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    k: int,
    task_type: Literal["KIS", "VQA", "TRAKE"],
) -> float:
    """
    Tính R@k cho một query.

    R@k là điểm r_score cao nhất đạt được trong top-k dự đoán đầu tiên.

    Args:
        ranked_list: Danh sách các dự đoán được xếp hạng giảm dần theo score.
        ground_truth: Ground truth của query.
        k: Mức lấy top.
        task_type: Loại task.

    Returns:
        float: Giá trị R@k (từ 0.0 đến 1.0).
    """
    if not ranked_list or k <= 0:
        return 0.0

    top_k_preds = ranked_list[:k]
    scores = [r_score(pred, ground_truth, task_type) for pred in top_k_preds]
    return max(scores, default=0.0)


def final_score(
    ranked_list: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    task_type: Literal["KIS", "VQA", "TRAKE"],
) -> float:
    """
    Tính điểm trung bình final_score cho một query.

    final_score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5

    Args:
        ranked_list: Danh sách kết quả dự đoán của hệ thống (đã được xếp hạng).
        ground_truth: Ground truth tương ứng.
        task_type: Loại task.

    Returns:
        float: Final score.
    """
    k_list = [1, 5, 20, 50, 100]
    recalls = [calculate_recall_at_k(ranked_list, ground_truth, k, task_type) for k in k_list]
    return sum(recalls) / len(recalls)
