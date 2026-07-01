"""
eval/skeletons/run_benchmark.py
================================
Script chạy benchmark so sánh chất lượng giữa các cấu hình khác nhau (Skeleton).

Cho phép đo đạc:
  - Latency trung bình mỗi query (ms)
  - Accuracy (Mean Final Score) trên tập test queries

Đọc tập queries và ground truth từ:
  - eval/queries/sample_queries_{task_type}.json

In ra bảng so sánh Markdown hoặc DataFrame.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

def load_benchmark_queries(queries_path: str) -> list[dict[str, Any]]:
    """Đọc file json chứa các câu hỏi test và ground truth tương ứng."""
    raise NotImplementedError

def evaluate_config(
    queries: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
) -> dict[str, float]:
    """
    Chạy thử nghiệm hệ thống với một cấu hình cụ thể và trả về các metrics.

    Metrics cần đo:
      - mean_final_score: Mean Final Score của tất cả queries.
      - avg_latency_ms: Thời gian xử lý trung bình mỗi query (ms).

    Args:
        queries: Danh sách các benchmark queries.
        pipeline_config: Cấu hình hệ thống (alpha, reranker_enabled, v.v.)

    Returns:
        Dict chứa kết quả đánh giá, ví dụ:
        {'mean_final_score': 0.765, 'avg_latency_ms': 120.4}
    """
    raise NotImplementedError

def run_benchmark(
    queries_dir: str,
    configs_to_test: list[dict[str, Any]],
) -> None:
    """
    Đầu vào chính để so sánh các cấu hình và in ra bảng so sánh.

    Args:
        queries_dir: Thư mục chứa các file json queries mẫu.
        configs_to_test: Danh sách các cấu hình cần benchmark.
    """
    raise NotImplementedError
