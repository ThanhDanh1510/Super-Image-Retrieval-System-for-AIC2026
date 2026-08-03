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

import json
import os
import time
from pathlib import Path
from eval.skeletons.metrics.final_score import final_score


def load_benchmark_queries(queries_path: str) -> list[dict[str, Any]]:
    """Đọc file json chứa các câu hỏi test và ground truth tương ứng."""
    path = Path(queries_path)
    if not path.exists():
        logger.warning(f"File queries không tồn tại: {queries_path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "queries" in data:
                return data["queries"]
            return []
    except Exception as e:
        logger.error(f"Lỗi khi đọc benchmark queries từ {queries_path}: {e}")
        return []


def evaluate_config(
    queries: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
) -> dict[str, float]:
    """
    Chạy thử nghiệm hệ thống với một cấu hình cụ thể và trả về các metrics.

    Args:
        queries: Danh sách các benchmark queries.
        pipeline_config: Cấu hình hệ thống (name, alpha, reranker_enabled, v.v.)

    Returns:
        Dict chứa kết quả đánh giá: {'mean_final_score': float, 'avg_latency_ms': float}
    """
    if not queries:
        return {"mean_final_score": 0.0, "avg_latency_ms": 0.0}

    total_score = 0.0
    total_time_ms = 0.0

    mock_search_func = pipeline_config.get("search_func")

    for q in queries:
        gt = q.get("ground_truth", {})
        task_type = q.get("task_type", "KIS")

        start_t = time.perf_counter()
        if callable(mock_search_func):
            ranked_preds = mock_search_func(q, pipeline_config)
        else:
            # Fallback mock predictions
            ranked_preds = q.get("mock_predictions", q.get("predictions", []))
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        q_score = final_score(ranked_preds, gt, task_type=task_type)

        total_score += q_score
        total_time_ms += elapsed_ms

    n = len(queries)
    return {
        "mean_final_score": round(total_score / n, 4),
        "avg_latency_ms": round(total_time_ms / n, 2),
    }


def run_benchmark(
    queries_dir: str,
    configs_to_test: list[dict[str, Any]],
) -> None:
    """
    Đầu vào chính để so sánh các cấu hình và in ra bảng so sánh Markdown.

    Args:
        queries_dir: Thư mục chứa các file json queries mẫu.
        configs_to_test: Danh sách các cấu hình cần benchmark.
    """
    q_dir = Path(queries_dir)
    all_queries = []

    if q_dir.exists():
        for q_file in q_dir.glob("*.json"):
            all_queries.extend(load_benchmark_queries(str(q_file)))

    if not all_queries:
        logger.warning(f"Không tìm thấy query nào trong thư mục {queries_dir}. Tạo sample queries giả lập.")
        all_queries = [
            {
                "task_type": "KIS",
                "ground_truth": {"video_id": "V001", "timestamp_start": 10.0, "timestamp_end": 20.0},
                "predictions": [{"video_id": "V001", "timestamp": 15.0}],
            }
        ]

    print("\n=======================================================")
    print("           BENCHMARK EVALUATION RESULTS                ")
    print("=======================================================\n")
    print("| Config Name | Mean Final Score | Avg Latency (ms) |")
    print("|-------------|------------------|------------------|")

    for cfg in configs_to_test:
        cfg_name = cfg.get("name", "Unnamed Config")
        res = evaluate_config(all_queries, cfg)
        print(f"| {cfg_name:<11} | {res['mean_final_score']:<16.4f} | {res['avg_latency_ms']:<16.2f} |")

    print("\n=======================================================\n")
