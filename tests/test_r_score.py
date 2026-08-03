"""
tests/test_r_score.py
======================
Unit tests cho Module Evaluation & Metrics (TASK-EVL-01, TASK-EVL-02, TASK-EVL-03).
"""

from __future__ import annotations

import pytest
from eval.skeletons.metrics.r_score import r_score
from eval.skeletons.metrics.final_score import calculate_recall_at_k, final_score
from eval.skeletons.run_benchmark import evaluate_config, load_benchmark_queries


# =============================================================================
# TASK-EVL-01: r_score Tests
# =============================================================================

def test_r_score_kis_correct():
    pred = {"video_id": "V001", "timestamp": 15.0}
    gt = {"video_id": "V001", "timestamp_start": 10.0, "timestamp_end": 20.0}
    assert r_score(pred, gt, task_type="KIS") == 1.0


def test_r_score_kis_wrong_video():
    pred = {"video_id": "V002", "timestamp": 15.0}
    gt = {"video_id": "V001", "timestamp_start": 10.0, "timestamp_end": 20.0}
    assert r_score(pred, gt, task_type="KIS") == 0.0


def test_r_score_kis_wrong_timestamp():
    pred = {"video_id": "V001", "timestamp": 25.0}
    gt = {"video_id": "V001", "timestamp_start": 10.0, "timestamp_end": 20.0}
    assert r_score(pred, gt, task_type="KIS") == 0.0


def test_r_score_vqa_correct():
    pred = {"video_id": "V001", "timestamp": 15.0, "answer": "Màu đỏ"}
    gt = {"video_id": "V001", "timestamp_start": 10.0, "timestamp_end": 20.0, "answer": "màu đỏ"}
    assert r_score(pred, gt, task_type="VQA") == 1.0


def test_r_score_vqa_wrong_answer():
    pred = {"video_id": "V001", "timestamp": 15.0, "answer": "Màu xanh"}
    gt = {"video_id": "V001", "timestamp_start": 10.0, "timestamp_end": 20.0, "answer": "màu đỏ"}
    assert r_score(pred, gt, task_type="VQA") == 0.0


def test_r_score_trake_partial_match():
    pred = {"video_id": "V001", "timestamps": [12.0, 45.0, 99.0]}
    gt = {
        "video_id": "V001",
        "events": [
            {"start": 10.0, "end": 15.0},
            {"start": 40.0, "end": 50.0},
            {"start": 60.0, "end": 70.0},
        ],
    }
    # Matches 2 out of 3 events -> 2 / 3
    assert pytest.approx(r_score(pred, gt, task_type="TRAKE"), 0.0001) == 2 / 3


# =============================================================================
# TASK-EVL-02: Final Score & Recall@K Tests
# =============================================================================

def test_recall_at_k():
    gt = {"video_id": "V001", "timestamp_start": 10.0, "timestamp_end": 20.0}
    ranked = [
        {"video_id": "V002", "timestamp": 15.0},
        {"video_id": "V001", "timestamp": 15.0},
    ]

    # R@1 is 0.0 because first pred is V002
    assert calculate_recall_at_k(ranked, gt, k=1, task_type="KIS") == 0.0
    # R@5 is 1.0 because second pred is V001
    assert calculate_recall_at_k(ranked, gt, k=5, task_type="KIS") == 1.0


def test_final_score_calculation():
    gt = {"video_id": "V001", "timestamp_start": 10.0, "timestamp_end": 20.0}
    ranked = [
        {"video_id": "V002", "timestamp": 15.0},
        {"video_id": "V001", "timestamp": 15.0},
    ]
    # R@1 = 0, R@5 = 1, R@20 = 1, R@50 = 1, R@100 = 1 -> Mean = 4/5 = 0.8
    score = final_score(ranked, gt, task_type="KIS")
    assert pytest.approx(score, 0.0001) == 0.8


# =============================================================================
# TASK-EVL-03: Evaluate Config Test
# =============================================================================

def test_evaluate_config():
    queries = [
        {
            "task_type": "KIS",
            "ground_truth": {"video_id": "V001", "timestamp_start": 10.0, "timestamp_end": 20.0},
            "predictions": [{"video_id": "V001", "timestamp": 15.0}],
        }
    ]
    res = evaluate_config(queries, {"name": "TestConfig"})
    assert res["mean_final_score"] == 1.0
    assert res["avg_latency_ms"] >= 0.0
