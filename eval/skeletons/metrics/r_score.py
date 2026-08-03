"""
eval/skeletons/metrics/r_score.py
==================================
Tính toán r_score cho từng loại task (KIS / VQA / TRAKE) (Skeleton).

CONSTRAINTS (CÔNG THỨC CHẤM ĐIỂM BẮT BUỘC):
  - KIS:
    r_score = 1.0 nếu (pred.video_id == gt.video_id) AND (gt.timestamp_start <= pred.timestamp <= gt.timestamp_end)
    Ngược lại = 0.0
    
  - VQA:
    r_score = 1.0 nếu đồng thời:
      1. pred.video_id == gt.video_id
      2. gt.timestamp_start <= pred.timestamp <= gt.timestamp_end
      3. pred.answer == gt.answer (so sánh text chính xác, lowercase + strip)
    Ngược lại = 0.0
    
  - TRAKE (Multi-event):
    Nếu pred.video_id != gt.video_id:
      r_score = 0.0
    Nếu đúng video_id:
      r_score = (1 / N) * tổng_số_event_đúng
      Trong đó N là số lượng sub-events (events) trong ground truth.
      Một event thứ j được coi là đúng nếu: gt.events[j].start <= pred.events[j].timestamp <= gt.events[j].end
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

def r_score(
    prediction: dict[str, Any],
    ground_truth: dict[str, Any],
    task_type: Literal["KIS", "VQA", "TRAKE"],
) -> float:
    """
    Tính điểm r_score cho một dự đoán đơn lẻ dựa trên ground truth.

    Args:
        prediction: Dict chứa kết quả dự đoán của mô hình.
        ground_truth: Dict chứa ground truth tương ứng.
        task_type: Loại task "KIS" | "VQA" | "TRAKE".

    Returns:
        float: Điểm số r_score trong khoảng [0.0, 1.0].
    """
    if not prediction or not ground_truth:
        return 0.0

    pred_video = str(prediction.get("video_id", "") or "").strip()
    gt_video = str(ground_truth.get("video_id", "") or "").strip()

    if not pred_video or not gt_video or pred_video != gt_video:
        return 0.0

    task = str(task_type).upper()

    if task == "KIS":
        pred_ts = prediction.get("timestamp")
        if pred_ts is None:
            pred_ts = prediction.get("timestamp_mid", 0.0)

        gt_start = float(ground_truth.get("timestamp_start", 0.0))
        gt_end = float(ground_truth.get("timestamp_end", float("inf")))

        if gt_start <= float(pred_ts) <= gt_end:
            return 1.0
        return 0.0

    elif task == "VQA":
        pred_ts = prediction.get("timestamp")
        if pred_ts is None:
            pred_ts = prediction.get("timestamp_mid", 0.0)

        gt_start = float(ground_truth.get("timestamp_start", 0.0))
        gt_end = float(ground_truth.get("timestamp_end", float("inf")))
        ts_valid = (gt_start <= float(pred_ts) <= gt_end)

        pred_ans = str(prediction.get("answer", "") or "").strip().lower()
        gt_ans = str(ground_truth.get("answer", "") or "").strip().lower()
        ans_valid = (pred_ans == gt_ans) and bool(gt_ans)

        if ts_valid and ans_valid:
            return 1.0
        return 0.0

    elif task == "TRAKE":
        gt_events = ground_truth.get("events", [])
        if not gt_events:
            return 0.0

        n_events = len(gt_events)

        pred_ts_list = prediction.get("timestamps")
        if pred_ts_list is None:
            pred_ts_list = prediction.get("sub_events_timestamps")
        if pred_ts_list is None:
            pred_events = prediction.get("events", [])
            if isinstance(pred_events, list):
                pred_ts_list = [
                    e.get("timestamp", e.get("timestamp_mid", 0.0)) if isinstance(e, dict) else float(e)
                    for e in pred_events
                ]

        if not pred_ts_list:
            return 0.0

        matched_count = 0
        for j in range(min(n_events, len(pred_ts_list))):
            ev = gt_events[j]
            ts = float(pred_ts_list[j])

            if isinstance(ev, dict):
                start = float(ev.get("start", ev.get("timestamp_start", 0.0)))
                end = float(ev.get("end", ev.get("timestamp_end", float("inf"))))
            elif isinstance(ev, (tuple, list)) and len(ev) >= 2:
                start, end = float(ev[0]), float(ev[1])
            else:
                continue

            if start <= ts <= end:
                matched_count += 1

        return float(matched_count / n_events)

    else:
        logger.warning(f"Unsupported task_type: {task_type}")
        return 0.0
