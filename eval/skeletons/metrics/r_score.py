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
            Ví dụ KIS: {'video_id': 'L01_V001', 'timestamp': 45.5}
            Ví dụ VQA: {'video_id': 'L01_V001', 'timestamp': 45.5, 'answer': 'màu xanh'}
            Ví dụ TRAKE: {'video_id': 'L01_V001', 'timestamps': [12.0, 45.5, 90.0]}
        ground_truth: Dict chứa ground truth tương ứng.
            Ví dụ KIS: {'video_id': 'L01_V001', 'timestamp_start': 40.0, 'timestamp_end': 50.0}
            Ví dụ VQA: {'video_id': 'L01_V001', 'timestamp_start': 40.0, 'timestamp_end': 50.0, 'answer': 'màu xanh'}
            Ví dụ TRAKE: {'video_id': 'L01_V001', 'events': [{'start': 10.0, 'end': 15.0}, {'start': 40.0, 'end': 50.0}]}
        task_type: Loại task "KIS" | "VQA" | "TRAKE".

    Returns:
        float: Điểm số r_score trong khoảng [0.0, 1.0].
    """
    raise NotImplementedError
