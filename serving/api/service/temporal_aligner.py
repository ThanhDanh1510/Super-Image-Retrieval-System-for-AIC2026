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

def align(
    event_candidates: list[list[CandidateResult]],
    max_results: int = 100,
) -> list[CandidateResult]:
    """
    Căn chỉnh thời gian giữa các sub-events để tìm ra chuỗi sự kiện hợp lệ trong cùng 1 video.

    Args:
        event_candidates: Danh sách các danh sách candidates.
            event_candidates[i] chứa danh sách CandidateResult trả về cho sub-event thứ i.
        max_results: Số lượng kết quả chuỗi tối đa cần trả về.

    Returns:
        list[CandidateResult]: Danh sách candidates đã được căn chỉnh và sắp xếp.
            Mỗi phần tử trong danh sách này đại diện cho một chuỗi sự kiện hợp lệ,
            hoặc danh sách các keyframes ứng với chuỗi sự kiện tối ưu được chọn.
    """
    raise NotImplementedError
