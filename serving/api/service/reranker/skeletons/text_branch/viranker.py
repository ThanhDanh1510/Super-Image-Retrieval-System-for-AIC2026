"""
serving/api/service/reranker/skeletons/text_branch/viranker.py
==============================================================
Nhánh rerank văn bản sử dụng ViRanker (BGE-M3 / MMARCO-VI tuned) (Skeleton).

CONSTRAINTS:
  - Lazy load: Khởi tạo model lần đầu khi gọi hàm, cache lại. KHÔNG khởi tạo ở module level.
  - Batch inference: Xử lý toàn bộ candidates trong 1 forward pass. KHÔNG loop từng candidate.
"""

from __future__ import annotations

import logging
from typing import Any
from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)

# Cache cho model và tokenizer
_MODEL_CACHE: dict[str, Any] = {}

def score(query: str, candidates: list[CandidateResult]) -> list[float]:
    """
    Tính điểm tương đồng ngữ nghĩa giữa query và ocr_text + asr_text của từng candidate.

    Args:
        query: Query text tiếng Anh hoặc tiếng Việt đã chuẩn hóa.
        candidates: Danh sách CandidateResult cần rerank.

    Returns:
        List[float]: Danh sách điểm số (cùng kích thước và thứ tự với candidates).
    """
    raise NotImplementedError
