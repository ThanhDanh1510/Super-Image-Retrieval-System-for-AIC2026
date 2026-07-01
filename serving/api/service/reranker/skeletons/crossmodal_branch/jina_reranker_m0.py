"""
serving/api/service/reranker/skeletons/crossmodal_branch/jina_reranker_m0.py
============================================================================
Nhánh rerank đa phương thức (multimodal) sử dụng Jina Reranker M0 (Skeleton).

Verify sự phù hợp giữa query và ảnh keyframe để lọc false positive.

CONSTRAINTS:
  - Lazy load: Khởi tạo model lần đầu khi gọi hàm, cache lại. KHÔNG khởi tạo ở module level.
  - Batch inference: Xử lý toàn bộ candidates trong 1 forward pass. KHÔNG loop từng candidate.
  - Ảnh keyframe được load dựa trên candidate.keyframe_path từ thư mục chứa data keyframe.
"""

from __future__ import annotations

import logging
from typing import Any
from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)

# Cache cho model
_MODEL_CACHE: dict[str, Any] = {}

def score(
    query: str,
    candidates: list[CandidateResult],
    keyframes_root_dir: str,
) -> list[float]:
    """
    Tính điểm tương đồng giữa text query và image keyframe tương ứng của từng candidate.

    Args:
        query: Query text (tiếng Anh).
        candidates: Danh sách CandidateResult cần rerank.
        keyframes_root_dir: Thư mục chứa toàn bộ ảnh keyframe của hệ thống.

    Returns:
        List[float]: Danh sách điểm số (cùng kích thước và thứ tự với candidates).
    """
    raise NotImplementedError
