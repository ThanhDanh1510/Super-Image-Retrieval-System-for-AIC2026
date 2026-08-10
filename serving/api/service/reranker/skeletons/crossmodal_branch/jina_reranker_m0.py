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

def _get_model():
    """Lazy load model Jina Reranker v1 (hoặc M0) vào cache."""
    if "model" not in _MODEL_CACHE:
        try:
            import torch
            from transformers import AutoModel

            model_name = "jinaai/jina-reranker-m0"
            logger.info(f"Loading Jina Reranker M0 model: {model_name} on CPU...")
            device = "cpu"
            dtype = torch.float32
            import transformers
            orig_verbosity = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()
            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
            finally:
                transformers.logging.set_verbosity(orig_verbosity)
            model.to(device)
            model = model.to(dtype)
            model.eval()

            _MODEL_CACHE["model"] = model
            _MODEL_CACHE["device"] = device
        except Exception as e:
            logger.warning(f"Không thể load model Jina Reranker M0 ({e}). Dùng fallback score.")
            _MODEL_CACHE["mock"] = True

    return _MODEL_CACHE


def _resolve_img_path(c: CandidateResult, keyframes_root_dir: str = "") -> Path | None:
    from pathlib import Path

    data_dir = Path(keyframes_root_dir) if keyframes_root_dir else Path("data")
    video_id = c.video_id
    frame_id = c.frame_id
    keyframe_path = c.keyframe_path
    frame_id_str = str(frame_id)

    parts = video_id.split("_")
    if len(parts) == 2:
        grp, vsub = parts[0], parts[1]
    else:
        grp, vsub = video_id, video_id

    candidates = [
        data_dir / "image" / grp / grp / vsub / f"{frame_id_str}.webp",
        data_dir / "image" / grp / grp / vsub / f"{frame_id_str}.jpg",
        data_dir / "image" / grp / f"{video_id}" / f"{frame_id_str}.webp",
        data_dir / "image" / f"{video_id}" / f"{frame_id_str}.webp",
        data_dir / "unified" / "keyframes" / keyframe_path if keyframe_path else None,
        data_dir / keyframe_path if keyframe_path else None,
        Path(keyframe_path) if keyframe_path else None,
    ]
    for p in candidates:
        if p and p.exists() and p.is_file():
            return p
    return None


def score(
    query: str,
    candidates: list[CandidateResult],
    keyframes_root_dir: str = "",
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
    if not candidates:
        return []

    cache = _get_model()

    if cache.get("mock", False):
        # Fallback cho unit test / offline
        return [float(c.score) for c in candidates]

    try:
        from pathlib import Path
        from PIL import Image

        model = cache["model"]

        # Chuẩn bị ảnh và text pairs
        sentence_pairs = []
        for c in candidates:
            img_path = _resolve_img_path(c, keyframes_root_dir)
            if img_path and img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    sentence_pairs.append([query, img])
                except Exception:
                    sentence_pairs.append([query, None])
            else:
                sentence_pairs.append([query, None])

        # Batch compute score bằng Jina Reranker model
        if hasattr(model, "compute_score"):
            # Filter out candidates with missing images if needed, or pass PIL images with doc_type="image"
            valid_pairs = []
            for pair in sentence_pairs:
                if pair[1] is not None:
                    valid_pairs.append(pair)

            if not valid_pairs:
                return [float(c.score) for c in candidates]

            raw_scores = model.compute_score(valid_pairs, doc_type="image")
            if isinstance(raw_scores, (list, float, int)):
                if not isinstance(raw_scores, list):
                    raw_scores = [raw_scores]
                
                # Map scores back to candidate list
                score_iter = iter(raw_scores)
                final_scores = []
                for pair, cand in zip(sentence_pairs, candidates):
                    if pair[1] is not None:
                        final_scores.append(float(next(score_iter)))
                    else:
                        final_scores.append(float(cand.score))
                return final_scores

        # Standard HuggingFace fallback nếu không có compute_score
        return [float(c.score) for c in candidates]
    except Exception as e:
        logger.error(f"Lỗi trong quá trình rerank Jina M0: {e}. Fallback score gốc.")
        return [float(c.score) for c in candidates]

