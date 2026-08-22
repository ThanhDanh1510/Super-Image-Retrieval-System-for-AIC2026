"""
serving/api/service/reranker/skeletons/crossmodal_branch/jina_reranker_m0.py
============================================================================
Nhánh rerank đa phương thức (multimodal) sử dụng Jina Reranker M0.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from serving.api.schema.retrieval import CandidateResult
from shared.utils.path_resolver import resolve_keyframe_path

logger = logging.getLogger(__name__)


def score(
    query: str,
    candidates: list[CandidateResult],
    keyframes_root_dir: str = "",
    model_bundle: Optional[dict[str, Any]] = None,
) -> list[float]:
    """
    Tính điểm tương đồng giữa text query và image keyframe tương ứng của từng candidate.

    Args:
        query: Query text (tiếng Anh).
        candidates: Danh sách CandidateResult cần rerank.
        keyframes_root_dir: Thư mục chứa toàn bộ ảnh keyframe của hệ thống.
        model_bundle: Dict chứa model, device (từ ModelRegistry).

    Returns:
        List[float]: Danh sách điểm số (cùng kích thước và thứ tự với candidates).
    """
    if not candidates:
        return []

    if model_bundle is None:
        try:
            import torch
            import transformers
            from transformers import AutoModel

            model_name = "jinaai/jina-reranker-m0"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            orig_verbosity = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()
            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                model.to(device)
                model.eval()
            finally:
                transformers.logging.set_verbosity(orig_verbosity)

            model_bundle = {"model": model, "device": device}
        except Exception as e:
            logger.error(f"Không thể load model Jina Reranker M0: {e}. Fallback score gốc.")
            return [float(c.score) for c in candidates]

    try:
        from pathlib import Path
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor

        model = model_bundle["model"]
        data_dir = Path(keyframes_root_dir) if keyframes_root_dir else None

        def _load_single_img(cand: CandidateResult):
            img_path = resolve_keyframe_path(
                video_id=cand.video_id,
                frame_id=cand.frame_id,
                keyframe_path=cand.keyframe_path,
                data_dir=data_dir,
            )
            if img_path and img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.thumbnail((384, 384))
                    return img
                except Exception:
                    return None
            return None

        # Đọc và resize song song 25 ảnh keyframe trên đa luồng CPU
        with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
            loaded_imgs = list(executor.map(_load_single_img, candidates))

        sentence_pairs = [[query, img] for img in loaded_imgs]

        # Batch compute score bằng Jina Reranker model
        if hasattr(model, "compute_score"):
            valid_pairs = [pair for pair in sentence_pairs if pair[1] is not None]

            if not valid_pairs:
                return [float(c.score) for c in candidates]

            import torch
            with torch.inference_mode():
                raw_scores = model.compute_score(valid_pairs, doc_type="image", max_query_length=256)
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

        return [float(c.score) for c in candidates]
    except Exception as e:
        logger.error(f"Lỗi trong quá trình rerank Jina M0: {e}. Fallback score gốc.")
        return [float(c.score) for c in candidates]
