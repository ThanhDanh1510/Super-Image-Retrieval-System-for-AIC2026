"""
serving/api/service/retrieval/query_expander.py
===============================================
Mở rộng truy vấn ngữ nghĩa (Semantic Query Expansion) & Prompt Ensembling cho CLIP.

Nghiên cứu gốc của OpenAI CLIP chứng minh việc trung bình hóa vector embedding
của nhiều biến thể prompt (Prompt Ensembling) làm mượt không gian hình học,
giúp tăng Zero-Shot Retrieval Recall từ 3% đến 5%.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


class PromptEnsembleExpander:
    """Bộ tạo biến thể prompt và hợp nhất vector CLIP đa góc nhìn."""

    @staticmethod
    def generate_clip_prompt_variants(query_en: str) -> list[str]:
        """Tạo các biến thể câu prompt bổ trợ cho thị giác CLIP."""
        clean = query_en.strip().rstrip(".")
        if not clean:
            return []

        # Các mẫu khuôn chuẩn hóa không gian thị giác CLIP
        variants = [
            clean,
            f"a photo of {clean}",
            f"{clean}, high quality clear visual shot",
        ]
        return variants

    @classmethod
    def ensemble_text_embeddings(
        cls,
        query_en: str,
        image_embedder: Any,
    ) -> list[float]:
        """
        Tạo vector tổng hợp (Ensemble Embedding) cho CLIP:
        Mã hóa các biến thể trong 1 batch và tính vector trung bình có trọng số, chuẩn hóa L2.
        """
        variants = cls.generate_clip_prompt_variants(query_en)
        if not variants:
            return []

        try:
            if hasattr(image_embedder, "encode_text"):
                vecs = image_embedder.encode_text(variants)
                if isinstance(vecs, np.ndarray) and vecs.ndim == 2:
                    weights = np.array([0.6, 0.2, 0.2], dtype=np.float32)
                    if len(vecs) == len(weights):
                        fused = np.sum(vecs * weights[:, None], axis=0)
                    else:
                        fused = np.mean(vecs, axis=0)
                    norm = np.linalg.norm(fused)
                    return (fused / (norm + 1e-9)).tolist()
                elif hasattr(vecs, "tolist"):
                    return vecs.tolist()
        except Exception as e:
            logger.warning(f"Lỗi khi ensemble prompt variants: {e}")

        # Fallback to single text
        try:
            emb = image_embedder.encode_text(query_en)
            if hasattr(emb, "tolist"):
                return emb.tolist()
            return list(emb)
        except Exception as e:
            logger.warning(f"Fallback encode_text error: {e}")
            return []
