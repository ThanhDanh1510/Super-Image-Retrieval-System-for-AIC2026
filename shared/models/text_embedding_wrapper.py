"""
shared/models/text_embedding_wrapper.py
========================================
Wrapper chuẩn hóa cho text embedding models (gte-Qwen2).

File này được import bởi:
  - ingestion/ notebooks (trên Kaggle) — để tạo vector khi ingestion
  - serving/api/service/retrieval/text_query.py — để encode query lúc thi

RÀNG BUỘC CỨNG:
  - encode() phải trả về L2-normalized vector.
  - dim của output phải khớp với named_vector_spec.OCR_DENSE.dim và ASR_DENSE.dim.
  - Không thay đổi normalize logic mà không cập nhật vectors trong Qdrant.

SKELETON: Implement TextEmbeddingWrapper.__init__() và encode().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

ModelSize = Literal["1.5B", "7B"]


class TextEmbeddingWrapper:
    """
    Wrapper load và inference gte-Qwen2 text embedding model.

    Supports 2 model sizes:
      - "1.5B" (gte-Qwen2-1.5B-instruct): dim=1536, dùng cho OCR text (ngắn)
      - "7B"   (gte-Qwen2-7B-instruct):   dim=3584, dùng cho ASR transcript (dài)

    Usage:
        wrapper_1b = TextEmbeddingWrapper(model_size="1.5B", device="cuda")
        wrapper_7b = TextEmbeddingWrapper(model_size="7B", device="cuda")
        ocr_vec = wrapper_1b.encode("Text extracted from keyframe")
        asr_vec = wrapper_7b.encode("Long ASR transcript text here...")
    """

    MODEL_IDS: dict[ModelSize, str] = {
        "1.5B": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "7B": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    }

    VECTOR_DIMS: dict[ModelSize, int] = {
        "1.5B": 1536,
        "7B": 3584,
    }

    def __init__(self, model_size: ModelSize = "1.5B", device: str = "cuda") -> None:
        """
        Load gte-Qwen2 model vào memory.

        Args:
            model_size: "1.5B" hoặc "7B". Xem MODEL_IDS để biết HuggingFace model ID.
            device: "cuda" hoặc "cpu". Dùng "cuda" nếu có GPU.

        Raises:
            ValueError: Nếu model_size không hợp lệ.
            RuntimeError: Nếu không load được model (thiếu VRAM, sai model ID...).
        """
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """Dimension của output vector. Phải khớp với named_vector_spec."""
        raise NotImplementedError

    def encode(self, text: str | list[str], batch_size: int = 16) -> np.ndarray:
        """
        Encode text thành L2-normalized embedding vector(s).

        Args:
            text: String hoặc list of strings để encode.
            batch_size: Số text encode song song. Điều chỉnh theo VRAM.

        Returns:
            np.ndarray, shape (dim,) nếu input là string,
                         shape (N, dim) nếu input là list[str].
            Luôn L2-normalized (mỗi vector có norm = 1.0).

        Raises:
            ValueError: Nếu text rỗng.
        """
        raise NotImplementedError
