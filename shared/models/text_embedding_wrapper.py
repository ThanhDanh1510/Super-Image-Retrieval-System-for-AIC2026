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

import numpy as np



class TextEmbeddingWrapper:
    """
    Wrapper load và inference gte-Qwen2-1.5B text embedding model.

    Dùng cho cả OCR text và ASR transcript — cùng 1 model, cùng dim=1536.

    Usage:
        wrapper = TextEmbeddingWrapper(device="cuda")
        ocr_vec = wrapper.encode("Text extracted from keyframe")
        asr_vec = wrapper.encode("Long ASR transcript text here...")
    """

    MODEL_ID = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    VECTOR_DIM = 1536

    def __init__(self, device: str = "cuda") -> None:
        """
        Load gte-Qwen2-1.5B model vào memory.

        Args:
            device: "cuda" hoặc "cpu".

        Raises:
            RuntimeError: Nếu không load được model.
        """
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """Dimension của output vector = 1536. Phải khớp với named_vector_spec."""
        return self.VECTOR_DIM

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
