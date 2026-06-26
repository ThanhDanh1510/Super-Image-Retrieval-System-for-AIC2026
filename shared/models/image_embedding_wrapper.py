"""
shared/models/image_embedding_wrapper.py
=========================================
Wrapper chuẩn hóa cho image embedding models (BEiT-large + CLIP).

File này được import bởi:
  - ingestion/ notebooks (trên Kaggle) — để tạo vector khi ingestion
  - serving/api/service/retrieval/image_query.py — để encode ảnh query lúc thi

RÀNG BUỘC CỨNG:
  - encode() phải trả về L2-normalized vector dim=1024.
  - Dim phải khớp với named_vector_spec.IMAGE_DENSE.dim = 1024.
  - Fusion strategy (BEiT + CLIP) phải nhất quán giữa ingestion và serving.

SKELETON: Implement ImageEmbeddingWrapper.__init__() và encode().
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageEmbeddingWrapper:
    """
    Wrapper load và inference BEiT-large + CLIP image embedding models.

    Fuses 2 models để tăng chất lượng retrieval:
      - BEiT-large-patch16-224 (semantic understanding)
      - CLIP ViT-L/14 (cross-modal alignment)

    Output: fused vector dim=1024, L2-normalized.

    Usage:
        wrapper = ImageEmbeddingWrapper(device="cuda")
        img = Image.open("keyframe.jpg")
        vec = wrapper.encode(img)  # shape (1024,), L2-normalized
    """

    # Model IDs phải khớp với configs/model_versions.yaml :: image_embedding
    BEIT_MODEL_ID = "microsoft/BEiT-large-patch16-224-pt22k-ft22k"
    CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
    VECTOR_DIM = 1024

    def __init__(self, device: str = "cuda") -> None:
        """
        Load BEiT và CLIP model vào memory.

        Args:
            device: "cuda" hoặc "cpu".

        Raises:
            RuntimeError: Nếu không load được model.
        """
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """Dimension của output vector = 1024."""
        return self.VECTOR_DIM

    def encode(
        self,
        image: Union[Image.Image, list[Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode ảnh thành L2-normalized embedding vector(s).

        Pipeline nội bộ:
          1. Preprocess ảnh (resize, normalize theo model config)
          2. Forward BEiT → beit_vec (dim tùy model, thường 1024)
          3. Forward CLIP → clip_vec (dim=768 cho ViT-L/14)
          4. Project/concat/weighted-average → fused_vec dim=1024
          5. L2-normalize

        Args:
            image: PIL Image hoặc list of PIL Images.
            batch_size: Số ảnh xử lý song song.

        Returns:
            np.ndarray, shape (1024,) nếu input là Image,
                         shape (N, 1024) nếu input là list.
            Luôn L2-normalized.

        Raises:
            ValueError: Nếu image không hợp lệ (wrong format, corrupt...).
        """
        raise NotImplementedError

    def encode_from_bytes(self, image_bytes: bytes, batch_size: int = 32) -> np.ndarray:
        """
        Convenience method: decode bytes → PIL Image → encode().

        Args:
            image_bytes: Raw bytes của ảnh (JPEG/PNG/WebP).
            batch_size: Số ảnh xử lý song song.

        Returns:
            np.ndarray shape (1024,), L2-normalized.

        Raises:
            ValueError: Nếu bytes không decode được thành ảnh hợp lệ.
        """
        raise NotImplementedError

    def encode_text(
        self,
        text: str | list[str],
    ) -> np.ndarray:
        """
        Encode text query thành L2-normalized image embedding vector(s) (Text-to-Image).
        Dùng mô hình CLIP (hoặc BEiT3 text encoder nếu có) để chiếu văn bản vào không gian chung.

        Args:
            text: Chuỗi văn bản truy vấn hoặc list các chuỗi văn bản.

        Returns:
            np.ndarray, shape (1024,) nếu input là str,
                         shape (N, 1024) nếu input là list.
            Luôn L2-normalized.
        """
        raise NotImplementedError

