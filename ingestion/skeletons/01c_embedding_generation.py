"""
ingestion/skeletons/01c_embedding_generation.py
=================================================
TASK-ING-03: Sinh vector embeddings cho ảnh và text.

SKELETON: Implement các hàm có `raise NotImplementedError`.

INPUT:
    - scene_metadata.parquet (từ TASK-ING-01)
    - ocr_text.parquet  (từ TASK-ING-02)
    - asr_text.parquet  (từ TASK-ING-02)
    - Keyframe JPG files (từ TASK-ING-01)

OUTPUT (đặt tại output_dir/):
    - image_vectors.npy  — shape (N, 1024), dtype float32, L2-normalized
    - ocr_vectors.npy    — shape (N, 1536), dtype float32, L2-normalized
    - asr_vectors.npy    — shape (N, 1536), dtype float32, L2-normalized
    - scene_ids.txt      — Danh sách scene_id tương ứng với thứ tự rows (1 scene_id mỗi dòng)
      (Bắt buộc để indexing/ biết scene_id nào ứng với row nào trong .npy)

CONSTRAINTS (KHÔNG ĐƯỢC THAY ĐỔI):
    - Model image embedding: BEiT-large + CLIP ViT-L/14
      → import từ shared/models/image_embedding_wrapper.py (ImageEmbeddingWrapper)
      → Output dim = 1024, L2-normalized
    - Model text embedding: gte-Qwen2-1.5B
      → import từ shared/models/text_embedding_wrapper.py (TextEmbeddingWrapper)
      → Dùng chung cho cả OCR lẫn ASR text (cùng 1 model instance)
      → Output dim = 1536, L2-normalized
    - Thứ tự rows trong .npy PHẢI khớp với scene_ids.txt

CÁCH CHẠY (Kaggle Notebook cell):
    >>> from ingestion.skeletons import run_embedding_generation  # noqa
    >>> run_embedding_generation(
    ...     metadata_path='/kaggle/working/shard_output/scene_metadata.parquet',
    ...     ocr_path='/kaggle/working/shard_output/ocr_text.parquet',
    ...     asr_path='/kaggle/working/shard_output/asr_text.parquet',
    ...     keyframe_dir='/kaggle/working/shard_output/keyframes',
    ...     output_dir='/kaggle/working/shard_output',
    ... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper
    from shared.models.text_embedding_wrapper import TextEmbeddingWrapper

# Trên Kaggle, cần mount shared/ package hoặc copy thủ công:
# !pip install -q /kaggle/input/aic2026-shared/shared-0.1.0-py3-none-any.whl
# from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper
# from shared.models.text_embedding_wrapper import TextEmbeddingWrapper

logger = logging.getLogger(__name__)


# =============================================================================
# SKELETON FUNCTIONS — Implement các hàm dưới đây
# =============================================================================

def load_image_model(device: str = "cuda") -> Any:
    """
    Load ImageEmbeddingWrapper (BEiT-large + CLIP ViT-L/14).

    QUAN TRỌNG: Import từ shared/models/image_embedding_wrapper.py
    KHÔNG tự khởi tạo model riêng — phải dùng wrapper dùng chung.

    Gợi ý implementation:
        from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper
        return ImageEmbeddingWrapper(device=device)

    Args:
        device: 'cuda' hoặc 'cpu'. Dùng 'cuda' trên Kaggle GPU.

    Returns:
        ImageEmbeddingWrapper instance đã load model lên GPU/CPU.

    Raises:
        RuntimeError: Nếu không load được model (thiếu weights, hết VRAM...).
    """
    raise NotImplementedError


def load_text_model(device: str = "cuda") -> Any:
    """
    Load TextEmbeddingWrapper (gte-Qwen2-1.5B-instruct).

    QUAN TRỌNG: Import từ shared/models/text_embedding_wrapper.py
    Dùng chung 1 instance cho cả OCR lẫn ASR text.

    Gợi ý implementation:
        from shared.models.text_embedding_wrapper import TextEmbeddingWrapper
        return TextEmbeddingWrapper(device=device)

    Args:
        device: 'cuda' hoặc 'cpu'.

    Returns:
        TextEmbeddingWrapper instance đã load model.
    """
    raise NotImplementedError


def generate_image_embeddings(
    image_model: Any,
    keyframe_dir: str,
    scene_ids: list[str],
    video_ids: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Sinh image embedding cho tất cả keyframes.

    Gợi ý implementation:
        from PIL import Image
        all_vecs = []
        for i in range(0, len(scene_ids), batch_size):
            batch_sids = scene_ids[i:i+batch_size]
            batch_vids = video_ids[i:i+batch_size]
            images = []
            for vid, sid in zip(batch_vids, batch_sids):
                path = Path(keyframe_dir) / vid / f'{sid}.jpg'
                images.append(Image.open(path).convert('RGB'))
            vecs = image_model.encode(images, batch_size=len(images))
            all_vecs.append(vecs)
            logger.info(f'Image embedding: {min(i+batch_size, len(scene_ids))}/{len(scene_ids)}')
        return np.vstack(all_vecs).astype(np.float32)

    Args:
        image_model: ImageEmbeddingWrapper từ load_image_model().
        keyframe_dir: Thư mục gốc chứa keyframes. Structure: {keyframe_dir}/{video_id}/{scene_id}.jpg
        scene_ids: List scene_id, theo đúng thứ tự rows cần lưu.
        video_ids: List video_id tương ứng với scene_ids.
        batch_size: Số ảnh xử lý song song. Điều chỉnh theo VRAM.

    Returns:
        np.ndarray, shape (N, 1024), dtype float32, L2-normalized.

    Raises:
        FileNotFoundError: Nếu có file keyframe không tồn tại.
    """
    raise NotImplementedError


def generate_text_embeddings(
    text_model: Any,
    texts: list[str],
    batch_size: int = 16,
) -> np.ndarray:
    """
    Sinh text embedding cho danh sách text (dùng cho cả OCR lẫn ASR).

    Gợi ý implementation:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Thay thế chuỗi rỗng bằng '[EMPTY]' để model không bị lỗi
            batch = [t if t.strip() else '[EMPTY]' for t in batch]
            vecs = text_model.encode(batch, batch_size=len(batch))
            all_vecs.append(vecs)
            logger.info(f'Text embedding: {min(i+batch_size, len(texts))}/{len(texts)}')
        return np.vstack(all_vecs).astype(np.float32)

    Args:
        text_model: TextEmbeddingWrapper từ load_text_model().
        texts: List text cần encode (OCR text hoặc ASR text).
        batch_size: Số text xử lý song song. Điều chỉnh theo VRAM (mặc định 16 cho gte-Qwen2-1.5B).

    Returns:
        np.ndarray, shape (N, 1536), dtype float32, L2-normalized.
    """
    raise NotImplementedError


def run_embedding_generation(
    metadata_path: str,
    ocr_path: str,
    asr_path: str,
    keyframe_dir: str,
    output_dir: str,
    device: str = "cuda",
) -> None:
    """
    Entry point chính: Sinh toàn bộ vectors và lưu ra file .npy.

    Luồng xử lý:
        1. Load metadata + ocr_text + asr_text, merge theo scene_id
        2. load_image_model() + load_text_model()
        3. generate_image_embeddings() → image_vectors.npy
        4. generate_text_embeddings(ocr_texts) → ocr_vectors.npy
        5. generate_text_embeddings(asr_texts) → asr_vectors.npy
        6. Lưu scene_ids.txt (thứ tự scene_id tương ứng với rows trong .npy)
        7. Log thống kê: số scene, dim của mỗi loại vector, thời gian chạy

    QUAN TRỌNG:
        - Đảm bảo thứ tự rows trong tất cả .npy file PHẢI giống nhau.
        - scene_ids.txt lưu theo thứ tự đó.
        - Nếu ocr_text hoặc asr_text bị thiếu cho một scene → dùng '' (chuỗi rỗng).

    Args:
        metadata_path: Đường dẫn tới scene_metadata.parquet.
        ocr_path: Đường dẫn tới ocr_text.parquet.
        asr_path: Đường dẫn tới asr_text.parquet.
        keyframe_dir: Thư mục chứa keyframe JPG.
        output_dir: Thư mục output. Nên là '/kaggle/working/shard_output/'.
        device: 'cuda' hoặc 'cpu'.

    Raises:
        FileNotFoundError: Nếu metadata_path, ocr_path, hoặc asr_path không tồn tại.
    """
    raise NotImplementedError
