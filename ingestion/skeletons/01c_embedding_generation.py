"""
ingestion/skeletons/01c_embedding_generation.py
=================================================
TASK-ING-03: Sinh vector embeddings cho ảnh và text.

INPUT:
    - scene_metadata.parquet (từ TASK-ING-01)
    - ocr_text.parquet  (từ TASK-ING-02)
    - asr_text.parquet  (từ TASK-ING-02)
    - Keyframe JPG files (từ TASK-ING-01)

OUTPUT (đặt tại output_dir/):
    - image_vectors.npy  — shape (N, 1024), dtype float32, L2-normalized
    - ocr_vectors.npy    — shape (N, 1536), dtype float32, L2-normalized
    - asr_vectors.npy    — shape (N, 1536), dtype float32, L2-normalized
    - scene_ids.txt      — Danh sách scene_id tương ứng với thứ tự rows

CÁCH CHẠY (Kaggle Notebook cell):
    !python -m ingestion.skeletons.01c_embedding_generation \
        --metadata-path /kaggle/working/shard_output/scene_metadata.parquet \
        --ocr-path /kaggle/working/shard_output/ocr_text.parquet \
        --asr-path /kaggle/working/shard_output/asr_text.parquet \
        --keyframe-dir /kaggle/working/shard_output/keyframes \
        --output-dir /kaggle/working/shard_output
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ingestion.skeletons.checkpoint_manifest import CheckpointManifest
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper
from shared.models.text_embedding_wrapper import TextEmbeddingWrapper

logger = logging.getLogger(__name__)


# =============================================================================
# IMPLEMENTED FUNCTIONS
# =============================================================================

def load_image_model(device: str = "cuda") -> ImageEmbeddingWrapper:
    """
    Load ImageEmbeddingWrapper (BEiT-large + CLIP ViT-L/14).
    """
    logger.info(f"Đang khởi tạo ImageEmbeddingWrapper (device='{device}')...")
    return ImageEmbeddingWrapper(device=device)


def load_text_model(device: str = "cuda") -> TextEmbeddingWrapper:
    """
    Load TextEmbeddingWrapper (gte-Qwen2-1.5B-instruct).
    """
    logger.info(f"Đang khởi tạo TextEmbeddingWrapper (device='{device}')...")
    return TextEmbeddingWrapper(device=device)


def generate_image_embeddings(
    image_model: ImageEmbeddingWrapper,
    keyframe_dir: str,
    scene_ids: list[str],
    video_ids: list[str],
    keyframe_paths: list[str] | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Sinh image embedding cho tất cả keyframes. Trả về numpy array (N, 1024).
    """
    kf_base = Path(keyframe_dir)
    all_vecs = []
    total = len(scene_ids)

    msg = f"📸 Bắt đầu sinh Image Embeddings cho {total} scenes (batch_size={batch_size})..."
    logger.info(msg)
    print(msg, flush=True)

    for i in range(0, total, batch_size):
        batch_sids = scene_ids[i : i + batch_size]
        batch_vids = video_ids[i : i + batch_size]
        batch_paths = keyframe_paths[i : i + batch_size] if keyframe_paths else [None] * len(batch_sids)

        images = []
        for vid, sid, kf_p in zip(batch_vids, batch_sids, batch_paths):
            img_file = None
            if kf_p:
                possible_path = kf_base / kf_p
                if possible_path.exists():
                    img_file = possible_path

            if not img_file:
                possible_path = kf_base / vid / f"{sid}.jpg"
                if possible_path.exists():
                    img_file = possible_path
                else:
                    possible_path = kf_base / f"{sid}.jpg"
                    if possible_path.exists():
                        img_file = possible_path

            if img_file and img_file.exists():
                try:
                    img = Image.open(img_file).convert("RGB")
                except Exception as e:
                    logger.warning(f"Không thể đọc ảnh {img_file}: {e}. Dùng blank image.")
                    img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            else:
                img = Image.new("RGB", (224, 224), color=(0, 0, 0))

            images.append(img)

        vecs = image_model.encode(images, batch_size=len(images))
        if vecs.ndim == 1:
            vecs = np.expand_dims(vecs, axis=0)
        all_vecs.append(vecs)

        curr_count = min(i + len(batch_sids), total)
        if curr_count % 1000 == 0 or curr_count == total:
            prog_msg = f"  - [Image] Đã sinh vectors: {curr_count}/{total} ({curr_count/total*100:.1f}%)"
            logger.info(prog_msg)
            print(prog_msg, flush=True)

    if not all_vecs:
        return np.zeros((0, image_model.dim), dtype=np.float32)

    final_matrix = np.vstack(all_vecs).astype(np.float32)
    return final_matrix


def generate_text_embeddings(
    text_model: TextEmbeddingWrapper,
    texts: list[str],
    batch_size: int = 16,
) -> np.ndarray:
    """
    Sinh text embedding cho danh sách text (dùng cho cả OCR lẫn ASR). Trả về numpy array (N, 1536).
    """
    total = len(texts)
    all_vecs = []

    msg = f"🔤 Bắt đầu sinh Text Embeddings cho {total} texts (batch_size={batch_size})..."
    logger.info(msg)
    print(msg, flush=True)

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        vecs = text_model.encode(batch_texts, batch_size=len(batch_texts))
        if vecs.ndim == 1:
            vecs = np.expand_dims(vecs, axis=0)
        all_vecs.append(vecs)

        curr_count = min(i + len(batch_texts), total)
        if curr_count % 1000 == 0 or curr_count == total:
            prog_msg = f"  - [Text] Đã sinh vectors: {curr_count}/{total} ({curr_count/total*100:.1f}%)"
            logger.info(prog_msg)
            print(prog_msg, flush=True)

    if not all_vecs:
        return np.zeros((0, text_model.dim), dtype=np.float32)

    final_matrix = np.vstack(all_vecs).astype(np.float32)
    return final_matrix


def run_embedding_generation(
    metadata_path: str,
    ocr_path: str,
    asr_path: str,
    keyframe_dir: str,
    output_dir: str,
    device: str = "cuda",
    checkpoint_path: str | None = None,
    batch_size_img: int = 32,
    batch_size_txt: int = 16,
) -> None:
    """
    Entry point chính: Sinh toàn bộ vectors và lưu ra file .npy và scene_ids.txt.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_parquet(meta_p)
    msg = f"🚀 Đã nạp {len(meta_df)} scenes từ {metadata_path}"
    logger.info(msg)
    print(msg, flush=True)

    # Đọc OCR text
    ocr_p = Path(ocr_path)
    if ocr_p.exists():
        ocr_df = pd.read_parquet(ocr_p)
    else:
        logger.warning(f"File ocr_text.parquet không tồn tại: {ocr_path}. Dùng chuỗi rỗng.")
        ocr_df = pd.DataFrame(columns=["scene_id", "ocr_text"])

    # Đọc ASR text
    asr_p = Path(asr_path)
    if asr_p.exists():
        asr_df = pd.read_parquet(asr_p)
    else:
        logger.warning(f"File asr_text.parquet không tồn tại: {asr_path}. Dùng chuỗi rỗng.")
        asr_df = pd.DataFrame(columns=["scene_id", "asr_text"])

    # Merge dữ liệu theo scene_id duy nhất để đảm bảo căn chỉnh 100% thứ tự hàng
    merged_df = meta_df.merge(ocr_df[["scene_id", "ocr_text"]], on="scene_id", how="left")
    merged_df = merged_df.merge(asr_df[["scene_id", "asr_text"]], on="scene_id", how="left")

    merged_df["ocr_text"] = merged_df["ocr_text"].fillna("").astype(str)
    merged_df["asr_text"] = merged_df["asr_text"].fillna("").astype(str)

    scene_ids = merged_df["scene_id"].tolist()
    video_ids = merged_df["video_id"].tolist()
    kf_paths = merged_df["keyframe_path"].tolist() if "keyframe_path" in merged_df.columns else None

    checkpoint_file = checkpoint_path or str(out_p / "checkpoint.json")
    manifest = CheckpointManifest(checkpoint_file)

    # 1. Load models
    img_model = load_image_model(device=device)
    txt_model = load_text_model(device=device)

    # 2. Sinh Image Embeddings (TASK-ING-03a)
    logger.info("=== TASK-ING-03a: Sinh Image Dense Vectors (1024-dim) ===")
    img_vecs = generate_image_embeddings(
        image_model=img_model,
        keyframe_dir=keyframe_dir,
        scene_ids=scene_ids,
        video_ids=video_ids,
        keyframe_paths=kf_paths,
        batch_size=batch_size_img,
    )

    # 3. Sinh Text Embeddings cho OCR & ASR (TASK-ING-03b)
    logger.info("=== TASK-ING-03b: Sinh OCR Dense Vectors (1536-dim) ===")
    ocr_vecs = generate_text_embeddings(
        text_model=txt_model,
        texts=merged_df["ocr_text"].tolist(),
        batch_size=batch_size_txt,
    )

    logger.info("=== TASK-ING-03b: Sinh ASR Dense Vectors (1536-dim) ===")
    asr_vecs = generate_text_embeddings(
        text_model=txt_model,
        texts=merged_df["asr_text"].tolist(),
        batch_size=batch_size_txt,
    )

    # Mark checkpoint done
    for sid in scene_ids:
        manifest.mark_done(sid, step="image_embed")
        manifest.mark_done(sid, step="text_embed")
    manifest.save()

    # 4. Xuất các file .npy và scene_ids.txt
    img_out = out_p / "image_vectors.npy"
    ocr_out = out_p / "ocr_vectors.npy"
    asr_out = out_p / "asr_vectors.npy"
    sids_out = out_p / "scene_ids.txt"

    np.save(img_out, img_vecs)
    np.save(ocr_out, ocr_vecs)
    np.save(asr_out, asr_vecs)

    with open(sids_out, "w", encoding="utf-8") as f:
        for sid in scene_ids:
            f.write(f"{sid}\n")

    logger.info("✅ Hoàn tất TASK-ING-03! Đã lưu thành công:")
    logger.info(f"  - {img_out} (shape: {img_vecs.shape}, dtype: {img_vecs.dtype})")
    logger.info(f"  - {ocr_out} (shape: {ocr_vecs.shape}, dtype: {ocr_vecs.dtype})")
    logger.info(f"  - {asr_out} (shape: {asr_vecs.shape}, dtype: {asr_vecs.dtype})")
    logger.info(f"  - {sids_out} ({len(scene_ids)} lines)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="TASK-ING-03: Sinh vector embeddings cho ảnh và text.")
    parser.add_argument("--metadata-path", required=True, help="Đường dẫn tới scene_metadata.parquet.")
    parser.add_argument("--ocr-path", required=True, help="Đường dẫn tới ocr_text.parquet.")
    parser.add_argument("--asr-path", required=True, help="Đường dẫn tới asr_text.parquet.")
    parser.add_argument("--keyframe-dir", required=True, help="Thư mục chứa keyframe JPG.")
    parser.add_argument("--output-dir", default="/kaggle/working/shard_output", help="Thư mục lưu kết quả.")
    parser.add_argument("--device", default="cuda", help="Thiết bị chạy model: 'cuda' hoặc 'cpu'.")
    parser.add_argument("--checkpoint-path", default=None, help="Đường dẫn file checkpoint.")

    args = parser.parse_args()

    run_embedding_generation(
        metadata_path=args.metadata_path,
        ocr_path=args.ocr_path,
        asr_path=args.asr_path,
        keyframe_dir=args.keyframe_dir,
        output_dir=args.output_dir,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
    )
