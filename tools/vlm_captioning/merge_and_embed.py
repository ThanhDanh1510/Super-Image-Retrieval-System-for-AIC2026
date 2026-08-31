"""
tools/vlm_captioning/merge_and_embed.py
======================================
Merges all VLM description shards, updates scene_descriptions.parquet,
and computes dense BGE-Large-EN embeddings (description_embeddings.npy).
"""

import sys
import time
import shutil
import logging
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Merge-and-Embed")

BASE_DIR = Path("d:/AIC2026")
UNIFIED_DIR = BASE_DIR / "data" / "unified"
SHARDS_DIR = UNIFIED_DIR / "vlm_shards"
METADATA_PARQUET = UNIFIED_DIR / "scene_metadata.parquet"
OUTPUT_PARQUET = UNIFIED_DIR / "scene_descriptions.parquet"
OUTPUT_EMBEDDINGS_NPY = UNIFIED_DIR / "description_embeddings.npy"
BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"


def merge_shards() -> pd.DataFrame:
    """Gộp toàn bộ các shard parquet lại thành 1 DataFrame đầy đủ."""
    shard_files = sorted(list(SHARDS_DIR.glob("*.parquet")))
    if not shard_files:
        raise FileNotFoundError(f"Không tìm thấy shard nào trong {SHARDS_DIR}")

    logger.info(f"📦 Đang nạp {len(shard_files)} shard files...")
    dfs = []
    for sf in shard_files:
        df_s = pd.read_parquet(sf)
        dfs.append(df_s)

    merged = pd.concat(dfs, ignore_index=True)
    # Deduplicate theo scene_id (giữ bản ghi mới nhất)
    merged = merged.drop_duplicates(subset=["scene_id"], keep="last")
    logger.info(f"✅ Đã gộp thành công {len(merged):,} scenes duy nhất từ {len(shard_files)} shards.")

    # Đọc metadata gốc để giữ đúng thứ tự index
    df_meta = pd.read_parquet(METADATA_PARQUET)
    merged_ordered = pd.merge(
        df_meta[["scene_id", "video_id", "frame_id", "timestamp_mid"]],
        merged[["scene_id", "description"]],
        on="scene_id",
        how="left",
    )
    merged_ordered["description"] = merged_ordered["description"].fillna("").astype(str)

    # Backup file cũ nếu có
    if OUTPUT_PARQUET.exists():
        backup_path = UNIFIED_DIR / f"scene_descriptions_backup_{int(time.time())}.parquet"
        shutil.copyfile(OUTPUT_PARQUET, backup_path)
        logger.info(f"🛡️ Đã tạo bản sao lưu an toàn: {backup_path.name}")

    merged_ordered.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info(f"💾 Đã lưu thành công {OUTPUT_PARQUET.name} với {len(merged_ordered):,} bản ghi.")
    return merged_ordered


def generate_bge_embeddings(df: pd.DataFrame, batch_size: int = 64):
    """Tính toán ma trận 88,899 vector BGE-Large-EN-v1.5 và lưu thành description_embeddings.npy."""
    logger.info(f"🚀 Đang khởi tạo mô hình Embedding {BGE_MODEL_ID} trên GPU...")
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_ID)
    model = AutoModel.from_pretrained(BGE_MODEL_ID, torch_dtype=torch.float16)
    model.to("cuda")
    model.eval()

    texts = df["description"].tolist()
    total = len(texts)
    all_embeddings = []

    logger.info(f"⚡ Bắt đầu sinh Dense Embeddings cho {total:,} scenes (Batch size: {batch_size})...")
    _t0 = time.time()

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        # Replace empty strings with a default placeholder
        batch_texts = [t if t.strip() else "general video scene" for t in batch_texts]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**encoded)
            # CLS token pooling & L2 normalization
            cls_repr = outputs[0][:, 0]
            norm_repr = torch.nn.functional.normalize(cls_repr, p=2, dim=1)
            all_embeddings.append(norm_repr.cpu().numpy().astype(np.float32))

        if (i // batch_size) % 100 == 0 or (i + batch_size) >= total:
            done = min(i + batch_size, total)
            logger.info(f"  Progress: [{done:,}/{total:,}] ({done/total*100:.1f}%) | Time: {time.time() - _t0:.1f}s")

    matrix = np.vstack(all_embeddings)
    logger.info(f"✅ Ma trận embeddings hoàn tất: Shape {matrix.shape} (dtype: {matrix.dtype}) trong {time.time() - _t0:.1f}s!")

    # Backup ma trận cũ
    if OUTPUT_EMBEDDINGS_NPY.exists():
        backup_emb = UNIFIED_DIR / f"description_embeddings_backup_{int(time.time())}.npy"
        shutil.copyfile(OUTPUT_EMBEDDINGS_NPY, backup_emb)
        logger.info(f"🛡️ Đã sao lưu ma trận cũ: {backup_emb.name}")

    np.save(OUTPUT_EMBEDDINGS_NPY, matrix)
    logger.info(f"💾 ĐÃ LƯU THÀNH CÔNG: {OUTPUT_EMBEDDINGS_NPY.name} ({matrix.nbytes / (1024*1024):.1f} MB).")


def main():
    df = merge_shards()
    generate_bge_embeddings(df)
    logger.info("🎉 TẤT CẢ QUÁ TRÌNH MERGE VÀ EMBEDDING ĐÃ HOÀN TẤT XUẤT SẮC!")


if __name__ == "__main__":
    main()
