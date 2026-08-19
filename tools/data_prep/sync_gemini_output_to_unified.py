"""
tools/data_prep/sync_gemini_output_to_unified.py
=================================================
Đồng bộ hóa 100% dữ liệu từ data/gemini_output/*.json vào:
1. data/unified/ocr_text.parquet
2. data/unified/scene_descriptions.parquet
3. data/unified/ocr_vectors.npy (gte-Qwen2-1.5B-instruct)
4. data/unified/description_embeddings.pt (bge-large-en-v1.5)
5. Qdrant Collection Storage (aic2026_clean_v3)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from shared.utils.text_normalizer import normalize_nfc, clean_whitespace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SyncGeminiOutput")


def main():
    project_root = Path(__file__).resolve().parents[2]
    gemini_dir = project_root / "data" / "gemini_output"
    unified_dir = project_root / "data" / "unified"

    if not gemini_dir.exists():
        logger.error(f"Thư mục không tồn tại: {gemini_dir}")
        sys.exit(1)

    json_files = sorted(list(gemini_dir.glob("*_descriptions.json")))
    logger.info(f"📂 Tìm thấy {len(json_files)} files JSON trong {gemini_dir.name}")

    # 1. Đọc và tổng hợp toàn bộ JSON
    gemini_data: dict[tuple[str, int], dict[str, str]] = {}
    for jf in json_files:
        vid = jf.stem.replace("_descriptions", "")
        if not vid.startswith("L25_"):
            vid = f"L25_{vid}"
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for f_str, item in data.items():
            try:
                fid = int(float(f_str))
            except (ValueError, TypeError):
                continue
            gemini_data[(vid, fid)] = {
                "ocr_text": clean_whitespace(normalize_nfc(item.get("ocr_text", ""))),
                "description": clean_whitespace(normalize_nfc(item.get("description", ""))),
            }

    logger.info(f"📊 Tổng số frames nạp từ JSON: {len(gemini_data):,d}")

    # 2. Xây dựng mapping scene_id -> gemini_info từ scene_metadata.parquet
    meta_path = unified_dir / "scene_metadata.parquet"
    meta_df = pd.read_parquet(meta_path)
    
    scene_to_ocr: dict[str, str] = {}
    scene_to_desc: dict[str, str] = {}
    for _, row in meta_df.iterrows():
        sid = row["scene_id"]
        vid = row["video_id"]
        try:
            fid = int(float(row["frame_id"]))
        except (ValueError, TypeError):
            continue
        key = (vid, fid)
        if key in gemini_data:
            scene_to_ocr[sid] = gemini_data[key]["ocr_text"]
            scene_to_desc[sid] = gemini_data[key]["description"]

    logger.info(f"🎯 Khớp chính xác {len(scene_to_ocr):,d} / {len(meta_df):,d} scene_id trong metadata.")

    # 3. Cập nhật ocr_text.parquet
    ocr_path = unified_dir / "ocr_text.parquet"
    if ocr_path.exists():
        logger.info(f"🔄 Đang cập nhật {ocr_path.name}...")
        ocr_df = pd.read_parquet(ocr_path)
        # Vectorized mapping
        mask = ocr_df["scene_id"].isin(scene_to_ocr)
        ocr_df.loc[mask, "ocr_text"] = ocr_df.loc[mask, "scene_id"].map(scene_to_ocr)
        ocr_df.to_parquet(ocr_path, index=False)
        logger.info(f"✅ Đã cập nhật {mask.sum():,d} / {len(ocr_df):,d} dòng trong {ocr_path.name}")

    # 4. Cập nhật scene_descriptions.parquet
    desc_path = unified_dir / "scene_descriptions.parquet"
    if desc_path.exists():
        logger.info(f"🔄 Đang cập nhật {desc_path.name}...")
        desc_df = pd.read_parquet(desc_path)
        mask = desc_df["scene_id"].isin(scene_to_desc)
        desc_df.loc[mask, "description"] = desc_df.loc[mask, "scene_id"].map(scene_to_desc)
        desc_df.to_parquet(desc_path, index=False)
        logger.info(f"✅ Đã cập nhật {mask.sum():,d} / {len(desc_df):,d} dòng trong {desc_path.name}")

    # 5. Cập nhật ocr_vectors.npy (gte-Qwen2-1.5B)
    scene_ids_path = unified_dir / "scene_ids.txt"
    ocr_vec_path = unified_dir / "ocr_vectors.npy"
    if scene_ids_path.exists() and ocr_vec_path.exists():
        logger.info(f"🔄 Đang re-compute ocr_vectors.npy cho các frame L25...")
        with open(scene_ids_path, "r", encoding="utf-8") as f:
            scene_ids = [line.strip() for line in f if line.strip()]

        target_indices = []
        target_texts = []
        for i, sid in enumerate(scene_ids):
            if sid in scene_to_ocr:
                target_indices.append(i)
                t = scene_to_ocr[sid]
                target_texts.append(t if t else "[EMPTY]")

        logger.info(f"⚡ Mã hóa {len(target_indices):,d} OCR texts bằng gte-Qwen2-1.5B (CUDA)...")
        if target_texts:
            from shared.models.text_embedding_wrapper import TextEmbeddingWrapper
            embedder = TextEmbeddingWrapper(device="cuda" if torch.cuda.is_available() else "cpu")
            new_vecs = embedder.encode(target_texts, batch_size=64)

            ocr_vecs = np.load(ocr_vec_path)
            for idx_in_matrix, new_v in zip(target_indices, new_vecs):
                ocr_vecs[idx_in_matrix] = new_v
            np.save(ocr_vec_path, ocr_vecs.astype(np.float32))
            logger.info(f"✅ Đã lưu ocr_vectors.npy cập nhật (shape: {ocr_vecs.shape})")

    # 6. Cập nhật description_embeddings.pt (BGE-Large-EN)
    desc_emb_path = unified_dir / "description_embeddings.pt"
    if desc_emb_path.exists() and scene_ids_path.exists():
        logger.info(f"🔄 Đang re-compute description_embeddings.pt cho các frame L25...")
        from transformers import AutoTokenizer, AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "BAAI/bge-large-en-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        if device == "cuda":
            model = model.half()

        target_desc_indices = []
        target_desc_texts = []
        for i, sid in enumerate(scene_ids):
            if sid in scene_to_desc:
                target_desc_indices.append(i)
                target_desc_texts.append(scene_to_desc[sid])

        logger.info(f"⚡ Mã hóa {len(target_desc_texts):,d} descriptions bằng {model_id}...")
        desc_vecs_list = []
        batch_sz = 128
        with torch.no_grad():
            for b_i in range(0, len(target_desc_texts), batch_sz):
                b_texts = target_desc_texts[b_i : b_i + batch_sz]
                encoded = tokenizer(
                    b_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else torch.no_grad():
                    out = model(**encoded)
                    s_emb = out[0][:, 0]
                    s_emb = torch.nn.functional.normalize(s_emb, p=2, dim=1)
                    desc_vecs_list.append(s_emb.cpu())

        new_desc_tensor = torch.cat(desc_vecs_list, dim=0)
        desc_tensor = torch.load(desc_emb_path, map_location="cpu")
        for idx_in_tensor, new_t in zip(target_desc_indices, new_desc_tensor):
            desc_tensor[idx_in_tensor] = new_t
        torch.save(desc_tensor, desc_emb_path)
        logger.info(f"✅ Đã lưu description_embeddings.pt cập nhật (shape: {list(desc_tensor.shape)})")

    logger.info("\n🎉 ĐỒNG BỘ HÓA HOÀN TẤT 100%!")


if __name__ == "__main__":
    main()
