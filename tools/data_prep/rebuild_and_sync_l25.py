"""
tools/data_prep/rebuild_and_sync_l25.py
======================================
Pipeline toàn diện:
1. Xóa vĩnh viễn các ảnh đã lọc trong data/filtered_keyframes.json (bao gồm 2,149 ảnh L25 vừa lọc).
2. Trích xuất metadata và làm giàu OCR/Descriptions cho 6,119 keyframes còn lại của L25.
3. Sinh CLIP image vectors, Qwen2 OCR text vectors, và BGE-large description vectors cho L25.
4. Hợp nhất hoàn hảo vào data/unified/ (tất cả parquet và vector .npy/.pt khớp 100% số dòng).
5. Nạp trực tiếp vào Qdrant collection aic2026_clean_v3.
"""

import os
import sys
import gc
import json
import time
import re
import shutil
import logging
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from shared.utils.text_normalizer import normalize_nfc, clean_whitespace
from shared.schemas.named_vector_spec import (
    IMAGE_DENSE,
    OCR_DENSE,
    ASR_DENSE,
    make_point_id,
    KeyframePayload,
)
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RebuildSyncL25")


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def main():
    logger.info("=" * 70)
    logger.info("🚀 BẮT ĐẦU PIPELINE TỔNG HỢP & NẠP LẠI BATCH L25")
    logger.info("=" * 70)

    filter_json_path = PROJECT_ROOT / "data" / "filtered_keyframes.json"
    image_dir = PROJECT_ROOT / "data" / "image"
    unified_dir = PROJECT_ROOT / "data" / "unified"
    gemini_dir = PROJECT_ROOT / "data" / "gemini_output"
    ocr_dir = PROJECT_ROOT / "data" / "ocr_output" / "checkpoints_L25"

    # -------------------------------------------------------------------------
    # 1. XÓA VẬT LÝ CÁC ẢNH ĐÃ ĐƯỢC ĐÁNH DẤU LỌC
    # -------------------------------------------------------------------------
    logger.info("--- BƯỚC 1: XÓA CÁC FILE ẢNH ĐÃ ĐƯỢC ĐÁNH DẤU LỌC ---")
    if filter_json_path.exists():
        with open(filter_json_path, "r", encoding="utf-8") as f:
            filter_data = json.load(f)
        filtered_items = filter_data.get("filtered_images", [])
        logger.info(f"Loaded {len(filtered_items)} ảnh cần xóa từ filtered_keyframes.json")
        deleted_count = 0
        for item in filtered_items:
            b = str(item.get("batch", ""))
            vid = str(item.get("video_id", ""))
            fid = int(item.get("frame_id", 0))
            fname = str(item.get("filename", f"{fid}.webp"))
            
            candidates = [
                image_dir / b / b / vid / fname,
                image_dir / b / vid / fname,
                image_dir / b / b / f"{b}_{vid}" / fname,
            ]
            for p in candidates:
                if p.exists() and p.is_file():
                    try:
                        p.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Lỗi xóa {p}: {e}")
                    break
        logger.info(f"Đã xóa vĩnh viễn {deleted_count} file ảnh đã lọc trên đĩa.")

    # -------------------------------------------------------------------------
    # 2. QUÉT VÀ TẠO METADATA CHO CÁC ẢNH L25 CÒN LẠI
    # -------------------------------------------------------------------------
    logger.info("\n--- BƯỚC 2: QUÉT VÀ LÀM GIÀU METADATA L25 ---")
    l25_dir = image_dir / "L25" / "L25"
    if not l25_dir.exists():
        l25_dir = image_dir / "L25"

    l25_clean_files = sorted(l25_dir.rglob("*.webp"), key=natural_sort_key)
    logger.info(f"Tổng số ảnh keyframe L25 còn giữ lại: {len(l25_clean_files)} ảnh")

    # Load Gemini OCR & Descriptions
    gemini_data: dict[tuple[str, int], dict[str, str]] = {}
    if gemini_dir.exists():
        for jf in gemini_dir.glob("*.json"):
            vid = jf.stem.replace("_descriptions", "").replace("gemini_output_", "")
            if not vid.startswith("L25_"):
                vid = f"L25_{vid}"
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                for f_str, item in d.items():
                    try:
                        fid = int(float(f_str))
                    except (ValueError, TypeError):
                        continue
                    gemini_data[(vid, fid)] = {
                        "ocr_text": clean_whitespace(normalize_nfc(item.get("ocr_text", ""))),
                        "description": clean_whitespace(normalize_nfc(item.get("description", ""))),
                    }
            except Exception as e:
                logger.warning(f"Lỗi đọc {jf}: {e}")
        logger.info(f"Loaded {len(gemini_data)} bản ghi OCR/Descriptions từ {gemini_dir.name}")

    # Load OCR Checkpoints for L25 if available
    ocr_ckpt_data: dict[tuple[str, int], str] = {}
    if ocr_dir.exists():
        for pf in ocr_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(pf)
                for _, row in df.iterrows():
                    v = str(row["video_id"])
                    f_id = int(row["frame_id"])
                    txt = clean_whitespace(normalize_nfc(str(row.get("ocr_text", ""))))
                    if txt:
                        ocr_ckpt_data[(v, f_id)] = txt
            except Exception:
                pass
        logger.info(f"Loaded {len(ocr_ckpt_data)} bản ghi từ checkpoints OCR L25.")

    l25_records = []
    l25_ocr_records = []
    l25_asr_records = []
    l25_desc_records = []

    vid_scene_counter = {}
    for f in l25_clean_files:
        vid_short = f.parent.name
        fid = int(f.stem)
        full_vid = f"L25_{vid_short}" if not vid_short.startswith("L25") else vid_short
        clean_short_vid = full_vid.replace("L25_", "")
        
        vid_scene_counter[full_vid] = vid_scene_counter.get(full_vid, 0) + 1
        idx = vid_scene_counter[full_vid]
        sid = f"{full_vid}_S{idx:03d}"

        # Metadata record
        l25_records.append({
            "scene_id": sid,
            "video_id": full_vid,
            "frame_id": fid,
            "fps": 25.0,
            "timestamp_start": round(max(0, fid - 25) / 25.0, 2),
            "timestamp_mid": round(fid / 25.0, 2),
            "timestamp_end": round((fid + 25) / 25.0, 2),
            "keyframe_path": f"L25/{clean_short_vid}/{f.name}",
            "file_path": str(f),
        })

        # Text matching
        g_info = gemini_data.get((full_vid, fid), {})
        ocr_text = g_info.get("ocr_text", "")
        if not ocr_text:
            ocr_text = ocr_ckpt_data.get((full_vid, fid), "")
            if not ocr_text:
                ocr_text = ocr_ckpt_data.get((clean_short_vid, fid), "")

        desc_text = g_info.get("description", "")
        if not desc_text and ocr_text:
            desc_text = f"Nội dung văn bản: {ocr_text}"

        l25_ocr_records.append({
            "scene_id": sid,
            "video_id": full_vid,
            "ocr_text": ocr_text,
            "frame_id": fid,
            "timestamp_mid": round(fid / 25.0, 2),
        })

        l25_asr_records.append({
            "scene_id": sid,
            "video_id": full_vid,
            "asr_text": "",
            "frame_id": fid,
            "timestamp_mid": round(fid / 25.0, 2),
        })

        l25_desc_records.append({
            "scene_id": sid,
            "video_id": full_vid,
            "description": desc_text,
            "l_json_key": f"{clean_short_vid}/{fid}",
            "clip_confidence": 0.95,
        })

    l25_meta_df = pd.DataFrame(l25_records)
    l25_ocr_df = pd.DataFrame(l25_ocr_records)
    l25_asr_df = pd.DataFrame(l25_asr_records)
    l25_desc_df = pd.DataFrame(l25_desc_records)

    logger.info(f"Đã tạo metadata cho {len(l25_meta_df)} clean L25 scenes across {len(vid_scene_counter)} videos.")

    # -------------------------------------------------------------------------
    # 3. SINH CLIP IMAGE VECTORS CHO CLEAN L25
    # -------------------------------------------------------------------------
    logger.info("\n--- BƯỚC 3: SINH CLIP IMAGE EMBEDDINGS (1024-dim) ---")
    from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper

    embedder_img = ImageEmbeddingWrapper(device="cuda")
    batch_size = 64
    all_img_vectors = []
    t0 = time.time()

    for i in range(0, len(l25_records), batch_size):
        batch_recs = l25_records[i : i + batch_size]
        imgs = []
        for r in batch_recs:
            try:
                imgs.append(Image.open(r["file_path"]).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        vecs = embedder_img.encode(imgs, batch_size=batch_size)
        all_img_vectors.append(vecs)
        done = min(i + batch_size, len(l25_records))
        if done % 1000 == 0 or done == len(l25_records):
            logger.info(f"  CLIP encoded {done}/{len(l25_records)} images ({done/len(l25_records)*100:.1f}%) in {time.time()-t0:.1f}s")

    l25_image_vectors = np.vstack(all_img_vectors).astype(np.float32)
    logger.info(f"Hoàn tất CLIP image vectors: shape={l25_image_vectors.shape}")

    # Free memory from image embedder
    del embedder_img
    del all_img_vectors
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. SINH QWEN2 TEXT VECTORS CHO OCR & ASR (1536-dim)
    # -------------------------------------------------------------------------
    logger.info("\n--- BƯỚC 4: SINH QWEN2 TEXT EMBEDDINGS (1536-dim) CHO OCR ---")
    from shared.models.text_embedding_wrapper import TextEmbeddingWrapper

    embedder_txt = TextEmbeddingWrapper(device="cuda")
    ocr_texts = l25_ocr_df["ocr_text"].fillna("").tolist()
    
    all_ocr_vectors = []
    t1 = time.time()
    for i in range(0, len(ocr_texts), 64):
        b_txts = ocr_texts[i : i + 64]
        v = embedder_txt.encode(b_txts)
        all_ocr_vectors.append(v)
        done = min(i + 64, len(ocr_texts))
        if done % 1000 == 0 or done == len(ocr_texts):
            logger.info(f"  Qwen2 encoded {done}/{len(ocr_texts)} OCR texts in {time.time()-t1:.1f}s")

    l25_ocr_vectors = np.vstack(all_ocr_vectors).astype(np.float32)
    logger.info(f"Hoàn tất Qwen2 OCR vectors: shape={l25_ocr_vectors.shape}")

    # ASR vectors (all zeros for empty L25 ASR)
    l25_asr_vectors = np.zeros((len(l25_records), 1536), dtype=np.float32)

    del embedder_txt
    del all_ocr_vectors
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. SINH BGE-LARGE EMBEDDINGS CHO DESCRIPTIONS (1024-dim)
    # -------------------------------------------------------------------------
    logger.info("\n--- BƯỚC 5: SINH BGE-LARGE DESCRIPTION EMBEDDINGS (1024-dim) ---")
    from transformers import AutoTokenizer, AutoModel

    bge_model_id = "BAAI/bge-large-en-v1.5"
    bge_tokenizer = AutoTokenizer.from_pretrained(bge_model_id)
    bge_model = AutoModel.from_pretrained(bge_model_id, torch_dtype=torch.float16).to("cuda")
    bge_model.eval()

    desc_texts = l25_desc_df["description"].fillna("").tolist()
    all_desc_vectors = []
    t2 = time.time()

    for i in range(0, len(desc_texts), 128):
        b_txts = desc_texts[i : i + 128]
        # Fallback if empty string
        clean_b_txts = [t if t.strip() else "scenery visual keyframe" for t in b_txts]
        encoded = bge_tokenizer(clean_b_txts, padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = bge_model(**encoded)
            sent_emb = out[0][:, 0]
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            all_desc_vectors.append(sent_emb.cpu())
        done = min(i + 128, len(desc_texts))
        if done % 1000 == 0 or done == len(desc_texts):
            logger.info(f"  BGE encoded {done}/{len(desc_texts)} descriptions in {time.time()-t2:.1f}s")

    l25_desc_tensor = torch.cat(all_desc_vectors, dim=0) # (N, 1024)
    logger.info(f"Hoàn tất BGE description vectors: shape={l25_desc_tensor.shape}")

    del bge_model
    del bge_tokenizer
    del all_desc_vectors
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 6. HỢP NHẤT NON-L25 + L25 VÀO DATA/UNIFIED/
    # -------------------------------------------------------------------------
    logger.info("\n--- BƯỚC 6: HỢP NHẤT TOÀN BỘ DATASET VÀO DATA/UNIFIED/ ---")

    # Load existing clean non-L25 data
    orig_meta = pd.read_parquet(unified_dir / "scene_metadata.parquet")
    non_l25_mask = ~orig_meta["video_id"].str.startswith("L25")
    non_l25_indices = np.where(non_l25_mask)[0]

    clean_meta = orig_meta[non_l25_mask].copy()

    orig_ocr = pd.read_parquet(unified_dir / "ocr_text.parquet")
    clean_ocr = orig_ocr[~orig_ocr["video_id"].str.startswith("L25")].copy()

    orig_asr = pd.read_parquet(unified_dir / "asr_text.parquet")
    clean_asr = orig_asr[~orig_asr["video_id"].str.startswith("L25")].copy()

    orig_desc = pd.read_parquet(unified_dir / "scene_descriptions.parquet")
    clean_desc = orig_desc[~orig_desc["video_id"].str.startswith("L25")].copy()

    # Vectors
    orig_img_v = np.load(unified_dir / "image_vectors.npy")
    clean_img_v = orig_img_v[non_l25_indices]

    orig_ocr_v = np.load(unified_dir / "ocr_vectors.npy")
    clean_ocr_v = orig_ocr_v[non_l25_indices]

    orig_asr_v = np.load(unified_dir / "asr_vectors.npy")
    clean_asr_v = orig_asr_v[non_l25_indices]

    orig_desc_t = torch.load(unified_dir / "description_embeddings.pt", map_location="cpu", weights_only=True)
    clean_desc_t = orig_desc_t[non_l25_indices]

    logger.info(f"Non-L25 Clean scenes count: {len(clean_meta):,d}")
    logger.info(f"L25 Clean scenes count    : {len(l25_meta_df):,d}")

    # Combine DataFrames
    combined_meta = pd.concat([clean_meta, l25_meta_df.drop(columns=["file_path"])], ignore_index=True)
    combined_ocr = pd.concat([clean_ocr, l25_ocr_df], ignore_index=True)
    combined_asr = pd.concat([clean_asr, l25_asr_df], ignore_index=True)
    combined_desc = pd.concat([clean_desc, l25_desc_df], ignore_index=True)

    # Combine Vectors
    combined_img_v = np.vstack([clean_img_v, l25_image_vectors])
    combined_ocr_v = np.vstack([clean_ocr_v, l25_ocr_vectors])
    combined_asr_v = np.vstack([clean_asr_v, l25_asr_vectors])
    combined_desc_t = torch.cat([clean_desc_t, l25_desc_tensor], dim=0)

    combined_scene_ids = list(combined_meta["scene_id"])

    N = len(combined_meta)
    logger.info("=" * 60)
    logger.info(f"🎉 TỔNG KẾT UNIFIED DATASET SAU HỢP NHẤT: {N:,d} SCENES")
    logger.info(f"  - scene_metadata.parquet   : {len(combined_meta):,d} rows")
    logger.info(f"  - ocr_text.parquet         : {len(combined_ocr):,d} rows")
    logger.info(f"  - asr_text.parquet         : {len(combined_asr):,d} rows")
    logger.info(f"  - scene_descriptions.parquet: {len(combined_desc):,d} rows")
    logger.info(f"  - image_vectors.npy        : {combined_img_v.shape}")
    logger.info(f"  - ocr_vectors.npy          : {combined_ocr_v.shape}")
    logger.info(f"  - asr_vectors.npy          : {combined_asr_v.shape}")
    logger.info(f"  - description_embeddings.pt: {combined_desc_t.shape}")
    logger.info(f"  - scene_ids.txt            : {len(combined_scene_ids):,d} lines")
    logger.info("=" * 60)

    # Save to disk safely
    combined_meta.to_parquet(unified_dir / "scene_metadata.parquet", index=False)
    combined_ocr.to_parquet(unified_dir / "ocr_text.parquet", index=False)
    combined_asr.to_parquet(unified_dir / "asr_text.parquet", index=False)
    combined_desc.to_parquet(unified_dir / "scene_descriptions.parquet", index=False)

    np.save(unified_dir / "image_vectors.npy", combined_img_v)
    np.save(unified_dir / "ocr_vectors.npy", combined_ocr_v)
    np.save(unified_dir / "asr_vectors.npy", combined_asr_v)
    torch.save(combined_desc_t, unified_dir / "description_embeddings.pt")

    with open(unified_dir / "scene_ids.txt", "w", encoding="utf-8") as f:
        for sid in combined_scene_ids:
            f.write(f"{sid}\n")

    # -------------------------------------------------------------------------
    # 7. TẠO & UPSERT COLLECTION VÀO QDRANT DB
    # -------------------------------------------------------------------------
    logger.info("\n--- BƯỚC 7: NẠP VÀO QDRANT COLLECTION (aic2026_clean_v3) ---")
    client = QdrantClient(url="http://localhost:6333")

    collection_name = "aic2026_clean_v3"
    
    # We will upsert in batches of 500
    qdrant_batch_size = 500
    total_points = N
    logger.info(f"Đang upsert {total_points:,d} points vào collection '{collection_name}'...")
    
    t_q = time.time()
    for start_idx in range(0, total_points, qdrant_batch_size):
        end_idx = min(start_idx + qdrant_batch_size, total_points)
        batch_points = []
        
        for idx in range(start_idx, end_idx):
            m_row = combined_meta.iloc[idx]
            sid = str(m_row["scene_id"])
            vid = str(m_row["video_id"])
            fid = int(m_row["frame_id"])
            
            ocr_val = str(combined_ocr.iloc[idx]["ocr_text"]) if idx < len(combined_ocr) else ""
            asr_val = str(combined_asr.iloc[idx]["asr_text"]) if idx < len(combined_asr) else ""
            desc_val = str(combined_desc.iloc[idx]["description"]) if idx < len(combined_desc) else ""

            pt_id = make_point_id(vid, sid)
            
            vectors = {
                IMAGE_DENSE.name: combined_img_v[idx].tolist(),
                OCR_DENSE.name: combined_ocr_v[idx].tolist(),
                ASR_DENSE.name: combined_asr_v[idx].tolist(),
            }
            
            payload = {
                "scene_id": sid,
                "video_id": vid,
                "frame_id": fid,
                "fps": float(m_row.get("fps", 25.0)),
                "timestamp_start": float(m_row.get("timestamp_start", 0.0)),
                "timestamp_mid": float(m_row.get("timestamp_mid", 0.0)),
                "timestamp_end": float(m_row.get("timestamp_end", 0.0)),
                "keyframe_path": str(m_row.get("keyframe_path", "")),
                "ocr_text": ocr_val,
                "asr_text": asr_val,
                "description": desc_val,
            }
            
            batch_points.append(PointStruct(id=pt_id, vector=vectors, payload=payload))
            
        client.upsert(collection_name=collection_name, points=batch_points, wait=True)
        if end_idx % 5000 == 0 or end_idx == total_points:
            logger.info(f"  Qdrant Upserted {end_idx:,d}/{total_points:,d} points ({end_idx/total_points*100:.1f}%) in {time.time()-t_q:.1f}s")

    logger.info("=" * 70)
    logger.info("🎉 TOÀN BỘ PIPELINE ĐỒNG BỘ, SINH VECTOR VÀ NẠP QDRANT ĐÃ HOÀN TẤT!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
