"""
tools/update_l25_descriptions.py
=================================
Cập nhật description L25 từ D:/AIC2026/data/gemini_output vào:
  1. data/unified/scene_descriptions.parquet (có backup)
  2. data/unified/description_embeddings.npy + .pt (re-encode BGE-Large-EN-v1.5)

Điều kiện bắt buộc:
  - Chỉ map frame nếu file ảnh thật (.webp) tồn tại trong data/image/L25/L25/{vid}/{frame_id}.webp
"""

import json
import shutil
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("update_l25")

BASE = Path("D:/AIC2026")
PARQUET_PATH     = BASE / "data/unified/scene_descriptions.parquet"
EMBED_NPY_PATH   = BASE / "data/unified/description_embeddings.npy"
EMBED_PT_PATH    = BASE / "data/unified/description_embeddings.pt"
SCENE_IDS_PATH   = BASE / "data/unified/scene_ids.txt"
GEMINI_DIR       = BASE / "data/gemini_output"
IMAGE_DIR        = BASE / "data/image/L25/L25"


def step1_update_parquet() -> pd.DataFrame:
    log.info("📂 Loading scene_descriptions.parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    total_rows = len(df)
    log.info(f"  Loaded {total_rows:,} total scenes")

    # Tạo backup
    ts = int(time.time())
    backup_path = PARQUET_PATH.parent / f"scene_descriptions_backup_{ts}.parquet"
    shutil.copy2(PARQUET_PATH, backup_path)
    log.info(f"  🛡️ Backup created: {backup_path.name}")

    log.info("🔍 Mapping Gemini descriptions for L25 scenes with existing image files...")
    updated_count = 0
    skipped_no_img = 0
    skipped_no_json = 0
    skipped_empty = 0

    json_cache: dict[str, dict] = {}

    for idx, row in df.iterrows():
        sid = str(row["scene_id"])
        if not sid.startswith("L25_"):
            continue

        vid = row["video_id"].replace("L25_", "")
        frame_str = str(row["frame_id"])

        # 1. Kiểm tra ảnh thật tồn tại
        img_path = IMAGE_DIR / vid / f"{frame_str}.webp"
        if not img_path.exists():
            skipped_no_img += 1
            continue

        # 2. Kiểm tra JSON tồn tại
        json_file = GEMINI_DIR / f"{vid}_descriptions.json"
        if not json_file.exists():
            skipped_no_json += 1
            continue

        if vid not in json_cache:
            with open(json_file, encoding="utf-8") as f:
                json_cache[vid] = json.load(f)
        data = json_cache[vid]

        if frame_str not in data:
            skipped_empty += 1
            continue

        new_desc = (data[frame_str].get("description") or "").strip()
        if not new_desc:
            skipped_empty += 1
            continue

        df.at[idx, "description"] = new_desc
        updated_count += 1

    log.info(f"  ✅ Successfully mapped: {updated_count:,} L25 descriptions")
    log.info(f"  ⏭️ Skipped (no image file): {skipped_no_img}")
    log.info(f"  ⏭️ Skipped (no JSON file): {skipped_no_json}")
    log.info(f"  ⏭️ Skipped (empty desc): {skipped_empty}")

    df.to_parquet(PARQUET_PATH, index=False)
    log.info(f"💾 Saved updated parquet to {PARQUET_PATH}")
    return df


def step2_reencode_embeddings(df: pd.DataFrame):
    log.info("⚡ Re-encoding L25 description embeddings with BGE-large-en-v1.5...")

    scene_ids = SCENE_IDS_PATH.read_text(encoding="utf-8").strip().splitlines()
    sid_to_idx = {sid: i for i, sid in enumerate(scene_ids)}

    if not EMBED_NPY_PATH.exists():
        raise FileNotFoundError(f"Matrix not found: {EMBED_NPY_PATH}")

    # Backup matrix cũ
    ts = int(time.time())
    backup_emb = EMBED_NPY_PATH.parent / f"description_embeddings_backup_{ts}.npy"
    shutil.copyfile(EMBED_NPY_PATH, backup_emb)
    log.info(f"  🛡️ Backup matrix created: {backup_emb.name}")

    embeds = np.load(EMBED_NPY_PATH).astype(np.float32)
    log.info(f"  Loaded existing embeddings: {embeds.shape}")

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    log.info(f"  Loading model {MODEL_NAME} on GPU/CPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.eval().to(device)
    log.info(f"  Model loaded on {device}")

    l25_df = df[df["scene_id"].str.startswith("L25_")].copy()
    l25_sids = l25_df["scene_id"].tolist()
    l25_descs = l25_df["description"].tolist()
    log.info(f"  Total L25 scenes to re-encode: {len(l25_sids):,}")

    BATCH_SIZE = 64
    _t0 = time.time()

    def encode_batch(texts: list[str]) -> np.ndarray:
        clean_texts = [t if t.strip() else "general video scene" for t in texts]
        enc = tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**enc)
            # CLS pooling + L2 normalization (standard BGE passage embedding)
            cls_repr = outputs[0][:, 0]
            norm_repr = torch.nn.functional.normalize(cls_repr, p=2, dim=1)
        return norm_repr.float().cpu().numpy()

    for start in range(0, len(l25_sids), BATCH_SIZE):
        batch_sids = l25_sids[start:start + BATCH_SIZE]
        batch_descs = l25_descs[start:start + BATCH_SIZE]
        vecs = encode_batch(batch_descs)
        for sid, vec in zip(batch_sids, vecs):
            if sid in sid_to_idx:
                embeds[sid_to_idx[sid]] = vec
        done = min(start + BATCH_SIZE, len(l25_sids))
        if (start // BATCH_SIZE) % 10 == 0 or done == len(l25_sids):
            elapsed = time.time() - _t0
            log.info(f"  Progress: [{done:,}/{len(l25_sids):,}] ({done/len(l25_sids)*100:.1f}%) | Time: {elapsed:.1f}s")

    np.save(EMBED_NPY_PATH, embeds)
    torch.save(torch.from_numpy(embeds), EMBED_PT_PATH)
    log.info(f"💾 Saved updated embeddings to: {EMBED_NPY_PATH} & {EMBED_PT_PATH}")
    log.info("🎉 All L25 descriptions and embeddings have been successfully updated!")


if __name__ == "__main__":
    updated_df = step1_update_parquet()
    step2_reencode_embeddings(updated_df)
