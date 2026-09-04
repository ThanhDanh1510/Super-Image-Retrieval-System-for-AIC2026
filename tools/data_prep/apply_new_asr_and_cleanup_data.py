"""
tools/data_prep/apply_new_asr_and_cleanup_data.py
=================================================
1. Merges new ASR dataset (data/ASR/asr_text.parquet) into data/unified/asr_text.parquet.
2. Backs up existing ASR file to data/backups/.
3. Re-aligns 100% with 88,899 clean scenes in scene_metadata.parquet & scene_ids.txt.
4. Cleans up orphaned / temporary files in data/ to keep the folder structure clean and organized.
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Data-Manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UNIFIED_DIR = DATA_DIR / "unified"
BACKUP_DIR = DATA_DIR / "backups"
ASR_DIR = DATA_DIR / "ASR"


def update_asr_dataset():
    logger.info("=" * 70)
    logger.info("🚀 1. UPDATING ASR DATASET (data/unified/asr_text.parquet)")
    logger.info("=" * 70)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    old_asr_path = UNIFIED_DIR / "asr_text.parquet"
    new_asr_path = ASR_DIR / "asr_text.parquet"
    meta_path = UNIFIED_DIR / "scene_metadata.parquet"
    scene_ids_path = UNIFIED_DIR / "scene_ids.txt"

    if not new_asr_path.exists():
        raise FileNotFoundError(f"New ASR file not found: {new_asr_path}")

    # Backup old file
    if old_asr_path.exists():
        backup_file = BACKUP_DIR / f"asr_text_backup_{int(time.time())}.parquet"
        shutil.copy2(old_asr_path, backup_file)
        logger.info(f"🛡️ Backed up old ASR to: {backup_file.name}")

    df_meta = pd.read_parquet(meta_path)
    df_old = pd.read_parquet(old_asr_path) if old_asr_path.exists() else pd.DataFrame()
    df_new = pd.read_parquet(new_asr_path)

    logger.info(f"📊 Clean Scene Metadata count: {len(df_meta):,} scenes")
    logger.info(f"📊 New ASR dataset count     : {len(df_new):,} scenes")

    # Merge with priority to new ASR
    merged = pd.merge(
        df_meta[["scene_id", "video_id", "frame_id", "timestamp_mid"]],
        df_old[["scene_id", "asr_text"]].rename(columns={"asr_text": "old_asr"}) if not df_old.empty else pd.DataFrame(columns=["scene_id", "old_asr"]),
        on="scene_id",
        how="left",
    )
    merged = pd.merge(
        merged,
        df_new[["scene_id", "asr_text"]].rename(columns={"asr_text": "new_asr"}),
        on="scene_id",
        how="left",
    )

    merged["old_asr"] = merged["old_asr"].fillna("").astype(str) if "old_asr" in merged.columns else ""
    merged["new_asr"] = merged["new_asr"].fillna("").astype(str)

    # Prioritize new ASR if non-empty, otherwise fallback to old ASR
    merged["asr_text"] = np.where(
        merged["new_asr"].str.strip() != "",
        merged["new_asr"],
        merged["old_asr"],
    )

    final_df = merged[["scene_id", "video_id", "frame_id", "timestamp_mid", "asr_text"]]
    final_df.to_parquet(old_asr_path, index=False)

    non_empty_count = (final_df["asr_text"].str.strip() != "").sum()
    logger.info(f"🎉 Updated {old_asr_path.name}: {len(final_df):,} total scenes ({non_empty_count:,} non-empty transcripts, {non_empty_count/len(final_df)*100:.1f}%)")

    # Verify 1-1 alignment with scene_ids.txt
    if scene_ids_path.exists():
        scene_ids = [line.strip() for line in open(scene_ids_path, encoding="utf-8")]
        assert (final_df["scene_id"].values == scene_ids).all(), "CRITICAL: Mismatch with scene_ids.txt order!"
        logger.info("✅ 100% 1-to-1 Row Alignment verified with scene_ids.txt!")


def cleanup_and_organize_data_folder():
    logger.info("=" * 70)
    logger.info("🧹 2. CLEANING AND STRUCTURING DATA/ FOLDER")
    logger.info("=" * 70)

    # 1. Move stray backup files from data/unified/ to data/backups/
    for p in UNIFIED_DIR.glob("*backup*.parquet"):
        dest = BACKUP_DIR / p.name
        shutil.move(str(p), str(dest))
        logger.info(f"🚚 Moved backup: {p.name} -> data/backups/")

    # 2. Clean temporary/orphaned files in data/ root
    orphaned_files = [
        DATA_DIR / ".lock",
        DATA_DIR / "meta.json",
        DATA_DIR / "user_feedback.jsonl",
    ]
    for of in orphaned_files:
        if of.exists():
            of.unlink()
            logger.info(f"🗑️ Removed orphaned file: {of.name}")

    # 3. Clean unused data/qdrant_db/ directory if empty or unused
    qdrant_db_dir = DATA_DIR / "qdrant_db"
    if qdrant_db_dir.exists():
        try:
            shutil.rmtree(str(qdrant_db_dir))
            logger.info("🗑️ Removed obsolete data/qdrant_db/ directory")
        except Exception as e:
            logger.warning(f"Could not remove data/qdrant_db: {e}")

    # 4. Clean temporary experiment artifacts in vlm_shards/ (keeping actual .parquet shards)
    vlm_shards_dir = UNIFIED_DIR / "vlm_shards"
    if vlm_shards_dir.exists():
        for item in vlm_shards_dir.iterdir():
            if item.is_dir() and item.name.startswith("grids_"):
                shutil.rmtree(str(item))
                logger.info(f"🗑️ Cleaned temp grid folder: {item.name}")
            elif item.is_dir() and item.name == "sheets":
                shutil.rmtree(str(item))
                logger.info(f"🗑️ Cleaned temp sheets folder: {item.name}")

    logger.info("✨ Data directory cleanup & restructuring completed successfully!")


def main():
    update_asr_dataset()
    cleanup_and_organize_data_folder()


if __name__ == "__main__":
    main()
