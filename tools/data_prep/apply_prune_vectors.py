"""
tools/data_prep/apply_prune_vectors.py
======================================
Cắt và lưu đồng bộ 4 bộ Vectors và scene_ids.txt theo đúng 88,899 scenes sạch:
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"

def main():
    print("=" * 70)
    print("🚀 BẮT ĐẦU CẮT ĐỒNG BỘ VECTORS & SCENE_IDS (88,899 SCENES)")
    print("=" * 70)

    # 1. Đọc metadata và scene_ids
    meta_df = pd.read_parquet(UNIFIED_DIR / "scene_metadata.parquet")
    print(f"📊 scene_metadata.parquet: {len(meta_df):,d} rows")

    with open(UNIFIED_DIR / "scene_ids.txt", "r", encoding="utf-8") as f:
        sids = [line.strip() for line in f if line.strip()]
    print(f"📊 scene_ids.txt (original): {len(sids):,d} lines")

    kept_sids_set = set(meta_df["scene_id"])
    keep_mask = np.array([sid in kept_sids_set for sid in sids], dtype=bool)
    print(f"✅ Mask: {np.sum(keep_mask):,d} KEEP | {len(sids) - np.sum(keep_mask):,d} PRUNED")

    # 2. Cắt và lưu image_vectors.npy
    img_path = UNIFIED_DIR / "image_vectors.npy"
    if img_path.exists():
        img_vecs = np.load(img_path)
        if img_vecs.shape[0] == len(sids):
            clean_img = img_vecs[keep_mask]
            np.save(img_path, clean_img)
            print(f"  + image_vectors.npy: {clean_img.shape}")
        else:
            print(f"  + image_vectors.npy already sliced: {img_vecs.shape}")

    # 3. Cắt và lưu ocr_vectors.npy
    ocr_path = UNIFIED_DIR / "ocr_vectors.npy"
    if ocr_path.exists():
        ocr_vecs = np.load(ocr_path)
        if ocr_vecs.shape[0] == len(sids):
            clean_ocr = ocr_vecs[keep_mask]
            np.save(ocr_path, clean_ocr)
            print(f"  + ocr_vectors.npy: {clean_ocr.shape}")
        else:
            print(f"  + ocr_vectors.npy already sliced: {ocr_vecs.shape}")

    # 4. Cắt và lưu asr_vectors.npy
    asr_path = UNIFIED_DIR / "asr_vectors.npy"
    if asr_path.exists():
        asr_vecs = np.load(asr_path)
        if asr_vecs.shape[0] == len(sids):
            clean_asr = asr_vecs[keep_mask]
            np.save(asr_path, clean_asr)
            print(f"  + asr_vectors.npy: {clean_asr.shape}")
        else:
            print(f"  + asr_vectors.npy already sliced: {asr_vecs.shape}")

    # 5. Cắt và lưu description_embeddings.pt
    desc_path = UNIFIED_DIR / "description_embeddings.pt"
    if desc_path.exists():
        desc_tensor = torch.load(desc_path, map_location="cpu", weights_only=False)
        if desc_tensor.shape[0] == len(sids):
            clean_desc = desc_tensor[keep_mask]
            torch.save(clean_desc, desc_path)
            print(f"  + description_embeddings.pt: {clean_desc.shape}")
        else:
            print(f"  + description_embeddings.pt already sliced: {desc_tensor.shape}")

    # 6. Lưu scene_ids.txt
    clean_sids = [sid for sid, k in zip(sids, keep_mask) if k]
    with open(UNIFIED_DIR / "scene_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(clean_sids) + "\n")
    print(f"  + scene_ids.txt: {len(clean_sids):,d} lines")

    # 7. Kiểm tra đồng bộ tất cả các file
    print("\n" + "=" * 70)
    print("🔍 KIỂM TRA ĐỒNG BỘ TOÀN BỘ 9 FILE TRONG data/unified/:")
    print("=" * 70)
    for p in sorted(UNIFIED_DIR.iterdir()):
        if p.name.endswith(".parquet"):
            df = pd.read_parquet(p)
            print(f"  📄 {p.name:<30}: {len(df):,d} rows")
        elif p.name.endswith(".npy"):
            arr = np.load(p)
            print(f"  📄 {p.name:<30}: shape {arr.shape}")
        elif p.name.endswith(".pt"):
            t = torch.load(p, map_location="cpu", weights_only=False)
            print(f"  📄 {p.name:<30}: shape {t.shape}")
        elif p.name.endswith(".txt"):
            with open(p, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            print(f"  📄 {p.name:<30}: {len(lines):,d} lines")
    print("=" * 70)

if __name__ == "__main__":
    main()
