"""
tools/data_prep/generate_l25_image_vectors.py
============================================
Sinh CLIP ViT-L/14 image vectors cho batch L25 (8,268 ảnh)
và merge vào data/unified/image_vectors.npy, scene_metadata.parquet, scene_ids.txt.
"""

import os
import sys
import time
import re
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def main():
    print("=" * 60)
    print("🚀 SINH CLIP EMBEDDING CHO BATCH L25")
    print("=" * 60)

    l25_dir = PROJECT_ROOT / "data" / "image" / "L25" / "L25"
    if not l25_dir.exists():
        l25_dir = PROJECT_ROOT / "data" / "image" / "L25"

    all_files = sorted(l25_dir.rglob("*.webp"), key=natural_sort_key)
    print(f"📁 Tìm thấy {len(all_files)} ảnh keyframe trong {l25_dir}")

    if not all_files:
        print("❌ Không tìm thấy ảnh nào!")
        return

    # Build metadata records
    records = []
    vid_counts = {}
    for f in all_files:
        vid = f.parent.name
        fid = int(f.stem)
        vid_counts[vid] = vid_counts.get(vid, 0) + 1
        idx = vid_counts[vid]
        full_vid = f"L25_{vid}" if not vid.startswith("L25") else vid
        short_vid = vid.replace("L25_", "")
        sid = f"{full_vid}_S{idx:03d}"
        records.append({
            "scene_id": sid,
            "video_id": full_vid,
            "frame_id": fid,
            "fps": 25.0,
            "timestamp_start": round(max(0, fid - 25) / 25.0, 2),
            "timestamp_mid": round(fid / 25.0, 2),
            "timestamp_end": round((fid + 25) / 25.0, 2),
            "keyframe_path": f"L25/{short_vid}/{f.name}",
            "file_path": str(f),
        })

    l25_meta_df = pd.DataFrame(records)
    print(f"✅ Đã tạo metadata cho {len(l25_meta_df)} scenes của {len(vid_counts)} videos.")

    print("\n⚡ Nạp mô hình CLIP ViT-L/14 trên CUDA...")
    embedder = ImageEmbeddingWrapper(device="cuda")

    print(f"\n🔄 Bắt đầu sinh vector cho {len(records)} ảnh (Batch size: 64)...")
    batch_size = 64
    all_vectors = []
    t0 = time.time()

    for i in range(0, len(records), batch_size):
        batch_recs = records[i : i + batch_size]
        imgs = []
        for r in batch_recs:
            try:
                img = Image.open(r["file_path"]).convert("RGB")
                imgs.append(img)
            except Exception as e:
                print(f"Lỗi đọc ảnh {r['file_path']}: {e}")
                imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        vecs = embedder.encode(imgs, batch_size=batch_size)
        all_vectors.append(vecs)

        done = min(i + batch_size, len(records))
        if done % 512 == 0 or done == len(records):
            elapsed = time.time() - t0
            speed = done / max(elapsed, 0.001)
            print(f"  Đã encode {done}/{len(records)} ảnh ({done/len(records)*100:.1f}%) — Tốc độ: {speed:.1f} img/s")

    l25_vecs = np.vstack(all_vectors)
    print(f"🎉 Hoàn tất sinh vector L25! Shape: {l25_vecs.shape} trong {time.time() - t0:.2f}s")

    print("\n💾 Hợp nhất vào data/unified/...")
    unified_dir = PROJECT_ROOT / "data" / "unified"

    meta_path = unified_dir / "scene_metadata.parquet"
    vec_path = unified_dir / "image_vectors.npy"
    scene_txt_path = unified_dir / "scene_ids.txt"

    orig_meta = pd.read_parquet(meta_path)
    # Loại bỏ L25 cũ nếu có
    clean_orig_meta = orig_meta[~orig_meta["video_id"].str.startswith("L25")].copy()
    clean_orig_indices = np.where(~orig_meta["video_id"].str.startswith("L25"))[0]

    orig_image_vecs = np.load(vec_path)
    clean_orig_image_vecs = orig_image_vecs[clean_orig_indices]

    # Nối L25 mới
    combined_meta = pd.concat([clean_orig_meta, l25_meta_df.drop(columns=["file_path"])], ignore_index=True)
    combined_image_vecs = np.vstack([clean_orig_image_vecs, l25_vecs])
    combined_scene_ids = list(combined_meta["scene_id"])

    print(f"- Metadata mới       : {len(combined_meta)} rows")
    print(f"- Image vectors mới  : {combined_image_vecs.shape}")
    print(f"- Scene IDs mới      : {len(combined_scene_ids)} lines")

    combined_meta.to_parquet(meta_path, index=False)
    np.save(vec_path, combined_image_vecs)
    with open(scene_txt_path, "w", encoding="utf-8") as f:
        for sid in combined_scene_ids:
            f.write(f"{sid}\n")

    print("=" * 60)
    print("🎉 ĐỒNG BỘ VECTOR & METADATA L25 CHO IMAGE FILTER SERVER THÀNH CÔNG!")
    print("=" * 60)

if __name__ == "__main__":
    main()
