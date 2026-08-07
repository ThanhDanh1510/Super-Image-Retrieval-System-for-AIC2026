import os
import sys
import re
import json
import argparse
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# Fix NumPy 2.0+ compatibility
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, bytes, str, object],
    }

from tqdm import tqdm
from ocr_local import _init_detector, _init_recognizer, ocr_image_quality, should_filter_text

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def run_batch_dataset(dataset_dir: str, output_dir: str, prefix: str):
    print("=" * 70)
    print(f"🚀 KHỞI CHẠY BATCH OCR GỘP TẬP TRUNG CHO DATASET {prefix}")
    print(f"📁 Dataset Dir: {dataset_dir}")
    print(f"💾 Output Dir : {output_dir}")
    print("=" * 70)

    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Thư mục không tồn tại: {dataset_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Engines
    detector = _init_detector()
    recognizer = _init_recognizer()

    # 2. Get all video folders
    all_entries = os.listdir(dataset_dir)
    video_folders = sorted(
        [f for f in all_entries if os.path.isdir(os.path.join(dataset_dir, f))],
        key=natural_sort_key
    )

    print(f"📹 Tìm thấy {len(video_folders)} video folders trong {prefix}.")

    all_records = []
    total_start = time.time()
    total_frames = 0

    for idx, v_folder in enumerate(video_folders, 1):
        folder_path = os.path.join(dataset_dir, v_folder)
        img_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))],
            key=natural_sort_key
        )

        v_id = f"{prefix}_{v_folder}" if not v_folder.startswith(prefix) else v_folder
        print(f"\n🎬 [{idx}/{len(video_folders)}] Xử lý {v_id} ({len(img_files)} frames)...")

        for img_file in tqdm(img_files, desc=f"Progress {v_id}"):
            frame_id = Path(img_file).stem
            img_path = os.path.join(folder_path, img_file)

            quality, err = ocr_image_quality(
                detector, recognizer, img_path,
                filter_spam=True,
                filter_exclude_region=True
            )

            ocr_text = quality.text.strip()
            scene_id = f"{v_id}_S{int(frame_id):03d}" if frame_id.isdigit() else f"{v_id}_{frame_id}"

            all_records.append({
                "scene_id": scene_id,
                "video_id": v_id,
                "frame_id": frame_id,
                "ocr_text": ocr_text
            })
            total_frames += 1

    # 3. Save single consolidated parquet for this prefix
    df_consolidated = pd.DataFrame(all_records)
    output_parquet_path = os.path.join(output_dir, f"{prefix}_ocr.parquet")
    df_consolidated.to_parquet(output_parquet_path, index=False)

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"✅ HOÀN THÀNH XỬ LÝ {prefix}!")
    print(f"📊 Tổng số frames: {total_frames} frames")
    print(f"⏱️ Tổng thời gian : {elapsed:.1f}s ({elapsed/max(1, total_frames):.2f}s/frame)")
    print(f"💾 File Parquet gộp duy nhất: {output_parquet_path}")
    print("=" * 70)

    # 4. Merge directly into data/unified/ocr_text.parquet
    unified_path = "data/unified/ocr_text.parquet"
    if os.path.exists(unified_path):
        try:
            df_unified = pd.read_parquet(unified_path)
            print(f"🔄 Đang cập nhật vào dữ liệu hệ thống API ({unified_path})...")
            df_unified_filtered = df_unified[~df_unified["video_id"].str.startswith(prefix)]
            new_unified_df = pd.concat([
                df_unified_filtered,
                df_consolidated[["scene_id", "video_id", "ocr_text"]]
            ], ignore_index=True)
            new_unified_df.to_parquet(unified_path, index=False)
            print(f"🎉 Cập nhật thành công {unified_path}! (Tổng số scenes: {len(new_unified_df)})")
        except Exception as e:
            print(f"⚠️ Không thể cập nhật {unified_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy OCR gộp file tập trung cho một prefix dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Thư mục chứa các folder video, vd: data/image/L21/L21")
    parser.add_argument("--output-dir", default="data/ocr_output", help="Thư mục lưu file parquet gộp duy nhất")
    parser.add_argument("--prefix", required=True, help="Tiền tố dataset, vd: L21 hoặc L22")

    args = parser.parse_args()
    run_batch_dataset(args.dataset_dir, args.output_dir, args.prefix)
