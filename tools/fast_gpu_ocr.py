import os
import sys
import re
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, bytes, str, object],
    }

from tqdm import tqdm
from shared.ocr.ocr_local import _init_detector, _init_recognizer, ocr_image_quality, should_filter_text

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def run_fast_gpu_ocr(dataset_dir: str, output_dir: str, prefix: str, num_threads: int = 8):
    print("=" * 75)
    print(f"⚡ FAST HIGH-THROUGHPUT GPU OCR (MULTI-THREADED PREFETCH + CUDA TENSOR BATCHING)")
    print(f"📁 Dataset Dir : {dataset_dir}")
    print(f"💾 Output Dir  : {output_dir}")
    print(f"🧵 CPU Workers : {num_threads} threads parallel prefetch")
    print("=" * 75)

    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Thư mục không tồn tại: {dataset_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Init engines on CUDA
    detector = _init_detector()
    recognizer = _init_recognizer()

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
        print(f"\n🎬 [{idx}/{len(video_folders)}] Processing {v_id} ({len(img_files)} frames)...")

        # Multi-thread prefetching image paths
        img_paths = [os.path.join(folder_path, f) for f in img_files]

        def process_single(path):
            frame_id = Path(path).stem
            quality, err = ocr_image_quality(
                detector, recognizer, path,
                filter_spam=True,
                filter_exclude_region=True
            )
            ocr_text = quality.text.strip()
            scene_id = f"{v_id}_S{int(frame_id):03d}" if frame_id.isdigit() else f"{v_id}_{frame_id}"
            return {
                "scene_id": scene_id,
                "video_id": v_id,
                "frame_id": frame_id,
                "ocr_text": ocr_text
            }

        # Run parallel batch processing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batch_results = list(tqdm(
                executor.map(process_single, img_paths),
                total=len(img_paths),
                desc=f"Parallel CUDA Batch {v_id}"
            ))

        all_records.extend(batch_results)
        total_frames += len(batch_results)

    # Save consolidated Parquet
    df_consolidated = pd.DataFrame(all_records)
    output_parquet_path = os.path.join(output_dir, f"{prefix}_ocr.parquet")
    df_consolidated.to_parquet(output_parquet_path, index=False)

    elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"✅ HOÀN THÀNH XỬ LÝ FAST GPU OCR CHO {prefix}!")
    print(f"📊 Tổng số frames : {total_frames} frames")
    print(f"⏱️ Tổng thời gian  : {elapsed:.1f}s ({elapsed/max(1, total_frames):.3f}s/frame)")
    print(f"🚀 Tốc độ trích xuất: {total_frames/max(1, elapsed):.1f} frames/giây")
    print(f"💾 Parquet File   : {output_parquet_path}")
    print("=" * 75)

    # Update unified parquet
    unified_path = "data/unified/ocr_text.parquet"
    if os.path.exists(unified_path):
        try:
            df_unified = pd.read_parquet(unified_path)
            df_filtered = df_unified[~df_unified["video_id"].str.startswith(prefix)]
            new_unified = pd.concat([
                df_filtered,
                df_consolidated[["scene_id", "video_id", "ocr_text"]]
            ], ignore_index=True)
            new_unified.to_parquet(unified_path, index=False)
            print(f"🎉 Cập nhật thành công {unified_path}! (Tổng scenes: {len(new_unified)})")
        except Exception as e:
            print(f"⚠️ Error updating unified parquet: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Multi-threaded High-Throughput GPU OCR Pipeline")
    parser.add_argument("--dataset-dir", required=True, help="Thư mục chứa các folder video")
    parser.add_argument("--output-dir", default="data/ocr_output", help="Thư mục lưu file parquet gộp")
    parser.add_argument("--prefix", required=True, help="Prefix dataset, vd L28")
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU nạp ảnh song song cho GPU")

    args = parser.parse_args()
    run_fast_gpu_ocr(args.dataset_dir, args.output_dir, args.prefix, num_threads=args.threads)
