import os
import sys
import re
import json
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, bytes, str, object],
    }

from tqdm import tqdm
from shared.ocr.ocr_local import _init_detector, _init_recognizer, should_filter_text

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def run_ultra_gpu_ocr(dataset_dir: str, output_dir: str, prefix: str):
    print("=" * 75)
    print(f"🔥 ULTRA GPU CUDA BATCH ENGINE (MAX CUDA CORE SATURATION)")
    print(f"📁 Dataset Dir : {dataset_dir}")
    print(f"💾 Output Dir  : {output_dir}")
    print(f"⚡ GPU Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 75)

    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Thư mục không tồn tại: {dataset_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Initialize engines
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
        print(f"\n🎬 [{idx}/{len(video_folders)}] Batch Processing {v_id} ({len(img_files)} frames)...")

        # Process frames in large batches to saturate GPU
        batch_size = 16
        for i in tqdm(range(0, len(img_files), batch_size), desc=f"CUDA Saturation {v_id}"):
            chunk = img_files[i:i + batch_size]
            for img_file in chunk:
                frame_id = Path(img_file).stem
                img_path = os.path.join(folder_path, img_file)

                # Detection on GPU
                dt_boxes, _ = detector.predict(img_path)
                if not dt_boxes:
                    ocr_text = ""
                else:
                    # Crop boxes
                    import cv2
                    img = cv2.imread(img_path)
                    crops = []
                    if img is not None:
                        for box in dt_boxes:
                            xs = [int(p[0]) for p in box]
                            ys = [int(p[1]) for p in box]
                            x1, x2 = max(0, min(xs)), min(img.shape[1], max(xs))
                            y1, y2 = max(0, min(ys)), min(img.shape[0], max(ys))
                            if x2 > x1 and y2 > y1:
                                crop = img[y1:y2, x1:x2]
                                from PIL import Image
                                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

                    if crops:
                        # Batch recognition on GPU CUDA
                        texts = recognizer.predict_batch(crops)
                        clean_lines = [t.strip() for t in texts if t and not should_filter_text(t)]
                        ocr_text = "\n".join(clean_lines)
                    else:
                        ocr_text = ""

                scene_id = f"{v_id}_S{int(frame_id):03d}" if frame_id.isdigit() else f"{v_id}_{frame_id}"
                all_records.append({
                    "scene_id": scene_id,
                    "video_id": v_id,
                    "frame_id": frame_id,
                    "ocr_text": ocr_text
                })
                total_frames += 1

    # Save consolidated Parquet
    df_consolidated = pd.DataFrame(all_records)
    output_parquet_path = os.path.join(output_dir, f"{prefix}_ocr.parquet")
    df_consolidated.to_parquet(output_parquet_path, index=False)

    elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"✅ HOÀN THÀNH XỬ LÝ ULTRA GPU OCR CHO {prefix}!")
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
    parser = argparse.ArgumentParser(description="Ultra GPU CUDA Batch OCR Pipeline")
    parser.add_argument("--dataset-dir", required=True, help="Thư mục chứa các folder video")
    parser.add_argument("--output-dir", default="data/ocr_output", help="Thư mục lưu file parquet gộp")
    parser.add_argument("--prefix", required=True, help="Prefix dataset, vd L28")

    args = parser.parse_args()
    run_ultra_gpu_ocr(args.dataset_dir, args.output_dir, args.prefix)
