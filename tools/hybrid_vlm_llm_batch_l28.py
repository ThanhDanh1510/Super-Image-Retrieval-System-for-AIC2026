import os
import sys
import re
import time
import argparse
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
from shared.ocr.ocr_local import _init_detector, _init_recognizer, ocr_image_hybrid_quality, get_vlm_engine

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def run_hybrid_batch_l28(dataset_dir: str, output_dir: str, only_folder: str = None):
    print("=" * 80)
    print("🚀 KHỞI CHẠY BATCH OCR 2 TẦNG SOTA (VIETOCR + VINTERN VLM + LLM POST-PROCESSING)")
    print(f"📁 Dataset Dir : {dataset_dir}")
    print(f"💾 Output Dir  : {output_dir}")
    if only_folder:
        print(f"🎯 Test Folder  : {only_folder}")
    print(f"⚡ GPU Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Thư mục không tồn tại: {dataset_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Initialize engines (LOAD WEIGHTS 1 LẦN DUY NHẤT VÀO VRAM)
    detector = _init_detector()
    recognizer = _init_recognizer()
    vlm_model, vlm_tokenizer = get_vlm_engine()

    # Maximize PyTorch CUDA hardware performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    all_entries = os.listdir(dataset_dir)
    video_folders = sorted(
        [f for f in all_entries if os.path.isdir(os.path.join(dataset_dir, f))],
        key=natural_sort_key
    )

    if only_folder:
        video_folders = [f for f in video_folders if f == only_folder or f == f"V{only_folder.lstrip('V')}"]
        if not video_folders:
            print(f"❌ Không tìm thấy folder: {only_folder}")
            return

    print(f"📹 Xử lý Siêu Tốc 2 Tầng cho {len(video_folders)} video folders trong L28.")

    all_records = []
    total_start = time.time()
    total_frames = 0

    for idx, v_folder in enumerate(video_folders, 1):
        folder_path = os.path.join(dataset_dir, v_folder)
        img_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))],
            key=natural_sort_key
        )

        v_id = f"L28_{v_folder}" if not v_folder.startswith("L28") else v_folder
        print(f"\n🎬 [{idx}/{len(video_folders)}] Xử lý CUDA Stream 2 Tầng cho {v_id} ({len(img_files)} frames)...")

        for img_file in tqdm(img_files, desc=f"Dual-Track SOTA CUDA {v_id}"):
            frame_id = Path(img_file).stem
            img_path = os.path.join(folder_path, img_file)

            quality, err, candidates = ocr_image_hybrid_quality(
                detector,
                recognizer,
                img_path,
                backend="hybrid",
                filter_spam=True,
                filter_exclude_region=True,
            )

            ocr_text = quality.text.strip()
            scene_id = f"{v_id}_S{int(frame_id):03d}" if frame_id.isdigit() else f"{v_id}_{frame_id}"

            all_records.append({
                "scene_id": scene_id,
                "video_id": v_id,
                "frame_id": frame_id,
                "ocr_text": ocr_text,
                "local_raw": candidates.get("local", ""),
                "vlm_raw": candidates.get("vlm", "")
            })
            total_frames += 1

    # Save consolidated Parquet
    df_consolidated = pd.DataFrame(all_records)
    output_parquet_path = os.path.join(output_dir, "L28_V001_test_hybrid.parquet" if only_folder else "L28_ocr.parquet")
    df_consolidated.to_parquet(output_parquet_path, index=False)

    elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH XỬ LÝ OCR 2 TẦNG SOTA CHO TEST FOLDER!")
    print(f"📊 Tổng số frames  : {total_frames} frames")
    print(f"⏱️ Tổng thời gian   : {elapsed:.1f}s ({elapsed/max(1, total_frames):.2f}s/frame)")
    print(f"💾 File Parquet Gộp: {output_parquet_path}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-folder", default="V001", help="Folder video test")
    args = parser.parse_args()

    dataset_dir = os.path.join(PROJECT_ROOT, "data", "image", "L28", "L28")
    output_dir = os.path.join(PROJECT_ROOT, "data", "ocr_output")
    run_hybrid_batch_l28(dataset_dir, output_dir, only_folder=args.only_folder)
