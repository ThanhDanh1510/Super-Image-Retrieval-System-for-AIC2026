import os
import sys
import re
import json
import argparse
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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

    # 2. Get image targets (flat image files or video subfolders)
    all_entries = os.listdir(dataset_dir)
    flat_img_files = sorted(
        [f for f in all_entries if f.startswith(f"{prefix}_") and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))],
        key=natural_sort_key
    )

    all_records = []
    total_start = time.time()
    total_frames = 0

    if flat_img_files:
        print(f"🖼️ Tìm thấy {len(flat_img_files)} ảnh phẳng trong {dataset_dir} cho {prefix}. Đang chạy OCR GPU...")
        from concurrent.futures import ThreadPoolExecutor

        def process_flat_file(img_file):
            stem = Path(img_file).stem
            img_path = os.path.join(dataset_dir, img_file)
            quality, err = ocr_image_quality(
                detector, recognizer, img_path,
                filter_spam=True,
                filter_exclude_region=True
            )
            ocr_text = quality.text.strip()

            if "-" in stem:
                parts = stem.split("-")
                scene_id = parts[0]
                frame_id = parts[1]
                v_parts = scene_id.split("_")
                v_id = f"{v_parts[0]}_{v_parts[1]}" if len(v_parts) >= 2 else prefix
            else:
                scene_id = stem
                v_id = f"{prefix}_{stem.split('_')[1]}" if "_" in stem else prefix
                frame_id = stem

            return {
                "scene_id": scene_id,
                "video_id": v_id,
                "frame_id": frame_id,
                "ocr_text": ocr_text
            }

        with ThreadPoolExecutor(max_workers=8) as executor:
            all_records = list(tqdm(
                executor.map(process_flat_file, flat_img_files),
                total=len(flat_img_files),
                desc=f"Fast CUDA Flat Batch {prefix}"
            ))
        total_frames = len(all_records)
    else:
        video_folders = sorted(
            [f for f in all_entries if os.path.isdir(os.path.join(dataset_dir, f))],
            key=natural_sort_key
        )

        if video_folders:
            print(f"📹 Tìm thấy {len(video_folders)} video folders trong {prefix}.")
            from concurrent.futures import ThreadPoolExecutor

            for idx, v_folder in enumerate(video_folders, 1):
                folder_path = os.path.join(dataset_dir, v_folder)
                img_files = sorted(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))],
                    key=natural_sort_key
                )

                v_id = f"{prefix}_{v_folder}" if not v_folder.startswith(prefix) else v_folder
                print(f"\n🎬 [{idx}/{len(video_folders)}] Xử lý Siêu Tốc {v_id} ({len(img_files)} frames)...")

                def process_frame(img_file):
                    frame_id = Path(img_file).stem
                    img_path = os.path.join(folder_path, img_file)
                    quality, err = ocr_image_quality(
                        detector, recognizer, img_path,
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

                with ThreadPoolExecutor(max_workers=8) as executor:
                    folder_records = list(tqdm(
                        executor.map(process_frame, img_files),
                        total=len(img_files),
                        desc=f"Fast CUDA Batch {v_id}"
                    ))

                all_records.extend(folder_records)
                total_frames += len(folder_records)

    if total_frames == 0:
        print(f"⚠️ Không tìm thấy khung ảnh thô. Đang trích xuất dữ liệu từ OCR JSONs cho {prefix}...")
        ocr_json_dir = os.path.join(PROJECT_ROOT, "data", "data", "OCR")
        if not os.path.exists(os.path.join(ocr_json_dir, prefix)):
            ocr_json_dir = os.path.join(PROJECT_ROOT, "data", "OCR")

        prefix_json_dir = os.path.join(ocr_json_dir, prefix)
        if os.path.exists(prefix_json_dir):
            import glob
            json_files = glob.glob(os.path.join(prefix_json_dir, "*.json"))
            if json_files:
                print(f"📄 Tìm thấy {len(json_files)} file JSON trong {prefix_json_dir}. Đang khớp với scene_metadata...")
                for m_rel in ["data/data/unified/scene_metadata.parquet", "data/unified/scene_metadata.parquet"]:
                    abs_m_path = os.path.join(PROJECT_ROOT, m_rel)
                    if os.path.exists(abs_m_path):
                        meta_df = pd.read_parquet(abs_m_path)
                        prefix_meta = meta_df[meta_df["video_id"].str.startswith(prefix)].copy()
                        if not prefix_meta.empty:
                            from aggregation.build_unified_from_raw import extract_ocr_for_metadata
                            ocr_sub_df = extract_ocr_for_metadata(prefix_meta, ocr_json_dir)
                            all_records = ocr_sub_df.to_dict(orient="records")
                            total_frames = len(all_records)
                            non_empty_count = (ocr_sub_df["ocr_text"] != "").sum()
                            print(f"📦 Trích xuất thành công {total_frames} scenes ({non_empty_count} non-empty OCR) từ {prefix_json_dir}.")
                            break

        if total_frames == 0:
            for u_path in ["data/data/unified/ocr_text.parquet", "data/unified/ocr_text.parquet"]:
                abs_u_path = os.path.join(PROJECT_ROOT, u_path)
                if os.path.exists(abs_u_path):
                    try:
                        df_u = pd.read_parquet(abs_u_path)
                        df_prefix = df_u[df_u["video_id"].str.startswith(prefix)]
                        if not df_prefix.empty and (df_prefix["ocr_text"] != "").sum() > 0:
                            all_records = df_prefix.to_dict(orient="records")
                            total_frames = len(all_records)
                            print(f"📦 Trích xuất thành công {total_frames} scenes từ {u_path} cho {prefix}.")
                            break
                    except Exception as e:
                        print(f"⚠️ Error reading {u_path}: {e}")

    # 3. Save single consolidated parquet for this prefix
    df_consolidated = pd.DataFrame(all_records)
    output_parquet_path = os.path.join(output_dir, f"{prefix}_ocr.parquet")
    df_consolidated.to_parquet(output_parquet_path, index=False)

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"✅ HOÀN THÀNH XỬ LÝ {prefix}!")
    print(f"📊 Tổng số frames/scenes: {total_frames}")
    print(f"⏱️ Tổng thời gian : {elapsed:.1f}s ({elapsed/max(1, total_frames):.2f}s/scene)")
    print(f"💾 File Parquet gộp duy nhất: {output_parquet_path}")
    print("=" * 70)

    # 4. Merge directly into unified ocr_text.parquet files
    unified_paths = ["data/data/unified/ocr_text.parquet", "data/unified/ocr_text.parquet"]
    for u_rel in unified_paths:
        unified_path = os.path.join(PROJECT_ROOT, u_rel)
        if os.path.exists(unified_path):
            try:
                df_unified = pd.read_parquet(unified_path)
                print(f"🔄 Đang cập nhật vào dữ liệu hệ thống API ({unified_path})...")
                df_unified_filtered = df_unified[~df_unified["video_id"].str.startswith(prefix)]
                cols_to_concat = [c for c in ["scene_id", "video_id", "ocr_text"] if c in df_consolidated.columns]
                new_unified_df = pd.concat([
                    df_unified_filtered,
                    df_consolidated[cols_to_concat]
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
