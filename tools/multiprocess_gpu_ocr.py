import os
import sys
import re
import time
import argparse
from pathlib import Path
import torch.multiprocessing as mp

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

from ocr_local import _init_detector, _init_recognizer, ocr_image_hybrid_quality, get_vlm_engine

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def worker_process(worker_id: int, folder_chunk: list, dataset_dir: str, return_dict: dict):
    print(f"🚀 Worker GPU/CPU Process #{worker_id} đã khởi động với {len(folder_chunk)} folders...")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    detector = _init_detector()
    recognizer = _init_recognizer()
    vlm_model, vlm_tokenizer = get_vlm_engine()

    worker_records = []
    for idx, v_folder in enumerate(folder_chunk, 1):
        folder_path = os.path.join(dataset_dir, v_folder)
        img_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))],
            key=natural_sort_key
        )
        v_id = f"L28_{v_folder}" if not v_folder.startswith("L28") else v_folder
        print(f"⚡ [Worker #{worker_id}] [{idx}/{len(folder_chunk)}] Xử lý {v_id} ({len(img_files)} frames)...")

        for img_file in img_files:
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

            worker_records.append({
                "scene_id": scene_id,
                "video_id": v_id,
                "frame_id": frame_id,
                "ocr_text": ocr_text,
                "local_raw": candidates.get("local", ""),
                "vlm_raw": candidates.get("vlm", "")
            })

    return_dict[worker_id] = worker_records
    print(f"✅ Worker #{worker_id} hoàn thành ({len(worker_records)} frames)!")

def run_multiprocess_l28(num_workers: int = 6, only_folder: str = None):
    dataset_dir = os.path.join(PROJECT_ROOT, "data", "image", "L28", "L28")
    output_dir = os.path.join(PROJECT_ROOT, "data", "ocr_output")

    print("=" * 80)
    print(f"🔥 MULTIPROCESSING MAX HARDWARE ENGINE ({num_workers} PARALLEL GPU/CPU WORKERS)")
    print(f"📁 Dataset Dir : {dataset_dir}")
    print(f"💾 Output Dir  : {output_dir}")
    print(f"⚡ GPU Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    all_entries = os.listdir(dataset_dir)
    video_folders = sorted(
        [f for f in all_entries if os.path.isdir(os.path.join(dataset_dir, f))],
        key=natural_sort_key
    )

    if only_folder:
        video_folders = [f for f in video_folders if f == only_folder or f == f"V{only_folder.lstrip('V')}"]

    print(f"📹 Tổng số video folders cần xử lý: {len(video_folders)}")

    # Split video folders across workers
    chunks = [[] for _ in range(num_workers)]
    for i, folder in enumerate(video_folders):
        chunks[i % num_workers].append(folder)

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    total_start = time.time()

    for w_id in range(num_workers):
        if not chunks[w_id]:
            continue
        p = mp.Process(target=worker_process, args=(w_id, chunks[w_id], dataset_dir, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    all_records = []
    for w_id, records in return_dict.items():
        all_records.extend(records)

    df_consolidated = pd.DataFrame(all_records)
    output_parquet_path = os.path.join(output_dir, "L28_V001_test_hybrid.parquet" if only_folder else "L28_ocr.parquet")
    df_consolidated.to_parquet(output_parquet_path, index=False)

    elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"✅ HOÀN THÀNH MULTIPROCESSING MAX HARDWARE FOR L28!")
    print(f"📊 Tổng số frames  : {len(all_records)} frames")
    print(f"⏱️ Tổng thời gian   : {elapsed:.1f}s ({elapsed/max(1, len(all_records)):.2f}s/frame)")
    print(f"🚀 Tốc độ trích xuất: {len(all_records)/max(1, elapsed):.2f} frames/giây")
    print(f"💾 File Parquet Gộp: {output_parquet_path}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2, help="Số tiến trình GPU/CPU chạy song song")
    parser.add_argument("--only-folder", default=None, help="Chạy test cho 1 folder cụ thể")
    args = parser.parse_args()

    run_multiprocess_l28(num_workers=args.workers, only_folder=args.only_folder)
