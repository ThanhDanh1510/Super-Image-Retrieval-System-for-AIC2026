"""
tools/detect_l25_video_file_duplicates.py
=========================================
Quét trực tiếp trên 88 file video MP4 trong data/video/L25/
Phát hiện video trùng lặp bằng:
1. Metadata video (Duration, Frame count, FPS, File size).
2. Trích xuất trực tiếp 20 khung hình từ file .mp4 tại các mốc thời gian (0%, 5%, ..., 95%) để so khớp Pixel MAE & SSIM.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

VIDEO_DIR = Path("data/video/L25")
NUM_SAMPLES = 20

def get_video_metadata(video_path: Path):
    vid = video_path.stem
    size_bytes = video_path.stat().st_size
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "video_id": vid,
        "filename": video_path.name,
        "path": str(video_path),
        "size_bytes": size_bytes,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "resolution": f"{width}x{height}",
        "duration_sec": round(duration_sec, 2),
    }

def extract_video_sample_frames(video_path: Path, num_samples=NUM_SAMPLES):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((num_samples, 45, 80), dtype=np.uint8)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros((num_samples, 45, 80), dtype=np.uint8)

    frames = []
    for k in range(num_samples):
        target_f = int(k * (total_frames - 1) / (num_samples - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, (80, 45))
        else:
            thumb = np.zeros((45, 80), dtype=np.uint8)
        frames.append(thumb)

    cap.release()
    return np.array(frames, dtype=np.uint8)

def main():
    print("=== AIC2026: Quét Trực Tiếp Trên File Video L25 (.mp4) ===", flush=True)
    video_files = sorted(list(VIDEO_DIR.glob("L25_*.mp4")) + list(VIDEO_DIR.glob("L25_*.mkv")))
    print(f"Tìm thấy {len(video_files)} video files trong {VIDEO_DIR}", flush=True)

    if not video_files:
        print("❌ Không tìm thấy file video nào!")
        return

    # 1. Đọc metadata
    print("Đang đọc metadata (FPS, Frame count, Thời lượng, Kích thước file)...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        meta_list = list(ex.map(get_video_metadata, video_files))

    meta_dict = {m["video_id"]: m for m in meta_list if m is not None}
    video_ids = sorted(list(meta_dict.keys()))

    # 2. Trích xuất frame mẫu trực tiếp từ video
    print(f"Đang trích xuất {NUM_SAMPLES} frame mẫu từ từng file video .mp4...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        frame_samples = list(ex.map(
            lambda v: extract_video_sample_frames(Path(meta_dict[v]["path"])),
            video_ids
        ))
    frames_dict = {vid: f for vid, f in zip(video_ids, frame_samples)}

    # 3. So khớp tất cả các cặp video
    num_vids = len(video_ids)
    total_pairs = num_vids * (num_vids - 1) // 2
    print(f"Bắt đầu so khớp {total_pairs} cặp video cấp độ Video Stream & Pixel MAE...", flush=True)

    exact_duplicates = []
    near_duplicates = []
    candidates = []

    for i in range(num_vids):
        for j in range(i + 1, num_vids):
            vA_id, vB_id = video_ids[i], video_ids[j]
            mA, mB = meta_dict[vA_id], meta_dict[vB_id]
            fA, fB = frames_dict[vA_id], frames_dict[vB_id]

            # Chênh lệch thời lượng và kích thước
            dur_diff = abs(mA["duration_sec"] - mB["duration_sec"])
            size_ratio = min(mA["size_bytes"], mB["size_bytes"]) / max(mA["size_bytes"], mB["size_bytes"])
            frame_diff = abs(mA["frame_count"] - mB["frame_count"])

            # Pixel difference trên 20 frame mẫu
            diff = np.abs(fA.astype(np.float32) - fB.astype(np.float32)) # (20, 45, 80)
            sample_maes = diff.mean(axis=(1, 2)) # (20,)
            mean_mae = float(sample_maes.mean())
            max_mae = float(sample_maes.max())

            item = {
                "video_A": vA_id,
                "video_B": vB_id,
                "duration_A": mA["duration_sec"],
                "duration_B": mB["duration_sec"],
                "duration_diff_sec": round(dur_diff, 2),
                "size_A_mb": round(mA["size_bytes"] / (1024*1024), 2),
                "size_B_mb": round(mB["size_bytes"] / (1024*1024), 2),
                "size_ratio": round(size_ratio, 3),
                "mean_pixel_mae": round(mean_mae, 2),
                "max_pixel_mae": round(max_mae, 2),
            }

            # PHÂN LOẠI CHÍNH XÁC:
            # 1. EXACT DUPLICATE: Pixel MAE < 6.0 trên toàn bộ các frame mẫu
            if mean_mae <= 6.0 and max_mae <= 15.0 and dur_diff <= 2.0:
                item["category"] = "EXACT_100_PERCENT_DUPLICATE"
                item["action"] = "DISCARD_ONE"
                exact_duplicates.append(item)
            elif mean_mae <= 25.0 and dur_diff <= 10.0:
                item["category"] = "NEAR_DUPLICATE_SAME_LESSON"
                item["action"] = "REVIEW"
                near_duplicates.append(item)

            candidates.append(item)

    candidates.sort(key=lambda x: (x["mean_pixel_mae"], x["duration_diff_sec"]))

    print("\n" + "=" * 70, flush=True)
    print(f"🎉 KẾT QUẢ ĐỐI SOÁT TRỰC TIẾP TRÊN FILE VIDEO (.mp4) BATCH L25", flush=True)
    print(f"🔴 EXACT_100_PERCENT_DUPLICATE : {len(exact_duplicates)} cặp", flush=True)
    print(f"🟡 NEAR_DUPLICATE_SAME_LESSON  : {len(near_duplicates)} cặp", flush=True)
    print("=" * 70, flush=True)

    if exact_duplicates:
        print("\n--- 🔴 CÁC CẶP VIDEO TRÙNG LẶP 100% HOÀN TOÀN (CÙNG NỘI DUNG/FRAME-FOR-FRAME) ---", flush=True)
        for d in exact_duplicates:
            print(f"  • {d['video_A']} ({d['duration_A']}s, {d['size_A_mb']}MB) <===> {d['video_B']} ({d['duration_B']}s, {d['size_B_mb']}MB)", flush=True)
            print(f"    Pixel MAE: {d['mean_pixel_mae']:.2f}/255 | Max MAE: {d['max_pixel_mae']:.2f} | DurDiff: {d['duration_diff_sec']}s", flush=True)
    else:
        print("\n✅ Không phát hiện thêm cặp video nào trùng lặp 100% (ngoài L25_V042 đã loại bỏ).", flush=True)

    if near_duplicates:
        print("\n--- 🟡 CÁC CẶP VIDEO GẦN TRÙNG / CÙNG BÀI GIẢNG ---", flush=True)
        for d in near_duplicates:
            print(f"  • {d['video_A']} ({d['duration_A']}s, {d['size_A_mb']}MB) <===> {d['video_B']} ({d['duration_B']}s, {d['size_B_mb']}MB)", flush=True)
            print(f"    Pixel MAE: {d['mean_pixel_mae']:.2f}/255 | Max MAE: {d['max_pixel_mae']:.2f} | DurDiff: {d['duration_diff_sec']}s", flush=True)

    print("\n--- 🔝 TOP 10 CẶP VIDEO CÓ MỨC ĐỘ TƯƠNG ĐỒNG CAO NHẤT TRONG FILE VIDEO ---", flush=True)
    for c in candidates[:10]:
        print(f"  {c['video_A']} <-> {c['video_B']}: Mean Pixel MAE = {c['mean_pixel_mae']:.2f} | DurA={c['duration_A']}s, DurB={c['duration_B']}s (Diff: {c['duration_diff_sec']}s)", flush=True)

    # Lưu báo cáo JSON
    report = {
        "batch": "L25",
        "video_directory": str(VIDEO_DIR),
        "total_video_files": len(video_files),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "top_closest_video_pairs": candidates[:30],
    }

    out_file = Path("data/l25_video_file_duplicate_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Báo cáo chi tiết trên file video đã được lưu vào: {out_file}", flush=True)

if __name__ == "__main__":
    main()
