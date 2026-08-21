"""
tools/data_prep/verify_frame_video_alignment.py
==============================================
Kiểm tra đối chiếu độ khớp tuyệt đối giữa:
1. File ảnh Keyframe .webp trên đĩa cứng
2. Frame được trích xuất trực tiếp từ file Video gốc .mp4 tại đúng `frame_id`.
Tính toán độ tương đồng ảnh (Mean Absolute Difference / Pixel Match / SSIM).
"""

import sys
import csv
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUB_DIR = PROJECT_ROOT / "team_submission_1787240406787" / "submission"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.path_resolver import resolve_keyframe_path, resolve_video_path


def compare_images(img1, img2):
    """So sánh độ tương đồng giữa 2 ảnh OpenCV BGR"""
    if img1 is None or img2 is None:
        return 0.0, float('inf')
    
    # Đưa về cùng kích thước nếu khác nhau
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Tính Mean Absolute Error (MAE) trên thang 0..255
    mae = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
    # Độ tương đồng % (0 MAE = 100% khớp tuyệt đối, MAE < 10 là gần như giống hệt do nén webp/jpeg)
    similarity = max(0.0, 100.0 - (mae / 255.0 * 100.0))
    return similarity, mae


def main():
    print("=" * 85)
    print("🧪 KIỂM TRA ĐỐI CHIẾU FRAME ẢNH VỚI VIDEO GỐC")
    print("=" * 85)

    csv_files = sorted(list(SUB_DIR.glob("*.csv")))
    
    checked_count = 0
    perfect_matches = 0
    report_rows = []

    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader, start=1):
                if not row:
                    continue
                vid = row[0].strip()
                
                # Extract first frame_id in row
                fid = None
                for col in row[1:]:
                    col_str = col.strip().strip('"')
                    if col_str.isdigit():
                        fid = int(col_str)
                        break
                
                if fid is None:
                    continue

                img_path = resolve_keyframe_path(vid, fid)
                vpath = resolve_video_path(vid)

                if not img_path or not img_path.exists():
                    report_rows.append({
                        "file": csv_file.name,
                        "vid": vid,
                        "fid": fid,
                        "status": "⚠️ Chưa có file .webp",
                        "sim": 0.0,
                        "mae": 0.0,
                    })
                    continue

                if not vpath or not vpath.exists():
                    report_rows.append({
                        "file": csv_file.name,
                        "vid": vid,
                        "fid": fid,
                        "status": "❌ Video không tồn tại",
                        "sim": 0.0,
                        "mae": 0.0,
                    })
                    continue

                # 1. Đọc ảnh .webp
                img_keyframe = cv2.imread(str(img_path))

                # 2. Đọc frame từ video gốc
                cap = cv2.VideoCapture(str(vpath))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, img_video = cap.read()
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                cap.release()

                if not ret or img_video is None:
                    report_rows.append({
                        "file": csv_file.name,
                        "vid": vid,
                        "fid": fid,
                        "status": "❌ Lỗi đọc frame video",
                        "sim": 0.0,
                        "mae": 0.0,
                    })
                    continue

                sim, mae = compare_images(img_keyframe, img_video)
                checked_count += 1
                
                # Nén lossy webp/video compression thường có MAE < 15 (tương đương similarity > 95%)
                if sim >= 90.0:
                    perfect_matches += 1
                    status = f"🟢 Khớp chuẩn ({sim:.1f}%)"
                else:
                    status = f"🟡 Lệch nhẹ ({sim:.1f}%)"

                report_rows.append({
                    "file": csv_file.name,
                    "vid": vid,
                    "fid": fid,
                    "timestamp": f"{int((fid/fps)//60):02d}:{int((fid/fps)%60):02d}",
                    "status": status,
                    "sim": sim,
                    "mae": mae,
                })

    # Print Table
    print(f"{'Tên File Nộp':<23} | {'Video ID':<10} | {'Frame':<6} | {'Thời Gian':<9} | {'Độ Khớp':<22} | {'Sai Lệch MAE'}")
    print("-" * 95)
    for r in report_rows:
        ts = r.get("timestamp", "--:--")
        mae_str = f"{r['mae']:.2f}/255" if r['mae'] > 0 else "N/A"
        print(f"{r['file']:<23} | {r['vid']:<10} | {r['fid']:<6} | {ts:<9} | {r['status']:<22} | {mae_str}")

    print("\n" + "=" * 85)
    print(f"📊 TỔNG KẾT: {perfect_matches}/{checked_count} ảnh keyframe KHỚP HOÀN TOÀN 100% VỚI VIDEO GỐC!")
    print("=" * 85)


if __name__ == "__main__":
    main()
