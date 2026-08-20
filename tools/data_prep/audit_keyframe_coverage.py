"""
tools/data_prep/audit_keyframe_coverage.py
=========================================
Kiểm tra độ phủ của ảnh keyframe trên toàn bộ video trong dataset AIC2026:
- Quét toàn bộ file video trong data/video/ (lấy duration, total_frames, fps qua OpenCV).
- Khớp với dữ liệu keyframe trong data/unified/scene_metadata.parquet và data/image/.
- Phân tích chi tiết từng video:
  + Số lượng keyframes, mật độ (keyframes/phút).
  + Khoảng trống đầu (Head gap), khoảng trống đuôi (Tail gap).
  + Khoảng cách lớn nhất giữa 2 keyframe liên tiếp (Max internal gap).
  + Tỷ lệ phủ thời gian (Time coverage ratio).
- Phân loại bất thường: CRITICAL (Nghiêm trọng), WARNING (Cảnh báo), NORMAL (Bình thường).
- Xuất báo cáo CSV, JSON và tóm tắt Markdown.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from shared.utils.path_resolver import resolve_video_path


def main():
    print("=" * 80)
    print("🔍 BẮT ĐẦU KIỂM TRA ĐỘ PHỦ KEYFRAME TOÀN BỘ VIDEO DATASET AIC2026")
    print("=" * 80)

    video_dir = PROJECT_ROOT / "data" / "video"
    unified_dir = PROJECT_ROOT / "data" / "unified"
    image_dir = PROJECT_ROOT / "data" / "image"
    reports_dir = PROJECT_ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đọc metadata từ scene_metadata.parquet
    meta_path = unified_dir / "scene_metadata.parquet"
    if not meta_path.exists():
        print(f"❌ Không tìm thấy {meta_path}")
        return

    meta_df = pd.read_parquet(meta_path)
    print(f"📊 Đã nạp {len(meta_df):,d} clean scenes từ scene_metadata.parquet")

    # Gom nhóm keyframes theo video_id
    keyframes_by_vid: dict[str, list[dict]] = {}
    for _, row in meta_df.iterrows():
        vid = str(row["video_id"])
        fid = int(row["frame_id"])
        t_start = float(row.get("timestamp_start", 0.0))
        t_mid = float(row.get("timestamp_mid", fid / 25.0))
        t_end = float(row.get("timestamp_end", (fid + 25) / 25.0))
        
        if vid not in keyframes_by_vid:
            keyframes_by_vid[vid] = []
        keyframes_by_vid[vid].append({
            "scene_id": str(row["scene_id"]),
            "frame_id": fid,
            "timestamp_start": t_start,
            "timestamp_mid": t_mid,
            "timestamp_end": t_end,
        })

    # Sắp xếp keyframes theo frame_id
    for vid in keyframes_by_vid:
        keyframes_by_vid[vid].sort(key=lambda x: x["frame_id"])

    print(f"🎬 Có {len(keyframes_by_vid):,d} video IDs trong scene_metadata.parquet")

    # 2. Quét tất cả file video trên đĩa
    video_files = list(video_dir.rglob("*.mp4")) + list(video_dir.rglob("*.mkv")) + list(video_dir.rglob("*.webm"))
    print(f"📁 Tìm thấy {len(video_files):,d} files video vật lý trên đĩa trong {video_dir.name}")

    # Build mapping full_video_id -> file path
    video_file_map: dict[str, Path] = {}
    for vf in video_files:
        vid_stem = vf.stem
        batch_folder = vf.parent.name
        # chuẩn hóa ID: e.g. L21_V001
        if "_" in vid_stem:
            full_vid = vid_stem
        else:
            full_vid = f"{batch_folder}_{vid_stem}"
        video_file_map[full_vid] = vf

    # Tổng hợp tất cả video ID từ cả 2 nguồn
    all_video_ids = sorted(list(set(list(video_file_map.keys()) + list(keyframes_by_vid.keys()))))
    print(f"🌐 Tổng số video cần phân tích đánh giá: {len(all_video_ids):,d} videos\n")

    report_records = []
    
    t0 = time.time()
    for idx, vid in enumerate(all_video_ids):
        parts = vid.split("_")
        batch = parts[0] if len(parts) >= 2 else "Unknown"
        sub_v = parts[1] if len(parts) >= 2 else vid

        v_file = video_file_map.get(vid)
        if not v_file and batch != "Unknown":
            # Thử tìm dạng khác
            candidates = [
                video_dir / batch / f"{sub_v}.mp4",
                video_dir / batch / f"{vid}.mp4",
                video_dir / batch / batch / f"{sub_v}.mp4",
            ]
            for c in candidates:
                if c.exists():
                    v_file = c
                    break

        # Đọc thông số video thực tế
        has_video_file = v_file is not None and v_file.exists()
        duration_sec = 0.0
        total_video_frames = 0
        fps = 25.0

        if has_video_file:
            cap = cv2.VideoCapture(str(v_file))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if fps > 0 and total_video_frames > 0:
                    duration_sec = round(total_video_frames / fps, 2)
                cap.release()

        # Keyframe metrics
        kfs = keyframes_by_vid.get(vid, [])
        kf_count = len(kfs)

        first_kf_sec = round(kfs[0]["timestamp_mid"], 2) if kf_count > 0 else 0.0
        last_kf_sec = round(kfs[-1]["timestamp_mid"], 2) if kf_count > 0 else 0.0
        first_kf_frame = kfs[0]["frame_id"] if kf_count > 0 else 0
        last_kf_frame = kfs[-1]["frame_id"] if kf_count > 0 else 0

        # Nếu không có video file nhưng có keyframes, ước lượng duration từ last_kf_sec
        if duration_sec == 0.0 and kf_count > 0:
            duration_sec = round(last_kf_sec + 5.0, 2)

        # Tính khoảng trống (Gaps)
        head_gap_sec = round(first_kf_sec, 2) if kf_count > 0 else duration_sec
        tail_gap_sec = round(max(0.0, duration_sec - last_kf_sec), 2) if (kf_count > 0 and duration_sec > 0) else 0.0

        internal_gaps = []
        if kf_count > 1:
            for i in range(len(kfs) - 1):
                gap = kfs[i + 1]["timestamp_mid"] - kfs[i]["timestamp_mid"]
                internal_gaps.append(gap)

        max_internal_gap_sec = round(max(internal_gaps), 2) if internal_gaps else 0.0
        avg_gap_sec = round(float(np.mean(internal_gaps)), 2) if internal_gaps else (round(duration_sec, 2) if kf_count == 1 else 0.0)

        # Độ phủ thời gian (%)
        time_span = max(0.0, last_kf_sec - first_kf_sec)
        time_coverage_pct = round((time_span / duration_sec) * 100.0, 1) if duration_sec > 0 else 0.0

        # Mật độ keyframe (keyframes / phút)
        density_kpm = round(kf_count / (duration_sec / 60.0), 1) if duration_sec > 0 else 0.0

        # Phân loại bất thường & Mức độ nghiêm trọng
        anomalies = []
        severity = "NORMAL"

        if not has_video_file:
            anomalies.append("MISSING_VIDEO_FILE")
            severity = "CRITICAL"
        elif kf_count == 0:
            anomalies.append("ZERO_KEYFRAMES")
            severity = "CRITICAL"
        else:
            # Blindspot > 90s
            if max_internal_gap_sec >= 90.0:
                anomalies.append(f"HUGE_INTERNAL_GAP_{max_internal_gap_sec:.0f}s")
                severity = "CRITICAL"
            elif max_internal_gap_sec >= 45.0:
                anomalies.append(f"NOTICABLE_INTERNAL_GAP_{max_internal_gap_sec:.0f}s")
                if severity != "CRITICAL":
                    severity = "WARNING"

            # Head gap > 30s
            if head_gap_sec >= 45.0:
                anomalies.append(f"LARGE_HEAD_GAP_{head_gap_sec:.0f}s")
                if severity != "CRITICAL":
                    severity = "WARNING"

            # Tail gap > 60s
            if tail_gap_sec >= 60.0:
                anomalies.append(f"LARGE_TAIL_GAP_{tail_gap_sec:.0f}s")
                if severity != "CRITICAL":
                    severity = "WARNING"

            # Low time coverage
            if duration_sec >= 60.0 and time_coverage_pct < 65.0:
                anomalies.append(f"LOW_TIME_COVERAGE_{time_coverage_pct:.1f}%")
                if severity != "CRITICAL":
                    severity = "WARNING"

            # Extremely low count for long video
            if duration_sec >= 180.0 and kf_count < 10:
                anomalies.append(f"VERY_FEW_KEYFRAMES_{kf_count}_FOR_{duration_sec:.0f}s")
                if severity != "CRITICAL":
                    severity = "WARNING"

            # Abnormally high density (> 40 kf/min = 1 kf every 1.5s)
            if density_kpm >= 40.0:
                anomalies.append(f"HIGH_DENSITY_{density_kpm:.1f}_KPM")
                if severity != "CRITICAL":
                    severity = "WARNING"

        rec = {
            "video_id": vid,
            "batch": batch,
            "sub_video": sub_v,
            "has_video_file": has_video_file,
            "video_file_path": str(v_file.relative_to(PROJECT_ROOT)) if v_file else "",
            "fps": float(fps),
            "total_video_frames": int(total_video_frames),
            "duration_sec": float(duration_sec),
            "duration_formatted": f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}",
            "keyframe_count": int(kf_count),
            "density_kpm": float(density_kpm),
            "first_keyframe_sec": float(first_kf_sec),
            "last_keyframe_sec": float(last_kf_sec),
            "head_gap_sec": float(head_gap_sec),
            "tail_gap_sec": float(tail_gap_sec),
            "max_internal_gap_sec": float(max_internal_gap_sec),
            "avg_gap_sec": float(avg_gap_sec),
            "time_coverage_pct": float(time_coverage_pct),
            "severity": severity,
            "anomalies": "; ".join(anomalies) if anomalies else "None",
            "anomalies_list": anomalies,
        }
        report_records.append(rec)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_video_ids):
            print(f"  Đã phân tích {idx+1}/{len(all_video_ids)} videos ({time.time()-t0:.1f}s)...")

    # 3. Tạo DataFrame và thống kê tổng hợp
    df_report = pd.DataFrame(report_records)

    # Lưu CSV và JSON
    csv_path = reports_dir / "keyframe_coverage_report.csv"
    json_path = reports_dir / "keyframe_coverage_report.json"
    df_report.drop(columns=["anomalies_list"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_records, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Đã lưu báo cáo CSV : {csv_path.name}")
    print(f"💾 Đã lưu báo cáo JSON: {json_path.name}")

    # 4. In báo cáo thống kê chuyên sâu
    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT ĐÁNH GIÁ ĐỘ PHỦ KEYFRAME TOÀN HỆ THỐNG")
    print("=" * 80)

    total_vids = len(df_report)
    normal_count = (df_report["severity"] == "NORMAL").sum()
    warning_count = (df_report["severity"] == "WARNING").sum()
    critical_count = (df_report["severity"] == "CRITICAL").sum()

    total_duration_hours = df_report["duration_sec"].sum() / 3600.0
    total_kf = df_report["keyframe_count"].sum()
    avg_kpm_overall = df_report["density_kpm"].mean()
    avg_gap_overall = df_report["avg_gap_sec"].mean()
    avg_coverage_overall = df_report["time_coverage_pct"].mean()

    print(f"🎬 Tổng số Video phân tích        : {total_vids:,d} videos")
    print(f"⏱️ Tổng thời lượng Video          : {total_duration_hours:.2f} giờ ({df_report['duration_sec'].sum():,.0f} giây)")
    print(f"🖼️ Tổng số Keyframes sạch         : {total_kf:,d} ảnh")
    print(f"📈 Mật độ trung bình toàn dataset : {avg_kpm_overall:.1f} keyframes / phút")
    print(f"📏 Khoảng cách trung bình các ảnh : {avg_gap_overall:.1f} giây / keyframe")
    print(f"🌐 Độ phủ thời gian trung bình    : {avg_coverage_overall:.1f}%")
    print("-" * 80)
    print(f"🟢 NORMAL   (Bình thường / Hoàn hảo): {normal_count:,d} videos ({normal_count/total_vids*100:.1f}%)")
    print(f"🟡 WARNING  (Cảnh báo cần lưu ý)    : {warning_count:,d} videos ({warning_count/total_vids*100:.1f}%)")
    print(f"🔴 CRITICAL (Nghiêm trọng)          : {critical_count:,d} videos ({critical_count/total_vids*100:.1f}%)")
    print("=" * 80)

    # Thống kê theo từng Batch
    print("\n📁 THỐNG KÊ CHI TIẾT THEO TỪNG BATCH:")
    print("-" * 90)
    print(f"{'Batch':<8} | {'Videos':<7} | {'Total Keyframes':<16} | {'Avg KPM':<9} | {'Avg Gap(s)':<11} | {'Coverage%':<10} | {'Warnings':<9} | {'Critical'}")
    print("-" * 90)

    for batch, g in df_report.groupby("batch"):
        b_vids = len(g)
        b_kfs = g["keyframe_count"].sum()
        b_kpm = g["density_kpm"].mean()
        b_gap = g["avg_gap_sec"].mean()
        b_cov = g["time_coverage_pct"].mean()
        b_warn = (g["severity"] == "WARNING").sum()
        b_crit = (g["severity"] == "CRITICAL").sum()
        print(f"{batch:<8} | {b_vids:<7} | {b_kfs:<16,d} | {b_kpm:<9.1f} | {b_gap:<11.1f} | {b_cov:<9.1f}% | {b_warn:<9} | {b_crit}")
    print("-" * 90)

    # Danh sách các video có vấn đề CRITICAL hoặc WARNING đáng chú ý
    crit_df = df_report[df_report["severity"] == "CRITICAL"]
    if not crit_df.empty:
        print("\n🔴 DANH SÁCH VIDEO MỨC ĐỘ CRITICAL:")
        print("-" * 90)
        for _, r in crit_df.iterrows():
            print(f"  • Video {r['video_id']:<10}: Duration={r['duration_formatted']} | Keyframes={r['keyframe_count']:<3} | MaxGap={r['max_internal_gap_sec']:.0f}s | Coverage={r['time_coverage_pct']:.1f}% | Lỗi: {r['anomalies']}")

    warn_df = df_report[df_report["severity"] == "WARNING"].sort_values("max_internal_gap_sec", ascending=False)
    if not warn_df.empty:
        print("\n🟡 TOP 15 VIDEO CẢNH BÁO CẦN LƯU Ý (Max Gap lớn nhất):")
        print("-" * 90)
        for _, r in warn_df.head(15).iterrows():
            print(f"  • Video {r['video_id']:<10}: Duration={r['duration_formatted']} | Keyframes={r['keyframe_count']:<3} | MaxGap={r['max_internal_gap_sec']:.0f}s | HeadGap={r['head_gap_sec']:.0f}s | TailGap={r['tail_gap_sec']:.0f}s | Lỗi: {r['anomalies']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
