"""
tools/data_prep/compare_submissions.py
======================================
So sánh chi tiết kết quả giữa 2 folder nộp bài:
Folder A (Của Bạn)    : D:\\AIC2026\\team_submission_1787240406787\\submission (24 files)
Folder B (Của Đồng Đội): D:\\team_submission_1787235000514\\submission (20 files)
"""

import csv
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIR_A = PROJECT_ROOT / "team_submission_1787240406787" / "submission"
DIR_B = Path("D:/team_submission_1787235000514/submission")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.path_resolver import resolve_video_path


def read_submission_csv(fpath):
    rows = []
    if not fpath.exists():
        return rows
    with open(fpath, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for r in reader:
            if r:
                rows.append([col.strip() for col in r])
    return rows


def get_video_fps(vid):
    vpath = resolve_video_path(vid)
    if not vpath or not vpath.exists():
        return 25.0
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)


def main():
    print("=" * 95)
    print("📊 SO SÁNH KẾT QUẢ NỘP BÀI: BẠN (DIR A) vs ĐỒNG ĐỘI (DIR B)")
    print("=" * 95)
    print(f"📁 Folder Bạn     : {DIR_A.parent.name} ({len(list(DIR_A.glob('*.csv')))} files)")
    print(f"📁 Folder Đồng Đội: {DIR_B.parent.name} ({len(list(DIR_B.glob('*.csv')))} files)")
    print("=" * 95)

    all_csv_names = sorted(list(set([f.name for f in DIR_A.glob("*.csv")] + [f.name for f in DIR_B.glob("*.csv")])))

    comparison_summary = []

    for fname in all_csv_names:
        file_a = DIR_A / fname
        file_b = DIR_B / fname

        rows_a = read_submission_csv(file_a)
        rows_b = read_submission_csv(file_b)

        exists_a = file_a.exists()
        exists_b = file_b.exists()

        if not exists_b:
            comparison_summary.append({
                "file": fname,
                "status": "⭐ Chỉ Bạn Có (Đồng đội chưa nộp)",
                "detail": f"Bạn nộp {len(rows_a)} dòng: {rows_a[0] if rows_a else 'N/A'}",
            })
            continue

        if not exists_a:
            comparison_summary.append({
                "file": fname,
                "status": "⚠️ Chỉ Đồng Đội Có",
                "detail": f"Đồng đội nộp {len(rows_b)} dòng: {rows_b[0] if rows_b else 'N/A'}",
            })
            continue

        # Compare contents
        # Check if identical
        if rows_a == rows_b:
            comparison_summary.append({
                "file": fname,
                "status": "🟢 TRÙNG KHỚP 100%",
                "detail": f"Cả 2 cùng chọn: {rows_a[0]} ({len(rows_a)} dòng)",
            })
            continue

        # Check video match
        vids_a = [r[0] for r in rows_a]
        vids_b = [r[0] for r in rows_b]
        common_vids = set(vids_a).intersection(set(vids_b))

        if not common_vids:
            comparison_summary.append({
                "file": fname,
                "status": "🔴 KHÁC VIDEO HOÀN TOÀN",
                "detail": f"Bạn: {vids_a} | Đồng đội: {vids_b}",
            })
        else:
            # Check frame closeness
            details = []
            for r_a in rows_a:
                vid_a = r_a[0]
                fid_a = int(r_a[1]) if len(r_a) > 1 and r_a[1].isdigit() else None
                fps = get_video_fps(vid_a)
                
                # Find matching row in B
                for r_b in rows_b:
                    if r_b[0] == vid_a:
                        fid_b = int(r_b[1]) if len(r_b) > 1 and r_b[1].isdigit() else None
                        if fid_a is not None and fid_b is not None:
                            diff_sec = abs(fid_a - fid_b) / fps
                            ts_a = f"{int((fid_a/fps)//60):02d}:{int((fid_a/fps)%60):02d}"
                            ts_b = f"{int((fid_b/fps)//60):02d}:{int((fid_b/fps)%60):02d}"
                            details.append(f"{vid_a}: Bạn f{fid_a}({ts_a}) vs Đội f{fid_b}({ts_b}) [Lệch {diff_sec:.1f}s]")

            comparison_summary.append({
                "file": fname,
                "status": "🟡 CÙNG VIDEO (LỆCH FRAME)",
                "detail": "; ".join(details) if details else f"Bạn: {rows_a} vs Đội: {rows_b}",
            })

    print(f"{'Tên File':<23} | {'Đánh Giá Khớp':<28} | {'Chi Tiết So Sánh'}")
    print("-" * 115)
    for c in comparison_summary:
        print(f"{c['file']:<23} | {c['status']:<28} | {c['detail']}")

    print("\n" + "=" * 95)
    print("📋 CHI TIẾT TỪNG CÂU HỎI:")
    print("=" * 95)
    for fname in all_csv_names:
        file_a = DIR_A / fname
        file_b = DIR_B / fname
        rows_a = read_submission_csv(file_a)
        rows_b = read_submission_csv(file_b)
        
        print(f"\n📄 File: {fname}")
        print(f"   👤 Của Bạn ({len(rows_a)} dòng)    : {rows_a}")
        print(f"   👥 Của Đồng Đội ({len(rows_b)} dòng): {rows_b}")


if __name__ == "__main__":
    main()
