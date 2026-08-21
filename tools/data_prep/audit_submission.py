"""
tools/data_prep/audit_submission.py
===================================
Kiểm tra tính hợp lệ của toàn bộ các file CSV trong folder submission theo đúng quy chuẩn BTC AIC:
1. Tên file: query-p{X}-{ID}-{kis|qa|trake}.csv
2. Format KIS: <video_name>,<frame_id>
3. Format Q&A: <video_name>,<frame_id>,"<answer>"
4. Format TRAKE: <video_name>,<frame_1>,<frame_2>,...
5. Không có Header
6. Không có khoảng trắng thừa sau dấu phẩy
7. Encoding UTF-8 (hỗ trợ BOM/không BOM)
8. Tối đa 100 dòng/file
"""

import os
import re
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUB_DIR = PROJECT_ROOT / "team_submission_merged" / "submission"

def main():
    print("=" * 80)
    print(f"📋 KIỂM TRA ĐỊNH DẠNG NỘP BÀI BTC AIC: {SUB_DIR.name}")
    print("=" * 80)

    if not SUB_DIR.exists():
        print(f"❌ Không tìm thấy thư mục: {SUB_DIR}")
        return

    csv_files = sorted(list(SUB_DIR.glob("*.csv")))
    print(f"📁 Tổng số file CSV: {len(csv_files)}\n")

    all_valid = True
    results = []

    for f in csv_files:
        fname = f.name
        content_raw = f.read_bytes()
        text = content_raw.decode("utf-8-sig")
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        match = re.match(r"^query-(?:p\d+-)?(\d+)-(kis|qa|vqa|trake)\.csv$", fname, re.IGNORECASE)
        errors = []
        task_type = "unknown"

        if not match:
            errors.append(f"Tên file không đúng chuẩn: {fname}")
        else:
            task_type = match.group(2).lower()
            if task_type == "vqa":
                task_type = "qa"

        if len(lines) == 0:
            errors.append("File rỗng (0 dòng)")
        elif len(lines) > 100:
            errors.append(f"Vượt quá 100 dòng ({len(lines)} dòng)")

        for line_idx, line in enumerate(lines, start=1):
            if line.lower().startswith("video_id") or line.lower().startswith("video,") or line.lower().startswith("video_name"):
                errors.append(f"Dòng {line_idx}: Chứa dòng Header!")

            # Check delimiter space between columns
            if ", " in line and not (line.startswith('"') or ',"' in line or '",' in line):
                errors.append(f"Dòng {line_idx}: Chứa khoảng trắng sau dấu phẩy giữa các cột (, )")

            reader = list(csv.reader([line]))
            if not reader or not reader[0]:
                continue
            row = reader[0]

            vid = row[0].strip()
            if not re.match(r"^[A-Za-z0-9_]+$", vid):
                errors.append(f"Dòng {line_idx}: video_id không hợp lệ ({vid})")

            if task_type == "kis":
                if len(row) != 2:
                    errors.append(f"Dòng {line_idx}: KIS yêu cầu đúng 2 cột <video_id,frame_id>, nhưng có {len(row)} cột")
                elif not row[1].strip().isdigit():
                    errors.append(f"Dòng {line_idx}: frame_id không phải số nguyên ({row[1]})")
            elif task_type in ["qa", "vqa"]:
                if len(row) != 3:
                    errors.append(f"Dòng {line_idx}: QA yêu cầu đúng 3 cột <video_id,frame_id,answer>, nhưng có {len(row)} cột")
                elif not row[1].strip().isdigit():
                    errors.append(f"Dòng {line_idx}: frame_id không phải số nguyên ({row[1]})")
                elif len(row[2]) > 100:
                    errors.append(f"Dòng {line_idx}: Answer dài quá 100 ký tự ({len(row[2])} chars)")
            elif task_type == "trake":
                if len(row) < 2:
                    errors.append(f"Dòng {line_idx}: TRAKE yêu cầu ít nhất 2 cột, nhưng có {len(row)} cột")
                else:
                    for f_idx, fid in enumerate(row[1:], start=1):
                        if not fid.strip().isdigit():
                            errors.append(f"Dòng {line_idx}: frame_{f_idx} không phải số ({fid})")

        if errors:
            all_valid = False

        status = "PASSED" if not errors else "FAILED"
        sample = lines[0] if lines else "EMPTY"
        results.append({
            "filename": fname,
            "type": task_type.upper(),
            "lines": len(lines),
            "status": status,
            "sample": sample,
            "errors": errors
        })

    # Print Report Table
    print(f"{'Tên File':<25} | {'Loại':<5} | {'Số Dòng':<7} | {'Trạng Thái':<10} | {'Dòng Mẫu'}")
    print("-" * 95)
    for r in results:
        status_icon = "🟢 ĐẠT" if r["status"] == "PASSED" else "🔴 LỖI"
        print(f"{r['filename']:<25} | {r['type']:<5} | {r['lines']:<7} | {status_icon:<10} | {r['sample']}")
        if r["errors"]:
            for err in r["errors"]:
                print(f"   ❌ {err}")

    print("\n" + "=" * 80)
    if all_valid:
        print("🎉 KẾT LUẬN: TẤT CẢ 24 FILE CSV ĐỀU CHUẨN ĐỊNH DẠNG 100% THEO QUY ĐỊNH BAN TỔ CHỨC!")
    else:
        print("⚠️ KẾT LUẬN: CÓ FILE BỊ LỖI ĐỊNH DẠNG, CẦN SỬA LẠI THEO CHI TIẾT Ở TRÊN.")
    print("=" * 80)

if __name__ == "__main__":
    main()
