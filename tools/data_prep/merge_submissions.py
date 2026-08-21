"""
tools/data_prep/merge_submissions.py
====================================
Gộp kết quả nộp bài từ 2 folder:
- Folder A (Bạn)     : team_submission_1787240406787/submission (24 files)
- Folder B (Đồng Đội): D:/team_submission_1787235000514/submission (20 files)

Chiến lược Rerank/Xếp hạng Tối ưu theo Quy chế BTC:
1. TOP 1 (Ưu tiên cao nhất): Các frame xuất hiện ở CẢ 2 bên (hoặc trùng khớp phân cảnh thời gian).
2. TOP 2: Các frame được chọn lọc kỹ càng bởi Bạn (Folder A).
3. TOP 3: Các candidate còn lại từ Đồng Đội (Folder B) theo đúng thứ tự xếp hạng ban đầu.
4. Giới hạn tối đa <= 100 dòng cho mỗi file CSV.
5. Đảm bảo 100% chuẩn định dạng BTC AIC (UTF-8, không Header, không khoảng trắng thừa).
"""

import csv
import sys
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIR_A = PROJECT_ROOT / "team_submission_1787240406787" / "submission"
DIR_B = Path("D:/team_submission_1787235000514/submission")
OUTPUT_DIR = PROJECT_ROOT / "team_submission_merged" / "submission"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_csv_rows(fpath):
    rows = []
    if not fpath.exists():
        return rows
    with open(fpath, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for r in reader:
            if r and any(col.strip() for col in r):
                rows.append([c.strip() for c in r])
    return rows


def merge_file(fname):
    file_a = DIR_A / fname
    file_b = DIR_B / fname

    rows_a = read_csv_rows(file_a)
    rows_b = read_csv_rows(file_b)

    task_type = "kis"
    if "-qa" in fname or "-vqa" in fname:
        task_type = "qa"
    elif "-trake" in fname:
        task_type = "trake"

    # If only one exists
    if not rows_b:
        return rows_a[:100], "Chỉ có bên A"
    if not rows_a:
        return rows_b[:100], "Chỉ có bên B"

    merged_rows = []
    seen_keys = set()

    def make_key(r):
        # Unique identifier for deduplication
        if task_type == "trake":
            return tuple(r)
        return (r[0], r[1])  # (video_id, frame_id)

    # 1. Tìm các phần tử TRÙNG KHỚP TUYỆT ĐỐI (có ở cả A và B) -> Đưa lên TOP ĐẦU
    set_b_keys = {make_key(r) for r in rows_b}
    overlap_in_a = []
    for r in rows_a:
        k = make_key(r)
        if k in set_b_keys:
            overlap_in_a.append(r)
            seen_keys.add(k)

    # Cho các dòng trùng khớp lên đầu tiên
    for r in overlap_in_a:
        merged_rows.append(r)

    # 2. Đưa các dòng còn lại của A (chọn lọc kỹ) lên tiếp theo
    for r in rows_a:
        k = make_key(r)
        if k not in seen_keys:
            merged_rows.append(r)
            seen_keys.add(k)

    # 3. Đưa các dòng còn lại từ B (xếp hạng của đồng đội) theo thứ tự
    for r in rows_b:
        k = make_key(r)
        if k not in seen_keys:
            merged_rows.append(r)
            seen_keys.add(k)

    return merged_rows[:100], f"Gộp {len(overlap_in_a)} trùng + {len(rows_a)-len(overlap_in_a)} từ A + {len(merged_rows)-len(rows_a)} từ B"


def main():
    print("=" * 85)
    print("🚀 BẮT ĐẦU GỘP 2 KẾT QUẢ NỘP BÀI VỚI CHIẾN LƯỢC TOP CONSENSUS RERANKING")
    print("=" * 85)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted(list(set([f.name for f in DIR_A.glob("*.csv")] + [f.name for f in DIR_B.glob("*.csv")])))
    print(f"📊 Tổng số câu hỏi cần gộp: {len(all_files)} file CSV\n")

    report = []

    for fname in all_files:
        task_type = "kis"
        if "-qa" in fname or "-vqa" in fname:
            task_type = "qa"
        elif "-trake" in fname:
            task_type = "trake"

        merged_rows, log_info = merge_file(fname)
        out_file = OUTPUT_DIR / fname

        # Write CSV conforming to BTC rule
        with open(out_file, "w", encoding="utf-8-sig", newline="") as f:
            if task_type == "qa":
                lines = []
                for r in merged_rows:
                    if len(r) >= 3:
                        ans = r[2].strip().strip('"').replace('"', '""')
                        if len(ans) > 100:
                            ans = ans[:100]
                        lines.append(f"{r[0]},{r[1]},\"{ans}\"")
                    else:
                        lines.append(",".join(r))
                f.write("\r\n".join(lines) + ("\r\n" if lines else ""))
            else:
                lines = [",".join(r) for r in merged_rows]
                f.write("\r\n".join(lines) + ("\r\n" if lines else ""))

        report.append({
            "file": fname,
            "count": len(merged_rows),
            "top1": merged_rows[0] if merged_rows else "N/A",
            "info": log_info
        })

    # Print Table
    print(f"{'Tên File CSV':<23} | {'Số Dòng':<7} | {'Top 1 Frame':<20} | {'Chi Tiết Gộp'}")
    print("-" * 90)
    for r in report:
        top1_str = str(r['top1'][:2]) if isinstance(r['top1'], list) else str(r['top1'])
        print(f"{r['file']:<23} | {r['count']:<7} | {top1_str:<20} | {r['info']}")

    # Zip for instant portal submission
    zip_path = PROJECT_ROOT / "team_submission_merged.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUTPUT_DIR.glob("*.csv"):
            z.write(f, arcname=f"submission/{f.name}")

    print("\n" + "=" * 85)
    print(f"🎉 GỘP THÀNH CÔNG TOÀN BỘ {len(all_files)} FILE CSV VÀO THƯ MỤC:")
    print(f"📁 Thư mục: {OUTPUT_DIR.resolve()}")
    print(f"📦 File Zip sẵn sàng nộp BTC: {zip_path.resolve()} ({zip_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 85)


if __name__ == "__main__":
    main()
