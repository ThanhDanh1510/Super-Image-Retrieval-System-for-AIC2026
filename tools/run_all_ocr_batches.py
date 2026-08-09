import os
import sys
import argparse
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_dataset_dir(prefix: str) -> str:
    unified_dir = os.path.join(PROJECT_ROOT, "data", "unified")
    if os.path.exists(unified_dir):
        import glob
        matches = glob.glob(os.path.join(unified_dir, f"{prefix}_*"))
        if matches:
            return unified_dir

    candidates = [
        os.path.join(PROJECT_ROOT, "data", "data", "image", prefix, prefix),
        os.path.join(PROJECT_ROOT, "data", "image", prefix, prefix),
        os.path.join(PROJECT_ROOT, "data", "data", "image", prefix),
        os.path.join(PROJECT_ROOT, "data", "image", prefix),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            files = [f for f in os.listdir(cand) if os.path.isfile(os.path.join(cand, f)) or (os.path.isdir(os.path.join(cand, f)) and os.listdir(os.path.join(cand, f)))]
            if files:
                return cand
    return candidates[0]

def run_queue(dataset_prefixes):
    python_exe = sys.executable
    batch_script = os.path.join(PROJECT_ROOT, "tools", "ocr_batch_dataset.py")
    output_dir = os.path.join(PROJECT_ROOT, "data", "ocr_output")

    print("=" * 70)
    print(f"🚀 KHỞI CHẠY HÀNG ĐỢI XỬ LÝ OCR GỘP TẬP TRUNG ({', '.join(dataset_prefixes)})")
    print("=" * 70)

    for prefix in dataset_prefixes:
        dataset_dir = get_dataset_dir(prefix)
        print(f"\n▶️ ĐANG CHẠY BATCH OCR CHO: {prefix} ({dataset_dir})")
        cmd = [
            python_exe,
            batch_script,
            "--dataset-dir", dataset_dir,
            "--output-dir", output_dir,
            "--prefix", prefix
        ]

        try:
            res = subprocess.run(cmd, check=True)
            print(f"✅ Hoàn thành batch {prefix} với exit code {res.returncode}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi xử lý {prefix}: {e}")

    print("\n" + "=" * 70)
    print(f"🎉 TẤT CẢ CÁC TẬP DỮ LIỆU {', '.join(dataset_prefixes)} ĐÃ HOÀN THÀNH XỬ LÝ!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy OCR hàng loạt cho các tập dữ liệu.")
    parser.add_argument("--datasets", nargs="+", default=["L28", "L29", "L30"], help="Danh sách các prefix dataset (vd: L28 L29 L30)")

    args = parser.parse_args()
    run_queue(args.datasets)

