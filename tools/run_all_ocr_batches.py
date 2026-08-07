import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("L28", os.path.join(PROJECT_ROOT, "data", "image", "L28", "L28")),
    ("L29", os.path.join(PROJECT_ROOT, "data", "image", "L29", "L29")),
    ("L30", os.path.join(PROJECT_ROOT, "data", "image", "L30", "L30")),
]

def run_queue():
    python_exe = sys.executable
    batch_script = os.path.join(PROJECT_ROOT, "tools", "ocr_batch_dataset.py")
    output_dir = os.path.join(PROJECT_ROOT, "data", "ocr_output")

    print("=" * 70)
    print("🚀 KHỞI CHẠY HÀNG ĐỢI XỬ LÝ OCR GỘP TẬP TRUNG (L28 -> L30)")
    print("=" * 70)

    for prefix, dataset_dir in DATASETS:
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
    print("🎉 TẤT CẢ CÁC TẬP DỮ LIỆU L28, L29, L30 ĐÃ HOÀN THÀNH XỬ LÝ!")
    print("=" * 70)

if __name__ == "__main__":
    run_queue()
