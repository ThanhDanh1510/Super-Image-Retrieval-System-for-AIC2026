import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("L21", os.path.join(PROJECT_ROOT, "data", "image", "L21", "L21")),
    ("L22", os.path.join(PROJECT_ROOT, "data", "image", "L22", "L22")),
    ("L23", os.path.join(PROJECT_ROOT, "data", "image", "L23", "L23")),
    ("L24", os.path.join(PROJECT_ROOT, "data", "image", "L24", "L24")),
    ("L25", os.path.join(PROJECT_ROOT, "data", "image", "L25", "L25")),
]

def run_queue():
    python_exe = sys.executable
    batch_script = os.path.join(PROJECT_ROOT, "tools", "ocr_batch_dataset.py")
    output_dir = os.path.join(PROJECT_ROOT, "data", "ocr_output")

    print("=" * 70)
    print("🚀 KHỞI CHẠY HÀNG ĐỢI XỬ LÝ OCR GỘP TẬP TRUNG (L21 -> L25)")
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
    print("🎉 TẤT CẢ CÁC TẬP DỮ LIỆU L21, L22, L23, L24, L25 ĐÃ HOÀN THÀNH XỬ LÝ!")
    print("=" * 70)

if __name__ == "__main__":
    run_queue()
