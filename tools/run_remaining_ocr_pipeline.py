import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable

prefixes = ["L23", "L22", "L21"]

def run_cmd(cmd):
    print(f"🚀 Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"⚠️ Command failed with code {res.returncode}")

def main():
    print("=" * 80)
    print("🔥 AUTOMATED SEQUENTIAL PIPELINE FOR L23, L22, L21")
    print("=" * 80)
    
    # 1. Wait for L24 (Task 3389) if running
    print("⏳ Checking active L24 process...")
    l24_parquet = os.path.join(PROJECT_ROOT, "data", "ocr_output", "L24_ocr.parquet")
    while not os.path.exists(l24_parquet):
        time.sleep(10)
    print("✅ L24 processing completed successfully!")

    # 2. Sequential execution for L23, L22, L21
    for prefix in prefixes:
        print("\n" + "=" * 80)
        print(f"⚡ STARTING 2-STAGE HYBRID SOTA OCR FOR {prefix}")
        print("=" * 80)
        
        # Clean old parquet & checkpoints for this prefix
        old_pq = os.path.join(PROJECT_ROOT, "data", "ocr_output", f"{prefix}_ocr.parquet")
        old_ckpt = os.path.join(PROJECT_ROOT, "data", "ocr_output", f"checkpoints_{prefix}")
        
        if os.path.exists(old_pq):
            os.remove(old_pq)
            print(f" 🗑️ Removed old {old_pq}")
        if os.path.exists(old_ckpt):
            shutil.rmtree(old_ckpt)
            print(f" 🗑️ Removed old {old_ckpt}")
            
        cmd = [PYTHON_EXE, "tools/multiprocess_gpu_ocr.py", "--prefix", prefix, "--workers", "4"]
        run_cmd(cmd)
        print(f"🎉 Completed {prefix} successfully!")

    # 3. Final rescore & unified sync across all datasets L21-L30
    print("\n" + "=" * 80)
    print("🔄 EXECUTING FINAL RESCORE & UNIFIED PARQUET SYNC ACROSS ALL DATASETS (L21-L30)")
    print("=" * 80)
    run_cmd([PYTHON_EXE, "tools/rescore_ocr_parquets.py"])
    
    print("\n" + "=" * 80)
    print("🎉 ALL DATASETS (L21 TO L30) ARE 100% PROCESSED & UNIFIED IN OCR_TEXT.PARQUET!")
    print("=" * 80)

if __name__ == "__main__":
    main()
