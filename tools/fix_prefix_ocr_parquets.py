import os
import glob
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

prefixes = ["L26", "L27", "L29", "L30"]
ocr_dir = os.path.join(PROJECT_ROOT, "data", "ocr_output")

print("=" * 80)
print("🛠️ SỬA CÁC FILE PARQUET & CHECKPOINTS BỊ HARDCODE PREFIX L28_")
print("=" * 80)

for pfix in prefixes:
    pfix_upper = pfix.upper()
    parquet_path = os.path.join(ocr_dir, f"{pfix_upper}_ocr.parquet")
    ckpt_dir = os.path.join(ocr_dir, f"checkpoints_{pfix_upper}")

    print(f"\n📂 Đang kiểm tra & sửa {pfix_upper}...")

    # 1. Fix consolidated Parquet file if exists
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        # Check if video_id starts with L28_ instead of pfix_upper
        wrong_count = (df['video_id'].str.startswith('L28_')).sum()
        if wrong_count > 0:
            print(f" ⚠️  Phát hiện {wrong_count:,d} / {len(df):,d} rows có video_id L28_ trong {os.path.basename(parquet_path)}")
            df['video_id'] = df['video_id'].apply(lambda v: f"{pfix_upper}_{v.split('_', 1)[1]}" if v.startswith("L28_") else v)
            df['scene_id'] = df['scene_id'].apply(lambda s: f"{pfix_upper}_{s.split('_', 1)[1]}" if s.startswith("L28_") else s)
            df.to_parquet(parquet_path, index=False)
            print(f" ✅ Đã sửa xong file {os.path.basename(parquet_path)}")
        else:
            print(f" ✅ File {os.path.basename(parquet_path)} đã chuẩn xác!")

    # 2. Fix checkpoint files inside checkpoint_dir
    if os.path.exists(ckpt_dir):
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.parquet"))
        fixed_ckpts = 0
        for cf in ckpt_files:
            bname = os.path.basename(cf)
            df_c = pd.read_parquet(cf)
            is_modified = False
            if (df_c['video_id'].str.startswith('L28_')).any():
                df_c['video_id'] = df_c['video_id'].apply(lambda v: f"{pfix_upper}_{v.split('_', 1)[1]}" if v.startswith("L28_") else v)
                df_c['scene_id'] = df_c['scene_id'].apply(lambda s: f"{pfix_upper}_{s.split('_', 1)[1]}" if s.startswith("L28_") else s)
                is_modified = True

            target_filename = bname
            if bname.startswith("L28_"):
                target_filename = f"{pfix_upper}_{bname.split('_', 1)[1]}"
                is_modified = True

            target_path = os.path.join(ckpt_dir, target_filename)
            if is_modified or target_filename != bname:
                df_c.to_parquet(target_path, index=False)
                if target_filename != bname and os.path.exists(cf):
                    os.remove(cf)
                fixed_ckpts += 1

        print(f" ✅ Đã sửa xong {fixed_ckpts} / {len(ckpt_files)} files checkpoint trong {os.path.basename(ckpt_dir)}")

# 3. Synchronize data/unified/ocr_text.parquet
print("\n🔄 Đang hợp nhất lại 100% dữ liệu chuẩn hóa vào data/unified/ocr_text.parquet...")
all_parquets = sorted(glob.glob(os.path.join(ocr_dir, "L*_ocr.parquet")))
dfs = []
for pf in all_parquets:
    df_p = pd.read_parquet(pf)
    dfs.append(df_p[['scene_id', 'video_id', 'ocr_text']])

df_unified = pd.concat(dfs, ignore_index=True)
df_unified.to_parquet(os.path.join(PROJECT_ROOT, "data", "unified", "ocr_text.parquet"), index=False)
print(f"🎉 Đã cập nhật xong 100% {len(df_unified):,d} scenes chuẩn prefix vào data/unified/ocr_text.parquet!")
