# 📦 AIC2026 Aggregation Layer

Tầng tổng hợp, chuẩn hóa và đóng gói **Unified Dataset** phục vụ lập chỉ mục Qdrant và phục vụ tìm kiếm.

---

## 📁 Cấu trúc thư mục

```text
aggregation/
├── __init__.py                     # Package entry & exports
├── common.py                       # Hằng số chuẩn (FPS, extensions), Schema và Tiện ích dùng chung
├── pipeline.py                     # Coordinator điều phối toàn bộ chu trình Aggregation
│
├── [Kaggle Distributed Shards Workflow]
│   ├── pull_shards.py              # Tải shards phân tán từ Kaggle
│   ├── verify_shards.py            # Xác thực tính toàn vẹn (schema, counts, vectors)
│   ├── merge_datasets.py           # Gộp các shard thành Unified Dataset
│   └── publish_unified.py          # Đóng gói và đẩy Unified Dataset lên Kaggle
│
└── [Unified Dataset & Filtering Workflow]
    ├── build_unified_from_raw.py   # Xây dựng Unified Dataset trực tiếp từ raw images + JSONs
    └── apply_filter.py             # Cắt tỉa ảnh đĩa cứng, Parquet metadata & Vector matrices
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Xây dựng Unified Dataset từ dữ liệu thô (Local Raw Builder)
Quét toàn bộ ảnh trong `data/image/`, khớp OCR JSON và ASR Transcript để tạo metadata Parquet:
```bash
python -m aggregation.build_unified_from_raw \
    --image-dir ./data/image \
    --ocr-dir ./data/OCR \
    --asr-json ./data/ASR/asr.json \
    --output-dir ./data/unified
```

---

### 2. Quy trình Kaggle Distributed Shards (Phân tán nhiều thành viên)
1. **Tải toàn bộ shard từ Kaggle:**
   ```bash
   python -m aggregation.pull_shards --shards-dir ./data/raw_shards --owner thanh-danh
   # Hoặc: make pull-shards
   ```
2. **Xác thực tính toàn vẹn:**
   ```bash
   python -m aggregation.verify_shards --shards-dir ./data/raw_shards
   # Hoặc: make verify-shards
   ```
3. **Gộp thành Unified Dataset:**
   ```bash
   python -m aggregation.merge_datasets --shards-dir ./data/raw_shards --output-dir ./data/unified
   # Hoặc: make merge-datasets
   ```
4. **Publish lên Kaggle:**
   ```bash
   python -m aggregation.publish_unified --unified-dir ./data/unified --version 1 --owner thanh-danh
   # Hoặc: make publish-unified
   ```

---

### 3. Áp dụng Bộ Lọc Ảnh & Cắt Tỉa Dataset (`apply_filter`)
Áp dụng danh sách `filtered_keyframes.json` để xóa ảnh rác trên đĩa và đồng bộ toàn bộ file Parquet + Vector matrices:
* **Kiểm tra thử (Dry-run):**
  ```bash
  python -m aggregation.apply_filter --dry-run
  ```
* **Áp dụng thực tế (Xóa ảnh & Rebuild Unified Dataset):**
  ```bash
  python -m aggregation.apply_filter --delete-images
  # Hoặc: make apply-filter
  ```
