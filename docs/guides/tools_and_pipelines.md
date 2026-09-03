# 🛠️ Hướng Dẫn Sử Dụng Bộ Công Cụ & Pipelines (Tools & Pipelines)

Dự án trang bị bộ công cụ tiện ích mạnh mẽ phục vụ chuẩn bị dữ liệu, lọc ảnh rác, kiểm toán độ phủ video và tối ưu hóa index.

---

## 1. Web UI Lọc Ảnh & Gom Cụm Trùng Lặp (`tools/image_filter`)

Ứng dụng web trực quan giúp duyệt từng video, đánh dấu loại bỏ ảnh mờ/hỏng/thừa và tìm kiếm gom cụm trùng lặp:

```powershell
python -m tools.image_filter.server
# Truy cập: http://localhost:8501
```

### Tính năng chính:
- **🔍 Bộ lọc phạm vi quét:** Chọn quét theo Video hiện tại (`L25_V001`), theo Batch (`L25`), hoặc toàn bộ Dataset.
- **⚡ Phím tắt nhanh:** `Shift + Click` để chọn dải ảnh liên tiếp, 1-Click tìm ảnh tương đồng bằng vector CLIP.
- **💾 Tự động đồng bộ:** Lưu trạng thái lọc vào `data/filtered_keyframes.json`.

```powershell
# Áp dụng xóa vĩnh viễn các file ảnh đã lọc:
python -m aggregation.apply_filter --delete-images
```

---

## 2. Kiểm Toán Độ Phủ Timeline Video (`tools/data_prep/audit_keyframe_coverage.py`)

Tự động quét toàn bộ 872 video trong `data/video/`, phân tích mốc thời gian của từng keyframe và phát hiện các khoảng trống (gaps):

```powershell
python tools/data_prep/audit_keyframe_coverage.py
```
- Báo cáo chi tiết được lưu tại: [`data/reports/keyframe_coverage_report.csv`](data/reports/keyframe_coverage_report.csv).

---

## 3. Quản Trị Zero-Downtime Alias Qdrant (`indexing/collection_alias.py`)

Hỗ trợ chuyển đổi collection phục vụ tìm kiếm tức thì mà không cần dừng server:

```powershell
# Xem collection đang nhận alias 'production'
python -m indexing.collection_alias --action get

# Đổi alias 'production' sang collection mới
python -m indexing.collection_alias --action set --collection aic2026_clean_v3
```

---

## 4. Tự Động Hóa Cập Nhật Mô Tả VLM L25 (`tools/update_l25_descriptions.py`)

Đọc 87 file JSON mô tả từ `data/gemini_output/`, ánh xạ vào 6,119 scenes của L25 và tự động mã hóa lại vector BGE-Large trên GPU:

```powershell
python tools/update_l25_descriptions.py
```
