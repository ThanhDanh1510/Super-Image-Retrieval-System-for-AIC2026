# 🧠 Hướng Dẫn & Prompt Tối Ưu: OCR & Sinh Mô Tả Chi Tiết Bằng Gemini API (AIC2026)

Tài liệu này cung cấp **System Prompt, JSON Schema và Script Python hoàn chỉnh** để trích xuất chữ viết (OCR) và sinh mô tả hình ảnh (Scene Description) bằng Gemini API phục vụ bài toán **AI Video Keyframe Retrieval**.

---

## 1. 🎯 Thiết Kế JSON Schema Tối Ưu Cho Tìm Kiếm

Mỗi ảnh keyframe sau khi phân tích sẽ trả về đúng định dạng JSON có cấu trúc sau:

```json
{
  "ocr_text": "Toàn bộ chữ viết trên màn hình, bảng hiệu, công thức toán, số hiệu, logo...",
  "short_caption": "Tóm tắt 1 câu súc tích về hành động và đối tượng chính trong khung hình.",
  "detailed_description": "Mô tả toàn diện: Nhân vật (trang phục, độ tuổi), Hành động, Đồ vật, Không gian/Bối cảnh, Màu sắc chủ đạo, Góc máy quay.",
  "objects": ["đối tượng 1", "đối tượng 2", "đối tượng 3"],
  "scene_type": "lớp học / đường phố / trường quay / chợ / trong nhà / ngoài trời",
  "tags": ["từ khóa 1", "từ khóa 2", "từ khóa tiếng Anh", "thể loại"]
}
```

---

## 2. 📝 System Prompt & User Prompt Cho Gemini

### 🤖 System Instruction:
```text
Bạn là chuyên gia thị giác máy tính và phân tích dữ liệu đa phương tiện hàng đầu cho hệ thống AI Video Retrieval.
Nhiệm vụ của bạn là phân tích ảnh chụp từ video (keyframe) với độ chi tiết và độ chính xác tối đa.
Tuyệt đối không suy đoán những gì không có trong ảnh.

Quy tắc phân tích:
1. OCR_TEXT: Đọc CHÍNH XÁC 100% tất cả văn bản xuất hiện trong ảnh (bảng hiệu, phụ đề, chữ viết tay, công thức toán học/hình học, biển số, số điện thoại, nhãn dán, logo). Giữ nguyên dấu tiếng Việt và các ký hiệu toán học. Nếu không có chữ, trả về chuỗi rỗng "".
2. SHORT_CAPTION: Một câu tiếng Việt ngắn gọn mô tả chính xác nội dung trọng tâm của bức ảnh.
3. DETAILED_DESCRIPTION: Đoạn văn tiếng Việt mô tả đầy đủ:
   - Chủ thể chính (người, động vật, phương tiện) kèm hành động đang thực hiện.
   - Bối cảnh không gian, địa điểm, thời gian (ban ngày/đêm), thời tiết, màu sắc và ánh sáng.
   - Các vật thể phụ xung quanh và vị trí tương đối (trái, phải, trung tâm, tiền cảnh, hậu cảnh).
4. OBJECTS: Danh sách các thực thể, đồ vật, con người có mặt trong ảnh.
5. SCENE_TYPE: Phân loại bối cảnh ngắn gọn (ví dụ: "lớp học bài giảng", "đường phố đông đúc", "trường quay thời sự", "phòng thí nghiệm").
6. TAGS: Mảng các từ khóa quan trọng cả tiếng Việt và tiếng Anh để phục vụ tra cứu ngược.
```

### 💬 User Prompt:
```text
Hãy phân tích bức ảnh keyframe đính kèm và trả về kết quả theo đúng cấu trúc JSON yêu cầu.
```

---

## 3. 🐍 Script Python Chạy Hết 1 Folder Rồi Dừng (`gemini_folder_extractor.py`)

Script dưới đây tự động quét tất cả ảnh trong **1 folder video cụ thể**, gọi Gemini API theo batch, lưu file kết quả JSON và tự động dừng lại khi xong thư mục đó.

```python
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# =============================================================================
# 1. Pydantic Response Schema cho Structured Output
# =============================================================================

class KeyframeAnalysis(BaseModel):
    ocr_text: str = Field(
        default="", 
        description="Toàn bộ chữ viết, văn bản, công thức, số hiệu, logo xuất hiện trong ảnh."
    )
    short_caption: str = Field(
        ..., 
        description="Câu tóm tắt ngắn gọn súc tích 1 câu về nội dung chính."
    )
    detailed_description: str = Field(
        ..., 
        description="Mô tả chi tiết nhân vật, hành động, đồ vật, không gian, màu sắc, bối cảnh."
    )
    objects: List[str] = Field(
        default_factory=list, 
        description="Danh sách các đồ vật, thực thể xuất hiện trong ảnh."
    )
    scene_type: str = Field(
        ..., 
        description="Phân loại bối cảnh (ví dụ: lớp học, đường phố, trường quay, thể thao...)"
    )
    tags: List[str] = Field(
        default_factory=list, 
        description="Các từ khóa tra cứu quan trọng bằng tiếng Việt và tiếng Anh."
    )

# =============================================================================
# 2. Hàm Xử Lý Phân Tích Ảnh Với Gemini API
# =============================================================================

SYSTEM_PROMPT = """Bạn là chuyên gia thị giác máy tính hàng đầu cho hệ thống AI Video Retrieval.
Nhiệm vụ của bạn là phân tích ảnh keyframe với độ chi tiết và chính xác tối đa.

Quy tắc:
1. OCR_TEXT: Đọc CHÍNH XÁC 100% tất cả văn bản trong ảnh (bảng hiệu, phụ đề, chữ viết tay, công thức toán học/hình học Oxyz, số hiệu, logo). Giữ nguyên dấu tiếng Việt và ký hiệu. Nếu không có chữ, trả về "".
2. SHORT_CAPTION: Một câu tiếng Việt ngắn gọn mô tả sự kiện/hành động chính.
3. DETAILED_DESCRIPTION: Đoạn văn mô tả chi tiết người, đồ vật, hành động, không gian, màu sắc, ánh sáng.
4. OBJECTS: Danh sách các thực thể/vật thể có trong ảnh.
5. SCENE_TYPE: Thể loại bối cảnh (ví dụ: 'bài giảng toán', 'giao thông', 'phim tài liệu'...).
6. TAGS: Danh sách từ khóa song ngữ (Việt - Anh) phục vụ tìm kiếm.
"""

def analyze_single_image(client: genai.Client, model_name: str, image_path: Path) -> Dict[str, Any]:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Nhận diện MIME type
    mime_type = "image/webp" if image_path.suffix.lower() == ".webp" else "image/jpeg"

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Hãy phân tích ảnh keyframe đính kèm và trả về JSON theo đúng schema."
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=KeyframeAnalysis,
            temperature=0.1,  # Nhiệt độ thấp để OCR chính xác tuyệt đối
        )
    )
    return json.loads(response.text)

# =============================================================================
# 3. Hàm Chạy Hết 1 Folder Video
# =============================================================================

def process_video_folder(folder_path: str, output_dir: str, model_name: str = "gemini-2.5-flash"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ LỖI: Chưa thiết lập biến môi trường GEMINI_API_KEY!")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    folder = Path(folder_path).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"❌ LỖI: Thư mục không tồn tại: {folder}")
        sys.exit(1)

    video_name = folder.name
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{video_name}_gemini_descriptions.json"

    # Đọc kết quả đã chạy trước đó nếu có (Resume checkpoint)
    results: Dict[str, Any] = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"🔄 Đã nạp checkpoint: Đã có {len(results)} ảnh được xử lý trước đó.")
        except Exception:
            results = {}

    # Lấy danh sách toàn bộ ảnh trong folder
    valid_exts = {".webp", ".jpg", ".jpeg", ".png"}
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in valid_exts])
    total = len(images)
    print(f"\n📂 Bắt đầu xử lý folder: {video_name} (Tổng cộng: {total} ảnh)")
    print(f"💾 File xuất kết quả: {out_file}\n")

    success_count = 0
    skip_count = 0

    for idx, img_path in enumerate(images, 1):
        frame_key = img_path.stem  # Ví dụ '18316'

        if frame_key in results:
            skip_count += 1
            continue

        print(f"[{idx}/{total}] 🖼️ Đang xử lý: {img_path.name} ... ", end="", flush=True)
        try:
            t0 = time.time()
            data = analyze_single_image(client, model_name, img_path)
            duration = time.time() - t0

            data["image_path"] = str(img_path.relative_to(folder.parents[2])) if len(folder.parents) >= 3 else str(img_path.name)
            results[frame_key] = data
            success_count += 1
            print(f"✅ Xong ({duration:.2f}s) | OCR: {len(data.get('ocr_text', ''))} ký tự")

            # Lưu định kỳ sau mỗi ảnh để không mất dữ liệu
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # Nghỉ nhẹ 0.3s để giữ Rate Limit ổn định
            time.sleep(0.3)

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            time.sleep(2.0)

    print(f"\n🎉 HOÀN THÀNH FOLDER {video_name}!")
    print(f"📊 Tổng: {total} ảnh | Mới xử lý: {success_count} | Đã bỏ qua: {skip_count}")
    print(f"📁 Kết quả đã lưu tại: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất OCR & Mô tả ảnh bằng Gemini API cho 1 folder")
    parser.add_argument("--folder", type=str, required=True, help="Đường dẫn tới thư mục video (ví dụ: ./data/image/L25/L25/V017)")
    parser.add_argument("--output-dir", type=str, default="./data/gemini_output", help="Thư mục lưu file JSON kết quả")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Tên model Gemini")
    args = parser.parse_args()

    process_video_folder(args.folder, args.output_dir, args.model)
```

---

## 4. 🚀 Hướng Dẫn Sử Dụng 1-Click

### Bước 1: Cài đặt thư viện Google GenAI
```bash
pip install google-genai pydantic
```

### Bước 2: Thiết lập API Key
```bash
# Trên macOS / Linux:
export GEMINI_API_KEY="AIzaSy..."

# Trên Windows (PowerShell):
$env:GEMINI_API_KEY="AIzaSy..."
```

### Bước 3: Chạy cho 1 folder video bất kỳ
```bash
python gemini_folder_extractor.py --folder ./data/image/L25/L25/V017 --output-dir ./data/gemini_output
```
👉 Script sẽ chạy qua toàn bộ ảnh của đúng thư mục `V017`, ghi file `V017_gemini_descriptions.json` và **dừng lại ngay khi hoàn tất**!
