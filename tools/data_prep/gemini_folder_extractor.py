"""
tools/data_prep/gemini_folder_extractor.py
============================================
Trích xuất OCR & Mô tả ảnh chi tiết (Scene Description) bằng Gemini API.
Cơ chế: Chạy hết tất cả ảnh trong 1 folder video rồi tự động dừng lại.
"""

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
# 2. Hàm Phân Tích Ảnh Bằng Gemini API
# =============================================================================

SYSTEM_PROMPT = """Bạn là chuyên gia thị giác máy tính hàng đầu cho hệ thống AI Video Retrieval.
Nhiệm vụ của bạn là phân tích ảnh keyframe với độ chi tiết và chính xác tối đa.

Quy tắc:
1. OCR_TEXT: Đọc CHÍNH XÁC 100% tất cả văn bản trong ảnh (bảng hiệu, phụ đề, chữ viết tay, công thức toán học/hình học Oxyz, số hiệu, logo). Giữ nguyên dấu tiếng Việt và ký hiệu toán. Nếu không có chữ, trả về "".
2. SHORT_CAPTION: Một câu tiếng Việt ngắn gọn mô tả sự kiện/hành động chính.
3. DETAILED_DESCRIPTION: Đoạn văn mô tả chi tiết người, đồ vật, hành động, không gian, màu sắc, ánh sáng.
4. OBJECTS: Danh sách các thực thể/vật thể có trong ảnh.
5. SCENE_TYPE: Thể loại bối cảnh (ví dụ: 'bài giảng toán', 'giao thông', 'phim tài liệu'...).
6. TAGS: Danh sách từ khóa song ngữ (Việt - Anh) phục vụ tìm kiếm.
"""

def analyze_single_image(client: genai.Client, model_name: str, image_path: Path) -> Dict[str, Any]:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

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
            temperature=0.1,
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
        print("💡 Hãy chạy: export GEMINI_API_KEY=\"...\" hoặc $env:GEMINI_API_KEY=\"...\"")
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

    # Đọc checkpoint trước đó nếu có (Resume)
    results: Dict[str, Any] = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"🔄 Đã nạp checkpoint: Đã có {len(results)} ảnh được xử lý trước đó.")
        except Exception:
            results = {}

    valid_exts = {".webp", ".jpg", ".jpeg", ".png"}
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in valid_exts])
    total = len(images)
    print(f"\n📂 Bắt đầu xử lý folder: {video_name} (Tổng cộng: {total} ảnh)")
    print(f"💾 File xuất kết quả: {out_file}\n")

    success_count = 0
    skip_count = 0

    for idx, img_path in enumerate(images, 1):
        frame_key = img_path.stem

        if frame_key in results:
            skip_count += 1
            continue

        print(f"[{idx}/{total}] 🖼️ Đang xử lý: {img_path.name} ... ", end="", flush=True)
        try:
            t0 = time.time()
            data = analyze_single_image(client, model_name, img_path)
            duration = time.time() - t0

            data["image_path"] = str(img_path.name)
            results[frame_key] = data
            success_count += 1
            print(f"✅ Xong ({duration:.2f}s) | OCR: {len(data.get('ocr_text', ''))} ký tự")

            # Lưu ngay sau mỗi ảnh
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

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
