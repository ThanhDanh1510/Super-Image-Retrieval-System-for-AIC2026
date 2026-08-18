"""
tools/data_prep/gemini_folder_extractor.py
============================================
Trích xuất OCR & Mô tả ảnh chi tiết (Scene Description) bằng Gemini API.
Cấu trúc tinh gọn: Chỉ gồm 2 trường cốt lõi `ocr_text` và `description`.
Cơ chế: Chạy hết tất cả ảnh trong 1 folder video rồi tự động dừng lại.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# =============================================================================
# 1. Pydantic Response Schema (2 Trường Cốt Lõi)
# =============================================================================

class KeyframeAnalysis(BaseModel):
    ocr_text: str = Field(
        default="", 
        description="Toàn bộ chữ viết, văn bản, công thức toán Oxyz, số hiệu, logo xuất hiện trong ảnh."
    )
    description: str = Field(
        ..., 
        description="Đoạn văn mô tả chi tiết: Câu đầu tóm tắt trọng tâm, các câu sau mô tả nhân vật, hành động, đồ vật, không gian, màu sắc."
    )


# =============================================================================
# 2. Hàm Phân Tích Ảnh Bằng Gemini API
# =============================================================================

SYSTEM_PROMPT = """Bạn là chuyên gia thị giác máy tính hàng đầu cho hệ thống AI Video Retrieval.
Nhiệm vụ của bạn là phân tích ảnh keyframe với độ chi tiết và chính xác tối đa.

Quy tắc:
1. OCR_TEXT: Đọc CHÍNH XÁC 100% tất cả văn bản nội dung trong ảnh (bài giảng, bảng viết, công thức toán học/hình học Oxyz, biển hiệu cửa hàng, băng rôn, số xe, phụ đề, nhãn mác). Giữ nguyên dấu tiếng Việt và ký hiệu toán.
   ⚠️ QUY TẮC LOẠI TRỪ: TUYỆT ĐỐI KHÔNG đọc logo/watermark của đài truyền hình, kênh báo chí ở các góc màn hình (ví dụ: 'Thanh Niên', 'Báo Thanh Niên', 'VTV', 'VTV1', 'HTV', 'Tuổi Trẻ', 'VTC', 'VOV'). Nếu không có chữ nội dung nào khác, trả về chuỗi rỗng "".
2. DESCRIPTION: Viết 1 đoạn văn tiếng Việt hoàn chỉnh. Câu đầu tiên tóm tắt hành động/đối tượng chính; các câu tiếp theo mô tả chi tiết người, đồ vật, hành động, không gian, màu sắc, ánh sáng.
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
    out_file = out_dir / f"{video_name}_descriptions.json"

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

            results[frame_key] = data
            success_count += 1
            print(f"✅ Xong ({duration:.2f}s) | OCR: {len(data.get('ocr_text', ''))} ký tự")

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
