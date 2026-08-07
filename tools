import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Fix NumPy 2.0+ compatibility
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, bytes, str, object],
    }

# Patch transformers compatibility
from transformers import PreTrainedModel

def get_tied(self):
    return getattr(self, "_all_tied_weights_keys_val", {})

def set_tied(self, val):
    self._all_tied_weights_keys_val = val

PreTrainedModel.all_tied_weights_keys = property(get_tied, set_tied)

from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_dynamic_tiles(pil_img, input_size=448):
    w, h = pil_img.size
    mid_w = w // 2
    mid_h = h // 2
    
    tile_tl = pil_img.crop((0, 0, mid_w + 50, mid_h + 50))
    tile_tr = pil_img.crop((mid_w - 50, 0, w, mid_h + 50))
    tile_bl = pil_img.crop((0, mid_h - 50, mid_w + 50, h))
    tile_br = pil_img.crop((mid_w - 50, mid_h - 50, w, h))
    
    transform = build_transform(input_size=input_size)
    t_global = transform(pil_img).unsqueeze(0)
    t_tl = transform(tile_tl).unsqueeze(0)
    t_tr = transform(tile_tr).unsqueeze(0)
    t_bl = transform(tile_bl).unsqueeze(0)
    t_br = transform(tile_br).unsqueeze(0)
    
    pixel_values = torch.cat([t_global, t_tl, t_tr, t_bl, t_br], dim=0).to(torch.bfloat16).cuda()
    return pixel_values

class SOTAOCRBatchPipeline:
    def __init__(self):
        print("🚀 Khởi tạo SOTA OCR Batch Pipeline...")
        # Load VietOCR
        print("🔹 Nạp VietOCR...")
        from ocr_local import TextDetection, Predictor, Cfg, ocr_image
        self.det = TextDetection()
        cfg = Cfg.load_config_from_name('vgg_transformer')
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rec = Predictor(cfg)
        self.ocr_image_fn = ocr_image

        # Load Vintern VLM
        vlm_id = "5CD-AI/Vintern-1B-v3_5"
        print(f"🔹 Nạp Vintern VLM ({vlm_id})...")
        self.vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_id, trust_remote_code=True)
        self.vlm_model = AutoModel.from_pretrained(
            vlm_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()

        # Load LLM Fusion
        llm_id = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"🔹 Nạp LLM Fusion ({llm_id})...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_id, trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        ).eval()

    def process_image(self, img_path: str) -> str:
        # Branch 1: VietOCR
        try:
            vietocr_text, _ = self.ocr_image_fn(self.det, self.rec, img_path, filter_spam=False, filter_exclude_region=False)
        except Exception as e:
            vietocr_text = ""

        # Branch 2: Vintern VLM Zero-Shot
        try:
            pil_img = Image.open(img_path).convert('RGB')
            pixel_values = load_dynamic_tiles(pil_img, input_size=448)
            vlm_prompt = "<image>\nHãy liệt kê tất cả các văn bản tiếng Việt xuất hiện trong ảnh (chữ in, chữ viết tay, bảng biểu, tiêu đề, chữ hiển thị trên màn hình):"
            generation_config = dict(max_new_tokens=512, do_sample=False)
            vintern_text = self.vlm_model.chat(self.vlm_tokenizer, pixel_values, vlm_prompt, generation_config=generation_config)
        except Exception as e:
            vintern_text = ""

        if not vintern_text and not vietocr_text:
            return ""

        # Branch 3: General LLM Fusion Zero-Shot
        try:
            system_prompt = (
                "Bạn là chuyên gia hợp nhất và khôi phục văn bản tiếng Việt từ kết quả OCR đa nguồn. "
                "Nhiệm vụ của bạn là kết hợp thông tin, giữ lại toàn bộ các dòng chữ có ý nghĩa, loại bỏ ký tự rác, "
                "sửa các từ bị đọc chệch nét thành tiếng Việt chuẩn có dấu đầy đủ. "
                "Chỉ xuất ra danh sách các dòng văn bản theo đúng nội dung trong ảnh. "
                "Tuyệt đối không tự bịa ra thông tin không có trong kết quả OCR thô."
            )
            
            user_prompt = (
                f"Dưới đây là 2 kết quả OCR thô từ hình ảnh:\n\n"
                f"[Nguồn VLM]:\n{vintern_text}\n\n"
                f"[Nguồn VietOCR]:\n{vietocr_text}\n\n"
                f"Hãy xuất ra danh sách văn bản tiếng Việt làm sạch hoàn chỉnh:"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            prompt_text = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.llm_tokenizer([prompt_text], return_tensors="pt").to("cuda")

            with torch.no_grad():
                generated_ids = self.llm_model.generate(**model_inputs, max_new_tokens=512, do_sample=False)

            input_len = model_inputs.input_ids.shape[1]
            response_ids = generated_ids[0][input_len:]
            final_text = self.llm_tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            return final_text
        except Exception as e:
            return vintern_text or vietocr_text

def run_batch_l22_v001():
    folder_path = r"D:\AIC2026\data\image\L22\L22\V001"
    output_dir = r"D:\AIC2026\data\ocr_test_output"
    os.makedirs(output_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.webp')])
    print(f"📁 Tìm thấy {len(img_files)} ảnh trong thư mục L22_V001.")

    pipeline = SOTAOCRBatchPipeline()
    results = []

    # Process first 10 sample keyframes in L22_V001 to show immediate progress
    sample_files = img_files[:10]
    for idx, fname in enumerate(sample_files, 1):
        img_path = os.path.join(folder_path, fname)
        frame_id = os.path.splitext(fname)[0]
        print(f"\n[{idx}/{len(sample_files)}] Đang xử lý keyframe {fname}...")
        extracted_text = pipeline.process_image(img_path)
        print(f"   ➔ Text extracted:\n{extracted_text}")
        results.append({
            "video_id": "L22_V001",
            "frame_id": frame_id,
            "image_path": img_path,
            "ocr_text": extracted_text
        })

    # Save to Parquet and JSON
    df = pd.DataFrame(results)
    parquet_path = os.path.join(output_dir, "L22_V001_sota_ocr.parquet")
    json_path = os.path.join(output_dir, "L22_V001_sota_ocr.json")
    
    df.to_parquet(parquet_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ ĐÃ HOÀN THÀNH XỬ LÝ BATCH MẪU L22_V001!")
    print(f"💾 Parquet: {parquet_path}")
    print(f"💾 JSON: {json_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_batch_l22_v001()
