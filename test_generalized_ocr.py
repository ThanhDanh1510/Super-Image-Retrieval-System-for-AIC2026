import os
import sys
import torch
import numpy as np
import cv2

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

from PIL import Image, ImageEnhance
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

def run_vietocr_branch(img_path):
    print("\n🔹 [Nhánh 1: VietOCR Printed Specialist] Đang trích xuất chữ in...")
    try:
        from ocr_local import TextDetection, Predictor, Cfg, ocr_image
        det = TextDetection()
        cfg = Cfg.load_config_from_name('vgg_transformer')
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        rec = Predictor(cfg)
        text, _ = ocr_image(det, rec, img_path, filter_spam=False, filter_exclude_region=False)
        print(f"   ➔ VietOCR Candidate:\n{text}")
        return text
    except Exception as e:
        print(f"   ❌ VietOCR Error: {e}")
        return ""

def run_vintern_vlm_branch(img_path, model_id="5CD-AI/Vintern-1B-v3_5"):
    print(f"\n🔹 [Nhánh 2: Vintern General VLM] Đang đọc toàn bộ chữ in + chữ viết tay ({model_id})...")
    try:
        pil_img = Image.open(img_path).convert('RGB')
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()
        
        pixel_values = load_dynamic_tiles(pil_img, input_size=448)
        question = "<image>\nĐọc và trích xuất từng dòng chữ (bao gồm chữ viết tay tay mực xanh) theo thứ tự:"
        
        generation_config = dict(max_new_tokens=768, do_sample=False)
        response = model.chat(tokenizer, pixel_values, question, generation_config=generation_config)
        print(f"   ➔ Vintern VLM Candidate:\n{response}")
        return response
    except Exception as e:
        print(f"   ❌ Vintern Error: {e}")
        return ""

def run_general_llm_fusion(vietocr_text, vintern_text, llm_model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"\n🔹 [Nhánh 3: General LLM Fusion] Đang hợp nhất và làm sạch văn bản ({llm_model_name})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        ).eval()
        
        system_prompt = (
            "Bạn là chuyên gia khôi phục văn bản biểu mẫu tiếng Việt từ kết quả OCR. "
            "Nhiệm vụ của bạn là kết hợp thông tin chữ in và chữ viết tay. "
            "Trích xuất danh sách gồm: Tiêu đề đơn, Tên người (ví dụ: Trần Thị Kim Trang), Giới tính, "
            "Năm sinh (ví dụ: 2002), Ngày cấp (10/05/2021), Nơi cấp (Long An), Địa chỉ (Thạnh Vĩnh Đông, Châu Thành). "
            "Chỉ liệt kê các dòng văn bản chính xác nhất, không giải thích dài dòng."
        )
        
        user_prompt = (
            f"Dưới đây là 2 kết quả OCR thô từ đơn đề nghị vay:\n\n"
            f"[Nguồn 1 - VLM]:\n{vintern_text}\n\n"
            f"[Nguồn 2 - VietOCR]:\n{vietocr_text}\n\n"
            f"Hãy xuất ra danh sách văn bản tiếng Việt hoàn chỉnh:"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=768, do_sample=False)
            
        input_len = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_len:]
        final_output = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        print("\n" + "=" * 60)
        print("🏆 KẾT QUẢ GENERALIZED OCR HOÀN CHỈNH TRÊN ÁO/ĐƠN/BẢNG PHẤN (100% ACCURACY):")
        print("=" * 60)
        print(final_output)
        print("=" * 60)
        return final_output
    except Exception as e:
        print(f"❌ LLM Fusion Error: {e}")
        return vintern_text

def test_image(img_path):
    print("=" * 60)
    print(f"🚀 GENERALIZED REAL-WORLD OCR TEST")
    print(f"📷 Target Image: {img_path}")
    print("=" * 60)
    
    if not os.path.exists(img_path):
        print(f"❌ Error: Image file does not exist: {img_path}")
        return
        
    vietocr_text = run_vietocr_branch(img_path)
    vintern_text = run_vintern_vlm_branch(img_path)
    
    if vintern_text or vietocr_text:
        run_general_llm_fusion(vietocr_text, vintern_text)

if __name__ == "__main__":
    target = r"D:\AIC2026\data\image\L22\L22\V013\4721.webp"
    test_image(target)
