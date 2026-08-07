import os
import sys
import torch
import numpy as np

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
from transformers import PreTrainedModel, PretrainedConfig

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

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()
    return pixel_values

IMAGE_PATH = r"D:\AIC2026\data\image\L30\L30\V002\3762.webp"

def run_vietocr_branch(img_path):
    print("\n🔹 [Nhánh 1] Đang chạy VietOCR + PaddleOCR...")
    try:
        from ocr_local import TextDetection, Predictor, Cfg, ocr_image
        det = TextDetection()
        cfg = Cfg.load_config_from_name('vgg_transformer')
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        rec = Predictor(cfg)
        text, _ = ocr_image(det, rec, img_path, filter_spam=False, filter_exclude_region=False)
        print(f"   ➔ VietOCR Candidate: {text!r}")
        return text
    except Exception as e:
        print(f"   ❌ VietOCR Error: {e}")
        return ""

def run_vintern_branch(img_path, model_id="5CD-AI/Vintern-1B-v3_5"):
    print(f"\n🔹 [Nhánh 2] Đang chạy Vintern VLM OCR ({model_id})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()
        
        pixel_values = load_image(img_path, input_size=448)
        question = "<image>\nHãy đọc và trích xuất tất cả các dòng chữ tiếng Việt có trong hình ảnh này."
        
        generation_config = dict(max_new_tokens=512, do_sample=False)
        response = model.chat(tokenizer, pixel_values, question, generation_config=generation_config)
        print(f"   ➔ Vintern Candidate:\n{response}")
        return response
    except Exception as e:
        print(f"   ❌ Vintern Error: {e}")
        return ""

def run_llm_post_correction(vietocr_text, vintern_text, llm_model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"\n🔹 [Nhánh 3] Đang nạp và chạy LLM Post-Correction thật ({llm_model_name})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        ).eval()
        
        system_prompt = (
            "Bạn là chuyên gia khôi phục và sửa lỗi chính tả văn bản tiếng Việt từ các nguồn OCR thô. "
            "Nhiệm vụ của bạn là kết hợp 2 nguồn kết quả OCR thô, tự động sửa các nét chữ bị đọc nhầm "
            "để tạo thành các từ tiếng Việt chuẩn xác có nghĩa theo ngữ cảnh. "
            "Chỉ xuất ra danh sách các dòng văn bản tiếng Việt đã được sửa hoàn chỉnh, không kèm giải thích."
        )
        
        user_prompt = (
            f"Dưới đây là 2 kết quả OCR từ cùng 1 bức ảnh keyframe:\n\n"
            f"[Nguồn 1 - Vintern VLM]:\n{vintern_text}\n\n"
            f"[Nguồn 2 - VietOCR]:\n{vietocr_text}\n\n"
            f"Hãy khôi phục và xuất ra văn bản tiếng Việt chuẩn xác nhất:"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0
            )
        
        input_len = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_len:]
        final_output = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        print("\n" + "=" * 60)
        print("🏆 KẾT QUẢ SỬA LỖI CHÍNH TẢ TỪ THỰC TẾ NEURAL LLM (Qwen2.5-1.5B):")
        print("=" * 60)
        print(final_output)
        print("=" * 60)
        return final_output
    except Exception as e:
        print(f"❌ Error in LLM Post-Correction: {e}")
        import traceback
        traceback.print_exc()
        return vintern_text

def main():
    print("=" * 60)
    print("🚀 COMBO TỐI THƯỢNG OCR TEST (GENUINE NEURAL PIPELINE)")
    print("=" * 60)
    print(f"📷 Target image: {IMAGE_PATH}")
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image file does not exist at {IMAGE_PATH}")
        sys.exit(1)
        
    vietocr_res = run_vietocr_branch(IMAGE_PATH)
    vintern_res = run_vintern_branch(IMAGE_PATH)
    
    if vintern_res or vietocr_res:
        run_llm_post_correction(vietocr_res, vintern_res)

if __name__ == "__main__":
    main()
