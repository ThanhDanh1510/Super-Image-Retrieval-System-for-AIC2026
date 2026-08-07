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
from transformers import AutoModel, AutoTokenizer
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

def test_vintern(model_id="5CD-AI/Vintern-1B-v3_5"):
    print(f"\n📦 Loading Vintern Model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()
        
        print("✅ Model loaded successfully!")
        pixel_values = load_image(IMAGE_PATH, input_size=448)
        question = "<image>\nHãy đọc và trích xuất chính xác toàn bộ các dòng chữ tiếng Việt có trên bảng phấn trong bức hình này."
        
        generation_config = dict(max_new_tokens=512, do_sample=False)
        response = model.chat(tokenizer, pixel_values, question, generation_config=generation_config)
        
        print("\n" + "=" * 60)
        print(f"📝 {model_id} OCR RESULT FOR 3762.webp:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        return True
    except Exception as e:
        print(f"❌ Error loading {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🚀 VLM OCR TEST: Vintern-1B-v3_5")
    print("=" * 60)
    print(f"📷 Target image: {IMAGE_PATH}")
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image file does not exist at {IMAGE_PATH}")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Device: {device} | PyTorch: {torch.__version__}")
    
    test_vintern("5CD-AI/Vintern-1B-v3_5")

if __name__ == "__main__":
    main()
