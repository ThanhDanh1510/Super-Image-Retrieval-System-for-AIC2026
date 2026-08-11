import os
import sys
import yaml
from pathlib import Path

# Đưa project root vào python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 80)
    print("📥 DOWNLOADING ALL REGISTERED AI MODELS FROM MODEL_VERSIONS.YAML")
    print("=" * 80)

    # 1. Đọc model versions config
    config_path = PROJECT_ROOT / "configs" / "model_versions.yaml"
    if not config_path.exists():
        print(f"❌ Không tìm thấy file cấu hình tại {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Thu thập danh sách model IDs
    model_ids = []
    
    # Text embedding & reranker & description
    models_section = config.get("models", {})
    for category in ["text_embedding", "text_reranker", "crossmodal_reranker", "description_embedding"]:
        info = models_section.get(category, {})
        if "model_id" in info:
            model_ids.append(info["model_id"])

    # Image embedding
    clip_info = models_section.get("image_embedding", {}).get("clip", {})
    if "model_id" in clip_info:
        model_ids.append(clip_info["model_id"])

    # ASR model
    asr_info = models_section.get("asr", {}).get("phowhisper", {})
    if "model_id" in asr_info:
        model_ids.append(asr_info["model_id"])

    # Loại bỏ trùng lặp nếu có
    model_ids = sorted(list(set(model_ids)))

    print(f"📋 Tìm thấy {len(model_ids)} models cần tải về:")
    for m in model_ids:
        print(f"   - {m}")
    print()

    # 3. Tiến hành tải bằng HuggingFace transformers
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

    # Thiết lập cache directory (tùy chọn)
    # Nếu muốn dùng cache chung trong project, có thể set environment variable HF_HOME
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print(f"📂 Sử dụng HF_HOME cache directory: {hf_home}")

    for model_id in model_ids:
        print("-" * 80)
        print(f"🚀 Đang tải: {model_id}...")
        try:
            # Tải tokenizer
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Tải model weights
            try:
                # Thử với SequenceClassification trước (phù hợp với Rerankers)
                AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
            except Exception:
                # Fallback về AutoModel thông thường
                AutoModel.from_pretrained(model_id, trust_remote_code=True)
                
            print(f"✅ Tải thành công: {model_id}")
        except Exception as e:
            print(f"❌ Lỗi khi tải {model_id}: {e}")

    print("=" * 80)
    print("🎉 Hoàn tất quá trình tải tất cả AI Models!")
    print("=" * 80)

if __name__ == "__main__":
    main()
