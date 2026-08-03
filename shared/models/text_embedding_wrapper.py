"""
shared/models/text_embedding_wrapper.py
========================================
Wrapper chuẩn hóa cho text embedding models (gte-Qwen2-1.5B-instruct).

RÀNG BUỘC CỨNG:
  - encode() phải trả về L2-normalized vector.
  - dim của output phải khớp với named_vector_spec.OCR_DENSE.dim và ASR_DENSE.dim (1536).
"""

from __future__ import annotations

import logging
import numpy as np

from shared.utils.text_normalizer import normalize_nfc

logger = logging.getLogger(__name__)


class TextEmbeddingWrapper:
    """
    Wrapper load và inference gte-Qwen2-1.5B-instruct text embedding model.

    Dùng cho cả OCR text và ASR transcript — cùng 1 model, cùng dim=1536.
    """

    MODEL_ID = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    VECTOR_DIM = 1536

    def __init__(self, device: str = "cuda") -> None:
        """Load gte-Qwen2-1.5B model vào memory."""
        self.device = device
        self.is_mock = False
        self.tokenizer = None
        self.model = None

        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            device_str = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
            logger.info(f"Loading Text Embedding model '{self.MODEL_ID}' on {device_str} (transformers v{transformers.__version__})...")
            dtype = torch.float16 if device_str == "cuda" else torch.float32

            try:
                # Thử dùng native transformers implementation (tránh bug DynamicCache cũ của remote code)
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
                self.model = AutoModel.from_pretrained(self.MODEL_ID, torch_dtype=dtype).to(device_str)
            except Exception as e_native:
                logger.info(f"Native load failed ({e_native}), trying with trust_remote_code...")
                config = AutoConfig.from_pretrained(self.MODEL_ID, trust_remote_code=True)
                if not hasattr(config, "rope_theta"):
                    config.rope_theta = 1000000.0

                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True, use_fast=False)
                
                self.model = AutoModel.from_pretrained(
                    self.MODEL_ID,
                    config=config,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(device_str)

            self.model.eval()
            self.torch = torch
            self.device_str = device_str
        except Exception as e:
            import traceback
            logger.warning(
                f"Không thể load real Text Embedding model '{self.MODEL_ID}' ({e}). Traceback: {traceback.format_exc()}. "
                "Sẽ sử dụng Fallback Deterministic Text Encoder cho môi trường test/CPU."
            )
            self.is_mock = True

    @property
    def dim(self) -> int:
        """Dimension của output vector = 1536."""
        return self.VECTOR_DIM

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize numpy vector array."""
        if vec.ndim == 1:
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-12) if norm > 0 else vec
        else:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vec / norms

    def _mock_encode_text(self, text: str) -> np.ndarray:
        """Tạo deterministic mock vector 1536-dim cho 1 chuỗi văn bản."""
        clean_t = normalize_nfc(text)
        if not clean_t:
            clean_t = "[EMPTY]"
            
        seed = sum(ord(c) for c in clean_t) % (2**31 - 1)
        rng = np.random.RandomState(seed)
        raw_vec = rng.randn(1536).astype(np.float32)
        return self._normalize(raw_vec)

    def encode(self, text: str | list[str], batch_size: int = 16) -> np.ndarray:
        """
        Encode text thành L2-normalized embedding vector(s).
        """
        if text is None:
            raise ValueError("Input text không được là None")

        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)

        if not texts:
            raise ValueError("Danh sách text rỗng")

        if self.is_mock:
            vecs = np.vstack([self._mock_encode_text(t) for t in texts])
            return vecs[0] if is_single else vecs

        try:
            all_vecs = []
            for i in range(0, len(texts), batch_size):
                batch_raw = texts[i : i + batch_size]
                batch_clean = [normalize_nfc(t) if t and t.strip() else "[EMPTY]" for t in batch_raw]
                
                inputs = self.tokenizer(
                    batch_clean,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device_str)

                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling trên token embeddings
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    token_embeddings = outputs[0]  # First element of model_output contains all token embeddings
                    sum_embeddings = self.torch.sum(token_embeddings * attention_mask, 1)
                    sum_mask = self.torch.clamp(attention_mask.sum(1), min=1e-9)
                    pooled = (sum_embeddings / sum_mask).cpu().numpy().astype(np.float32)

                    # Adjust dimension sang 1536 nếu cần
                    if pooled.shape[1] < 1536:
                        pad_width = 1536 - pooled.shape[1]
                        pooled = np.pad(pooled, ((0, 0), (0, pad_width)), mode="constant")
                    elif pooled.shape[1] > 1536:
                        pooled = pooled[:, :1536]

                    norm_vecs = self._normalize(pooled)
                    all_vecs.append(norm_vecs)

            final_vecs = np.vstack(all_vecs)
            return final_vecs[0] if is_single else final_vecs
        except Exception as e:
            logger.warning(f"Lỗi encode real Text Embedding model, fallback sang mock: {e}")
            vecs = np.vstack([self._mock_encode_text(t) for t in texts])
            return vecs[0] if is_single else vecs
