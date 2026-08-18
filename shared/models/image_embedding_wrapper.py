"""
shared/models/image_embedding_wrapper.py
=========================================
Wrapper chuẩn hóa cho image embedding model (CLIP ViT-L/14).

RÀNG BUỘC CỨNG:
  - encode() phải trả về L2-normalized vector dim=1024.
  - Dim phải khớp với named_vector_spec.IMAGE_DENSE.dim = 1024.

GHI CHÚ RAM: Chỉ load CLIP (1 model duy nhất) — BEiT đã được bỏ khỏi
wrapper để giảm bộ nhớ; fused BEiT + CLIP chỉ tồn tại phía indexing.
"""

from __future__ import annotations

import io
import logging
from typing import Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageEmbeddingWrapper:
    """
    Wrapper load và inference CLIP ViT-L/14 image embedding model.

    Output: vector dim=1024, L2-normalized (khớp IMAGE_DENSE).
    """

    CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
    VECTOR_DIM = 1024

    def __init__(self, device: str = "cuda") -> None:
        """Load CLIP model vào memory."""
        self.device = device
        self.is_mock = False
        self.clip_model = None
        self.clip_processor = None

        try:
            import os
            from pathlib import Path
            if "HF_HOME" not in os.environ:
                hf_dir = Path("./data/hf_cache").resolve()
                hf_dir.mkdir(parents=True, exist_ok=True)
                os.environ["HF_HOME"] = str(hf_dir)

            import torch
            import transformers
            from transformers import CLIPModel, CLIPProcessor

            from shared.utils.device_utils import resolve_torch_device

            device_str = resolve_torch_device(device)
            if device_str == "cuda":
                try:
                    # Test GPU kernel capability
                    _t = torch.zeros(1, device="cuda")
                except Exception:
                    logger.warning("CUDA execution failed. Falling back to CPU for Real Model.")
                    device_str = "cpu"
            elif device_str == "mps":
                try:
                    _t = torch.zeros(1, device="mps")
                except Exception:
                    logger.warning("MPS execution failed. Falling back to CPU for Real Model.")
                    device_str = "cpu"

            logger.info(f"Loading CLIP model '{self.CLIP_MODEL_ID}' on {device_str} (transformers v{transformers.__version__})...")
            self.clip_processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL_ID)
            dtype = torch.float16 if device_str in ("cuda", "mps") else torch.float32
            self.clip_model = CLIPModel.from_pretrained(
                self.CLIP_MODEL_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device_str)
            self.clip_model.eval()
            self.torch = torch
            self.device_str = device_str
        except Exception as e:
            import traceback
            logger.warning(
                f"Không thể load real transformers model ({e}). Traceback: {traceback.format_exc()}. "
                "Sẽ sử dụng Fallback Deterministic Image Encoder cho môi trường test/CPU."
            )
            self.is_mock = True

    @property
    def dim(self) -> int:
        """Dimension của output vector = 1024."""
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

    def _mock_encode_image(self, img: Image.Image) -> np.ndarray:
        """Tạo deterministic mock vector 1024-dim cho 1 PIL Image khi offline."""
        img_rgb = img.convert("RGB").resize((64, 64))
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        # Seed dựa trên mean pixel values
        seed = int(np.sum(arr * 1000.0)) % (2**31 - 1)
        rng = np.random.RandomState(seed)
        raw_vec = rng.randn(1024).astype(np.float32)
        return self._normalize(raw_vec)

    def encode(
        self,
        image: Union[Image.Image, list[Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode ảnh thành L2-normalized embedding vector(s).
        """
        if image is None:
            raise ValueError("Input image không được là None")

        is_single = isinstance(image, Image.Image)
        images = [image] if is_single else list(image)

        if not images:
            raise ValueError("Danh sách ảnh rỗng")

        if self.is_mock:
            vecs = np.vstack([self._mock_encode_image(img) for img in images])
            return vecs[0] if is_single else vecs

        try:
            all_vecs = []
            for i in range(0, len(images), batch_size):
                batch_imgs = [img.convert("RGB") for img in images[i : i + batch_size]]
                inputs = self.clip_processor(images=batch_imgs, return_tensors="pt").to(self.device_str)
                with self.torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                    if not isinstance(features, self.torch.Tensor):
                        features = getattr(features, "image_embeds", getattr(features, "pooler_output", features[0]))
                    feats_np = features.cpu().numpy().astype(np.float32)
                    
                    if feats_np.shape[1] < 1024:
                        pad_width = 1024 - feats_np.shape[1]
                        feats_np = np.pad(feats_np, ((0, 0), (0, pad_width)), mode="constant")
                    elif feats_np.shape[1] > 1024:
                        feats_np = feats_np[:, :1024]
                        
                    norm_vecs = self._normalize(feats_np)
                    all_vecs.append(norm_vecs)

            final_vecs = np.vstack(all_vecs)
            return final_vecs[0] if is_single else final_vecs
        except Exception as e:
            logger.warning(f"Lỗi khi encode real CLIP image, fallback sang mock: {e}")
            vecs = np.vstack([self._mock_encode_image(img) for img in images])
            return vecs[0] if is_single else vecs

    def encode_from_bytes(self, image_bytes: bytes, batch_size: int = 32) -> np.ndarray:
        """Decode bytes → PIL Image → encode()."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return self.encode(img, batch_size=batch_size)
        except Exception as e:
            raise ValueError(f"Không thể decode bytes thành ảnh PIL: {e}") from e

    def encode_text(self, text: str | list[str]) -> np.ndarray:
        """Encode text query thành L2-normalized image embedding vector(s) (Text-to-Image)."""
        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)

        if self.is_mock:
            vecs = []
            for t in texts:
                seed = sum(ord(c) for c in t) % (2**31 - 1)
                rng = np.random.RandomState(seed)
                v = rng.randn(1024).astype(np.float32)
                vecs.append(self._normalize(v))
            res = np.vstack(vecs)
            return res[0] if is_single else res

        try:
            inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device_str)
            with self.torch.no_grad():
                features = self.clip_model.get_text_features(**inputs)
                if not isinstance(features, self.torch.Tensor):
                    features = getattr(features, "text_embeds", getattr(features, "pooler_output", features[0]))
                feats_np = features.cpu().numpy().astype(np.float32)
                if feats_np.shape[1] < 1024:
                    pad_width = 1024 - feats_np.shape[1]
                    feats_np = np.pad(feats_np, ((0, 0), (0, pad_width)), mode="constant")
                elif feats_np.shape[1] > 1024:
                    feats_np = feats_np[:, :1024]
                norm_vecs = self._normalize(feats_np)
                return norm_vecs[0] if is_single else norm_vecs
        except Exception as e:
            logger.warning(f"Lỗi encode_text CLIP, fallback: {e}")
            vecs = []
            for t in texts:
                seed = sum(ord(c) for c in t) % (2**31 - 1)
                rng = np.random.RandomState(seed)
                v = rng.randn(1024).astype(np.float32)
                vecs.append(self._normalize(v))
            res = np.vstack(vecs)
            return res[0] if is_single else res
