"""
serving/api/core/models.py
============================
Unified ModelRegistry — quản lý vòng đời toàn bộ 7 AI models và Search Engines (RAM saver).

Hỗ trợ:
  - Lazy load: model chỉ được khởi tạo khi endpoint đầu tiên thực sự cần.
  - Idle unload: nếu MODEL_UNLOAD_TTL_S > 0, model không được dùng trong
    khoảng thời gian đó sẽ bị giải phóng khỏi RAM/VRAM (gc + cuda.empty_cache).
  - unload_all(): giải phóng toàn bộ models khi shutdown server.
  - Thread-safe & Async-safe: đồng thời nhiều requests chỉ load 1 lần.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def free_memory() -> None:
    """Thu hồi RAM/VRAM của các model đã bị unload."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


_GLOBAL_REGISTRY_INSTANCE: ModelRegistry | None = None
_GLOBAL_REGISTRY_LOCK = threading.Lock()


class ModelRegistry:
    """
    Unified Model Registry cho toàn bộ các AI model và engine nặng của serving layer.

    Keys:
      - "text":          TextEmbeddingWrapper (gte-Qwen2-1.5B-instruct, dim=1536)
      - "image":         ImageEmbeddingWrapper (CLIP ViT-L/14, dim=1024)
      - "asr":           PhoWhisper ASR pipeline (vinai/PhoWhisper-medium)
      - "viranker":      ViRanker CrossEncoder (BAAI/bge-reranker-v2-m3)
      - "jina_m0":       Jina Multimodal Reranker (jinaai/jina-reranker-m0)
      - "event_splitter": Qwen2.5 Event Splitter (Qwen/Qwen2.5-1.5B-Instruct)
      - "hybrid_engine": HybridSearchEngine (BGE-Large-EN + BM25)
      - "marian_mt":     MarianMT Translation (Helsinki-NLP/opus-mt-vi-en)
    """

    def __new__(cls, *args, **kwargs):
        global _GLOBAL_REGISTRY_INSTANCE
        if _GLOBAL_REGISTRY_INSTANCE is None:
            with _GLOBAL_REGISTRY_LOCK:
                if _GLOBAL_REGISTRY_INSTANCE is None:
                    _GLOBAL_REGISTRY_INSTANCE = super().__new__(cls)
                    _GLOBAL_REGISTRY_INSTANCE._initialized = False
        return _GLOBAL_REGISTRY_INSTANCE

    def __init__(
        self,
        device: str = "cuda",
        text_embedding_batch_size: int = 16,
        image_embedding_batch_size: int = 32,
        unload_ttl_s: int = 0,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        self._device = device
        self._text_batch_size = text_embedding_batch_size
        self._image_batch_size = image_embedding_batch_size
        self._unload_ttl_s = max(0, unload_ttl_s)

        self._models: dict[str, Any] = {}
        self._custom_loaders: dict[str, Callable[[], Any]] = {}
        self._custom_unloaders: dict[str, Callable[[Any], None]] = {}
        self._last_used: dict[str, float] = {}
        self._load_lock = threading.Lock()
        self._async_locks: dict[str, asyncio.Lock] = {}
        self._initialized = True

    def _get_async_lock(self, key: str) -> asyncio.Lock:
        if key not in self._async_locks:
            self._async_locks[key] = asyncio.Lock()
        return self._async_locks[key]

    def register(
        self,
        key: str,
        loader: Callable[[], Any],
        unloader: Callable[[Any], None] | None = None,
    ) -> None:
        """Đăng ký dynamic model loader & unloader."""
        with self._load_lock:
            self._custom_loaders[key] = loader
            if unloader:
                self._custom_unloaders[key] = unloader

    # =========================================================================
    # Standard Model Getters (Sync & Async)
    # =========================================================================

    def get_text_embedder(self):
        return self.get("text")

    def get_image_embedder(self):
        return self.get("image")

    def get_asr_model(self):
        return self.get("asr")

    def get_viranker(self):
        return self.get("viranker")

    def get_jina_m0(self):
        return self.get("jina_m0")

    def get_event_splitter(self):
        return self.get("event_splitter")

    def get_hybrid_engine(self):
        return self.get("hybrid_engine")

    async def aget_text_embedder(self):
        return await self.aget("text")

    async def aget_image_embedder(self):
        return await self.aget("image")

    async def aget_asr_model(self):
        return await self.aget("asr")

    async def aget_viranker(self):
        return await self.aget("viranker")

    async def aget_jina_m0(self):
        return await self.aget("jina_m0")

    async def aget_event_splitter(self):
        return await self.aget("event_splitter")

    async def aget_hybrid_engine(self):
        return await self.aget("hybrid_engine")

    # =========================================================================
    # Generic Getters
    # =========================================================================

    def get(self, key: str) -> Any:
        return self._load(key)

    async def aget(self, key: str) -> Any:
        return await self._aload(key)

    # =========================================================================
    # Warmup / Unload Lifecycle
    # =========================================================================

    async def warmup_async(self, exclude_keys: tuple[str, ...] = ("jina_m0", "marian_mt")) -> None:
        """
        Preload đồng thời tất cả các model đã đăng ký (trừ các model trong exclude_keys).
        Mặc định loại trừ 'jina_m0' và 'marian_mt' để ưu tiên Google Translate và tiết kiệm VRAM.
        """
        keys = [k for k in self._loaders().keys() if k not in exclude_keys]
        tasks = [self.aget(k) for k in keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for k, res in zip(keys, results):
            if isinstance(res, Exception):
                logger.error(f"ModelRegistry: Lỗi khi warmup model '{k}': {res}")
        logger.info(f"ModelRegistry: warmup hoàn tất đồng thời cho {len(keys)} models/engines (loại trừ: {list(exclude_keys)}).")

        # Chạy inference mẫu (dummy forward pass) để CUDA compile kernels trước
        try:
            logger.info("🔥 Chạy warm-up forward pass để biên dịch CUDA kernels...")
            # 1. Warmup CLIP image embedder
            img_emb = self.get("image")
            if img_emb and hasattr(img_emb, "encode_text"):
                _ = img_emb.encode_text("warmup query")
            # 2. Warmup HybridSearchEngine
            hybrid = self.get("hybrid_engine")
            if hybrid and hasattr(hybrid, "warmup"):
                hybrid.warmup()
            # 3. Warmup ViRanker
            from serving.api.service.reranker.rerank_service import warmup_reranker
            await warmup_reranker(self)
            logger.info("⚡ Toàn bộ CUDA kernels và pipeline đã được compile & warmup sẵn sàng!")
        except Exception as e:
            logger.warning(f"Lỗi khi chạy warmup inference: {e}")

    def unload(self, key: str) -> None:
        """Giải phóng 1 model khỏi RAM/VRAM."""
        with self._load_lock:
            model = self._models.pop(key, None)
            if model is not None:
                if key in self._custom_unloaders:
                    try:
                        self._custom_unloaders[key](model)
                    except Exception as e:
                        logger.warning(f"Error in custom unloader for '{key}': {e}")
                del model
        self._last_used.pop(key, None)
        free_memory()
        logger.info(f"ModelRegistry: đã unload model '{key}' (RAM đã giải phóng).")

    def unload_idle(self) -> list[str]:
        """Unload các model không được dùng trong MODEL_UNLOAD_TTL_S giây."""
        if self._unload_ttl_s <= 0:
            return []
        now = time.time()
        to_unload = [
            key
            for key in list(self._models)
            if now - self._last_used.get(key, now) > self._unload_ttl_s
        ]
        for key in to_unload:
            self.unload(key)
        return to_unload

    def unload_all(self) -> None:
        """Giải phóng toàn bộ model (gọi lúc shutdown server)."""
        for key in list(self._models):
            self.unload(key)

    # =========================================================================
    # Internal Loading Logic
    # =========================================================================

    def _load(self, key: str) -> Any:
        with self._load_lock:
            model = self._models.get(key)
            if model is None:
                loaders = self._loaders()
                if key not in loaders:
                    raise KeyError(f"No model loader registered for key '{key}'")
                loader = loaders[key]
                model = loader()
                self._models[key] = model
            self._last_used[key] = time.time()
            return model

    async def _aload(self, key: str) -> Any:
        lock = self._get_async_lock(key)
        async with lock:
            return await asyncio.to_thread(self._load, key)

    def _loaders(self) -> dict[str, Callable[[], Any]]:
        base = {
            "text": self._load_text_embedder,
            "image": self._load_image_embedder,
            "asr": self._load_asr_model,
            "viranker": self._load_viranker,
            "jina_m0": self._load_jina_m0,
            "event_splitter": self._load_event_splitter,
            "hybrid_engine": self._load_hybrid_engine,
            "marian_mt": self._load_marian_mt,
        }
        base.update(self._custom_loaders)
        return base

    def _load_text_embedder(self):
        from shared.models.text_embedding_wrapper import TextEmbeddingWrapper

        logger.info(f"ModelRegistry: lazy-loading text embedder (device={self._device})...")
        return TextEmbeddingWrapper(device=self._device)

    def _load_image_embedder(self):
        from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper

        logger.info(f"ModelRegistry: lazy-loading image embedder (device={self._device})...")
        return ImageEmbeddingWrapper(device=self._device)

    def _load_asr_model(self):
        try:
            import torch
            from faster_whisper import WhisperModel

            device = "cuda" if torch.cuda.is_available() and self._device == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(f"ModelRegistry: lazy-loading faster-whisper (model='small', device={device}, compute_type={compute_type})...")
            return WhisperModel("small", device=device, compute_type=compute_type)
        except Exception as e_fw:
            logger.info(f"faster-whisper load failed ({e_fw}), trying fallback PhoWhisper...")
            try:
                from transformers import pipeline

                logger.info("ModelRegistry: lazy-loading ASR model (vinai/PhoWhisper-medium)...")
                return pipeline(
                    "automatic-speech-recognition",
                    model="vinai/PhoWhisper-medium",
                    device=-1,
                )
            except Exception as e:
                logger.warning(f"Không thể load ASR model: {e}. Voice queries sẽ dùng fallback.")
                return None

    def _load_viranker(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = "BAAI/bge-reranker-v2-m3"
        logger.info(f"ModelRegistry: lazy-loading ViRanker ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = self._device if torch.cuda.is_available() and self._device == "cuda" else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        model.eval()
        return {"model": model, "tokenizer": tokenizer, "device": device}

    def _load_jina_m0(self):
        import torch
        import transformers
        from transformers import AutoModel

        model_name = "jinaai/jina-reranker-m0"
        device = self._device if torch.cuda.is_available() and self._device == "cuda" else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info(f"ModelRegistry: lazy-loading Jina Reranker M0 ({model_name}) on {device}...")

        orig_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            model.to(device)
            model.eval()
        finally:
            transformers.logging.set_verbosity(orig_verbosity)
        return {"model": model, "device": device}

    def _load_event_splitter(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        device = self._device if torch.cuda.is_available() and self._device == "cuda" else "cpu"
        logger.info(f"ModelRegistry: lazy-loading Event Splitter ({model_id})...")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        return {"model": model, "tokenizer": tokenizer, "device": device}

    def _load_hybrid_engine(self):
        from pathlib import Path
        from serving.api.service.retrieval.hybrid_search import HybridSearchEngine
        from serving.api.core.config import get_settings

        data_dir = str(Path(get_settings().DATA_DIR) / "unified")
        logger.info(f"ModelRegistry: lazy-loading HybridSearchEngine from {data_dir}...")
        return HybridSearchEngine(data_dir=data_dir)

    def _load_marian_mt(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        model_name = "Helsinki-NLP/opus-mt-vi-en"
        device = self._device if torch.cuda.is_available() and self._device == "cuda" else "cpu"
        logger.info(f"ModelRegistry: lazy-loading MarianMT ({model_name}) on {device}...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        if device == "cuda":
            model = model.to(device).half()
        model.eval()
        return {"model": model, "tokenizer": tokenizer, "device": device}
