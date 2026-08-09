"""
serving/api/core/models.py
============================
ModelRegistry — quản lý vòng đời các model nặng (RAM saver).

Thay vì load TẤT CẢ model ngay khi server khởi động (tốn hàng GB RAM),
registry này:
  - Lazy load: model chỉ được khởi tạo khi endpoint đầu tiên thực sự cần.
  - Idle unload: nếu MODEL_UNLOAD_TTL_S > 0, model không được dùng trong
    khoảng thời gian đó sẽ bị giải phóng khỏi RAM (gc + cuda.empty_cache).
  - unload_all(): gọi khi shutdown server để giải phóng toàn bộ.

Thread-safe: load một model nhiều lần đồng thời (nhiều request) chỉ
khởi tạo đúng 1 lần.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _free_memory() -> None:
    """Thu hồi RAM/VRAM của các model đã bị unload."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class ModelRegistry:
    """
    Registry lazy-load + unload cho các model nặng của serving layer.

    Keys:
      - "text":  TextEmbeddingWrapper (gte-Qwen2-1.5B-instruct, dim=1536)
      - "image": ImageEmbeddingWrapper (CLIP ViT-L/14, dim=1024)
      - "asr":   PhoWhisper ASR pipeline (vinai/PhoWhisper-medium)
    """

    def __init__(
        self,
        device: str = "cuda",
        text_embedding_batch_size: int = 16,
        image_embedding_batch_size: int = 32,
        unload_ttl_s: int = 0,
    ) -> None:
        self._device = device
        self._text_batch_size = text_embedding_batch_size
        self._image_batch_size = image_embedding_batch_size
        self._unload_ttl_s = max(0, unload_ttl_s)

        self._models: dict[str, Any] = {}
        self._last_used: dict[str, float] = {}
        self._load_lock = threading.Lock()
        self._async_locks: dict[str, asyncio.Lock] = {
            "text": asyncio.Lock(),
            "image": asyncio.Lock(),
            "asr": asyncio.Lock(),
        }

    # =========================================================================
    # Sync getters — dùng cho warmup / shutdown / background tasks
    # =========================================================================

    def get_text_embedder(self):
        """Lazy-load và trả về TextEmbeddingWrapper (mock nếu load lỗi)."""
        return self._load("text")

    def get_image_embedder(self):
        """Lazy-load và trả về ImageEmbeddingWrapper (mock nếu load lỗi)."""
        return self._load("image")

    def get_asr_model(self):
        """Lazy-load và trả về ASR pipeline, None nếu load lỗi."""
        return self._load("asr")

    # =========================================================================
    # Async getters — dùng cho FastAPI dependencies (không block event loop)
    # =========================================================================

    async def aget_text_embedder(self):
        return await self._aload("text")

    async def aget_image_embedder(self):
        return await self._aload("image")

    async def aget_asr_model(self):
        return await self._aload("asr")

    # =========================================================================
    # Warmup / Unload
    # =========================================================================

    async def warmup_async(self) -> None:
        """Preload toàn bộ model (dùng khi PRELOAD_MODELS=true)."""
        from serving.api.service.reranker.skeletons.text_branch import viranker
        from serving.api.service.reranker.skeletons.crossmodal_branch import jina_reranker_m0
        from serving.api.service.retrieval.hybrid_search import get_hybrid_search_engine

        await asyncio.gather(
            self.aget_text_embedder(),
            self.aget_image_embedder(),
            self.aget_asr_model(),
            asyncio.to_thread(viranker._get_model),
            asyncio.to_thread(jina_reranker_m0._get_model),
            asyncio.to_thread(get_hybrid_search_engine),
        )
        logger.info("ModelRegistry: warmup hoàn tất (toàn bộ AI Model + Hybrid Engine đã nạp sẵn vào GPU/RAM).")

    def unload(self, key: str) -> None:
        """Giải phóng 1 model khỏi RAM/VRAM."""
        with self._load_lock:
            model = self._models.pop(key, None)
            del model
        self._last_used.pop(key, None)
        _free_memory()
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
    # Internal
    # =========================================================================

    def _load(self, key: str) -> Any:
        with self._load_lock:
            model = self._models.get(key)
            if model is None:
                loader = self._loaders()[key]
                model = loader()
                self._models[key] = model
            self._last_used[key] = time.time()
            return model

    async def _aload(self, key: str) -> Any:
        async with self._async_locks[key]:
            return await asyncio.to_thread(self._load, key)

    def _loaders(self) -> dict[str, Callable[[], Any]]:
        return {
            "text": self._load_text_embedder,
            "image": self._load_image_embedder,
            "asr": self._load_asr_model,
        }

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
            from transformers import pipeline

            device_id = 0 if (self._device == "cuda" and torch.cuda.is_available()) else -1
            logger.info(
                f"ModelRegistry: lazy-loading ASR model (vinai/PhoWhisper-medium, device={device_id})..."
            )
            return pipeline(
                "automatic-speech-recognition",
                model="vinai/PhoWhisper-medium",
                device=device_id,
            )
        except Exception as e:
            logger.warning(
                f"Không thể load ASR model: {e}. Voice queries sẽ dùng transcript fallback."
            )
            return None
