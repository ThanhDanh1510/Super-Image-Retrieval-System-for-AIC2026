"""
serving/api/core/dependencies.py
==================================
FastAPI Dependency Injection functions.

Cung cấp các model singleton và database client từ app.state
cho các routes endpoints.

Model được lazy-load qua ModelRegistry (chỉ tải khi endpoint đầu tiên
cần đến) để tiết kiệm RAM — xem serving/api/core/models.py.
"""

from __future__ import annotations

from fastapi import Request
from qdrant_client import AsyncQdrantClient
from serving.api.core.models import ModelRegistry

ModelT = object


def _get_registry(request: Request) -> ModelRegistry:
    """Lấy ModelRegistry singleton từ app.state."""
    return request.app.state.model_registry


def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    """Inject AsyncQdrantClient singleton."""
    return request.app.state.qdrant_client


async def get_text_embedder(request: Request) -> ModelT:
    """Inject TextEmbeddingWrapper — lazy-load lần đầu dùng."""
    return await _get_registry(request).aget_text_embedder()


async def get_image_embedder(request: Request) -> ModelT:
    """Inject ImageEmbeddingWrapper — lazy-load lần đầu dùng."""
    return await _get_registry(request).aget_image_embedder()


async def get_asr_model(request: Request) -> ModelT:
    """Inject ASR model (PhoWhisper) — lazy-load lần đầu dùng."""
    return await _get_registry(request).aget_asr_model()
