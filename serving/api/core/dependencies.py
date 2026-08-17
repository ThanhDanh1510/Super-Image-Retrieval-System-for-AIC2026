"""
serving/api/core/dependencies.py
==================================
FastAPI Dependency Injection providers.
"""

from fastapi import Request
from qdrant_client import AsyncQdrantClient
from serving.api.core.models import ModelRegistry


def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    """Inject AsyncQdrantClient từ app.state."""
    return request.app.state.qdrant_client


def get_model_registry(request: Request) -> ModelRegistry:
    """Inject ModelRegistry từ app.state."""
    return request.app.state.model_registry


async def get_text_embedder(request: Request):
    """Inject TextEmbeddingWrapper singleton."""
    registry: ModelRegistry = request.app.state.model_registry
    return await registry.aget_text_embedder()


async def get_image_embedder(request: Request):
    """Inject ImageEmbeddingWrapper singleton."""
    registry: ModelRegistry = request.app.state.model_registry
    return await registry.aget_image_embedder()


async def get_asr_model(request: Request):
    """Inject ASR model singleton."""
    registry: ModelRegistry = request.app.state.model_registry
    return await registry.aget_asr_model()
