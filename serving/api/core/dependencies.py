"""
serving/api/core/dependencies.py
=================================
FastAPI Dependency Injection functions.

Cung cấp các model singleton và database client từ app.state
cho các routes endpoints.
"""

from __future__ import annotations

from fastapi import Request
from qdrant_client import AsyncQdrantClient
from shared.models.text_embedding_wrapper import TextEmbeddingWrapper
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper


def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    """Inject AsyncQdrantClient singleton."""
    return request.app.state.qdrant_client


def get_text_embedder(request: Request) -> TextEmbeddingWrapper:
    """Inject TextEmbeddingWrapper singleton."""
    return request.app.state.text_embedder


def get_image_embedder(request: Request) -> ImageEmbeddingWrapper:
    """Inject ImageEmbeddingWrapper singleton."""
    return request.app.state.image_embedder


def get_asr_model(request: Request):
    """Inject ASR model singleton (PhoWhisper)."""
    return request.app.state.asr_model
