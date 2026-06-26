"""
serving/api/main.py
====================
Điểm khởi tạo ứng dụng FastAPI — entry point duy nhất.

TRÁCH NHIỆM:
    - Tạo FastAPI app instance
    - Mount thư mục static files (ảnh keyframe)
    - Đăng ký tất cả routers (controller/routes_*.py)
    - Cấu hình CORS middleware
    - Định nghĩa lifespan events (startup/shutdown)

CHẠY:
    uvicorn serving.api.main:app --host 0.0.0.0 --port 8000 --reload

IMPORT RULES:
    - Import từ controller/ để đăng ký routes.
    - Import từ core/config.py để lấy settings.
    - KHÔNG import từ service/ hay repo/ trực tiếp.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from serving.api.core.config import get_settings
from serving.api.controller import routes_kis, routes_similar_image

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan — startup / shutdown events
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý lifecycle của ứng dụng.

    STARTUP:
        1. Load tất cả AI models vào GPU (text_embedder, image_embedder, asr_model)
        2. Khởi tạo Qdrant async client
        3. Lưu vào app.state để inject vào endpoints qua dependencies

    SHUTDOWN:
        1. Đóng Qdrant client connection
        2. Giải phóng GPU memory (del models, torch.cuda.empty_cache)

    SKELETON: Implement startup và shutdown logic.
    """
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Qdrant: {settings.QDRANT_URL}")
    logger.info(f"Device: {settings.DEVICE}")

    # --- STARTUP: Load models và kết nối Qdrant ---
    # TODO: Implement
    #   app.state.qdrant_client = await get_async_qdrant_client(
    #       settings.QDRANT_URL, settings.QDRANT_API_KEY
    #   )
    #   app.state.text_embedder = TextEmbeddingWrapper(device=settings.DEVICE)
    #   app.state.image_embedder = ImageEmbeddingWrapper(device=settings.DEVICE)
    #   app.state.asr_model = load_asr_model(device=settings.DEVICE)  # hoặc None
    logger.warning("SKELETON: Models chưa được load — implement lifespan startup.")

    yield

    # --- SHUTDOWN: Cleanup ---
    # TODO: Implement
    #   await app.state.qdrant_client.close()
    #   del app.state.text_embedder
    #   del app.state.image_embedder
    #   torch.cuda.empty_cache()
    logger.info("Shutdown complete.")


# =============================================================================
# App Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Factory function tạo FastAPI app instance.

    Returns:
        FastAPI app đã cấu hình đầy đủ routes, middleware, và static files.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "API phục vụ truy vấn video keyframe cho AI Challenge HCMC 2026. "
            "Hỗ trợ KIS (Known-Item Search), VQA (Visual QA), "
            "TRAKE (Temporal Retrieval), và Image-to-Image Search."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- CORS Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Mount Static Files (ảnh keyframe) ---
    # URL: /static/unified_keyframes/{video_id}/{frame_idx:06d}.jpg
    if os.path.isdir(settings.DATA_DIR):
        app.mount("/static", StaticFiles(directory=settings.DATA_DIR), name="static")
        logger.info(f"Static files mounted: {settings.DATA_DIR}")
    else:
        logger.warning(
            f"DATA_DIR không tồn tại: {settings.DATA_DIR}. "
            f"Static files sẽ không khả dụng. Tạo thư mục data/ và thêm ảnh keyframe."
        )

    # --- Register Routers ---
    app.include_router(routes_kis.router)
    app.include_router(routes_similar_image.router)

    # --- Health Check ---
    @app.get("/health", tags=["System"])
    async def health_check():
        """Endpoint kiểm tra server còn sống không."""
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


# Uvicorn sẽ import biến `app` này
app = create_app()
