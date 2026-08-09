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

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from serving.api.core.config import get_settings
from serving.api.core.models import ModelRegistry
from serving.api.controller import routes_kis, routes_similar_image

logger = logging.getLogger(__name__)


async def _idle_unload_loop(registry: ModelRegistry, interval_s: float) -> None:
    """
    Background task: định kỳ unload các model không dùng trong TTL (nếu cấu hình).

    Args:
        registry: ModelRegistry singleton.
        interval_s: Chu kỳ kiểm tra (giây).
    """
    while True:
        await asyncio.sleep(interval_s)
        try:
            unloaded = registry.unload_idle()
            if unloaded:
                logger.info(f"Idle unload: đã giải phóng {len(unloaded)} model: {unloaded}")
        except Exception:
            logger.warning("Idle unload loop gặp lỗi, tiếp tục vòng lặp.", exc_info=True)


# =============================================================================
# Lifespan — startup / shutdown events
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý lifecycle của ứng dụng.

    STARTUP:
        1. Khởi tạo Qdrant async client
        2. Tạo ModelRegistry — model CHỈ được load khi endpoint đầu tiên cần
           (lazy loading, tiết kiệm RAM). Nếu PRELOAD_MODELS=true thì warmup.
        3. Nếu MODEL_UNLOAD_TTL_S > 0, chạy background task unload model idle.

    SHUTDOWN:
        1. Hủy background task idle unload
        2. Giải phóng toàn bộ model (unload_all) + GPU memory
        3. Đóng Qdrant client connection
    """
    # Tắt log kiểm tra cache mạng rườm rà của httpx/huggingface
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Qdrant: {settings.QDRANT_URL}")
    logger.info(f"Device: {settings.DEVICE} | PRELOAD_MODELS={settings.PRELOAD_MODELS} | MODEL_UNLOAD_TTL_S={settings.MODEL_UNLOAD_TTL_S}")

    # --- STARTUP ---
    from serving.api.repo.qdrant_repo import get_async_qdrant_client

    # 1. Khởi tạo Qdrant client
    app.state.qdrant_client = get_async_qdrant_client(
        settings.QDRANT_URL, settings.QDRANT_API_KEY
    )

    # 2. ModelRegistry — lazy loading, không tốn RAM lúc khởi động
    registry = ModelRegistry(
        device=settings.DEVICE,
        text_embedding_batch_size=settings.TEXT_EMBEDDING_BATCH_SIZE,
        image_embedding_batch_size=settings.IMAGE_EMBEDDING_BATCH_SIZE,
        unload_ttl_s=settings.MODEL_UNLOAD_TTL_S,
    )
    app.state.model_registry = registry

    if settings.PRELOAD_MODELS:
        logger.info("PRELOAD_MODELS=true — warmup toàn bộ model ngay lúc startup...")
        await registry.warmup_async()
    else:
        logger.info(
            "PRELOAD_MODELS=false — model sẽ được lazy-load khi endpoint đầu tiên cần đến (tiết kiệm RAM)."
        )

    # 3. Background task: unload model idle (RAM saver)
    idle_task: asyncio.Task | None = None
    if settings.MODEL_UNLOAD_TTL_S > 0:
        idle_task = asyncio.create_task(
            _idle_unload_loop(registry, interval_s=settings.MODEL_IDLE_CHECK_S)
        )
        logger.info(f"Idle unload loop started (TTL={settings.MODEL_UNLOAD_TTL_S}s).")

    yield

    # --- SHUTDOWN: Cleanup ---
    if idle_task is not None:
        idle_task.cancel()
        try:
            await idle_task
        except asyncio.CancelledError:
            pass
        logger.info("Stopped idle unload loop.")

    registry.unload_all()
    logger.info("Released all models (RAM/VRAM freed).")

    if hasattr(app.state, "qdrant_client"):
        await app.state.qdrant_client.close()
        logger.info("Closed Qdrant client connection.")

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

    # --- Mount Static Files (ảnh keyframe & UI) ---
    if os.path.isdir(settings.DATA_DIR):
        app.mount("/static", StaticFiles(directory=settings.DATA_DIR), name="static")
        logger.info(f"Static files mounted: {settings.DATA_DIR}")

    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")
        logger.info(f"Web UI mounted: {web_dir}")

    # --- Register Routers ---
    app.include_router(routes_kis.router)
    app.include_router(routes_similar_image.router)

    # --- Root Redirect to UI ---
    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/ui/index.html")

    # --- Health Check ---
    @app.get("/health", tags=["System"])
    async def health_check():
        """Endpoint kiểm tra server còn sống không."""
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


# Uvicorn sẽ import biến `app` này
app = create_app()
