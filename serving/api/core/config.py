"""
serving/api/core/config.py
===========================
Cấu hình trung tâm cho FastAPI Serving Layer.

Đọc biến môi trường (.env) và YAML configs, cung cấp một Settings object
duy nhất cho toàn bộ ứng dụng.

CÁCH SỬ DỤNG:
    from serving.api.core.config import get_settings
    settings = get_settings()
    print(settings.QDRANT_URL)

IMPORT RULES:
    - File này KHÔNG import từ service/, repo/, hay controller/.
    - Chỉ import từ shared/ và standard library.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file nếu tồn tại (dùng cho local development)
load_dotenv()


class Settings(BaseSettings):
    """
    Tập trung tất cả cấu hình runtime cho serving layer.

    Ưu tiên đọc từ biến môi trường. Nếu không có, dùng giá trị mặc định.
    Trên production (lúc thi), set biến môi trường qua .env hoặc docker-compose.
    """

    # --- App ---
    APP_NAME: str = "AIC2026 Retrieval API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # --- Qdrant ---
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_ALIAS: str = "production"

    # --- Model Loading ---
    DEVICE: str = "cuda"   # "cuda" hoặc "cpu"
    TEXT_EMBEDDING_BATCH_SIZE: int = 16
    IMAGE_EMBEDDING_BATCH_SIZE: int = 32

    # --- RAM Saver ---
    # Warmup toàn bộ models ngay khi server khởi động (giúp request đầu tiên phản hồi siêu tốc ~0.3s)
    PRELOAD_MODELS: bool = True
    # Unload model không dùng sau N giây (0 = không tự unload). Càng nhỏ càng tiết kiệm RAM.
    MODEL_UNLOAD_TTL_S: int = 0
    # Chu kỳ (giây) background task kiểm tra model idle để unload.
    MODEL_IDLE_CHECK_S: int = 30

    # --- Static Files ---
    DATA_DIR: str = str(Path(__file__).resolve().parents[3] / "data")
    KEYFRAME_DIR: str = str(
        Path(__file__).resolve().parents[3] / "data" / "unified"
    )

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1   # Giữ 1 worker vì model load nặng (GPU singleton)

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["*"]   # Cho phép tất cả origin lúc thi

    model_config = {
        "env_prefix": "AIC_",   # Biến môi trường prefix: AIC_QDRANT_URL, AIC_DEVICE, ...
        "env_file": ".env",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton factory cho Settings.

    Returns:
        Settings instance (cached — chỉ khởi tạo 1 lần).
    """
    return Settings()
