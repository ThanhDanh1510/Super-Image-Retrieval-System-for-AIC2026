# =============================================================================
# Dockerfile: AIC2026 Super Image Retrieval Serving API
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIC_DEVICE=cpu \
    QDRANT_URL=http://qdrant:6333 \
    UV_SYSTEM_PYTHON=1

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và xử lý ảnh
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt uv package manager
RUN pip install --no-cache-dir uv

# Sao chép file cấu hình và cài đặt dependencies
COPY pyproject.toml .
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install -e ".[test,macos-server]"

# Sao chép toàn bộ mã nguồn
COPY . .

# Mở cổng API Server
EXPOSE 8000

# Khởi động FastAPI server
CMD ["python", "start_api.py"]
