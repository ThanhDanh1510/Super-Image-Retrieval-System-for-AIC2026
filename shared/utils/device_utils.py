"""
shared/utils/device_utils.py
==============================
Hàm hỗ trợ tự động nhận diện và chuẩn hóa PyTorch execution device:
  - "cuda": NVIDIA GPU (CUDA)
  - "mps": Apple Silicon GPU (Metal Performance Shaders)
  - "cpu": CPU fallback
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


def resolve_torch_device(requested_device: str | None = None) -> str:
    """
    Chuẩn hóa và kiểm tra tính khả thi của device được yêu cầu.

    Args:
        requested_device: Chuỗi tên device (ví dụ: "cuda", "mps", "gpu", "cpu").

    Returns:
        Tên device chuẩn: "cuda", "mps", hoặc "cpu".
    """
    if not requested_device:
        requested_device = "cpu"

    req = str(requested_device).strip().lower()

    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but not available. Falling back...")

    if req in ("mps", "gpu", "metal"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("MPS/GPU requested but not available. Falling back to CPU...")

    return "cpu"
