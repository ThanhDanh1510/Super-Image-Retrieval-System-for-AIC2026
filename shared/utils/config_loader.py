"""
shared/utils/config_loader.py
==============================
Tiện ích load YAML configs từ thư mục configs/ ở root dự án.

Mọi module (ingestion, serving, indexing) đều dùng file này
để đọc cấu hình — KHÔNG tự load YAML riêng lẻ.

CÁCH SỬ DỤNG:
    from shared.utils.config_loader import load_model_versions, load_fusion_config
    models = load_model_versions()
    fusion = load_fusion_config()
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml

# Root dự án — từ shared/utils/config_loader.py đi lên 3 cấp
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


def _load_yaml(filename: str) -> dict[str, Any]:
    """
    Load một file YAML từ thư mục configs/.

    Args:
        filename: Tên file YAML (ví dụ: 'model_versions.yaml').

    Returns:
        Dict chứa nội dung YAML đã parse.

    Raises:
        FileNotFoundError: Nếu file không tồn tại trong configs/.
        yaml.YAMLError: Nếu file YAML không hợp lệ.
    """
    filepath = _CONFIGS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Config file không tồn tại: {filepath}. "
            f"Kiểm tra lại thư mục configs/ tại {_CONFIGS_DIR}"
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_model_versions() -> dict[str, Any]:
    """
    Load configs/model_versions.yaml — thông tin phiên bản tất cả model AI.

    Returns:
        Dict chứa cấu hình model. Ví dụ:
        {
            'schema_version': '1.0.0',
            'image_embedding': {'model': 'microsoft/BEiT-large-...', 'vector_dim': 1024, ...},
            'text_embedding': {'model': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', ...},
            ...
        }

    Note:
        Kết quả được cache — chỉ đọc file 1 lần duy nhất trong suốt process.
    """
    return _load_yaml("model_versions.yaml")


@lru_cache(maxsize=1)
def load_fusion_config() -> dict[str, Any]:
    """
    Load configs/fusion_config.yaml — tham số RRF fusion và query routing.

    Returns:
        Dict chứa cấu hình fusion. Ví dụ:
        {
            'rrf': {'k': 60},
            'prefetch': {'image_dense': 200, ...},
            'query_routing': {'text_query': {'enabled_branches': [...]}, ...},
            ...
        }
    """
    return _load_yaml("fusion_config.yaml")


@lru_cache(maxsize=1)
def load_shard_assignment() -> dict[str, Any]:
    """
    Load configs/shard_assignment.yaml — phân công video cho từng thành viên.

    Returns:
        Dict chứa thông tin phân công shard.
    """
    return _load_yaml("shard_assignment.yaml")


def get_project_root() -> Path:
    """Trả về đường dẫn tuyệt đối tới root dự án."""
    return _PROJECT_ROOT


def get_data_dir() -> Path:
    """Trả về đường dẫn tới thư mục data/ (chứa unified_keyframes/)."""
    return _PROJECT_ROOT / "data"
