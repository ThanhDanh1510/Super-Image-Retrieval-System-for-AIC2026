"""
shared/utils/text_normalizer.py
=================================
Module dùng chung xử lý và chuẩn hóa văn bản tiếng Việt cho toàn bộ hệ thống.

Cung cấp các hàm:
  - normalize_nfc: Chuẩn hóa Unicode NFC (tránh vỡ font tiếng Việt)
  - clean_whitespace: Làm sạch khoảng trắng rác, xuống dòng thừa
  - sanitize_search_query: Tối ưu câu query tìm kiếm từ người dùng
"""

from __future__ import annotations

import re
import unicodedata


def clean_whitespace(text: str) -> str:
    """Loại bỏ khoảng trắng rác, ký tự xuống dòng thừa và gộp các khoảng trắng kép."""
    if not text:
        return ""
    # Thay thế các ký tự điều khiển / xuống dòng bằng khoảng trắng
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Gộp nhiều khoảng trắng liên tiếp thành 1
    return " ".join(text.split()).strip()


def normalize_nfc(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt sang dạng Unicode NFC chuẩn.

    Giúp xử lý các chuỗi ký tự bị mã hóa theo dạng NFD (phân rã dấu)
    về cùng một chuẩn thống nhất NFC trước khi lưu trữ hoặc tìm kiếm.
    """
    if not text:
        return ""
    # Chuyển đổi NFD/NFKC -> NFC
    nfc_text = unicodedata.normalize("NFC", str(text))
    return clean_whitespace(nfc_text)


def sanitize_search_query(text: str) -> str:
    """
    Làm sạch và chuẩn hóa câu truy vấn từ người dùng (User Query).
    
    Loại bỏ các ký tự đặc biệt không hỗ trợ tìm kiếm vector/BM25.
    """
    if not text:
        return ""
    text = normalize_nfc(text)
    # Loại bỏ các ký tự rác không in được
    text = "".join(ch for ch in text if ch.isprintable())
    return clean_whitespace(text)
