"""
serving/api/service/retrieval/query_analyzer.py
===============================================
Module phân tích ý đồ tìm kiếm (Query Understanding & Intent Detection).

Xác định xem câu truy vấn đang tập trung vào:
1. OCR Text: Chữ trên biển hiệu, số xe, logo, khẩu hiệu, bảng tên...
2. ASR Speech: Lời nói, phát biểu, giới thiệu, âm thanh, bài hát...
3. Visual Scene: Hành động, màu sắc, góc quay, khung cảnh, trang phục...
4. QA Info Extraction: Hỏi số lượng, tên gọi, thực thể, nồng độ, trọng lượng...

Từ đó tự động sinh trọng số Weighted RRF động (Dynamic Fusion Weights)
để ưu tiên nhánh tìm kiếm phù hợp nhất thay vì dùng trọng số cố định.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryIntent:
    """Kết quả phân tích ý đồ truy vấn."""
    raw_query: str
    has_ocr_intent: bool
    has_asr_intent: bool
    has_visual_intent: bool
    has_qa_intent: bool = False
    extracted_ocr_terms: list[str] = field(default_factory=list)
    extracted_entities: list[str] = field(default_factory=list)
    custom_weights: dict[str, float] = field(default_factory=dict)
    primary_intent: str = "balanced"  # "ocr" | "asr" | "visual" | "qa" | "balanced"


class QueryAnalyzer:
    """Bộ phân tích cú pháp và ngữ nghĩa ý đồ câu hỏi."""

    # Từ khóa chỉ ý đồ tìm chữ / văn bản hiển thị
    OCR_CUES = {
        "chữ", "biển hiệu", "bảng hiệu", "bảng tên", "biển báo",
        "số xe", "biển số", "áo số", "dòng chữ", "logo", "banner",
        "poster", "khẩu hiệu", "slogan", "tiêu đề", "thương hiệu",
        "ghi chữ", "in chữ", "viết chữ", "bảng quảng cáo", "tên đường",
        "biển chỉ dẫn", "bảng giá", "nhãn", "tem"
    }

    # Từ khóa chỉ ý đồ tìm lời thoại / âm thanh phát biểu (Chỉ dùng các từ chỉ rõ giọng nói/phát biểu)
    ASR_CUES = {
        "phát biểu", "lời bài hát", "hát bài", "thuyết minh viên",
        "phát thanh viên", "nói rằng", "tuyên bố", "tiếng nói", "giọng nói",
        "lời thoại", "trả lời phỏng vấn"
    }

    # Từ khóa thuần thị giác, góc máy, đối tượng và hành động thực tế
    VISUAL_CUES = {
        "cận cảnh", "toàn cảnh", "từ trên cao", "flycam", "ảnh cận",
        "mặc áo", "màu", "áo dài", "nón lá", "đang chạy", "đang đi",
        "trên bàn", "trong phòng", "bờ hồ", "bầu trời", "hoàng hôn",
        "vẫy tay", "khom người", "bước vào", "chỉ tay", "ngã xe", "cầm",
        "cảnh phim", "máy quay", "quay từ", "cú máy", "chuyển cảnh",
        "nguyên liệu", "món ăn", "hải sản", "thực phẩm", "nhiều màu sắc",
        "đồ ăn", "nấu ăn", "chế biến", "dĩa", "đĩa", "tô", "bát", "bếp"
    }

    # Từ khóa dạng hỏi đáp thông tin (QA)
    QA_CUES = {
        "bao nhiêu", "mấy", "ai", "tên gì", "loại gì", "thứ mấy",
        "ở đâu", "gì", "bao lâu", "gồm những", "như thế nào",
        "nặng bao nhiêu", "giá bao nhiêu", "chiếc nào", "con gì"
    }

    @classmethod
    def _load_adaptive_config(cls) -> dict:
        """Đọc cấu hình adaptive_fusion từ fusion_config.yaml."""
        try:
            from shared.utils.config_loader import load_fusion_config
            cfg = load_fusion_config()
            return cfg.get("adaptive_fusion", {})
        except Exception:
            return {}

    @classmethod
    def analyze(cls, raw_query: str, task_type: str | None = None) -> QueryIntent:
        """Phân tích raw_query và trả về QueryIntent kèm trọng số nhánh động."""
        if not raw_query or not raw_query.strip():
            return QueryIntent(
                raw_query="",
                has_ocr_intent=False,
                has_asr_intent=False,
                has_visual_intent=False,
                has_qa_intent=False,
                extracted_ocr_terms=[],
                extracted_entities=[],
                custom_weights={"image_dense": 2.5, "description_dense": 2.0, "ocr_dense": 0.5, "asr_dense": 0.3},
                primary_intent="balanced",
            )

        q_lower = raw_query.lower()
        adaptive_cfg = cls._load_adaptive_config()

        # 1. Trích xuất cụm từ trong ngoặc kép (Exact Text Intent)
        quoted_terms = re.findall(r'["\']([^"\']+)["\']', raw_query)
        extracted_ocr_terms = [t.strip() for t in quoted_terms if t.strip()]

        # 2. Trích xuất thực thể số / đơn vị đo lường / biển số
        entities: list[str] = list(extracted_ocr_terms)
        plates = re.findall(r'\b[0-9]{2}[A-Za-z0-9]{1,3}[-\s]?[0-9]{3,5}\b', raw_query)
        for p in plates:
            if p.strip() not in entities:
                entities.append(p.strip())

        units = re.findall(r'\b[0-9]+(?:\.[0-9]+)?\s*(?:g|kg|gram|ml|l|m|cm|mm|cái|chiếc|người|phần nghìn)\b|[0-9]+(?:\.[0-9]+)?\s*[%‰]', raw_query, re.IGNORECASE)
        for u in units:
            if u.strip() not in entities:
                entities.append(u.strip())

        has_ocr = len(extracted_ocr_terms) > 0 or len(plates) > 0 or any(cue in q_lower for cue in cls.OCR_CUES)
        has_asr = any(cue in q_lower for cue in cls.ASR_CUES)
        has_visual = any(cue in q_lower for cue in cls.VISUAL_CUES)
        has_qa = (task_type is not None and "qa" in str(task_type).lower()) or any(cue in q_lower for cue in cls.QA_CUES)

        # 3. Quyết định Primary Intent và Dynamic Weights
        if has_qa:
            primary = "qa"
            weights = adaptive_cfg.get("qa_intent", {
                "image_dense": 2.0,
                "description_dense": 2.8,
                "ocr_dense": 1.8,
                "asr_dense": 1.5,
                "ocr_sparse": 0.5,
                "asr_sparse": 0.5,
            })
        elif has_ocr and not has_asr:
            primary = "ocr"
            weights = adaptive_cfg.get("ocr_intent", {
                "image_dense": 1.5,
                "description_dense": 1.5,
                "ocr_dense": 3.5,
                "asr_dense": 0.1,
                "ocr_sparse": 0.8,
                "asr_sparse": 0.1,
            })
        elif has_asr and not has_visual and not has_ocr:
            primary = "asr"
            weights = adaptive_cfg.get("asr_intent", {
                "image_dense": 1.5,
                "description_dense": 1.5,
                "ocr_dense": 0.2,
                "asr_dense": 3.2,
                "ocr_sparse": 0.1,
                "asr_sparse": 0.8,
            })
        elif has_visual:
            primary = "visual"
            weights = adaptive_cfg.get("visual_intent", {
                "image_dense": 3.0,
                "description_dense": 3.0,
                "ocr_dense": 0.2,
                "asr_dense": 0.1,
                "ocr_sparse": 0.1,
                "asr_sparse": 0.1,
            })
        else:
            primary = "balanced"
            weights = adaptive_cfg.get("balanced_intent", {
                "image_dense": 2.5,
                "description_dense": 2.5,
                "ocr_dense": 0.5,
                "asr_dense": 0.3,
                "ocr_sparse": 0.2,
                "asr_sparse": 0.2,
            })

        return QueryIntent(
            raw_query=raw_query,
            has_ocr_intent=has_ocr,
            has_asr_intent=has_asr,
            has_visual_intent=has_visual,
            has_qa_intent=has_qa,
            extracted_ocr_terms=extracted_ocr_terms,
            extracted_entities=entities,
            custom_weights=dict(weights),
            primary_intent=primary,
        )

