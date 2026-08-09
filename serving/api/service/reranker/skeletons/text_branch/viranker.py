"""
serving/api/service/reranker/skeletons/text_branch/viranker.py
==============================================================
Nhánh rerank văn bản sử dụng ViRanker (BGE-M3 / MMARCO-VI tuned) (Skeleton).

CONSTRAINTS:
  - Lazy load: Khởi tạo model lần đầu khi gọi hàm, cache lại. KHÔNG khởi tạo ở module level.
  - Batch inference: Xử lý toàn bộ candidates trong 1 forward pass. KHÔNG loop từng candidate.
"""

from __future__ import annotations

import logging
from typing import Any
from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)

# Cache cho model và tokenizer
_MODEL_CACHE: dict[str, Any] = {}

def _get_model():
    """Lazy load model và tokenizer BAAI/bge-reranker-v2-m3 vào cache."""
    if "model" not in _MODEL_CACHE or "tokenizer" not in _MODEL_CACHE:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "BAAI/bge-reranker-v2-m3"
            logger.info(f"Loading ViRanker text model: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
            model.to(device)
            model.eval()

            _MODEL_CACHE["tokenizer"] = tokenizer
            _MODEL_CACHE["model"] = model
            _MODEL_CACHE["device"] = device
        except Exception as e:
            logger.warning(f"Không thể load model ViRanker ({e}). Dùng CPU mock score.")
            _MODEL_CACHE["mock"] = True

    return _MODEL_CACHE


def score(query: str, candidates: list[CandidateResult]) -> list[float]:
    """
    Tính điểm tương đồng ngữ nghĩa giữa query và ocr_text + asr_text của từng candidate.

    Args:
        query: Query text tiếng Anh hoặc tiếng Việt đã chuẩn hóa.
        candidates: Danh sách CandidateResult cần rerank.

    Returns:
        List[float]: Danh sách điểm số (cùng kích thước và thứ tự với candidates).
    """
    if not candidates:
        return []

    cache = _get_model()

    if cache.get("mock", False):
        # Fallback cho unit test / offline: Trả về score gốc từ candidate
        return [float(c.score) for c in candidates]

    try:
        import torch
        model = cache["model"]
        tokenizer = cache["tokenizer"]
        device = cache["device"]

        # Chuẩn bị cặp (query, doc_text)
        pairs = []
        for c in candidates:
            desc = getattr(c, "description", "") or ""
            ocr = getattr(c, "ocr_text", "") or ""
            asr = getattr(c, "asr_text", "") or ""
            doc = f"{desc} {ocr} {asr}".strip()
            if not doc:
                doc = "[EMPTY]"
            pairs.append([query, doc])

        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            scores_tensor = torch.sigmoid(logits)
            scores = scores_tensor.cpu().numpy().tolist()

        if isinstance(scores, float):
            scores = [scores]

        return [float(s) for s in scores]
    except Exception as e:
        logger.error(f"Lỗi trong quá trình rerank ViRanker: {e}. Fallback score gốc.")
        return [float(c.score) for c in candidates]

