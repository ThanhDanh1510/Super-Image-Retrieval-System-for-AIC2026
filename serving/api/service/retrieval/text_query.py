"""
serving/api/service/retrieval/text_query.py
============================================
Service xử lý query dạng text cho KIS, VQA, và TRAKE.

LUỒNG XỬ LÝ:
  1. Dịch query sang tiếng Anh (nếu cần) + enhance
  2. Tách thành N sub-events (chỉ cho TRAKE)
  3. Encode text → ocr_vector + asr_vector + sparse vectors
  4. Gọi qdrant_repo.search_by_text_vectors() → danh sách CandidateResult thô

IMPORT RULES:
  - Được gọi bởi: retrieval_pipeline.py
  - Gọi: serving/api/repo/qdrant_repo.py
  - Không gọi reranker hay temporal_aligner — đó là việc của pipeline orchestrator.

SKELETON: Implement các hàm có `raise NotImplementedError`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from serving.api.schema.retrieval import CandidateResult, TaskType
from serving.api.repo import qdrant_repo
from shared.models.text_embedding_wrapper import TextEmbeddingWrapper
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TextQueryInput:
    """Input chuẩn hóa cho text_query.process()."""
    raw_query: str
    """Query gốc từ người dùng (tiếng Việt hoặc tiếng Anh)."""

    task_type: TaskType
    """KIS / VQA / TRAKE — ảnh hưởng đến cách xử lý (TRAKE cần tách sub-events)."""

    top_k: int = 100
    """Số candidates trả về. Tối đa 100."""

    search_mode: str = "all"
    """Chế độ tìm kiếm: 'all', 'embedding', 'ocr', 'asr'."""


@dataclass
class TextQueryOutput:
    """Output trả về từ text_query.process()."""
    candidates: list[CandidateResult]
    """Danh sách candidates thô từ Qdrant, chưa qua reranker."""

    enhanced_query: str
    """Query sau khi đã dịch và enhance (dùng để log và hiển thị trên UI)."""

    sub_events: list[str] | None = None
    """
    Danh sách sub-events đã tách (chỉ có giá trị nếu task_type == TRAKE).
    None với KIS và VQA.
    """


# =============================================================================
# Core Functions
# =============================================================================

async def process(
    query_input: TextQueryInput,
    qdrant_client,
    text_embedder: TextEmbeddingWrapper,
    image_embedder: ImageEmbeddingWrapper,
) -> TextQueryOutput:
    """
    Xử lý toàn bộ luồng text query: song ngữ EN/VI → encode (text + image vectors) → retrieve.
    """
    import asyncio
    import time

    _t0 = time.perf_counter()
    en_query, vi_query = await _enhance_query(query_input.raw_query)
    _t1 = time.perf_counter()
    logger.info(f"[PROFILE] enhance_query: {(_t1 - _t0) * 1000:.2f}ms")

    sub_events: list[str] | None = None
    if query_input.task_type == TaskType.TRAKE:
        sub_events = _split_events(en_query)
        logger.info(f"TRAKE: tách được {len(sub_events)} sub-events từ query.")

    # Tối ưu hóa: Nếu chỉ tìm kiếm bằng OCR hoặc ASR, không cần chạy model Qwen2 để sinh embedding text
    # vì hệ thống sẽ sử dụng cơ chế so khớp từ khóa văn bản thuần túy (MatchText).
    ocr_vector = []
    asr_vector = []
    image_vector = []

    if query_input.search_mode in ["all", "embedding"]:
        # Chạy song song sinh embedding cho Image (CLIP) và Text (Qwen2)
        (ocr_vector, asr_vector), image_vector = await asyncio.gather(
            _encode_text(vi_query, text_embedder),
            _encode_text_to_image_vector(en_query, image_embedder),
        )
    else:
        # Đối với chế độ ocr hoặc asr, chỉ cần chạy CLIP (nếu cần), hoặc bỏ qua nếu không dùng
        # Ở đây ta khởi tạo rỗng để tiết kiệm tối đa thời gian xử lý và tài nguyên GPU
        pass

    _t2 = time.perf_counter()
    logger.info(f"[PROFILE] encode_text+image: {(_t2 - _t1) * 1000:.2f}ms")

    ocr_sparse, asr_sparse = _encode_sparse(vi_query)

    candidates = await qdrant_repo.search_by_text_vectors(
        client=qdrant_client,
        ocr_vector=ocr_vector,
        asr_vector=asr_vector,
        ocr_sparse_vector=ocr_sparse,
        asr_sparse_vector=asr_sparse,
        image_vector=image_vector,
        top_k=query_input.top_k,
        search_mode=query_input.search_mode,
        query_text=vi_query, # Truyền query_text tiếng Việt đã làm sạch xuống repo
    )
    _t3 = time.perf_counter()
    logger.info(f"[PROFILE] qdrant_search: {(_t3 - _t2) * 1000:.2f}ms, candidates: {len(candidates)}")

    return TextQueryOutput(
        candidates=candidates,
        enhanced_query=en_query,
        sub_events=sub_events,
    )


# =============================================================================
# Internal Helper Functions (implementations)
# =============================================================================

# deep_translator.GoogleTranslator đôi khi trả về trang lỗi HTML của Google
# (VD: "Error 500 (Server Error)!!1500.That's an error...") dưới dạng một
# "bản dịch" thành công — KHÔNG raise exception. Những marker này dùng để
# phát hiện và loại bỏ output rác, tránh rò rỉ vào enhanced_query/query_text.
_GOOGLE_ERROR_MARKERS = (
    "error 500", "error 400", "error 502", "error 503",
    "that's an error", "that’s an error",
    "please try again later", "that's all we know", "that’s all we know",
    "the server encountered an error",
)

# deep_translator không set timeout cho requests.get() -> có thể treo vô hạn.
# Giới hạn cứng mỗi lần dịch để request API không bao giờ bị chặn lâu.
TRANSLATE_TIMEOUT_S = 15.0


def _is_valid_translation(text: str, source: str) -> bool:
    """
    Kiểm tra output của GoogleTranslator có phải bản dịch hợp lệ hay không.

    deep_translator đôi khi trả về trang lỗi HTML của Google như một "bản dịch"
    thành công (không raise). Hàm này lọc các chuỗi không phải dịch thật:
      - Rỗng / chỉ khoảng trắng.
      - Chứa marker lỗi của Google ("Error 500", "That's an error", ...).
      - Rò rỉ HTML (`<tag>`) hoặc ký tự không in được.
      - Không chứa từ tiếng Latinh thật (error page thường chỉ là symbol noise).

    Args:
        text: Output mà translator trả về.
        source: Query nguồn gốc (dùng để chặn output quá lệch).

    Returns:
        True nếu text được coi là bản dịch hợp lệ, ngược lại False.
    """
    if not text or not text.strip():
        return False

    lowered = text.lower()
    if any(marker in lowered for marker in _GOOGLE_ERROR_MARKERS):
        return False
    # Rò rỉ HTML markup -> đánh dấu là lỗi
    if "<" in text and ">" in text:
        return False
    # Phải chứa ít nhất một từ Latinh (2 ký tự chữ trở lên)
    if not re.search(r"[A-Za-zÀ-ỹ]{2,}", text):
        return False
    # Bản dịch hợp lệ hiếm khi dài hơn một bội số hợp lý so với nguồn
    if len(text) > 5 * max(len(source), 1) + 500:
        return False
    return True


async def _translate(query: str, target: str) -> str | None:
    """
    Dịch `query` sang ngôn ngữ `target` bằng deep_translator.GoogleTranslator,
    kèm validate + retry để trang lỗi HTML của Google không bao giờ lọt vào kết quả.

    Args:
        query: Query nguồn (tiếng Việt hoặc tiếng Anh).
        target: Mã ngôn ngữ đích ("en" / "vi").

    Returns:
        Bản dịch hợp lệ, hoặc None nếu mọi lần dịch đều lỗi/không hợp lệ.
    """
    import asyncio
    from deep_translator import GoogleTranslator

    attempts = 0
    while attempts < 2:
        attempts += 1
        try:
            translator = GoogleTranslator(source="auto", target=target)
            # deep_translator KHÔNG set timeout cho requests.get() -> có thể treo vô hạn.
            # Bọc thêm timeout cứng để không bao giờ chặn request > TRANSLATE_TIMEOUT_S.
            out = await asyncio.wait_for(
                asyncio.to_thread(translator.translate, query),
                timeout=TRANSLATE_TIMEOUT_S,
            )
        except Exception as e:
            logger.warning(f"GoogleTranslator (attempt {attempts} -> {target}) failed: {e}")
            out = None

        if out and _is_valid_translation(out, query):
            return out

        logger.warning(
            f"GoogleTranslator (attempt {attempts} -> {target}) returned non-translation "
            f"output (len={len(out or '')}): {str(out)[:120]!r}. Falling back."
        )

    return None


async def _enhance_query(raw_query: str) -> tuple[str, str]:
    """
    Dịch và chuẩn hóa query song ngữ (English cho CLIP, Tiếng Việt cho Qwen2/OCR/ASR).

    Nếu dịch thất bại hoặc Google trả về trang lỗi, hàm KHÔNG ghi đè query gốc:
    giữ nguyên query đã làm sạch để không làm hỏng enhanced_query/query_text.

    Args:
        raw_query: Query gốc từ người dùng.

    Returns:
        Tuple (en_query, vi_query).
    """
    from shared.utils.text_normalizer import sanitize_search_query
    cleaned_query = sanitize_search_query(raw_query)
    if not cleaned_query:
        return "", ""

    has_vietnamese_accent = bool(
        re.search(
            r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]',
            cleaned_query,
            re.IGNORECASE,
        )
    )

    en_query = cleaned_query
    vi_query = cleaned_query

    if has_vietnamese_accent:
        # Query gốc là tiếng Việt -> Dịch sang tiếng Anh cho CLIP
        # Với cụm từ ngắn (<= 6 từ), dùng Title Case khi dịch để bảo toàn danh từ riêng (Đào Sơn Tây -> Dao Son Tay)
        words = cleaned_query.split()
        query_to_translate = cleaned_query.title() if len(words) <= 6 else cleaned_query

        import unicodedata
        nfkd = unicodedata.normalize('NFKD', cleaned_query)
        unaccented = ''.join([c for c in nfkd if not unicodedata.combining(c)]).replace('đ', 'd').replace('Đ', 'D')
        title_cased = unaccented.title()

        translated_en = await _translate(query_to_translate, "en")
        if not translated_en or not _is_valid_translation(translated_en, query_to_translate):
            translated_en = await _translate(cleaned_query, "en")

        if translated_en and _is_valid_translation(translated_en, cleaned_query):
            en_clean = sanitize_search_query(translated_en)
            if en_clean.lower() != unaccented.lower():
                en_query = en_clean
            else:
                en_query = f"{title_cased} {unaccented}"
        else:
            en_query = f"{title_cased} {unaccented}"
    else:
        # Query gốc là tiếng Anh -> Dịch sang tiếng Việt cho Qwen2/OCR/ASR
        translated_vi = await _translate(cleaned_query, "vi")
        if translated_vi and _is_valid_translation(translated_vi, cleaned_query):
            vi_query = sanitize_search_query(translated_vi)

    logger.info(f"Enhanced queries — EN (CLIP): '{en_query}' | VI (Qwen2): '{vi_query}'")
    return en_query, vi_query


def _split_events(query: str) -> list[str]:
    """
    Tách TRAKE query thành N sub-event descriptions.
    Hybrid approach: Pattern-based (nhanh) → LLM fallback (chính xác).

    Args:
        query: Query tiếng Anh đã enhanced (hoặc tiếng Việt).

    Returns:
        List các sub-event strings. Mỗi string mô tả 1 khoảnh khắc.

    Raises:
        ValueError: Nếu không thể tách được ít nhất 2 sub-events từ query.
    """
    import re

    # --- Strategy 1: Pattern-based (nhanh, ~0s) ---

    # 1a. Explicit labels: E1:, E2:, 1., 2., - , *
    # Hỗ trợ cả single-line (cách nhau bởi space) và multi-line
    explicit_pattern = r"(?:E?\d+[\.:\)]\s*)(.+?)(?=(?:E?\d+[\.:\)]\s*)|\Z)"
    matches = re.findall(explicit_pattern, query, flags=re.DOTALL)
    if len(matches) >= 2:
        return [m.strip(" .,;-") for m in matches if m.strip(" .,;-")]

    # 1b. Bullet points: -, *, •
    bullet_pattern = r"(?:^|\n)\s*[\-\*\•]\s*(.+?)(?=(?:\n\s*[\-\*\•]\s*)|\Z)"
    matches = re.findall(bullet_pattern, query, flags=re.MULTILINE | re.DOTALL)
    if len(matches) >= 2:
        return [m.strip(" .,;-") for m in matches if m.strip(" .,;-")]

    # 1b. Newline split (có số dòng >= 2)
    if "\n" in query:
        lines = [line.strip(" .,;-•*") for line in query.split("\n") if line.strip(" .,;-•*")]
        if len(lines) >= 2:
            return lines

    # 1c. Temporal markers / connectors
    connectors_pattern = (
        r"\b(?:bắt đầu(?: bằng| với)?|tiếp theo|sau đó|rồi|cuối cùng|"
        r"trước tiếp|cuối cùng|finally|then|after(?: that|wards)?|followed by|next|subsequently|later|"
        r"và|and)\b"
    )
    segments = re.split(connectors_pattern, query, flags=re.IGNORECASE)
    sub_events = [s.strip(" .,;-") for s in segments if s and len(s.strip(" .,;-")) > 5]

    if len(sub_events) >= 2:
        return sub_events

    # 1d. Phân tách theo dấu chấm/phẩy (events dài > 5 từ)
    punct_segments = re.split(r"[.;,]", query)
    sub_events = [s.strip(" .,;-") for s in punct_segments if s and len(s.strip(" .,;-")) > 5]
    if len(sub_events) >= 2:
        return sub_events

    # --- Strategy 2: LLM-based fallback (~1-2s) ---
    llm_events = _split_events_llm(query)
    if len(llm_events) >= 2:
        return llm_events

    raise ValueError(f"Không thể tách query thành ít nhất 2 sub-events: '{query}'")


def _split_events_llm(query: str) -> list[str]:
    """
    Dùng Qwen2.5-1.5B để tách query thành events (fallback khi pattern fail).

    Args:
        query: Query text.

    Returns:
        List các sub-event strings. Trả về [] nếu LLM fail.
    """
    try:
        from serving.api.core.models import ModelRegistry

        # Standalone loader nếu chưa nạp
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        prompt = f"""Tách câu sau thành các sự kiện riêng biệt, mỗi sự kiện trên 1 dòng. Chỉ liệt kê sự kiện, không giải thích.

Query: "{query}"

Sự kiện:"""

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Parse response → events
        events = []
        for line in response.strip().split("\n"):
            line = line.strip(" .,;-•*1234567890.)")
            if line and len(line) > 5:
                events.append(line)

        return events
    except Exception as e:
        logger.warning(f"LLM event splitting failed: {e}")
        return []


async def _encode_text(
    text: str,
    embedder: TextEmbeddingWrapper,
) -> tuple[list[float], list[float]]:
    """
    Encode text thành 2 dense vectors: ocr_vector và asr_vector (dim=1536).

    Args:
        text: Text đã enhanced.
        embedder: TextEmbeddingWrapper singleton (gte-Qwen2-1.5B).

    Returns:
        Tuple (ocr_vector, asr_vector) list[float], len=1536.
    """
    import asyncio
    vec = await asyncio.to_thread(embedder.encode, text)
    if hasattr(vec, "tolist"):
        vec_list = vec.tolist()
    else:
        vec_list = list(vec)

    return vec_list, vec_list


def _encode_sparse(text: str) -> tuple[object | None, object | None]:
    """
    Encode text thành sparse vectors (BM42) cho keyword matching.
    Phase 1: Trả về (None, None).
    """
    return None, None


async def _encode_text_to_image_vector(
    text: str,
    embedder: ImageEmbeddingWrapper,
) -> list[float]:
    """
    Encode text query thành image dense vector (Text-to-Image) dùng CLIP text encoder (dim=1024).

    Args:
        text: Text đã enhanced.
        embedder: ImageEmbeddingWrapper singleton.

    Returns:
        list[float], len=1024, L2-normalized.
    """
    import asyncio
    vec = await asyncio.to_thread(embedder.encode_text, text)
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)

