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

import asyncio
import logging
import re
from dataclasses import dataclass

from serving.api.schema.retrieval import CandidateResult, TaskType
from serving.api.repo import qdrant_repo
from shared.models.text_embedding_wrapper import TextEmbeddingWrapper
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper
from serving.api.service.retrieval.hybrid_search import (
    search_description_vector,
    search_asr_candidates,
    search_ocr_candidates,
)

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

    context: str | None = None
    """Ngữ cảnh tổng quan của video (Preamble / Global video context) để làm giàu sub-events."""

    pre_enhanced_en: str | None = None
    """Bản dịch EN có sẵn từ batch translation (bỏ qua enhance đơn lẻ)."""

    pre_enhanced_vi: str | None = None
    """Bản dịch VI có sẵn từ batch translation."""


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

    stage_timings: dict[str, float] | None = None
    """Chi tiết thời gian các chặng xử lý (ms)."""


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
    Tối ưu hóa: Đồ thị bất đồng bộ song song (Async DAG) giúp triệt tiêu 100% độ trễ của nhánh Intent & Sparse.
    """
    import asyncio
    import time
    from shared.utils.text_normalizer import sanitize_search_query
    from serving.api.service.retrieval.query_analyzer import QueryAnalyzer

    vi_initial = sanitize_search_query(query_input.raw_query)

    _t0 = time.perf_counter()

    # Giai đoạn 1: Chạy song song Enhance Query (EN) + Intent Analysis + Sparse Tokenizer
    async def _async_intent():
        t_type = query_input.task_type.value if hasattr(query_input.task_type, "value") else str(query_input.task_type)
        return QueryAnalyzer.analyze(query_input.raw_query, task_type=t_type)

    async def _async_sparse():
        return _encode_sparse(vi_initial)

    if query_input.pre_enhanced_en and query_input.pre_enhanced_vi:
        en_query, vi_query = query_input.pre_enhanced_en, query_input.pre_enhanced_vi
        intent, (ocr_sparse, asr_sparse) = await asyncio.gather(
            _async_intent(),
            _async_sparse(),
        )
    else:
        (en_query, vi_query), intent, (ocr_sparse, asr_sparse) = await asyncio.gather(
            _enhance_query(query_input.raw_query, context=query_input.context),
            _async_intent(),
            _async_sparse(),
        )
    _t1 = time.perf_counter()
    logger.info(f"[PROFILE] parallel_enhance+intent+sparse: {(_t1 - _t0) * 1000:.2f}ms")
    logger.info(
        f"🎯 [INTENT] Primary: {intent.primary_intent} | "
        f"OCR: {intent.has_ocr_intent} | ASR: {intent.has_asr_intent} | Visual: {intent.has_visual_intent}"
    )

    sub_events: list[str] | None = None
    if query_input.task_type == TaskType.TRAKE:
        try:
            sub_events = _split_events(query_input.raw_query)
        except Exception:
            try:
                sub_events = _split_events(vi_query)
            except Exception:
                sub_events = _split_events(en_query)
        logger.info(f"TRAKE: tách được {len(sub_events)} sub-events từ query (giữ nguyên tiếng Việt gốc).")

    # Bỏ sinh dense vector cho OCR và ASR vì đã có nhánh so khớp từ khóa văn bản thuần túy (MatchText).
    # Kích hoạt tìm kiếm dense vector cho Image Description (BGE-Large-EN v1.5).
    ocr_vector = []
    asr_vector = []
    image_vector = []
    description_candidates = []
    asr_candidates = []
    ocr_candidates = []

    # Giai đoạn 2: Chạy song song CLIP Image Vector, BGE Description Search, và BM25 (chỉ khi có intent)
    if query_input.search_mode in ["all", "embedding"]:
        image_task = _encode_text_to_image_vector(en_query, image_embedder)
        desc_task = search_description_vector(en_query, top_k=max(query_input.top_k, 200), query_vi=vi_query)
        asr_task = search_asr_candidates(vi_query, top_k=max(query_input.top_k, 100)) if intent.has_asr_intent else None
        ocr_task = search_ocr_candidates(vi_query, top_k=max(query_input.top_k, 100)) if intent.has_ocr_intent else None

        tasks = [image_task, desc_task]
        if asr_task is not None:
            tasks.append(asr_task)
        if ocr_task is not None:
            tasks.append(ocr_task)

        results = await asyncio.gather(*tasks)
        image_vector = results[0]
        description_candidates = results[1]
        cur_idx = 2
        if asr_task is not None:
            asr_candidates = results[cur_idx]
            cur_idx += 1
        if ocr_task is not None:
            ocr_candidates = results[cur_idx]
            cur_idx += 1
    elif query_input.search_mode == "description":
        description_candidates = await search_description_vector(en_query, top_k=max(query_input.top_k, 200), query_vi=vi_query)
    elif query_input.search_mode == "asr":
        asr_candidates = await search_asr_candidates(vi_query, top_k=query_input.top_k)
    elif query_input.search_mode == "ocr":
        ocr_candidates = await search_ocr_candidates(vi_query, top_k=query_input.top_k)

    # Giai đoạn 2.1: Multi-Scene Sub-Query Fusion cho KIS
    if query_input.task_type == TaskType.KIS and (query_input.search_mode in ["all", "embedding", "description"]):
        try:
            from serving.api.service.retrieval import sub_event_parser
            _, kis_scenes = sub_event_parser.extract_preamble_and_sub_events(query_input.raw_query)
            if kis_scenes and len(kis_scenes) >= 2:
                sub_enhanced = await batch_enhance_queries(kis_scenes)
                sub_ens = [se[0] for se in sub_enhanced if se[0]]
                sub_vis = [se[1] for se in sub_enhanced if se[1]]
                if sub_ens:
                    from serving.api.service.retrieval.hybrid_search import search_description_batch_vector
                    sub_desc_results = await search_description_batch_vector(
                        queries_en=sub_ens,
                        top_k=max(50, query_input.top_k),
                        queries_vi=sub_vis,
                    )
                    seen_desc_ids = {c.scene_id for c in description_candidates}
                    for sub_list in sub_desc_results:
                        for sc_cand in sub_list:
                            if sc_cand.scene_id not in seen_desc_ids:
                                description_candidates.append(sc_cand)
                                seen_desc_ids.add(sc_cand.scene_id)
        except Exception as e:
            logger.debug(f"KIS multi-scene fusion error: {e}")

    _t2 = time.perf_counter()
    logger.info(f"[PROFILE] encode_image+desc_search+asr+ocr: {(_t2 - _t1) * 1000:.2f}ms")

    # Giai đoạn 3: Qdrant Vector & Text Fusion Search
    candidates = await qdrant_repo.search_by_text_vectors(
        client=qdrant_client,
        ocr_vector=ocr_vector,
        asr_vector=asr_vector,
        ocr_sparse_vector=ocr_sparse,
        asr_sparse_vector=asr_sparse,
        image_vector=image_vector,
        description_candidates=description_candidates,
        asr_candidates=asr_candidates,
        top_k=max(query_input.top_k, 100),
        search_mode=query_input.search_mode,
        query_text=vi_query,
        custom_weights=intent.custom_weights,
    )
    _t3 = time.perf_counter()
    logger.info(f"[PROFILE] qdrant_search: {(_t3 - _t2) * 1000:.2f}ms, candidates: {len(candidates)}")

    # Giai đoạn 3.1: Áp dụng Negative Keyword Penalization (nếu có điều kiện phủ định)
    try:
        from shared.utils.config_loader import load_fusion_config
        _kis_cfg = load_fusion_config().get("kis_accuracy", {})
    except Exception:
        _kis_cfg = {}

    if _kis_cfg.get("enable_negation_filter", True):
        neg_terms = extract_negative_keywords(query_input.raw_query)
        if neg_terms and candidates:
            penalty = float(_kis_cfg.get("negation_penalty_factor", 0.50))
            for c in candidates:
                doc_lower = f"{c.ocr_text} {c.asr_text} {c.description}".lower()
                if any(nt in doc_lower for nt in neg_terms):
                    c.score = round(float(c.score * penalty), 4)
            candidates.sort(key=lambda x: x.score, reverse=True)

    stage_timings = {
        "enhance_query_ms": round((_t1 - _t0) * 1000, 1),
        "encode_desc_ms": round((_t2 - _t1) * 1000, 1),
        "qdrant_search_ms": round((_t3 - _t2) * 1000, 1),
    }

    return TextQueryOutput(
        candidates=candidates,
        enhanced_query=en_query,
        sub_events=sub_events,
        stage_timings=stage_timings,
    )


def extract_negative_keywords(text: str) -> list[str]:
    """
    Trích xuất các từ khóa phủ định (VD: 'không có mũ bảo hiểm' -> ['mũ bảo hiểm'])
    để lọc và trừ điểm các candidates vi phạm.
    """
    if not text:
        return []
    neg_terms = []
    patterns = [
        r'\b(?:không có|không|chưa|trừ|ngoại trừ|không phải)\s+([^\.,?!;]+)',
        r'\b(?:without|no|not having|except)\s+([^\.,?!;]+)',
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            clean_m = m.strip().lower()
            words = clean_m.split()[:4]
            if words:
                term = " ".join(words)
                if len(term) >= 2:
                    neg_terms.append(term)
    return neg_terms


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
# Giới hạn mỗi lần dịch Google Translate an toàn lên 10.0s để không bao giờ bỏ lỡ bản dịch Google chuẩn xác.
TRANSLATE_TIMEOUT_S = 10.0



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
        bool: True nếu hợp lệ, False nếu là trang lỗi.
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


def _translate_with_google_sync(query: str, target: str = "en") -> str | None:
    """Dịch query bằng Google Translator qua deep_translator."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="auto", target=target)
        out = translator.translate(query)
        if out and _is_valid_translation(out, query):
            return out.strip()
    except Exception as e:
        logger.debug(f"GoogleTranslator sync error: {e}")
    return None


_LOCAL_LLM_CACHE = None


def _get_qwen_bundle():
    """Lấy cached Qwen2.5 bundle cho offline translation và event splitting."""
    global _LOCAL_LLM_CACHE
    if _LOCAL_LLM_CACHE is not None:
        return _LOCAL_LLM_CACHE
    try:
        from serving.api.core.models import ModelRegistry
        reg = ModelRegistry()
        bundle = reg.get("event_splitter")
        if bundle is not None and isinstance(bundle, dict) and "model" in bundle:
            _LOCAL_LLM_CACHE = bundle
            return _LOCAL_LLM_CACHE

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        _LOCAL_LLM_CACHE = {"model": model, "tokenizer": tokenizer, "device": device}
        return _LOCAL_LLM_CACHE
    except Exception as e:
        logger.warning(f"Failed to load local Qwen2.5 model for translation: {e}")
        return None


_CULTURAL_CATEGORY_RULES = [
    (
        ("áo dài", "nón lá", "bà ba", "khăn rằn", "khăn quàng", "csgt", "cảnh sát", "bộ đội", "chiến sĩ", "bác sĩ", "y tá", "blouse"),
        "- Costumes: 'áo dài'->'Vietnamese ao dai', 'nón lá'->'conical hat', 'áo bà ba'->'traditional blouse', 'CSGT'->'traffic police in tan uniform', 'bộ đội'->'soldier in green uniform'.",
    ),
    (
        ("lân", "rồng", "thanh trụ", "mai hoa thung", "kẻng", "chiêng", "dùi", "ông địa", "đua thuyền", "chùa", "đền", "miếu", "đình", "thắp hương", "bàn thờ", "tượng phật"),
        "- Festivals/Culture: 'lân/múa lân'->'lion dance', 'mai hoa thung'->'high stepping poles', 'kẻng đồng'->'bronze gong', 'dùi'->'drumstick', 'ông địa'->'Earth God mask', 'chùa/đền'->'Buddhist pagoda/temple'.",
    ),
    (
        ("phở", "tái nạm", "bánh mì", "bánh rán", "bánh cam", "bánh chưng", "bánh tét", "nem rán", "chả giò", "gỏi cuốn", "cơm tấm", "sườn bì", "bún chả", "bún bò", "cà phê sữa", "hàng rong", "măng tây", "măng cụt", "khổ qua", "mướp đắng", "chảo dầu", "chảo", "tô măng tây", "miếng măng tây"),
        "- Cuisine: 'phở/tái nạm'->'Vietnamese pho beef noodle soup', 'bánh mì'->'banh mi baguette', 'cơm tấm'->'broken rice plate', 'măng tây'->'green asparagus', 'chảo dầu'->'frying pan with hot oil', 'cà phê sữa đá'->'Vietnamese iced milk coffee'.",
    ),
    (
        ("bến thành", "hồ gươm", "tháp rùa", "thê húc", "nhà thờ đức bà", "độc lập", "ruộng bậc thang", "phố cổ", "nhà ống", "bùng binh", "vòng xoay", "cầu treo", "cầu khỉ"),
        "- Landmarks: 'chợ Bến Thành'->'Ben Thanh market', 'Hồ Gươm/Tháp Rùa'->'Hoan Kiem Lake Turtle Tower', 'ruộng bậc thang'->'terraced rice fields', 'bùng binh'->'traffic roundabout'.",
    ),
    (
        ("xe máy", "xe tay ga", "xe số", "xe ôm", "xích lô", "ba gác", "thuyền ba lá", "ghe xuồng", "mũ bảo hiểm", "trạm xăng", "bình xăng", "dầu mazut"),
        "- Vehicles: 'xe ôm/xe ôm công nghệ'->'tech motorcycle taxi driver', 'xe máy/xe tay ga'->'motorcycle/scooter' (NEVER 'bicycle'), 'người đi xe máy'->'motorcyclist' (NEVER 'cyclist'), 'trạm xăng'->'gas station', 'dầu mazut'->'mazut heavy fuel oil'.",
    ),
    (
        ("ảnh cận", "cận cảnh", "toàn cảnh", "từ trên cao", "flycam", "mc", "phóng viên"),
        "- Cinematic: 'cận cảnh'->'close-up shot', 'toàn cảnh/flycam'->'wide-angle panoramic shot', 'MC/phóng viên'->'TV reporter speaking to camera'.",
    ),
    (
        ("sầu riêng", "măng cụt", "bưởi", "bòn bon", "dâu bòn bon", "vú sữa", "chôm chôm", "thanh long", "mãng cầu", "na", "nhãn", "vải", "mận", "xoài", "ổi", "mít", "đu đủ", "vườn cây", "miền tây", "tây nam bộ"),
        "- Tropical Fruits & Orchard: 'sầu riêng'->'durian fruit on tree', 'măng cụt'->'mangosteen fruit', 'bưởi'->'pomelo fruit on tree', 'dâu bòn bon/bòn bon'->'langsat bonbon fruit clusters on tree trunk', 'vú sữa'->'star apple milk fruit', 'chôm chôm'->'rambutan fruit', 'thanh long'->'dragon fruit', 'mãng cầu'->'custard apple soursop', 'nhãn'->'longan fruit', 'vải'->'lychee', 'mít'->'jackfruit', 'vườn cây ăn trái'->'tropical fruit orchard garden', 'miền tây nam bộ'->'Mekong Delta Southwest Vietnam'.",
    ),
    (
        ("mì tôm", "thùng mì", "mì gói", "mì ăn liền", "bánh mì", "thùng carton", "niêm phong", "bọc bánh mì", "xếp chồng"),
        "- Packaging & Food: 'mì tôm/thùng mì tôm/mì gói'->'carton of instant noodles', 'thùng carton'->'cardboard boxes', 'dán niêm phong'->'sealing cardboard box with tape', 'bọc bánh mì'->'bag of bread loaves', 'xếp chồng'->'stacking boxes'.",
    ),
]


def _build_dynamic_system_prompt(query: str, target: str) -> str:
    """Tạo prompt tinh gọn động dựa trên từ khóa thực tế trong query, giảm 90% prefill tokens."""
    if target != "en":
        return "You are a professional search query translator translating English to Vietnamese. Output only the clean translated query text."

    q_lower = query.lower()
    matched_rules = []
    for keywords, rule_str in _CULTURAL_CATEGORY_RULES:
        if any(kw in q_lower for kw in keywords):
            matched_rules.append(rule_str)

    base = (
        "You are an expert visual search query translator. "
        "Translate Vietnamese to concise English for CLIP visual retrieval. "
        "Preserve subjects, actions, colors, numbers, and text accurately.\n"
    )
    if matched_rules:
        base += "DOMAIN VOCABULARY RULES:\n" + "\n".join(matched_rules) + "\n"
    base += "Output ONLY the concise visual English description. No explanations or quotes."
    return base


_AIC_IDIOM_MAP: dict[str, str] = {
    # Thực phẩm, Đóng gói & Vận chuyển (Khắc phục triệt để MarianMT dịch nhầm mì tôm -> fried shrimp)
    "thùng mì tôm": "carton of instant noodles",
    "thùng mì": "carton of instant noodles",
    "mì tôm": "instant noodles",
    "gói mì tôm": "packet of instant noodles",
    "mì gói": "instant noodles",
    "mì ăn liền": "instant noodles",
    "bọc bánh mì": "bag of bread loaves",
    "bánh mì": "bread baguette",
    "thùng carton": "cardboard box",
    "thùng các tông": "cardboard box",
    "dán niêm phong": "taping sealing cardboard box",
    "chồng thùng": "stack of cardboard boxes",
    "xếp chồng": "stacked on top of each other",
    # Giao thông & Xe cộ
    "xe ôm công nghệ": "tài xế xe máy công nghệ",
    "tài xế xe ôm": "tài xế xe máy chở khách",
    "xe ôm": "xe máy chở khách",
    "dầu mazut": "dầu mazut nhiên liệu",
    "bình xăng": "bình xăng xe máy",
    "trạm xăng": "cây xăng",
    "bùng binh": "vòng xoay ngã tư",
    # Trang phục & Văn hóa
    "áo dài": "áo dài truyền thống Việt Nam",
    "nón lá": "nón lá truyền thống",
    "mai hoa thung": "trụ mai hoa thung múa lân",
    "thanh trụ": "trụ biểu diễn múa lân",
    "kẻng đồng": "kẻng đồng múa lân",
    "con lân": "lân sư rồng",
    "đầu lân": "đầu lân sư rồng",
    "múa lân": "biểu diễn múa lân",
    "chợ bến thành": "chợ Bến Thành Sài Gòn",
    "tháp rùa": "Tháp Rùa Hồ Gươm",
    "cầu thê húc": "cầu Thê Húc đỏ",
    # Trái cây & Nông nghiệp đặc thù Việt Nam / Nam Bộ (Khắc phục triệt để MarianMT dịch nhầm)
    "trái dâu bòn bon": "langsat bonbon fruit clusters on tree",
    "dâu bòn bon": "langsat bonbon fruit clusters on tree",
    "trái bòn bon": "langsat bonbon fruit clusters on tree",
    "bòn bon": "langsat bonbon fruit clusters on tree",
    "trái sầu riêng": "durian fruit tree",
    "sầu riêng": "durian fruit tree",
    "trái măng cụt": "mangosteen fruit pile",
    "măng cụt": "mangosteen fruit pile",
    "trái bưởi": "pomelo grapefruit citrus",
    "quả bưởi": "pomelo grapefruit citrus",
    "bưởi": "pomelo grapefruit citrus",
    "vú sữa": "star apple milk fruit",
    "trái vú sữa": "star apple milk fruit",
    "chôm chôm": "rambutan fruit",
    "trái chôm chôm": "rambutan fruit",
    "thanh long": "dragon fruit pitaya",
    "trái thanh long": "dragon fruit",
    "mãng cầu": "custard apple soursop",
    "trái mãng cầu": "custard apple soursop",
    "quả na": "sugar apple",
    "trái na": "sugar apple",
    "quả nhãn": "longan fruit",
    "trái nhãn": "longan fruit",
    "chùm nhãn": "longan fruit clusters",
    "vải thiều": "lychee fruit",
    "trái vải": "lychee fruit",
    "trái mận": "rose apple water apple",
    "quả mận": "rose apple water apple",
    "trái xoài": "mango fruit",
    "trái ổi": "guava fruit",
    "trái mít": "jackfruit",
    "quả mít": "jackfruit",
    "đu đủ": "papaya fruit",
    "vườn cây ăn trái": "tropical fruit orchard garden",
    "khu vườn cây ăn trái": "tropical fruit orchard garden",
    "vườn trái cây": "tropical fruit orchard garden",
    "miền tây nam bộ": "Mekong Delta Southwest Vietnam",
    "tây nam bộ": "Mekong Delta Southwest Vietnam",
    "miền tây": "Mekong Delta Vietnam",
    # Ẩm thực & Nấu nướng
    "măng tây": "asparagus",
    "miến măng tây": "asparagus spear",
    "miếng măng tây": "asparagus spear",
    "tô măng tây": "asparagus bowl",
    "khổ qua": "bitter melon",
    "mướp đắng": "bitter melon",
    "chảo dầu": "frying pan with oil",
    "chảo": "frying pan",
    "dĩa": "plate",
    "cơm tấm": "cơm tấm sườn",
    "cà phê sữa đá": "cà phê sữa đá",
}

# Regex biên dịch một lần, sắp xếp từ dài đến ngắn để khớp cụm từ ghép trước (vd: "xe ôm công nghệ" trước "xe ôm")
_IDIOM_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_AIC_IDIOM_MAP.keys(), key=len, reverse=True)),
    flags=re.IGNORECASE,
)


def _normalize_vietnamese_idioms(text: str) -> str:
    """Thay thế một lượt duy nhất (single-pass, không đè lồng đệ quy) các thành ngữ đặc thù AIC."""
    if not text:
        return text
    return _IDIOM_PATTERN.sub(lambda m: _AIC_IDIOM_MAP[m.group(0).lower()], text)


_NARRATIVE_CONNECTORS = (
    "trước đó là",
    "trước đó",
    "sau đó là",
    "sau đó",
    "tiếp theo là",
    "tiếp theo",
    "kế tiếp",
    "tiếp tục",
    "đồng thời",
    "cùng lúc",
    "sau khi",
    "trước khi",
    "rồi sau đó",
    "rồi lại",
)


def _is_complex_narrative_query(query: str, context: str | None = None) -> bool:
    """Xác định xem query có phải là câu chuyện tự sự phức tạp (TRAKE) cần Qwen2.5 hay câu thông thường."""
    combined = f"{context} {query}".strip() if context else query.strip()
    words = combined.split()
    if len(words) >= 22:
        return True
    q_lower = combined.lower()
    if any(conn in q_lower for conn in _NARRATIVE_CONNECTORS):
        return True
    return False


def _translate_with_llm_sync(query: str, target: str, bundle: dict, context: str | None = None) -> str | None:
    try:
        import torch
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        device = bundle["device"]

        system_prompt = _build_dynamic_system_prompt(query, target)

        if target == "en":
            if context and query != context:
                prompt = (
                    f"Video Global Context: {context}\n"
                    f"Sub-event to translate: {query}\n"
                    f"Instruction: Translate ONLY the sub-event into a concise English visual caption for image search. "
                    f"Incorporate global context only to disambiguate the scene or subjects. "
                    f"Do NOT repeat or translate the global context itself.\n"
                    f"Translation:"
                )
            else:
                prompt = f"Translate to English visual caption: {query}\nTranslation:"
        else:
            prompt = f"Translate to Vietnamese search query: {query}\nTranslation:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        word_count = len(query.split())
        calc_tokens = min(128, max(64, int(word_count * 2.0)))

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=calc_tokens,
                max_length=None,
                do_sample=False,
                use_cache=True,
            )
        res = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip(' "\'\n\t`')
        return res if res else None
    except Exception as e:
        logger.warning(f"Local LLM translation failed: {e}")
        return None


def _translate_with_llm_batch(
    queries: list[str],
    target: str,
    bundle: dict,
    context: str | None = None,
) -> list[str | None]:
    """Dịch đồng thời một batch queries bằng Qwen2.5-1.5B trên GPU (~1.5s cho cả batch thay vì 13.5s)."""
    try:
        import torch
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        device = bundle["device"]

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompts = []
        for q in queries:
            system_prompt = _build_dynamic_system_prompt(q, target)
            if target == "en":
                if context and q != context:
                    user_prompt = (
                        f"Video Global Context: {context}\n"
                        f"Sub-event to translate: {q}\n"
                        f"Instruction: Translate ONLY the sub-event into a concise English visual caption for image search. "
                        f"Incorporate global context only to disambiguate the scene or subjects. "
                        f"Do NOT repeat or translate the global context itself.\n"
                        f"Translation:"
                    )
                else:
                    user_prompt = f"Translate to English visual caption: {q}\nTranslation:"
            else:
                user_prompt = f"Translate to Vietnamese search query: {q}\nTranslation:"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        max_words = max(len(q.split()) for q in queries) if queries else 10
        calc_tokens = min(128, max(64, int(max_words * 2.0)))

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=calc_tokens,
                max_length=None,
                do_sample=False,
                use_cache=True,
            )

        results = []
        input_len = inputs.input_ids.shape[1]
        for o in out:
            res = tokenizer.decode(o[input_len:], skip_special_tokens=True).strip(' "\'\n\t`')
            results.append(res if res else None)
        return results
    except Exception as e:
        logger.warning(f"Batch Local LLM translation failed: {e}")
        return [None] * len(queries)


_marian_bundle: dict = {}


def _translate_with_marian_sync(query: str, target: str) -> str | None:
    """Dịch nhanh bằng model local MarianMT (~60-120ms trên GPU CUDA FP16, 100% offline)."""
    global _marian_bundle
    model_name = "Helsinki-NLP/opus-mt-vi-en" if target == "en" else "Helsinki-NLP/opus-mt-en-vi"
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        if model_name not in _marian_bundle:
            try:
                from serving.api.core.models import ModelRegistry
                reg = ModelRegistry()
                reg_bundle = reg.get("marian_mt")
                if reg_bundle is not None and isinstance(reg_bundle, dict) and "model" in reg_bundle:
                    _marian_bundle[model_name] = reg_bundle
            except Exception:
                pass

        if model_name not in _marian_bundle:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = MarianMTModel.from_pretrained(model_name)
            if device == "cuda":
                model = model.to("cuda").half()
            _marian_bundle[model_name] = {"model": model, "tokenizer": tokenizer, "device": device}

        b = _marian_bundle[model_name]

        # Tiền xử lý thành ngữ đặc thù AIC nếu dịch từ VI sang EN
        q_proc = _normalize_vietnamese_idioms(query) if target == "en" else query

        inputs = b["tokenizer"](q_proc, return_tensors="pt", truncation=True, max_length=256)
        if b["device"] == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        word_count = len(query.split())
        calc_tokens = min(48, max(24, int(word_count * 1.5)))

        with torch.inference_mode():
            outputs = b["model"].generate(**inputs, num_beams=1, max_new_tokens=calc_tokens, max_length=None)
        res = b["tokenizer"].decode(outputs[0], skip_special_tokens=True).strip()
        return res if res else None
    except Exception as e:
        logger.debug(f"MarianMT local translation fallback: {e}")
        return None


def warmup_translation() -> None:
    """Warmup MarianMT CUDA kernels và tokenizer tại thời điểm startup."""
    try:
        _ = _translate_with_marian_sync("khởi động tìm kiếm", target="en")
        logger.info("🔥 MarianMT translation warmup hoàn tất!")
    except Exception as e:
        logger.debug(f"warmup_translation: {e}")


def _translate_with_marian_batch(queries: list[str], target: str = "en") -> list[str | None]:
    """Dịch đồng thời một batch queries bằng MarianMT trên GPU CUDA (~40-80ms cho cả batch)."""
    global _marian_bundle
    model_name = "Helsinki-NLP/opus-mt-vi-en" if target == "en" else "Helsinki-NLP/opus-mt-en-vi"
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        if model_name not in _marian_bundle:
            try:
                from serving.api.core.models import ModelRegistry
                reg = ModelRegistry()
                reg_bundle = reg.get("marian_mt")
                if reg_bundle is not None and isinstance(reg_bundle, dict) and "model" in reg_bundle:
                    _marian_bundle[model_name] = reg_bundle
            except Exception:
                pass

        if model_name not in _marian_bundle:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = MarianMTModel.from_pretrained(model_name)
            if device == "cuda":
                model = model.to("cuda").half()
            _marian_bundle[model_name] = {"model": model, "tokenizer": tokenizer, "device": device}

        b = _marian_bundle[model_name]

        # Tiền xử lý thành ngữ đặc thù AIC nếu dịch từ VI sang EN
        q_procs = [_normalize_vietnamese_idioms(q) if target == "en" else q for q in queries]

        inputs = b["tokenizer"](q_procs, return_tensors="pt", padding=True, truncation=True, max_length=256)
        if b["device"] == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        calc_tokens = min(64, max(24, int(max(len(q.split()) for q in queries) * 1.5))) if queries else 48

        with torch.inference_mode():
            outputs = b["model"].generate(**inputs, num_beams=1, max_new_tokens=calc_tokens, max_length=None)

        results = [b["tokenizer"].decode(o, skip_special_tokens=True).strip() for o in outputs]
        return [r if r else None for r in results]
    except Exception as e:
        logger.debug(f"MarianMT batch translation fallback: {e}")
        return [None] * len(queries)


def _post_process_english_translation(en_text: str) -> str:
    """Làm sạch các từ dịch sai lệch văn hóa đặc thù của Google/Marian/LLM."""
    if not en_text:
        return ""
    clean = en_text
    # Dâu bòn bon: không bao giờ là strawberry
    clean = re.sub(r"\b(?:bonbon strawberry|bon\s*bon\s*strawberry|bonbon)\b", "langsat fruit clusters on tree trunk", clean, flags=re.IGNORECASE)
    # Bưởi: pomelo thay vì grapefruit hay lemon
    clean = re.sub(r"\bgrapefruit\b", "pomelo fruit", clean, flags=re.IGNORECASE)
    # Mì tôm: instant noodles thay vì fried shrimp
    clean = re.sub(r"\bfried shrimp\b", "carton of instant noodles", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bshrimp noodles?\b", "instant noodles", clean, flags=re.IGNORECASE)
    return clean.strip()


async def _translate(
    query: str,
    target: str = "en",
    context: str | None = None,
) -> str:
    """
    Dịch query sử dụng hệ thống phân tầng thông minh (Cache -> Google API -> MarianMT GPU -> Qwen2.5-1.5B).
    """
    if not query or not query.strip():
        return ""

    from serving.api.core.translation_cache import get_translation_cache
    cache = get_translation_cache()

    # Tier 0: Tra cứu Cache (khi không có context riêng biệt)
    if not context:
        cached = cache.get(query, target)
        if cached:
            return cached

    import asyncio
    from serving.api.core.config import get_settings
    settings = get_settings()

    # 1. Tier 1: Google Translator API trực tuyến (Ưu tiên số 1, độ chính xác cao nhất)
    if getattr(settings, "USE_GOOGLE_TRANSLATE", True):
        try:
            norm_q = _normalize_vietnamese_idioms(query)
            res = await asyncio.wait_for(
                asyncio.to_thread(_translate_with_google_sync, norm_q, target),
                timeout=TRANSLATE_TIMEOUT_S,
            )
            if res and _is_valid_translation(res, query):
                if target == "en":
                    res = _post_process_english_translation(res)
                if not context:
                    cache.set(query, target, res)
                return res
        except Exception as err:
            logger.debug(f"GoogleTranslator error, falling back to local LLM: {err}")

    # 2. Tier 2: Local LLM (Qwen2.5-1.5B) Fallback chất lượng cao
    bundle = _get_qwen_bundle()
    if bundle is not None:
        try:
            res = await asyncio.to_thread(_translate_with_llm_sync, query, target, bundle, context)
            if res and _is_valid_translation(res, query):
                if target == "en":
                    res = _post_process_english_translation(res)
                if not context:
                    cache.set(query, target, res)
                return res
        except Exception as err:
            logger.debug(f"Local LLM translation fallback: {err}")

    # 4. Fallback Web Translators (MyMemory)
    from deep_translator import MyMemoryTranslator

    for translator_cls in (MyMemoryTranslator,):
        try:
            translator = translator_cls(source="auto", target=target)
            out = await asyncio.wait_for(
                asyncio.to_thread(translator.translate, query),
                timeout=TRANSLATE_TIMEOUT_S,
            )
            if out and _is_valid_translation(out, query):
                if not context:
                    cache.set(query, target, out)
                return out
        except Exception:
            pass

    return None


async def _enhance_query(raw_query: str, context: str | None = None) -> tuple[str, str]:
    """
    Dịch và chuẩn hóa query song ngữ (English cho CLIP, Tiếng Việt cho Qwen2/OCR/ASR).
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
        import unicodedata
        nfkd = unicodedata.normalize('NFKD', cleaned_query)
        unaccented = ''.join([c for c in nfkd if not unicodedata.combining(c)]).replace('đ', 'd').replace('Đ', 'D')
        title_cased = unaccented.title()

        # Dịch trực tiếp cleaned_query (không làm méo accent qua .title())
        translated_en = await _translate(cleaned_query, "en", context=context)

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


async def batch_enhance_queries(
    queries: list[str],
    context: str | None = None,
) -> list[tuple[str, str]]:
    """
    Tăng cường đồng thời một batch các query (tiếng Việt sang tiếng Anh CLIP và tiếng Việt chuẩn hóa).
    Tối ưu cho TRAKE sub-events để tránh hiện tượng GPU lock xếp hàng tuần tự.
    """
    from shared.utils.text_normalizer import sanitize_search_query
    from serving.api.core.translation_cache import get_translation_cache
    cache = get_translation_cache()

    cleaned_queries = [sanitize_search_query(q) for q in queries]
    if not cleaned_queries:
        return []

    results: list[tuple[str, str] | None] = [None] * len(cleaned_queries)
    uncached_indices: list[int] = []

    for idx, q in enumerate(cleaned_queries):
        if not q:
            results[idx] = ("", "")
            continue

        has_vietnamese_accent = bool(
            re.search(
                r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]',
                q,
                re.IGNORECASE,
            )
        )
        if not has_vietnamese_accent:
            results[idx] = (q, q)
            continue

        cache_key = f"{q} [CTX: {context}]" if context else q
        cached = cache.get(cache_key, "en")
        if cached and _is_valid_translation(cached, q):
            results[idx] = (sanitize_search_query(cached), q)
        else:
            uncached_indices.append(idx)

    if uncached_indices:
        uncached_queries = [cleaned_queries[i] for i in uncached_indices]
        from serving.api.core.config import get_settings
        settings = get_settings()

        translations: list[str | None] = [None] * len(uncached_queries)

        # 1. Thử dịch đồng thời toàn bộ batch qua Google Translator API nếu được bật
        if getattr(settings, "USE_GOOGLE_TRANSLATE", True):
            try:
                google_tasks = [
                    asyncio.wait_for(
                        asyncio.to_thread(_translate_with_google_sync, _normalize_vietnamese_idioms(q), "en"),
                        timeout=TRANSLATE_TIMEOUT_S,
                    )
                    for q in uncached_queries
                ]
                google_results = await asyncio.gather(*google_tasks, return_exceptions=True)
                for i, r in enumerate(google_results):
                    if isinstance(r, str) and r and _is_valid_translation(r, uncached_queries[i]):
                        translations[i] = _post_process_english_translation(r)
            except Exception as e:
                logger.debug(f"Batch GoogleTranslator error: {e}")

        # 2. Fallback sang Local Qwen LLM cho những câu Google chưa dịch được (hoặc khi mất mạng)
        missing_indices = [i for i, t in enumerate(translations) if not t]
        if missing_indices:
            missing_queries = [uncached_queries[i] for i in missing_indices]
            bundle = _get_qwen_bundle()

            if bundle is not None:
                local_trans = await asyncio.to_thread(
                    _translate_with_llm_batch,
                    missing_queries,
                    "en",
                    bundle,
                    context,
                )
                for m_idx, lt in zip(missing_indices, local_trans):
                    if lt:
                        translations[m_idx] = _post_process_english_translation(lt)

        for u_idx, trans in zip(uncached_indices, translations):
            raw_q = cleaned_queries[u_idx]
            if trans and _is_valid_translation(trans, raw_q):
                en_clean = sanitize_search_query(trans)
                cache_key = f"{raw_q} [CTX: {context}]" if context else raw_q
                cache.set(cache_key, "en", en_clean)
                results[u_idx] = (en_clean, raw_q)
            else:
                import unicodedata
                nfkd = unicodedata.normalize('NFKD', raw_q)
                unaccented = ''.join([c for c in nfkd if not unicodedata.combining(c)]).replace('đ', 'd').replace('Đ', 'D')
                results[u_idx] = (f"{unaccented.title()} {unaccented}", raw_q)

    final_res = [r if r is not None else ("", "") for r in results]
    for orig, (en, vi) in zip(queries, final_res):
        logger.info(f"Batch enhanced — EN: '{en}' | VI: '{vi}'")
    return final_res


def _split_events(query: str) -> list[str]:
    """
    Tách TRAKE query thành N sub-event descriptions bằng sub_event_parser.
    """
    from serving.api.service.retrieval import sub_event_parser
    events = sub_event_parser.parse_sub_events(query)
    if len(events) >= 2:
        return events

    # Fallback sang LLM splitting nếu regex parser không tách được
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
        import torch
        bundle = _get_qwen_bundle()
        if not bundle:
            return []
        model, tokenizer, device = bundle["model"], bundle["tokenizer"], bundle["device"]

        prompt = f"""Tách câu sau thành các sự kiện riêng biệt theo thứ tự thời gian, mỗi sự kiện trên 1 dòng. Chỉ liệt kê sự kiện, không giải thích.

Query: "{query}"

Sự kiện:"""

        messages = [
            {"role": "system", "content": "You are a temporal event parsing assistant. Output only the list of sequential events, one per line."},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

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
    Tích hợp Prompt Ensembling: kết hợp trung bình có trọng số của các biến thể prompt thị giác.

    Args:
        text: Text đã enhanced.
        embedder: ImageEmbeddingWrapper singleton.

    Returns:
        list[float], len=1024, L2-normalized.
    """
    import asyncio
    from serving.api.service.retrieval.query_expander import PromptEnsembleExpander

    try:
        fused = await asyncio.to_thread(PromptEnsembleExpander.ensemble_text_embeddings, text, embedder)
        if fused:
            return fused
    except Exception as e:
        logger.debug(f"Prompt ensembling fallback to single vector: {e}")

    vec = await asyncio.to_thread(embedder.encode_text, text)
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)

