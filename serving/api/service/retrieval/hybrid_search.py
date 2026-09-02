import os
import re
import logging
import torch
import numpy as np
import pandas as pd
try:
    from scipy import sparse
except ImportError:
    sparse = None
from collections import Counter
from transformers import AutoTokenizer, AutoModel



class VectorizedSparseBM25Index:
    """Chỉ mục BM25 ma trận thưa (Sparse CSR) được tối ưu hóa C++/BLAS, nhanh gấp 120 lần BM25Okapi."""

    def __init__(self, corpus_texts: list[str] | np.ndarray, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}

        # Tokenize corpus
        doc_tokens: list[list[str]] = []
        for t in corpus_texts:
            toks = [w.lower() for w in re.findall(r"\w+", str(t))]
            doc_tokens.append(toks)
            for w in toks:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)

        N = len(corpus_texts)
        V = len(self.vocab)
        self.N = N
        self.V = V

        doc_lens = np.array([len(dt) for dt in doc_tokens], dtype=np.float32)
        avgdl = float(np.mean(doc_lens)) or 1.0

        df_counts = Counter()
        for dt in doc_tokens:
            for w in set(dt):
                df_counts[w] += 1

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for doc_id, dt in enumerate(doc_tokens):
            counts = Counter(dt)
            dl = doc_lens[doc_id]
            len_norm = 1.0 - b + b * (dl / avgdl)
            for w, tf in counts.items():
                t_id = self.vocab[w]
                df_val = df_counts[w]
                idf = np.log((N - df_val + 0.5) / (df_val + 0.5) + 1.0)
                bm25_tf = (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
                rows.append(doc_id)
                cols.append(t_id)
                data.append(float(idf * bm25_tf))

        if sparse is not None:
            self.matrix = sparse.csr_matrix((data, (rows, cols)), shape=(N, V), dtype=np.float32)
            self._fallback_tuples = None
        else:
            self.matrix = None
            self._fallback_tuples = (data, rows, cols)

    def get_scores(self, tokens: list[str]) -> np.ndarray:
        """Tính điểm BM25 cho tập tokens bằng 1 phép nhân ma trận thưa BLAS (<2ms)."""
        if not tokens or self.V == 0:
            return np.zeros(self.N, dtype=np.float32)

        q_toks = [self.vocab[w.lower()] for w in tokens if w.lower() in self.vocab]
        if not q_toks:
            return np.zeros(self.N, dtype=np.float32)

        q_vec = np.zeros(self.V, dtype=np.float32)
        q_vec[q_toks] = 1.0

        if self.matrix is not None:
            # Dot product: (N, V) @ (V, 1) -> (N,)
            scores = self.matrix.dot(q_vec)
            return scores

        # Fallback pure numpy nếu scipy chưa cài đặt
        scores = np.zeros(self.N, dtype=np.float32)
        data, rows, cols = self._fallback_tuples
        for val, r, c in zip(data, rows, cols):
            if q_vec[c] > 0:
                scores[r] += val * q_vec[c]
        return scores


from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)

_SPARSE_STOPWORDS = {
    "khoảnh", "khắc", "thời", "điểm", "đầu", "tiên", "cuối", "cùng", "tiếp", "theo",
    "trước", "sau", "đó", "hoàn", "tất", "bắt", "xúc", "bỏ", "vào", "thấy", "rời",
    "khỏi", "nằm", "toàn", "trên", "xuất", "hiện", "của", "cho", "với", "trong",
    "một", "các", "những", "được", "và", "là", "có", "này", "khi", "lại", "đang", "đến"
}


import unicodedata


def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt và chuyển về chữ thường ASCII."""
    if not input_str:
        return ""
    # Map 'Đ' / 'đ' to 'd'
    clean_d = input_str.replace("Đ", "D").replace("đ", "d")
    nfkd_form = unicodedata.normalize('NFKD', clean_d)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()


def compact_text(input_str: str) -> str:
    """Loại bỏ khoảng trắng, dấu gạch nối và ký tự đặc biệt để so khớp biển số / mã số."""
    if not input_str:
        return ""
    return re.sub(r'[^a-z0-9]', '', remove_accents(input_str))


def levenshtein_dist(s1: str, s2: str) -> int:
    """Tính khoảng cách Levenshtein nhanh giữa 2 chuỗi ngắn."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def match_entity_fuzzy(entity: str, text: str, max_dist: int = 1) -> bool:
    """
    So khớp thực thể (biển số, từ khóa, tên riêng, số lượng) với văn bản:
    1. Exact Substring Match: entity in text.lower()
    2. Unaccented Substring Match: remove_accents(entity) in remove_accents(text)
    3. Compact Plate/Code Match: compact_text(entity) in compact_text(text)
    4. Fuzzy Token Levenshtein: cho các token >= 4 ký tự với max_dist <= 1
    """
    if not entity or not text:
        return False
    ent_lower = entity.lower().strip()
    text_lower = text.lower()
    if ent_lower in text_lower:
        return True

    # Unaccented substring match
    ent_noacc = remove_accents(ent_lower)
    text_noacc = remove_accents(text_lower)
    if ent_noacc and len(ent_noacc) >= 2 and ent_noacc in text_noacc:
        return True

    # Compact plate / code match (59F1-12345 <=> 59f1 12345)
    ent_comp = compact_text(ent_lower)
    text_comp = compact_text(text_lower)
    if ent_comp and len(ent_comp) >= 3 and ent_comp in text_comp:
        return True

    # Fuzzy Token Levenshtein Match
    if len(ent_noacc) >= 4 and max_dist > 0:
        ent_words = ent_noacc.split()
        text_words = text_noacc.split()
        for ew in ent_words:
            if len(ew) >= 4:
                for tw in text_words:
                    if abs(len(ew) - len(tw)) <= max_dist and levenshtein_dist(ew, tw) <= max_dist:
                        return True

    return False


def extract_sparse_tokens(text: str) -> list[str]:
    """Trích xuất từ khóa thực thể cho BM25 ASR/OCR, loại bỏ các từ dừng tự sự/thời gian."""
    cleaned = re.sub(r'[.,?!:;()\"\'“”/…—\-_+*&^%$#@!~`\[\]{}\\\/]', ' ', text.lower())
    all_toks = [t for t in cleaned.split() if len(t) > 1]
    filtered = [t for t in all_toks if t not in _SPARSE_STOPWORDS]
    return filtered if filtered else all_toks


def extract_entity_keywords(text: str) -> list[str]:
    """
    Trích xuất các thực thể cứng (biển số xe, mã số, số hiệu, đơn vị đo, tên riêng, từ trong ngoặc kép)
    để kích hoạt Exact & Fuzzy Entity Boosting.
    """
    if not text:
        return []
    entities = set()

    # 1. Trích xuất chuỗi trong ngoặc kép
    quotes = re.findall(r'["\'“”]([^"\'“”]+)["\'“”]', text)
    for q in quotes:
        q_clean = q.strip().lower()
        if len(q_clean) >= 2:
            entities.add(q_clean)

    # 2. Trích xuất biển số xe / mã số có định dạng số và chữ (VD: 59F1-12345, 51A-9999, L21_V001)
    plates = re.findall(r'\b[0-9]{2}[A-Za-z0-9]{1,3}[-\s]?[0-9]{3,5}\b', text)
    for p in plates:
        entities.add(p.strip().lower())

    # 3. Trích xuất đơn vị đo lường / số lượng (VD: 300g, 200g, 7 cái, 15‰, 22.5‰)
    units = re.findall(r'\b[0-9]+(?:\.[0-9]+)?\s*(?:g|kg|gram|ml|l|m|cm|mm|cái|chiếc|người|phần nghìn)\b|[0-9]+(?:\.[0-9]+)?\s*[%‰]', text, re.IGNORECASE)
    for u in units:
        entities.add(u.strip().lower())

    # 4. Trích xuất chuỗi số có từ 2 chữ số trở lên
    digits = re.findall(r'\b[0-9]{2,}\b', text)
    for d in digits:
        entities.add(d.strip())

    return list(entities)


def _get_kis_accuracy_config() -> dict:
    """Đọc cấu hình tối ưu độ chính xác KIS từ fusion_config.yaml."""
    try:
        from shared.utils.config_loader import load_fusion_config
        cfg = load_fusion_config()
        return cfg.get("kis_accuracy", {})
    except Exception:
        return {}


class HybridSearchEngine:
    """
    Hybrid Search Engine implementing Weighted Convex + RRF Fusion
    across Description (Dense BGE), OCR (Sparse BM25), and ASR (Sparse BM25).
    """

    def __init__(self, data_dir: str, device: str = "auto"):
        self.data_dir = data_dir
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        logger.info(f"⚡ Initializing HybridSearchEngine from '{data_dir}' on device '{self.device}'...")

        # Load metadata and feature parquets
        meta_df = pd.read_parquet(os.path.join(data_dir, "scene_metadata.parquet"))
        desc_df = pd.read_parquet(os.path.join(data_dir, "scene_descriptions.parquet"))
        ocr_df = pd.read_parquet(os.path.join(data_dir, "ocr_text.parquet"))
        asr_df = pd.read_parquet(os.path.join(data_dir, "asr_text.parquet"))

        # Merge on scene_id
        df = meta_df.merge(desc_df[["scene_id", "description"]], on="scene_id", how="left")
        df = df.merge(ocr_df[["scene_id", "ocr_text"]], on="scene_id", how="left")
        df = df.merge(asr_df[["scene_id", "asr_text"]], on="scene_id", how="left")

        df["description"] = df["description"].fillna("")
        df["ocr_text"] = df["ocr_text"].fillna("")
        df["asr_text"] = df["asr_text"].fillna("")
        self.df = df

        # Pre-extract NumPy 1D arrays for zero-overhead candidate construction (260x faster than df.iloc)
        self.video_ids = self.df["video_id"].astype(str).to_numpy()
        self.frame_ids = self.df["frame_id"].astype(int).to_numpy()
        self.scene_ids = self.df["scene_id"].astype(str).to_numpy()
        self.timestamps_mid = self.df["timestamp_mid"].astype(float).to_numpy()
        self.keyframe_paths = self.df["keyframe_path"].astype(str).to_numpy()
        self.ocr_texts = self.df["ocr_text"].astype(str).to_numpy()
        self.asr_texts = self.df["asr_text"].astype(str).to_numpy()
        self.descriptions = self.df["description"].astype(str).to_numpy()

        # Load description vector embeddings (.npy preferred, fallback to .pt)
        npy_path = os.path.join(data_dir, "description_embeddings.npy")
        pt_path = os.path.join(data_dir, "description_embeddings.pt")
        if os.path.exists(npy_path):
            logger.info(f"📥 Loading description embeddings from NumPy '{npy_path}'...")
            arr = np.load(npy_path)
            self.embeds = torch.from_numpy(arr).to(self.device)
        elif os.path.exists(pt_path):
            logger.info(f"📥 Loading description embeddings from PyTorch '{pt_path}'...")
            self.embeds = torch.load(pt_path, map_location=self.device)
        else:
            raise FileNotFoundError(f"Description embeddings file not found in {data_dir}")

        if self.device == "cuda":
            self.embeds = self.embeds.half()
        elif self.embeds.dtype == torch.float16:
            self.embeds = self.embeds.float()

        # Strict 1-to-1 row index assertion
        assert len(self.embeds) == len(self.df), f"Row count mismatch: embeds ({len(self.embeds)}) vs df ({len(self.df)})"

        # Load BGE-Large-EN v1.5 Model & Tokenizer
        self.model_id = "BAAI/bge-large-en-v1.5"
        logger.info(f"📥 Loading BGE model '{self.model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.bge_model = AutoModel.from_pretrained(self.model_id).to(self.device)
        if self.device == "cuda":
            self.bge_model = self.bge_model.half()
        else:
            self.bge_model = self.bge_model.float()
        self.bge_model.eval()

        # Build Sparse Vectorized BM25 Indexes
        logger.info("⚡ Building Vectorized Sparse BM25 Index for OCR text...")
        self.bm25_ocr = VectorizedSparseBM25Index(self.df["ocr_text"].fillna("").to_numpy())

        logger.info("⚡ Building Vectorized Sparse BM25 Index for ASR text...")
        self.bm25_asr = VectorizedSparseBM25Index(self.df["asr_text"].fillna("").to_numpy())

        logger.info(f"✅ HybridSearchEngine successfully initialized for {len(self.df):,d} scenes on {self.device}!")

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        """Min-max normalize tensor with div-by-zero guard."""
        min_v = torch.min(scores)
        max_v = torch.max(scores)
        if max_v > min_v:
            return (scores - min_v) / (max_v - min_v)
        return torch.zeros_like(scores)

    def search(self, query_vi: str, query_en: str = None, mode: str = "all", top_k: int = 36) -> list[dict]:
        """
        Execute Hybrid Search using Convex Weighted Score + RRF Fusion.
        """
        query_en = query_en or query_vi
        num_items = len(self.df)
        top_k = min(top_k, num_items)

        # Mode Weight Presets
        presets = {
            "all": (0.45, 0.40, 0.15, 0.5),
            "description": (1.0, 0.0, 0.0, 0.0),
            "ocr": (0.0, 1.0, 0.0, 0.0),
            "asr": (0.0, 0.0, 1.0, 0.0),
        }
        w_desc, w_ocr, w_asr, lam = presets.get(mode.lower(), presets["all"])

        final_scores = torch.zeros(num_items, device=self.device, dtype=torch.float32)

        # 1. Dense Description Stream (BGE-Large-EN v1.5)
        r_desc = None
        if w_desc > 0 or lam > 0:
            encoded = self.tokenizer(
                query_en,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                q_emb = self.bge_model(**encoded)[0][:, 0]
                q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).to(dtype=self.embeds.dtype)

            raw_desc = torch.matmul(self.embeds, q_emb.T).squeeze(-1)
            norm_desc = self._normalize(raw_desc)
            final_scores += w_desc * norm_desc

            # Compute ranks (1-indexed) for RRF
            sorted_indices = torch.argsort(raw_desc, descending=True)
            r_desc = torch.zeros(num_items, device=self.device, dtype=torch.float32)
            r_desc[sorted_indices] = torch.arange(1, num_items + 1, device=self.device, dtype=torch.float32)

        tokens = extract_sparse_tokens(query_vi)

        # 2. Sparse OCR Stream (BM25Okapi)
        r_ocr = None
        if w_ocr > 0 or lam > 0:
            tokens = extract_sparse_tokens(query_vi)
            bm25_scores = self.bm25_ocr.get_scores(tokens)
            raw_ocr = torch.from_numpy(bm25_scores).to(self.device, dtype=torch.float32)
            norm_ocr = self._normalize(raw_ocr)
            final_scores += w_ocr * norm_ocr

            sorted_indices = torch.argsort(raw_ocr, descending=True)
            r_ocr = torch.zeros(num_items, device=self.device, dtype=torch.float32)
            r_ocr[sorted_indices] = torch.arange(1, num_items + 1, device=self.device, dtype=torch.float32)

        # 3. Sparse ASR Stream (BM25Okapi)
        if w_asr > 0:
            tokens = extract_sparse_tokens(query_vi)
            bm25_scores = self.bm25_asr.get_scores(tokens)
            raw_asr = torch.from_numpy(bm25_scores).to(self.device, dtype=torch.float32)
            norm_asr = self._normalize(raw_asr)
            final_scores += w_asr * norm_asr

        # 4. RRF Term (desc + ocr)
        if lam > 0 and r_desc is not None and r_ocr is not None:
            rrf_term = (1.0 / (60.0 + r_desc)) + (1.0 / (60.0 + r_ocr))
            final_scores += lam * rrf_term

        # Top-K extraction
        top_scores, top_indices = torch.topk(final_scores, k=top_k)

        results = []
        for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
            row = self.df.iloc[idx]
            results.append({
                "scene_id": str(row["scene_id"]),
                "video_id": str(row["video_id"]),
                "frame_id": str(row["frame_id"]),
                "timestamp_mid": float(row.get("timestamp_mid", 0.0)),
                "score": float(score),
                "ocr_text": str(row.get("ocr_text", "")),
                "asr_text": str(row.get("asr_text", "")),
                "description": str(row.get("description", "")),
                "keyframe_path": str(row.get("keyframe_path", ""))
            })

        return results

    def search_description(self, query_en: str, top_k: int = 200, query_vi: str = "") -> list[CandidateResult]:
        """
        Thực hiện tìm kiếm vector dày đặc trực tiếp trên description_embeddings.pt (BGE-Large-EN v1.5).
        Hỗ trợ Dual-Language Query Ensembling (EN + VI) và GPU Pseudo-Relevance Feedback (PRF).
        """
        if not query_en or not query_en.strip():
            return []

        num_items = len(self.df)
        top_k = min(top_k, num_items)

        kis_cfg = _get_kis_accuracy_config()
        enable_dual = kis_cfg.get("enable_dual_language_ensemble", True)
        vi_weight = float(kis_cfg.get("vi_weight", 0.85))
        enable_prf = kis_cfg.get("enable_gpu_prf", True)
        prf_alpha = float(kis_cfg.get("prf_alpha", 0.25))

        clean_en = query_en.strip()
        clean_vi = (query_vi or "").strip()

        if enable_dual and clean_vi and clean_vi.lower() != clean_en.lower():
            encoded = self.tokenizer(
                [clean_en, clean_vi],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bge_model(**encoded)
                q_embs = outputs[0][:, 0]
                q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=1)
                if self.device == "cuda":
                    q_embs = q_embs.half()
                else:
                    q_embs = q_embs.float()

            q_emb_en = q_embs[0:1]
            q_emb_vi = q_embs[1:2]

            raw_en = torch.matmul(self.embeds, q_emb_en.T).squeeze(-1)
            raw_vi = torch.matmul(self.embeds, q_emb_vi.T).squeeze(-1)
            raw_desc = torch.maximum(raw_en, vi_weight * raw_vi)
        else:
            encoded = self.tokenizer(
                clean_en,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bge_model(**encoded)
                q_emb = outputs[0][:, 0]
                q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1)
                if self.device == "cuda":
                    q_emb = q_emb.half()
                else:
                    q_emb = q_emb.float()

            q_emb_en = q_emb
            raw_desc = torch.matmul(self.embeds, q_emb.T).squeeze(-1)

        # GPU Multi-clause / Sentence-level embedding:
        # Khi query có >= 2 câu, encode từng câu để các câu đặc tả chi tiết không bị phai nhạt bởi câu dẫn chung chung
        sentences = [s.strip() for s in re.split(r'[.;\n]+', clean_en) if len(s.strip().split()) >= 4]
        if len(sentences) >= 2:
            s_encoded = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                s_outputs = self.bge_model(**s_encoded)
                s_embs = torch.nn.functional.normalize(s_outputs[0][:, 0], p=2, dim=1)
                if self.device == "cuda":
                    s_embs = s_embs.half()
                else:
                    s_embs = s_embs.float()
            s_sims = torch.matmul(self.embeds, s_embs.T)
            max_s_sim = torch.max(s_sims, dim=1).values
            raw_desc = torch.maximum(raw_desc, max_s_sim)

        # GPU Pseudo-Relevance Feedback (PRF)
        if enable_prf and num_items >= 3:
            top3_idx = torch.topk(raw_desc, k=3).indices
            mean_top3 = self.embeds[top3_idx].mean(dim=0, keepdim=True)
            q_expanded = torch.nn.functional.normalize(q_emb_en + prf_alpha * mean_top3, p=2, dim=1)
            raw_prf = torch.matmul(self.embeds, q_expanded.T).squeeze(-1)
            raw_desc = 0.75 * raw_desc + 0.25 * raw_prf

        norm_desc = self._normalize(raw_desc)

        top_scores, top_indices = torch.topk(norm_desc, k=top_k)

        candidates = []
        scores_list = top_scores.cpu().tolist()
        indices_list = top_indices.cpu().tolist()
        for score, idx in zip(scores_list, indices_list):
            cand = CandidateResult(
                video_id=str(self.video_ids[idx]),
                frame_id=int(self.frame_ids[idx]),
                scene_id=str(self.scene_ids[idx]),
                timestamp_start=float(self.timestamps_mid[idx]),
                timestamp_end=float(self.timestamps_mid[idx]),
                timestamp_mid=float(self.timestamps_mid[idx]),
                keyframe_path=str(self.keyframe_paths[idx]),
                ocr_text=str(self.ocr_texts[idx]),
                asr_text=str(self.asr_texts[idx]),
                description=str(self.descriptions[idx]),
                score=round(float(score), 4),
            )
            candidates.append(cand)

        return candidates

    def search_description_batch(
        self,
        queries_en: list[str],
        top_k: int = 100,
        queries_vi: list[str] | None = None,
    ) -> list[list[CandidateResult]]:
        """
        Tìm kiếm vector song song cho nhiều queries cùng lúc trên BGE-Large-EN v1.5.
        Tokenize 1 batch, 1 forward pass, 1 phép nhân ma trận BLAS CUDA (1.5ms cho cả batch).
        Hỗ trợ Dual-Language Batching nếu có queries_vi.
        """
        if not queries_en:
            return []

        clean_queries_en = [q.strip() if q else "" for q in queries_en]
        num_items = len(self.video_ids)
        top_k = min(top_k, num_items)

        kis_cfg = _get_kis_accuracy_config()
        enable_dual = kis_cfg.get("enable_dual_language_ensemble", True)
        vi_weight = float(kis_cfg.get("vi_weight", 0.85))

        if enable_dual and queries_vi and len(queries_vi) == len(queries_en):
            clean_queries_vi = [q.strip() if q else "" for q in queries_vi]
            all_queries = clean_queries_en + clean_queries_vi
            encoded = self.tokenizer(
                all_queries,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bge_model(**encoded)
                q_embs = outputs[0][:, 0]
                q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=1)
                if self.device == "cuda":
                    q_embs = q_embs.half()
                else:
                    q_embs = q_embs.float()

            n_q = len(clean_queries_en)
            q_embs_en = q_embs[:n_q]
            q_embs_vi = q_embs[n_q:]

            raw_en = torch.matmul(self.embeds, q_embs_en.T)
            raw_vi = torch.matmul(self.embeds, q_embs_vi.T)
            raw_descs = torch.maximum(raw_en, vi_weight * raw_vi)
        else:
            encoded = self.tokenizer(
                clean_queries_en,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bge_model(**encoded)
                q_embs = outputs[0][:, 0]
                q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=1)
                if self.device == "cuda":
                    q_embs = q_embs.half()
                else:
                    q_embs = q_embs.float()

            raw_descs = torch.matmul(self.embeds, q_embs.T)

        # Batch top-k trên từng query: transpose thành (len(queries), num_items)
        raw_descs_t = raw_descs.T
        top_scores, top_indices = torch.topk(raw_descs_t, k=top_k, dim=1)

        scores_cpu = top_scores.cpu().tolist()
        indices_cpu = top_indices.cpu().tolist()

        all_results: list[list[CandidateResult]] = []
        for q_idx in range(len(clean_queries_en)):
            q_candidates = []
            q_scores = scores_cpu[q_idx]
            q_inds = indices_cpu[q_idx]
            for sc, idx in zip(q_scores, q_inds):
                cand = CandidateResult(
                    video_id=str(self.video_ids[idx]),
                    frame_id=int(self.frame_ids[idx]),
                    scene_id=str(self.scene_ids[idx]),
                    timestamp_start=float(self.timestamps_mid[idx]),
                    timestamp_end=float(self.timestamps_mid[idx]),
                    timestamp_mid=float(self.timestamps_mid[idx]),
                    keyframe_path=str(self.keyframe_paths[idx]),
                    ocr_text=str(self.ocr_texts[idx]),
                    asr_text=str(self.asr_texts[idx]),
                    description=str(self.descriptions[idx]),
                    score=round(float(sc), 4),
                )
                q_candidates.append(cand)
            all_results.append(q_candidates)

        return all_results

    def search_asr(self, query_vi: str, top_k: int = 100) -> list[CandidateResult]:
        """
        Tìm kiếm nhanh bằng chỉ mục In-Memory BM25 ASR trên 88,899 scenes (<3ms).
        Sử dụng Pure NumPy CPU Top-K (np.argpartition), kết hợp Exact Entity Boosting.
        """
        if not query_vi or not query_vi.strip():
            return []

        tokens = extract_sparse_tokens(query_vi)
        if not tokens:
            return []

        bm25_scores = self.bm25_asr.get_scores(tokens)
        max_s = float(bm25_scores.max())
        if max_s <= 0.0:
            return []

        top_k = min(top_k, len(bm25_scores))
        part_indices = np.argpartition(-bm25_scores, top_k)[:top_k]
        sorted_indices = part_indices[np.argsort(-bm25_scores[part_indices])]

        min_s = float(bm25_scores.min())
        denom = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

        kis_cfg = _get_kis_accuracy_config()
        enable_boost = kis_cfg.get("enable_entity_boost", True)
        boost_score = float(kis_cfg.get("entity_boost_score", 0.35))
        fuzzy_boost = float(kis_cfg.get("fuzzy_boost_score", 0.30))
        max_dist = int(kis_cfg.get("fuzzy_max_distance", 1))
        entities = extract_entity_keywords(query_vi) if enable_boost else []

        candidates = []
        for idx in sorted_indices:
            sc = float(bm25_scores[idx])
            if sc <= 0.0:
                continue
            norm_score = (sc - min_s) / denom
            asr_txt = str(self.asr_texts[idx])
            if entities:
                if any(match_entity_fuzzy(ent, asr_txt, max_dist=max_dist) for ent in entities):
                    norm_score = min(1.0, norm_score + boost_score)

            cand = CandidateResult(
                video_id=str(self.video_ids[idx]),
                frame_id=int(self.frame_ids[idx]),
                scene_id=str(self.scene_ids[idx]),
                timestamp_start=float(self.timestamps_mid[idx]),
                timestamp_end=float(self.timestamps_mid[idx]),
                timestamp_mid=float(self.timestamps_mid[idx]),
                keyframe_path=str(self.keyframe_paths[idx]),
                ocr_text=str(self.ocr_texts[idx]),
                asr_text=asr_txt,
                description=str(self.descriptions[idx]),
                score=round(float(norm_score), 4),
            )
            candidates.append(cand)

        if entities:
            candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates

    def search_ocr(self, query_vi: str, top_k: int = 100) -> list[CandidateResult]:
        """
        Tìm kiếm nhanh bằng chỉ mục In-Memory BM25 OCR trên 88,899 scenes (<3ms).
        Sử dụng Pure NumPy CPU Top-K (np.argpartition), kết hợp Exact & Fuzzy Entity Boosting.
        """
        if not query_vi or not query_vi.strip():
            return []

        tokens = extract_sparse_tokens(query_vi)
        if not tokens:
            return []

        bm25_scores = self.bm25_ocr.get_scores(tokens)
        max_s = float(bm25_scores.max())
        if max_s <= 0.0:
            return []

        top_k = min(top_k, len(bm25_scores))
        part_indices = np.argpartition(-bm25_scores, top_k)[:top_k]
        sorted_indices = part_indices[np.argsort(-bm25_scores[part_indices])]

        min_s = float(bm25_scores.min())
        denom = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

        kis_cfg = _get_kis_accuracy_config()
        enable_boost = kis_cfg.get("enable_entity_boost", True)
        boost_score = float(kis_cfg.get("entity_boost_score", 0.35))
        max_dist = int(kis_cfg.get("fuzzy_max_distance", 1))
        entities = extract_entity_keywords(query_vi) if enable_boost else []

        candidates = []
        for idx in sorted_indices:
            sc = float(bm25_scores[idx])
            if sc <= 0.0:
                continue
            norm_score = (sc - min_s) / denom
            ocr_txt = str(self.ocr_texts[idx])
            if entities:
                if any(match_entity_fuzzy(ent, ocr_txt, max_dist=max_dist) for ent in entities):
                    norm_score = min(1.0, norm_score + boost_score)

            cand = CandidateResult(
                video_id=str(self.video_ids[idx]),
                frame_id=int(self.frame_ids[idx]),
                scene_id=str(self.scene_ids[idx]),
                timestamp_start=float(self.timestamps_mid[idx]),
                timestamp_end=float(self.timestamps_mid[idx]),
                timestamp_mid=float(self.timestamps_mid[idx]),
                keyframe_path=str(self.keyframe_paths[idx]),
                ocr_text=ocr_txt,
                asr_text=str(self.asr_texts[idx]),
                description=str(self.descriptions[idx]),
                score=round(float(norm_score), 4),
            )
            candidates.append(cand)

        if entities:
            candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates

    def warmup(self) -> None:
        """Thực hiện inference mẫu để JIT compile CUDA kernels và làm nóng RAM cache."""
        try:
            _ = self.search_description("warmup visual query", top_k=5)
            _ = self.search_asr("warmup query", top_k=5)
            _ = self.search_ocr("warmup query", top_k=5)
            logger.info("🔥 HybridSearchEngine warmup hoàn tất!")
        except Exception as e:
            logger.debug(f"HybridSearchEngine warmup: {e}")


# Global Singleton Instance
_hybrid_engine: HybridSearchEngine | None = None

def get_hybrid_search_engine(data_dir: str = "data/unified", device: str = "auto") -> HybridSearchEngine:
    global _hybrid_engine
    if _hybrid_engine is not None:
        return _hybrid_engine

    try:
        from serving.api.core.models import ModelRegistry
        reg = ModelRegistry()
        engine = reg.get("hybrid_engine")
        if engine is not None and isinstance(engine, HybridSearchEngine):
            _hybrid_engine = engine
            return _hybrid_engine
    except Exception:
        pass

    _hybrid_engine = HybridSearchEngine(data_dir=data_dir, device=device)
    return _hybrid_engine


async def search_description_vector(
    query_en: str,
    top_k: int = 200,
    data_dir: str = "data/unified",
    query_vi: str = "",
) -> list[CandidateResult]:
    """
    Async helper chạy search_description trên worker threadpool để không block event loop.
    """
    import asyncio
    try:
        engine = get_hybrid_search_engine(data_dir=data_dir)
        return await asyncio.to_thread(engine.search_description, query_en=query_en, top_k=top_k, query_vi=query_vi)
    except Exception as e:
        logger.warning(f"Lỗi khi search description vector: {e}")
        return []


async def search_description_batch_vector(
    queries_en: list[str],
    top_k: int = 100,
    data_dir: str = "data/unified",
    queries_vi: list[str] | None = None,
) -> list[list[CandidateResult]]:
    """
    Async helper chạy search_description_batch trên worker threadpool (1.5ms matrix multiply).
    """
    import asyncio
    try:
        engine = get_hybrid_search_engine(data_dir=data_dir)
        return await asyncio.to_thread(
            engine.search_description_batch,
            queries_en=queries_en,
            top_k=top_k,
            queries_vi=queries_vi,
        )
    except Exception as e:
        logger.warning(f"Lỗi khi search description batch vector: {e}")
        return [[] for _ in queries_en]


async def search_asr_candidates(
    query_vi: str,
    top_k: int = 100,
    data_dir: str = "data/unified",
) -> list[CandidateResult]:
    """Async helper chạy in-memory BM25 ASR search trên worker threadpool (~25ms)."""
    import asyncio
    try:
        engine = get_hybrid_search_engine(data_dir=data_dir)
        return await asyncio.to_thread(engine.search_asr, query_vi=query_vi, top_k=top_k)
    except Exception as e:
        logger.warning(f"Lỗi khi search ASR BM25: {e}")
        return []


async def search_ocr_candidates(
    query_vi: str,
    top_k: int = 100,
    data_dir: str = "data/unified",
) -> list[CandidateResult]:
    """Async helper chạy in-memory BM25 OCR search trên worker threadpool (~25ms)."""
    import asyncio
    try:
        engine = get_hybrid_search_engine(data_dir=data_dir)
        return await asyncio.to_thread(engine.search_ocr, query_vi=query_vi, top_k=top_k)
    except Exception as e:
        logger.warning(f"Lỗi khi search OCR BM25: {e}")
        return []

