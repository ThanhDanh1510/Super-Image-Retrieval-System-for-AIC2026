import os
import torch
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel

class HybridSearchEngine:
    """
    Hybrid Search Engine implementing Weighted Convex + RRF Fusion
    across Description (Dense BGE), OCR (Sparse BM25), and ASR (Sparse BM25).
    """

    def __init__(self, data_dir: str, device: str = "cuda"):
        self.data_dir = data_dir
        self.device = device if torch.cuda.is_available() else "cpu"

        print(f"⚡ Initializing HybridSearchEngine from '{data_dir}' on device '{self.device}'...")

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

        # Load description vector embeddings
        embed_path = os.path.join(data_dir, "description_embeddings.pt")
        if not os.path.exists(embed_path):
            raise FileNotFoundError(f"Description embeddings file not found: {embed_path}")

        print(f"📥 Loading description embeddings tensor from '{embed_path}'...")
        self.embeds = torch.load(embed_path, map_location=self.device)
        if self.embeds.dtype == torch.float16:
            self.embeds = self.embeds.float()

        # Strict 1-to-1 row index assertion
        assert len(self.embeds) == len(self.df), f"Row count mismatch: embeds ({len(self.embeds)}) vs df ({len(self.df)})"

        # Load BGE-Large-EN v1.5 Model & Tokenizer
        self.model_id = "BAAI/bge-large-en-v1.5"
        print(f"📥 Loading BGE model '{self.model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.bge_model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.bge_model.eval()

        # Build Sparse BM25 Indexes
        print("⚡ Building BM25 Index for OCR text...")
        ocr_corpus = [doc.lower().split() for doc in self.df["ocr_text"]]
        self.bm25_ocr = BM25Okapi(ocr_corpus)

        print("⚡ Building BM25 Index for ASR text...")
        asr_corpus = [doc.lower().split() for doc in self.df["asr_text"]]
        self.bm25_asr = BM25Okapi(asr_corpus)

        print(f"✅ HybridSearchEngine successfully initialized for {len(self.df):,d} scenes!")

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
                q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).float()

            raw_desc = torch.matmul(self.embeds, q_emb.T).squeeze(-1)
            norm_desc = self._normalize(raw_desc)
            final_scores += w_desc * norm_desc

            # Compute ranks (1-indexed) for RRF
            sorted_indices = torch.argsort(raw_desc, descending=True)
            r_desc = torch.zeros(num_items, device=self.device, dtype=torch.float32)
            r_desc[sorted_indices] = torch.arange(1, num_items + 1, device=self.device, dtype=torch.float32)

        # 2. Sparse OCR Stream (BM25Okapi)
        r_ocr = None
        if w_ocr > 0 or lam > 0:
            tokens = query_vi.lower().split()
            bm25_scores = self.bm25_ocr.get_scores(tokens)
            raw_ocr = torch.from_numpy(bm25_scores).to(self.device, dtype=torch.float32)
            norm_ocr = self._normalize(raw_ocr)
            final_scores += w_ocr * norm_ocr

            sorted_indices = torch.argsort(raw_ocr, descending=True)
            r_ocr = torch.zeros(num_items, device=self.device, dtype=torch.float32)
            r_ocr[sorted_indices] = torch.arange(1, num_items + 1, device=self.device, dtype=torch.float32)

        # 3. Sparse ASR Stream (BM25Okapi)
        if w_asr > 0:
            tokens = query_vi.lower().split()
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


# Global Singleton Instance
_hybrid_engine: HybridSearchEngine | None = None

def get_hybrid_search_engine(data_dir: str = "data/unified") -> HybridSearchEngine:
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridSearchEngine(data_dir=data_dir)
    return _hybrid_engine
