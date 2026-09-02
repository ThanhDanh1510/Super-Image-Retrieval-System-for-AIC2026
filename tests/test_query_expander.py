"""
tests/test_query_expander.py
============================
Unit tests cho PromptEnsembleExpander (CLIP Prompt Ensembling).
"""

from unittest.mock import MagicMock
import numpy as np
from serving.api.service.retrieval.query_expander import PromptEnsembleExpander


def test_prompt_variants_generation():
    q = "a woman wearing white ao dai"
    variants = PromptEnsembleExpander.generate_clip_prompt_variants(q)
    assert len(variants) == 3
    assert variants[0] == "a woman wearing white ao dai"
    assert "a photo of" in variants[1]
    assert "clear visual shot" in variants[2]


def test_ensemble_text_embeddings_normalization():
    mock_embedder = MagicMock()
    # Mock trả về 2D array (3 variants x 4 dims)
    mock_embedder.encode_text.return_value = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
    ], dtype=np.float32)

    fused = PromptEnsembleExpander.ensemble_text_embeddings("test query", mock_embedder)
    assert len(fused) == 4
    # Vector phải được chuẩn hóa L2 (norm == 1.0)
    norm = np.linalg.norm(np.array(fused))
    assert abs(norm - 1.0) < 1e-4
