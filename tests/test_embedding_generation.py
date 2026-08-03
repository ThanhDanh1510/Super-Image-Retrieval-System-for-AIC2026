import os
import importlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper
from shared.models.text_embedding_wrapper import TextEmbeddingWrapper

emb_gen_mod = importlib.import_module("ingestion.skeletons.01c_embedding_generation")
run_embedding_generation = emb_gen_mod.run_embedding_generation


def test_image_embedding_wrapper_shape_and_norm():
    wrapper = ImageEmbeddingWrapper(device="cpu")
    assert wrapper.dim == 1024

    # Single image test
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    vec = wrapper.encode(img)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1024,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)

    # Batch images test
    imgs = [Image.new("RGB", (50, 50), color=(i * 10, i * 20, i * 30)) for i in range(5)]
    vecs = wrapper.encode(imgs)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (5, 1024)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_text_embedding_wrapper_shape_and_norm():
    wrapper = TextEmbeddingWrapper(device="cpu")
    assert wrapper.dim == 1536

    # Single text test
    text = "Cảnh quay xe máy di chuyển trên đường phố"
    vec = wrapper.encode(text)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1536,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)

    # Batch text test with empty strings
    texts = ["Văn bản 1", "", "   ", "Văn bản 4"]
    vecs = wrapper.encode(texts)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (4, 1536)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_run_embedding_generation_pipeline(tmp_path):
    # 1. Fake metadata
    meta_df = pd.DataFrame([
        {"scene_id": "L21_V001_S001", "video_id": "L21_V001", "keyframe_path": "L21_V001/L21_V001_S001.jpg"},
        {"scene_id": "L21_V001_S002", "video_id": "L21_V001", "keyframe_path": "L21_V001/L21_V001_S002.jpg"},
    ])
    meta_p = tmp_path / "scene_metadata.parquet"
    meta_df.to_parquet(meta_p, index=False)

    # 2. Fake OCR
    ocr_df = pd.DataFrame([
        {"scene_id": "L21_V001_S001", "ocr_text": "TIN CHÍNH 60 GIÂY"},
        {"scene_id": "L21_V001_S002", "ocr_text": ""},
    ])
    ocr_p = tmp_path / "ocr_text.parquet"
    ocr_df.to_parquet(ocr_p, index=False)

    # 3. Fake ASR
    asr_df = pd.DataFrame([
        {"scene_id": "L21_V001_S001", "asr_text": "Lời thoại âm thanh video"},
        {"scene_id": "L21_V001_S002", "asr_text": "Cảnh quay không gian"},
    ])
    asr_p = tmp_path / "asr_text.parquet"
    asr_df.to_parquet(asr_p, index=False)

    # 4. Fake keyframes
    kf_dir = tmp_path / "keyframes"
    v_dir = kf_dir / "L21_V001"
    v_dir.mkdir(parents=True)
    
    img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))
    img1.save(v_dir / "L21_V001_S001.jpg")
    img2.save(v_dir / "L21_V001_S002.jpg")

    out_dir = tmp_path / "output"

    run_embedding_generation(
        metadata_path=str(meta_p),
        ocr_path=str(ocr_p),
        asr_path=str(asr_p),
        keyframe_dir=str(kf_dir),
        output_dir=str(out_dir),
        device="cpu",
    )

    # Verify generated output files
    assert (out_dir / "image_vectors.npy").exists()
    assert (out_dir / "ocr_vectors.npy").exists()
    assert (out_dir / "asr_vectors.npy").exists()
    assert (out_dir / "scene_ids.txt").exists()

    img_v = np.load(out_dir / "image_vectors.npy")
    ocr_v = np.load(out_dir / "ocr_vectors.npy")
    asr_v = np.load(out_dir / "asr_vectors.npy")

    assert img_v.shape == (2, 1024)
    assert ocr_v.shape == (2, 1536)
    assert asr_v.shape == (2, 1536)

    # Check L2-normalization
    assert np.allclose(np.linalg.norm(img_v, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(ocr_v, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(asr_v, axis=1), 1.0, atol=1e-3)

    # Check scene_ids.txt lines
    with open(out_dir / "scene_ids.txt", "r", encoding="utf-8") as f:
        sids = [line.strip() for line in f if line.strip()]
    assert sids == ["L21_V001_S001", "L21_V001_S002"]
