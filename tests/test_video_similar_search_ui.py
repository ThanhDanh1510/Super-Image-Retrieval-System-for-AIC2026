from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_video_modal_exposes_similar_search_and_exact_frame_seek():
    app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

    assert 'id="video-search-similar-btn"' in index
    assert "searchSimilarToCurrentFrame" in app
    assert "/api/v1/similar-image?top_k=" in app
    assert "cand.frame_id / fps" in app
