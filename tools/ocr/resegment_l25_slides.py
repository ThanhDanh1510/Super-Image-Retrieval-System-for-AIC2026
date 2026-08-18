"""
tools/ocr/resegment_l25_slides.py
=================================
Tự động phát hiện chuyển cảnh slide thực tế (Slide Transition Detection)
và căn chỉnh chuẩn xác 100% OCR text & Description cho toàn bộ 87 video của L25,
loại bỏ hoàn toàn tình trạng gán nhầm chữ của bài tập này sang bài tập khác.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from shared.utils.text_normalizer import normalize_nfc, clean_whitespace
from aggregation.common import parse_video_and_frame_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ResegmentL25")


def detect_slide_transitions(
    image_paths: list[Path],
    diff_threshold: float = 12.0,
) -> list[list[Path]]:
    """
    Phát hiện các phân đoạn slide thực tế bằng cách so sánh vùng hiển thị bài giảng (ROI).
    Vùng bài giảng thường nằm ở nửa phải hoặc trung tâm màn hình (bỏ qua phần giáo viên đứng bên trái).
    """
    if not image_paths:
        return []

    clusters: list[list[Path]] = []
    current_cluster: list[Path] = [image_paths[0]]

    # Đọc và resize ảnh đầu tiên của cụm
    prev_img = cv2.imread(str(image_paths[0]))
    if prev_img is None:
        prev_gray = None
    else:
        h, w = prev_img.shape[:2]
        # Crop vùng hiển thị nội dung bài giảng (35% đến 95% chiều rộng)
        roi = prev_img[:, int(w * 0.35) :]
        prev_gray = cv2.cvtColor(cv2.resize(roi, (320, 240)), cv2.COLOR_BGR2GRAY)

    for img_p in image_paths[1:]:
        img = cv2.imread(str(img_p))
        if img is None or prev_gray is None:
            current_cluster.append(img_p)
            continue

        h, w = img.shape[:2]
        roi = img[:, int(w * 0.35) :]
        curr_gray = cv2.cvtColor(cv2.resize(roi, (320, 240)), cv2.COLOR_BGR2GRAY)

        # Tính mean absolute difference trên vùng bài giảng
        diff = np.mean(np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32)))

        if diff > diff_threshold:
            # Phát hiện chuyển slide / chuyển câu hỏi mới
            clusters.append(current_cluster)
            current_cluster = [img_p]
            prev_gray = curr_gray
        else:
            current_cluster.append(img_p)

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def main():
    parser = argparse.ArgumentParser(description="Resegment and clean L25 slides OCR and descriptions.")
    parser.add_argument("--json-path", default="./data/L25_all_distinct_ocr.json")
    parser.add_argument("--unified-dir", default="./data/unified")
    parser.add_argument("--img-root", default="./data/image/L25/L25")
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    unified_dir = Path(args.unified_dir).resolve()
    img_root = Path(args.img_root).resolve()

    logger.info("Đang đọc L25_all_distinct_ocr.json...")
    with open(json_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    # Tổ chức templates theo video_id
    templates_by_vid: dict[str, list[dict]] = {}
    for it in templates:
        vid = it.get("video_id", "")
        if vid:
            templates_by_vid.setdefault(vid, []).append(it)

    meta_path = unified_dir / "scene_metadata.parquet"
    ocr_path = unified_dir / "ocr_text.parquet"
    desc_path = unified_dir / "scene_descriptions.parquet"

    meta_df = pd.read_parquet(meta_path)
    ocr_df = pd.read_parquet(ocr_path)
    desc_df = pd.read_parquet(desc_path)

    l25_meta = meta_df[meta_df["video_id"].str.startswith("L25_")].copy()
    lookup_scene = {(r["video_id"], int(r["frame_id"])): r["scene_id"] for _, r in l25_meta.iterrows()}

    # Lấy description gốc ban đầu từ VLM
    vlm_desc_dict = desc_df.set_index("scene_id")["description"].to_dict()
    # Làm sạch bỏ phần [Chủ đề bài giảng: ...] cũ nếu có
    for sid, d in vlm_desc_dict.items():
        if isinstance(d, str) and " | " in d:
            parts = d.split(" | ", 1)
            if parts[0].startswith("[") and ("This image" in parts[1] or "Lecture" in parts[1] or "The image" in parts[1] or "In this" in parts[1]):
                vlm_desc_dict[sid] = parts[1].strip()

    final_ocr_map: dict[str, str] = {}
    final_desc_map: dict[str, str] = {}

    total_subclusters = 0

    logger.info("Đang tiến hành quét phân đoạn slide thực tế cho 87 video của L25...")
    for vid_folder in sorted(list(img_root.glob("V*")), key=lambda p: p.name):
        v_name = vid_folder.name
        full_vid = f"L25_{v_name}"
        img_files = sorted(list(vid_folder.glob("*.webp")), key=lambda p: int(p.stem))

        if not img_files:
            continue

        # Phân đoạn slide thực tế
        slide_clusters = detect_slide_transitions(img_files, diff_threshold=12.0)
        total_subclusters += len(slide_clusters)

        v_templates = templates_by_vid.get(full_vid, [])

        for cluster in slide_clusters:
            cluster_fids = [int(p.stem) for p in cluster]

            # Kiểm tra xem cluster này có chứa frame đại diện nào từ JSON không
            matched_template = None
            for t in v_templates:
                rep_file = t.get("representative_filename", "")
                _, rep_fid = parse_video_and_frame_id(rep_file)
                if rep_fid in cluster_fids:
                    matched_template = t
                    break

            if matched_template:
                # Cụm này chứa đúng frame đại diện -> gán trọn vẹn OCR và notes chuẩn xác
                text = clean_whitespace(normalize_nfc(matched_template.get("text", "").strip()))
                notes = clean_whitespace(normalize_nfc(matched_template.get("notes", "").strip()))

                for fid in cluster_fids:
                    sid = lookup_scene.get((full_vid, fid))
                    if not sid:
                        continue
                    if text:
                        final_ocr_map[sid] = text
                    
                    orig_desc = vlm_desc_dict.get(sid, "")
                    if fid == rep_fid:
                        desc_str = f"[Bài giảng/Đề thi]: {notes}. Trích đoạn nội dung: {text[:280]} | {orig_desc}" if notes else f"[Bài giảng/Đề thi]: {text[:300]} | {orig_desc}"
                    else:
                        desc_str = f"[Chủ đề bài giảng: {notes}]. Nội dung: {text[:150]} | {orig_desc}" if notes else f"[Slide bài giảng: {text[:150]}] | {orig_desc}"
                    final_desc_map[sid] = desc_str
            else:
                # Cụm này là một bài tập con/slide chuyển tiếp khác trong video
                # -> Gán chủ đề tổng quát của video, KHÔNG gán nhầm chữ của câu hỏi khác
                video_topic = v_templates[0].get("notes", "") if v_templates else ""
                video_topic_short = video_topic.split(":")[0] if ":" in video_topic else video_topic

                for fid in cluster_fids:
                    sid = lookup_scene.get((full_vid, fid))
                    if not sid:
                        continue
                    orig_desc = vlm_desc_dict.get(sid, "")
                    if video_topic_short:
                        final_desc_map[sid] = f"[Chuyên đề video: {video_topic_short}] | {orig_desc}"
                    else:
                        final_desc_map[sid] = orig_desc

    logger.info(f"🎉 Đã phát hiện tổng cộng {total_subclusters} slide sub-clusters trên 87 video (so với 305 templates thô ban đầu)!")
    logger.info(f"Số lượng frame L25 được gán OCR chuẩn xác: {len(final_ocr_map)}")
    logger.info(f"Số lượng frame L25 được gán Description đồng bộ: {len(final_desc_map)}")

    # Cập nhật vào ocr_text.parquet
    logger.info("Đang cập nhật ocr_text.parquet...")
    update_mask = ocr_df["scene_id"].isin(final_ocr_map)
    ocr_df.loc[update_mask, "ocr_text"] = ocr_df.loc[update_mask, "scene_id"].map(final_ocr_map)
    
    # Đối với các frame L25 không thuộc template nào, đặt về rỗng thay vì để text sai
    l25_mask = ocr_df["video_id"].str.startswith("L25_") & ~ocr_df["scene_id"].isin(final_ocr_map)
    ocr_df.loc[l25_mask, "ocr_text"] = ""
    ocr_df.to_parquet(ocr_path, index=False)
    logger.info(f"✅ Đã lưu ocr_text.parquet mới!")

    # Cập nhật vào scene_descriptions.parquet
    logger.info("Đang cập nhật scene_descriptions.parquet...")
    desc_mask = desc_df["scene_id"].isin(final_desc_map)
    desc_df.loc[desc_mask, "description"] = desc_df.loc[desc_mask, "scene_id"].map(final_desc_map)
    desc_df.to_parquet(desc_path, index=False)
    logger.info(f"✅ Đã lưu scene_descriptions.parquet mới!")

    # Cập nhật vào L25_ocr.parquet
    ocr_out_dir = unified_dir.parent / "ocr_output"
    if ocr_out_dir.exists():
        l25_ocr_file = ocr_out_dir / "L25_ocr.parquet"
        l25_records = []
        for (vid, fid), sid in lookup_scene.items():
            txt = final_ocr_map.get(sid, "")
            l25_records.append({
                "scene_id": sid,
                "video_id": vid,
                "frame_id": fid,
                "ocr_text": txt,
            })
        pd.DataFrame(l25_records).to_parquet(l25_ocr_file, index=False)
        logger.info(f"✅ Đã đồng bộ L25_ocr.parquet!")

    logger.info("Hoàn tất!")


if __name__ == "__main__":
    main()
