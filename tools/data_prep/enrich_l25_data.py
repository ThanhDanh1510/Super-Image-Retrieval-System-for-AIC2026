"""
tools/data_prep/enrich_l25_data.py
==================================
Nâng cấp dữ liệu OCR, làm giàu mô tả ngữ nghĩa (Descriptions),
và trích xuất danh sách ảnh trùng lặp slide cho batch L25 từ data/L25_all_distinct_ocr.json.

CÁCH CHẠY:
    # 1. Làm giàu dữ liệu OCR + Description cho L25:
    python -m tools.data_prep.enrich_l25_data --enrich-metadata

    # 2. Làm giàu dữ liệu ĐỒNG THỜI xuất danh sách ảnh trùng slide vào filtered_keyframes.json:
    python -m tools.data_prep.enrich_l25_data --enrich-metadata --export-duplicates

    # 3. Xem trước thống kê (Dry-run):
    python -m tools.data_prep.enrich_l25_data --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
import pandas as pd

from shared.utils.text_normalizer import normalize_nfc, clean_whitespace
from aggregation.common import parse_video_and_frame_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("EnrichL25")


def parse_args():
    parser = argparse.ArgumentParser(description="Enrich L25 OCR and Descriptions from distinct templates JSON.")
    parser.add_argument(
        "--json-path",
        default="./data/L25_all_distinct_ocr.json",
        help="Đường dẫn file L25_all_distinct_ocr.json",
    )
    parser.add_argument(
        "--unified-dir",
        default="./data/unified",
        help="Thư mục chứa unified dataset (scene_metadata.parquet, ocr_text.parquet, ...)",
    )
    parser.add_argument(
        "--filter-json",
        default="./data/filtered_keyframes.json",
        help="Đường dẫn file filtered_keyframes.json",
    )
    parser.add_argument(
        "--enrich-metadata",
        action="store_true",
        help="Cập nhật trực tiếp vào ocr_text.parquet và scene_descriptions.parquet",
    )
    parser.add_argument(
        "--export-duplicates",
        action="store_true",
        help="Xuất và hợp nhất các frame trùng slide vào filtered_keyframes.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in báo cáo thống kê, không ghi file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json_path).resolve()
    unified_dir = Path(args.unified_dir).resolve()
    filter_json_path = Path(args.filter_json).resolve()

    if not json_path.exists():
        logger.error(f"Không tìm thấy file: {json_path}")
        return

    # 1. Đọc L25 distinct OCR JSON
    logger.info(f"Đang đọc {json_path.name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    logger.info(f"Loaded {len(templates)} slide templates từ {json_path.name}")

    # 2. Đọc ground-truth metadata
    meta_path = unified_dir / "scene_metadata.parquet"
    if not meta_path.exists():
        logger.error(f"Không tìm thấy {meta_path}")
        return

    meta_df = pd.read_parquet(meta_path)
    l25_meta = meta_df[meta_df["video_id"].str.startswith("L25_")].copy()
    logger.info(f"Tổng số frame L25 trong Unified DB hiện tại: {len(l25_meta)}")

    # Tạo mapping (video_id, frame_id) -> scene_id
    lookup_scene: dict[tuple[str, int], str] = {}
    for _, row in l25_meta.iterrows():
        lookup_scene[(row["video_id"], int(row["frame_id"]))] = row["scene_id"]

    # 3. Phân tích templates: frame đại diện vs frame trùng lặp
    ocr_updates: dict[str, str] = {}  # scene_id -> ocr_text
    desc_updates: dict[str, str] = {}  # scene_id -> enriched_description
    duplicate_entries: list[dict] = []

    matched_rep_count = 0
    matched_dup_count = 0

    for t_idx, item in enumerate(templates):
        video_id = item.get("video_id", "")
        text = clean_whitespace(normalize_nfc(item.get("text", "").strip()))
        notes = clean_whitespace(normalize_nfc(item.get("notes", "").strip()))
        rep_file = item.get("representative_filename", "")
        all_frames = item.get("all_matching_frames", [])

        # Xác định frame_id đại diện
        _, rep_fid = parse_video_and_frame_id(rep_file)

        # Duyệt qua tất cả các matching frames của slide này
        for f_path in all_frames:
            _, f_id = parse_video_and_frame_id(f_path)
            if f_id is None:
                continue

            scene_id = lookup_scene.get((video_id, f_id))
            if not scene_id:
                # Frame đã bị lọc từ trước -> tự động bỏ qua (Ground-truth alignment)
                continue

            is_rep = (f_id == rep_fid)

            # Cập nhật OCR
            if text:
                ocr_updates[scene_id] = text

            # Làm giàu Description
            if is_rep:
                matched_rep_count += 1
                desc_text = f"[Bài giảng/Đề thi]: {notes}. Trích đoạn nội dung: {text[:280]}" if notes else f"[Bài giảng/Đề thi]: {text[:300]}"
                desc_updates[scene_id] = desc_text
            else:
                matched_dup_count += 1
                dup_desc = f"[Chủ đề bài giảng: {notes}]. Nội dung: {text[:150]}" if notes else f"[Slide bài giảng: {text[:150]}]"
                desc_updates[scene_id] = dup_desc

                # Đóng gói entry duplicate để xuất vào filtered_keyframes.json
                v_short = video_id.replace("L25_", "")
                duplicate_entries.append({
                    "batch": "L25",
                    "video_id": v_short,
                    "full_video_id": video_id,
                    "filename": f"{f_id}.webp",
                    "frame_id": f_id,
                    "keyframe_path": f"L25/{v_short}/{f_id}.webp",
                    "scene_id": scene_id,
                    "template_id": t_idx,
                    "reason": "duplicate_slide_content",
                    "template_notes": notes[:120],
                })

    logger.info(f"Đã khớp thành công:")
    logger.info(f"  - Frame đại diện (Representative): {matched_rep_count}")
    logger.info(f"  - Frame lặp lại (Duplicate slide frames): {matched_dup_count}")
    logger.info(f"  - Tổng số frame L25 được làm giàu: {len(ocr_updates)} / {len(l25_meta)} (100%)")

    # 4. Xuất / Hợp nhất vào filtered_keyframes.json (nếu có --export-duplicates)
    if args.export_duplicates:
        logger.info(f"Đang chuẩn bị hợp nhất {len(duplicate_entries)} duplicate entries vào {filter_json_path.name}...")
        if filter_json_path.exists():
            with open(filter_json_path, "r", encoding="utf-8") as f:
                filter_master = json.load(f)
        else:
            filter_master = {"total_filtered_count": 0, "summary_by_batch": {}, "filtered_images": []}

        existing_filters = filter_master.get("filtered_images", [])
        existing_keys = set()
        for e in existing_filters:
            existing_keys.add((str(e.get("batch")), str(e.get("video_id")), int(e.get("frame_id", -1))))
            existing_keys.add((str(e.get("full_video_id")), int(e.get("frame_id", -1))))

        added_count = 0
        for dup in duplicate_entries:
            key1 = (dup["batch"], dup["video_id"], dup["frame_id"])
            key2 = (dup["full_video_id"], dup["frame_id"])
            if key1 not in existing_keys and key2 not in existing_keys:
                existing_filters.append(dup)
                existing_keys.add(key1)
                existing_keys.add(key2)
                added_count += 1

        # Cập nhật summary
        summary_by_batch = {}
        for item in existing_filters:
            b = str(item.get("batch", "UNKNOWN"))
            summary_by_batch[b] = summary_by_batch.get(b, 0) + 1

        filter_master["total_filtered_count"] = len(existing_filters)
        filter_master["summary_by_batch"] = summary_by_batch
        filter_master["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if not args.dry_run:
            with open(filter_json_path, "w", encoding="utf-8") as f:
                json.dump(filter_master, f, ensure_ascii=False, indent=2)
            logger.info(f"🎉 Đã thêm mới {added_count} duplicate entries vào {filter_json_path.name}! (Tổng hiện tại: {len(existing_filters)})")
        else:
            logger.info(f"[DRY-RUN] Dự kiến thêm mới {added_count} duplicate entries vào {filter_json_path.name}")

    # 5. Cập nhật Parquet metadata (nếu có --enrich-metadata)
    if args.enrich_metadata and not args.dry_run:
        # 5.1 ocr_text.parquet
        ocr_path = unified_dir / "ocr_text.parquet"
        if ocr_path.exists():
            logger.info("Đang cập nhật ocr_text.parquet...")
            ocr_df = pd.read_parquet(ocr_path)
            # Áp dụng updates
            update_mask = ocr_df["scene_id"].isin(ocr_updates)
            ocr_df.loc[update_mask, "ocr_text"] = ocr_df.loc[update_mask, "scene_id"].map(ocr_updates)
            ocr_df.to_parquet(ocr_path, index=False)
            non_empty_count = (ocr_df["ocr_text"].fillna("").str.strip() != "").sum()
            logger.info(f"✅ Đã lưu ocr_text.parquet mới! (Non-empty OCR toàn hệ thống: {non_empty_count}/{len(ocr_df)})")

        # 5.2 scene_descriptions.parquet
        desc_path = unified_dir / "scene_descriptions.parquet"
        if desc_path.exists():
            logger.info("Đang cập nhật scene_descriptions.parquet...")
            desc_df = pd.read_parquet(desc_path)
            update_mask = desc_df["scene_id"].isin(desc_updates)
            
            # Kết hợp mô tả gốc VLM + mô tả bài giảng
            for idx in desc_df[update_mask].index:
                sid = desc_df.loc[idx, "scene_id"]
                orig = str(desc_df.loc[idx, "description"] or "").strip()
                enr = desc_updates[sid]
                desc_df.loc[idx, "description"] = f"{enr} | {orig}" if orig else enr

            desc_df.to_parquet(desc_path, index=False)
            logger.info(f"✅ Đã lưu scene_descriptions.parquet mới với ngữ nghĩa bài giảng phong phú!")

        # 5.3 data/ocr_output/L25_ocr.parquet
        ocr_out_dir = unified_dir.parent / "ocr_output"
        if ocr_out_dir.exists():
            l25_ocr_file = ocr_out_dir / "L25_ocr.parquet"
            l25_records = []
            for (vid, fid), sid in lookup_scene.items():
                txt = ocr_updates.get(sid, "")
                l25_records.append({
                    "scene_id": sid,
                    "video_id": vid,
                    "frame_id": fid,
                    "ocr_text": txt,
                })
            l25_out_df = pd.DataFrame(l25_records)
            l25_out_df.to_parquet(l25_ocr_file, index=False)
            logger.info(f"✅ Đã đồng bộ {l25_ocr_file.name}: {len(l25_out_df)} rows")

    logger.info("Hoàn tất!")


if __name__ == "__main__":
    main()
