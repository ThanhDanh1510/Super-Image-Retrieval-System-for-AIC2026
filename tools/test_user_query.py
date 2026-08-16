import requests
import json

def main():
    url = "http://127.0.0.1:8000/api/v1/kis"
    payload = {
        "input_type": "text",
        "query": "Đoạn clip bắt đầu với cảnh một tác phẩm điêu khắc cát hoành tráng tại Lễ hội điêu khắc trên cát. Tác phẩm mô tả cảnh những thanh niên chơi thể thao đường phố: một người trượt patin, hai người trượt ván. Nền phía sau là họa tiết vòm cong lớn và có khắc chữ. Tiếp theo có cảnh nhiều tác phẩm khác bằng cát, và có 2 cột khói màu hồng",
        "search_mode": "all",
        "top_k": 10,
        "user_choices": [],
        "use_reranker": False
    }

    print("Sending query to API...", flush=True)
    resp = requests.post(url, json=payload, timeout=30)
    print(f"Status code: {resp.status_code}", flush=True)
    
    if resp.status_code == 200:
        data = resp.json()
        cands = data.get("results", [])
        print(f"Found {len(cands)} candidates:\n", flush=True)
        for idx, c in enumerate(cands, 1):
            vid = c.get("video_id")
            fid = c.get("frame_id")
            sc = c.get("score", 0)
            ts = c.get("timestamp_mid", 0)
            ocr = c.get("ocr_text", "")
            print(f"  #{idx:02d}: Video={vid:<8} | Frame=f{fid:<6} | Score={sc*100:5.1f}% | Time={ts:7.2f}s | OCR={ocr[:30]}", flush=True)

if __name__ == "__main__":
    main()
