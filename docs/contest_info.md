# HỘI THI THỬ THÁCH TRÍ TUỆ NHÂN TẠO THÀNH PHỐ HỒ CHÍ MINH NĂM 2026

Tài liệu này tổng hợp thông tin thể lệ, dữ liệu và phương pháp đánh giá vòng sơ tuyển dựa trên văn bản chính thức của cuộc thi.

---

## 1. NỘI DUNG CÁC TRUY VẤN VÒNG SƠ TUYỂN

Vòng sơ tuyển gồm 3 dạng truy vấn chính, được cung cấp sẵn và trọn vẹn dưới dạng văn bản tiếng Việt.

### 1.1. Truy vấn dạng 1: Tìm kiếm chính xác theo văn bản (Textual Known Item Search – Textual KIS)
Nhiệm vụ tìm kiếm sự kiện dựa trên mô tả bằng văn bản.
* **Nội dung truy vấn:** Mô tả bằng ngôn ngữ tự nhiên về một sự kiện. Cần định vị chính xác đoạn video chứa sự kiện bằng cách chỉ ra một khung hình bất kỳ thuộc đoạn video đó.
* **Định dạng câu trả lời nộp:** `<video_id>, <frame_id>`
* **Ví dụ:** *"Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh."*
  * **Kết quả nộp:** `video_id = video_abc(.mp4)`, `frame_id = 1500`.

### 1.2. Truy vấn dạng 2: Truy vấn dạng Hỏi–Đáp (Q&A / Visual Question Answering)
Nhiệm vụ tìm kiếm sự kiện và trích xuất thông tin cụ thể từ video.
* **Nội dung truy vấn:** Mô tả bằng ngôn ngữ tự nhiên của một sự kiện và một câu hỏi về thông tin trong sự kiện này. Cần tìm ra chính xác khoảnh khắc liên quan và câu trả lời (tiếng Việt hoặc tiếng Anh).
* **Định dạng câu trả lời nộp:** `<video_id>, <frame_id>, <answer>`
* **Ví dụ:** *"Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?"*
  * **Kết quả nộp:** `video_id = video_xyz(.mp4)`, `frame_id = 3450`, `answer = "5"` hoặc `"Năm"`.

### 1.3. Truy vấn dạng 3: Truy xuất và căn chỉnh sự kiện video theo thời gian (Temporal Retrieval and Alignment of Key Events – TRAKE)
Nhiệm vụ phức hợp đòi hỏi độ chính xác cao trong cả việc truy xuất video và căn chỉnh thời gian của các khoảnh khắc quan trọng. Gồm hai giai đoạn:
* **Giai đoạn 1 – Truy xuất (Retrieval):** Tìm ra một video duy nhất chứa chuỗi sự kiện khớp nhất với truy vấn từ kho dữ liệu lớn.
* **Giai đoạn 2 – Căn chỉnh (Alignment):** Đối với video đã truy xuất, xác định chính xác một khung hình ngữ nghĩa duy nhất (semantic keyframe) cho mỗi giai đoạn của chuỗi sự kiện.
* **Định dạng câu trả lời nộp:** `<video_id>, <frame_id₁>, ..., <frame_idₙ>`
* **Lưu ý quan trọng:** *"Khung hình ngữ nghĩa"* (semantic keyframe) là khoảnh khắc mang ý nghĩa về nội dung (ví dụ: chân chạm đất, bay qua xà), hoàn toàn khác với *"I-Frame"* (khung hình kỹ thuật trong nén video).
* **Ví dụ – hành động "Nhảy cao" (chuỗi 4 khoảnh khắc):**
  * `Event 1 – Chạy đà (Approach)`: Khoảnh khắc bàn chân đầu tiên chạm đất và bước qua khỏi vạch xuất phát.
  * `Event 2 – Giậm nhảy (Take-off)`: Khoảnh khắc đầu tiên bàn chân của chân giậm nhảy rời hoàn toàn khỏi mặt đất.
  * `Event 3 – Bay qua xà (Clearance)`: Khoảnh khắc phần hông của vận động viên ở vị trí cao nhất so với xà ngang.
  * `Event 4 – Tiếp đất (Landing)`: Khoảnh khắc đầu tiên bất kỳ bộ phận nào của lưng (từ vai đến hông) bắt đầu chạm vào đệm.

---

## 2. PHƯƠNG PHÁP ĐÁNH GIÁ VÒNG SƠ TUYỂN

Với mỗi truy vấn, đội thi được gửi **tối đa 100 câu trả lời**.

### 2.1. Điểm Tương Quan (R-Score)
Thang đo độ chính xác nhận giá trị từ 0 đến 1 cho mỗi câu trả lời $r_i$ nộp lên:
* **KIS:** $R\text{-}Score(r_i) = I(v_i = GT_v \land id_i \in [s, e])$
  * Đúng video và frame_id nằm trong khoảng đáp án đúng $[s, e]$.
* **Q&A:** $R\text{-}Score(r_i) = I(v_i = GT_v \land id_i \in [s, e] \land a_i = GT_a)$
  * Đúng video, đúng khoảng frame, và câu trả lời khớp ngữ nghĩa với ground truth $GT_a$.
* **TRAKE:**
  * Nếu sai video ($v_i \neq GT_v$): $R\text{-}Score(r_i) = 0$.
  * Nếu đúng video ($v_i = GT_v$): $R\text{-}Score(r_i) = \frac{1}{N} \sum_{j=1}^{N} I(id_{i,j} \in [s_j, e_j])$
  * Điểm được tính bằng tỉ lệ khung hình khớp với đáp án ($N$ là tổng số khoảnh khắc). Khoảng đáp án đúng cho mỗi khoảnh khắc thường rất ngắn (dưới 10 frames).

### 2.2. Điểm Cuối Cùng (Final Score)
Điểm cuối cùng cho một truy vấn là trung bình cộng của **R@k** tại 5 ngưỡng $k \in \{1, 5, 20, 50, 100\}$:
* **R@k:** Điểm R-Score cao nhất đạt được trong $k$ câu trả lời đầu tiên:
  $$R@k = \max_{1 \le i \le k} \{ R\text{-}Score(r_i) \}$$
* **Công thức Final Score:**
  $$Final\ Score = \frac{1}{5} \sum_{k \in \{1, 5, 20, 50, 100\}} R@k$$

> [!IMPORTANT]
> **Hệ quả thiết kế hệ thống:** Việc đẩy đáp án đúng lên những vị trí đầu tiên (ranking) cực kỳ quan trọng vì ngưỡng R@1 chiếm 1/5 tổng số điểm. Nếu tìm ra đáp án đúng nhưng để ở cuối danh sách (ví dụ hạng 80), bạn sẽ mất điểm ở các ngưỡng R@1, R@5, R@20, R@50 và chỉ nhận điểm ở R@100. Đây là lý do **Rerankers** đóng vai trò quyết định chất lượng hệ thống.

---

## 3. THÔNG TIN DỮ LIỆU VÒNG SƠ TUYỂN – ĐỢT 1

Dữ liệu chính thức được dùng để làm quen bao gồm các thành phần sau:
* **Videos:** Video gốc của cuộc thi. (Lưu ý: Dữ liệu thi chính thức là Video).
* **Keyframes:** Chứa toàn bộ keyframes được trích xuất từ video, lưu trong các thư mục tương ứng với tên video (đặt tên tăng dần, vị trí frame index được ghi trong file metadata).
* **Objects:** File JSON liệt kê các vật thể phát hiện từ mô hình Faster R-CNN (pretrained trên OpenImages V4). Định dạng file JSON tương ứng với tên file keyframe (ví dụ: `L01_V001/0000.json` cho `L01_V001/0000.jpg`).
* **CLIP features:** Các vector feature trích xuất từ mô hình `clip-ViT-B-32` của tất cả keyframe, lưu trong duy nhất một file `.npy` theo thứ tự tăng dần của keyframe.
* **Metadata:** Thông tin metadata của video lấy từ YouTube, lưu dưới dạng file JSON tương ứng với tên video.

### 🌐 Link tải dữ liệu chính thức:
Tải dữ liệu tại: [Google Spreadsheet Link](https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?usp=sharing)

> [!NOTE]
> Đây là tập dữ liệu batch 1 của AIC 2025. Dữ liệu đầy đủ của vòng sơ tuyển AIC 2026 sẽ bao gồm thêm **batch 2** được công bố bởi BTC trong thời gian tới.
