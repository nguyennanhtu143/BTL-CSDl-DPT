# Kế hoạch triển khai hệ thống tìm kiếm ảnh nền thiên nhiên (CBIR)

## Tổng quan dự án

**Mục tiêu:** Xây dựng hệ thống Content-Based Image Retrieval (CBIR) tìm kiếm 5 ảnh nền thiên nhiên giống nhất với ảnh truy vấn từ tập 500 ảnh có sẵn.

**Đặc trưng sử dụng:**
- **Color Histogram (Grid-based)** — đại diện cho sự tương đồng về màu sắc và bố cục.
- **Gradient Histogram (Sobel)** — đại diện cho sự khác biệt về kết cấu/hình dạng.

**Ràng buộc:** Chỉ được dùng thư viện để đọc ảnh (Pillow/OpenCV). Toàn bộ thuật toán xử lý phải tự code bằng vòng lặp + toán học (NumPy được phép cho mảng).

**Dataset:** 500 ảnh tại `Images_Dataset/01_thien_nhien/` (đã kiểm tra).

**Ngôn ngữ & công nghệ:**
- **Python 3.10+** cho toàn bộ source code.
- **Pillow (PIL)** — đọc ảnh `.jpg` thành mảng (được phép theo đề bài).
- **NumPy** — thao tác mảng và vector hóa (cho phép vì đây là cấu trúc dữ liệu, không phải thuật toán xử lý ảnh đóng gói).
- **SQLite3** (built-in Python) — lưu CSDL đặc trưng.
- **Streamlit** hoặc **Tkinter** — UI demo.
- **Matplotlib** — visualize gradient, debug.
- **KHÔNG** dùng `cv2.calcHist`, `cv2.Sobel`, `scipy.ndimage`, `skimage.feature.hog`, hay bất kỳ hàm xử lý ảnh đóng gói nào — phải tự code.

---

## Phase 0 — Khởi tạo dự án & cấu trúc thư mục

**Mục tiêu:** Chuẩn bị môi trường, cấu trúc code, dependency.

**Task:**
1. Tạo cấu trúc thư mục:
   ```
   BTL-CSDL-DPT/
   ├── Images_Dataset/01_thien_nhien/   (đã có 500 ảnh)
   ├── src/
   │   ├── preprocessing.py             (Phase 1)
   │   ├── color_features.py            (Phase 2)
   │   ├── gradient_features.py         (Phase 3)
   │   ├── feature_extractor.py         (Phase 4 - tích hợp)
   │   ├── matcher.py                   (Phase 4 - so sánh)
   │   ├── database.py                  (Phase 5 - CSDL)
   │   └── utils.py                     (helper: load ảnh, normalize, ...)
   ├── data/
   │   └── features.db                  (SQLite — sinh ra ở Phase 5)
   ├── notebooks/                       (test nhanh từng module)
   ├── ui/                              (Phase 6 — giao diện)
   ├── main.py                          (entry point: build DB / query)
   ├── requirements.txt
   └── README.md
   ```
2. Tạo `requirements.txt`: `pillow`, `numpy` (bắt buộc); `matplotlib` (hiển thị); tùy chọn `streamlit`/`tkinter` cho UI.
3. Cấu hình hằng số toàn cục (file `src/config.py`):
   - `IMG_WIDTH = 256`, `IMG_HEIGHT = 144` (giữ tỉ lệ 16:9 của ảnh gốc 640×360).
   - `GRID = 3` (chia 3×3 = 9 ô; mỗi ô ~85×48 pixel).
   - `COLOR_BINS = 4` (RGB 4×4×4 = 64 bin/ô).
   - `GRAD_BINS = 9` (chia 0°–180° thành 9 khoảng 20°).
   - `W_COLOR = 0.7`, `W_SHAPE = 0.3`.

**Output:**
- Cây thư mục dự án hoàn chỉnh.
- `requirements.txt`, `config.py` đã commit.
- Có thể `pip install -r requirements.txt` chạy không lỗi.

---

## Phase 1 — Tiền xử lý ảnh (Pre-processing)

**Mục tiêu:** Đọc và chuẩn hóa toàn bộ ảnh về cùng kích thước để các bước sau có cơ sở so sánh tương đương.

**Bối cảnh:** Toàn bộ 500 ảnh trong dataset đều có kích thước **640×360** (tỉ lệ 16:9 — chuẩn ảnh nền). Ta resize về **256×144** để giảm tải tính toán nhưng vẫn giữ nguyên tỉ lệ và bố cục — tránh bóp méo trời/đất.

**Task:**
1. Hàm `load_image(path) -> np.ndarray`: dùng Pillow đọc ảnh `.jpg`, trả về mảng 3D `(H, W, 3)` kiểu `uint8` ở không gian RGB.
2. Hàm `resize_image(img, target_w=256, target_h=144) -> np.ndarray`: tự code resize bằng nội suy song tuyến (bilinear) — hỗ trợ kích thước không vuông. Output shape `(144, 256, 3)`.
3. Hàm `to_grayscale(img) -> np.ndarray`: áp dụng công thức `Y = 0.299*R + 0.587*G + 0.114*B` (chỉ dùng cho Phase 3).
4. Viết unit test nhanh: load 5 ảnh bất kỳ, resize về 256×144, kiểm tra `shape == (144, 256, 3)`.

**Output:**
- Module `src/preprocessing.py` với 3 hàm trên đã test.
- Demo notebook hoặc script in ra shape của ảnh trước/sau khi xử lý: `(360, 640, 3) -> (144, 256, 3)`.

---

## Phase 2 — Trích xuất đặc trưng Màu sắc (Grid-based Color Histogram)

**Mục tiêu:** Tạo vector màu phản ánh cả phân bố màu lẫn bố cục không gian.

**Phương án đã chốt:** **RGB 4×4×4 = 64 bin/ô**, lưới 3×3 → vector màu **9 × 64 = 576 chiều**. Đơn giản, không cần chuyển không gian màu, đủ tốt cho ảnh thiên nhiên (xanh lá / xanh dương / vàng tách rõ trong RGB).

**Task:**
1. Hàm `split_into_grid(img, grid=3) -> List[np.ndarray]`: chia ảnh 256×144 thành 9 ô (mỗi ô ~85×48 pixel — 3 ô theo chiều rộng, 3 ô theo chiều cao). Lưu ý: 256 không chia hết cho 3 (256/3 = 85.33), 144 chia hết (144/3 = 48). Cách xử lý: 2 ô đầu rộng 85, ô cuối rộng 86 (hoặc ngược lại) — làm đúng cách trong code.
2. Hàm `quantize_pixel(r, g, b, bins=4) -> int`: lượng tử hóa — chia mỗi kênh `[0, 256)` thành 4 khoảng (rộng 64), gắn pixel vào bin tương ứng theo công thức:
   - `r_bin = r // 64`, `g_bin = g // 64`, `b_bin = b // 64` (mỗi giá trị ∈ {0,1,2,3}).
   - `bin_index = r_bin * 16 + g_bin * 4 + b_bin` → giá trị ∈ `[0, 64)`.
3. Hàm `cell_histogram(cell, bins=4) -> np.ndarray`: duyệt từng pixel trong ô, đếm số lượng vào từng bin → vector dài 64.
4. Hàm `color_histogram(img) -> np.ndarray`: ghép histogram của 9 ô → vector dài **576**.
5. Hàm `normalize(vec) -> np.ndarray`: chuẩn hóa L1 (chia mỗi phần tử cho tổng để tổng = 1).
6. Tối ưu tốc độ: dùng NumPy vector hóa — không loop Python thuần. Có thể tính `bin_index` cho cả mảng 1 lệnh: `bin_idx = (img[..., 0]//64)*16 + (img[..., 1]//64)*4 + (img[..., 2]//64)`, sau đó `np.bincount(bin_idx.ravel(), minlength=64)`.

**Output:**
- Module `src/color_features.py` với hàm `extract_color_feature(img) -> np.ndarray` trả về vector 576 chiều đã normalize.
- Test: chạy trên 1 ảnh và verify `vector.sum() ≈ 1.0`.

---

## Phase 3 — Trích xuất đặc trưng Hình dạng (Gradient Histogram)

**Mục tiêu:** Bổ sung thông tin kết cấu/cạnh để phân biệt các ảnh có cùng tông màu nhưng khác bản chất.

**Task:**
1. Định nghĩa kernel Sobel:
   - `Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]`
   - `Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]`
2. Hàm `convolve2d(gray_img, kernel) -> np.ndarray`: tự code phép tích chập 2D — vòng lặp trượt kernel 3×3 trên ảnh xám, padding 0 ở biên. Vector hóa bằng NumPy slicing để tăng tốc.
3. Hàm `compute_gradient(gray_img) -> (magnitude, angle)`:
   - `dx = convolve2d(gray, Gx)`, `dy = convolve2d(gray, Gy)`.
   - `magnitude = sqrt(dx² + dy²)`.
   - `angle = arctan2(dy, dx)` rồi chuyển về độ trong `[0°, 180°)` (lấy giá trị tuyệt đối).
4. Hàm `gradient_histogram(gray_img, bins=9) -> np.ndarray`:
   - Chia `[0°, 180°)` thành 9 bin (mỗi bin 20°).
   - Với từng pixel, cộng `magnitude` vào bin tương ứng theo `angle` (trọng số theo magnitude — giống HOG đơn giản hóa).
   - Chia ảnh thành lưới 3×3 (giống Phase 2, áp dụng trên ảnh xám 256×144) để giữ thông tin không gian → vector **9 × 9 = 81 chiều**.
5. Normalize L1.

**Output:**
- Module `src/gradient_features.py` với hàm `extract_gradient_feature(img) -> np.ndarray`.
- Test: visualize ma trận `magnitude` của 1 ảnh để đảm bảo các cạnh hiển thị đúng.

---

## Phase 4 — Tích hợp đặc trưng & Hàm so sánh

**Mục tiêu:** Kết hợp 2 vector thành 1 vector tổng và cài đặt hàm tính khoảng cách.

**Task:**
1. Hàm `extract_features(img) -> np.ndarray`:
   - Gọi `extract_color_feature` (đã normalize) → `v_color`.
   - Gọi `extract_gradient_feature` (đã normalize) → `v_shape`.
   - Áp trọng số: `v = concat(W_COLOR * v_color, W_SHAPE * v_shape)`.
   - Trả về vector cuối cùng (không normalize lại để giữ trọng số).
2. Hàm `euclidean_distance(v1, v2) -> float`: tự code `sqrt(sum((v1[i] - v2[i])²))`.
3. Hàm `find_top_k(query_vec, db_vectors, k=5) -> List[Tuple[image_id, distance]]`: tính khoảng cách query với toàn bộ DB, sort tăng dần, trả về top-k.
4. (Tùy chọn nâng cao) Cho phép so sánh từng phần: Color-only, Shape-only, Combined để báo cáo so sánh.

**Quyết định cần chốt:**
- Trọng số `W_COLOR : W_SHAPE` — bắt đầu với 0.7 : 0.3 theo gợi ý GV, có thể tune sau khi thử nghiệm.

**Output:**
- Module `src/feature_extractor.py` và `src/matcher.py`.
- Test: trích xuất feature 1 ảnh, query lại chính nó → khoảng cách phải = 0.

---

## Phase 5 — Tổ chức CSDL đặc trưng (Feature Database)

**Mục tiêu:** Tiền xử lý toàn bộ 500 ảnh, lưu vector vào CSDL để query nhanh.

**Task:**
1. Chọn định dạng lưu trữ: **SQLite** (gọn, query bằng Python `sqlite3` chuẩn, có thể lưu BLOB).
   - Schema bảng `images`: `id INTEGER PRIMARY KEY, filename TEXT, width INT, height INT, file_size INT, feature_vector BLOB, created_at TIMESTAMP`.
2. Script `build_database.py`:
   - Duyệt toàn bộ folder `Images_Dataset/01_thien_nhien/`.
   - Với mỗi ảnh: load → resize → extract features → lưu metadata + serialize vector (`numpy.tobytes()`) vào SQLite.
   - Hiển thị progress bar (in từng 50 ảnh) để dễ theo dõi.
   - Đo và log thời gian xử lý / ảnh.
3. Hàm `load_database() -> (ids, filenames, vectors_matrix)`: đọc toàn bộ CSDL về RAM thành ma trận NumPy `(500, D)` để query vector hóa.
4. Hàm `query(image_path, k=5)`: load ảnh truy vấn → extract feature → tính khoảng cách → trả top-5.

**Output:**
- File `data/features.db` chứa 500 record.
- Script `build_database.py` chạy được end-to-end.
- Log thời gian build (kỳ vọng < 5 phút cho 500 ảnh).

---

## Phase 6 — Giao diện web (Streamlit) & Console log

**Mục tiêu:** Cho phép chọn ảnh truy vấn qua web UI, hiển thị 5 ảnh kết quả + khoảng cách trên trang, đồng thời **log toàn bộ thông tin chi tiết ra console** để verify thuật toán.

**Framework:** **Streamlit** (`streamlit run ui/app.py`).

**Task:**
1. **Layout web UI** (`ui/app.py`):
   - Vùng upload ảnh (`st.file_uploader`) — chấp nhận `.jpg`/`.png`.
   - Hiển thị ảnh truy vấn (preview).
   - Nút "Tìm kiếm" → gọi `query()`.
   - Hiển thị top-5 ảnh kết quả dạng grid 5 cột (`st.columns(5)`), mỗi cột có:
     - Ảnh kết quả.
     - Tên file (`filename`).
     - Khoảng cách Euclidean (định dạng 4 chữ số thập phân).
   - (Tùy chọn) Slider điều chỉnh `W_COLOR` / `W_SHAPE` realtime để demo hiệu ứng trọng số.

2. **Console output (BẮT BUỘC)** — mỗi lần thực hiện query, in ra terminal:
   ```
   ============================================================
   [QUERY] Image: <tên file ảnh đầu vào>
   ============================================================
   [FEATURE VECTOR] Shape: (657,)
     - Color (576 dim): [0.0123, 0.0045, 0.0312, ..., 0.0089]
     - Shape (81 dim) : [0.0512, 0.1023, 0.0834, ..., 0.0271]
   ------------------------------------------------------------
   [TOP 5 RESULTS]
   Rank | Filename                    | Distance
     1  | 01_thien_nhien_127.jpg      | 0.0000
     2  | 01_thien_nhien_341.jpg      | 0.0234
     3  | 01_thien_nhien_088.jpg      | 0.0301
     4  | 01_thien_nhien_412.jpg      | 0.0356
     5  | 01_thien_nhien_205.jpg      | 0.0423
   ============================================================
   ```
   - In **toàn bộ vector** (có thể dùng `np.set_printoptions(threshold=np.inf, precision=6)` nếu giảng viên yêu cầu xem đủ; mặc định in `np.array2string` rút gọn để dễ đọc, kèm option toggle ở UI để in đầy đủ).
   - Tách rõ phần Color / Shape của vector.
   - In bảng top-5 theo format căn cột.
   - Implement trong `src/matcher.py` hoặc helper `src/logger.py`.

3. **Xử lý lỗi:** ảnh không hợp lệ, ảnh không phải `.jpg`/`.png`, file rỗng, etc.

**Output:**
- App web chạy được tại `localhost:8501`.
- Console log đúng format mỗi lần query (verify được vector và khoảng cách bằng mắt).
- Screenshot UI + screenshot console cho báo cáo.

---

## Phase 7 — Đánh giá & Báo cáo

**Mục tiêu:** Kiểm tra chất lượng kết quả, viết báo cáo theo yêu cầu đề bài.

**Task:**
1. Chọn 10 ảnh truy vấn đại diện (rừng, biển, núi, hoàng hôn, tuyết, ...).
2. Với mỗi ảnh, chạy query và đánh giá định tính top-5: bao nhiêu ảnh "thực sự giống" theo cảm nhận.
3. Lập bảng kết quả: query | top-5 result filenames | distance | nhận xét.
4. So sánh 3 cấu hình:
   - Chỉ Color Histogram.
   - Chỉ Gradient Histogram.
   - Kết hợp 0.7/0.3.
5. Viết báo cáo (file `report.md` hoặc Word) theo cấu trúc đề bài:
   - Phân tích bài toán.
   - Bộ thuộc tính: nhóm "tương đồng" (Global Color) vs nhóm "khác biệt" (Grid-Color, Gradient).
   - Thuật toán + công thức (Sobel, Euclidean, Quantization).
   - Cấu trúc CSDL.
   - Kết quả thực nghiệm + screenshots.
   - Hạn chế & hướng cải tiến.

**Output:**
- File báo cáo hoàn chỉnh (`report.md` / `.docx` / `.pdf`).
- Bảng kết quả thực nghiệm.
- Slide thuyết trình (nếu có yêu cầu).

---

## Lộ trình thời gian đề xuất

| Phase | Thời gian ước tính | Phụ thuộc |
|-------|-------------------|-----------|
| 0 — Khởi tạo | 0.5 ngày | — |
| 1 — Pre-processing | 1 ngày | Phase 0 |
| 2 — Color Features | 1.5 ngày | Phase 1 |
| 3 — Gradient Features | 1.5 ngày | Phase 1 |
| 4 — Tích hợp & So sánh | 1 ngày | Phase 2, 3 |
| 5 — CSDL | 1 ngày | Phase 4 |
| 6 — UI | 1 ngày | Phase 5 |
| 7 — Báo cáo | 1.5 ngày | Phase 6 |
| **Tổng** | **~9 ngày** | |

---

## Các quyết định đã chốt

| # | Vấn đề | Quyết định |
|---|--------|------------|
| 1 | Kích thước resize | **256×144** (giữ tỉ lệ 16:9 của ảnh gốc 640×360) |
| 2 | Không gian màu | **RGB 4×4×4 = 64 bin/ô** (vector màu 576 chiều) |
| 3 | Resize method | Tự code **bilinear interpolation** |
| 4 | Trọng số Color/Shape | **0.7 / 0.3** (có thể tune ở Phase 7) |
| 5 | CSDL | **SQLite** (file `data/features.db`, lưu vector dưới dạng BLOB) |
| 6 | UI | **Streamlit** (web, chạy `streamlit run ui/app.py`) |
| 7 | Yêu cầu console khi query | **In ra terminal:** vector đặc trưng đầy đủ của ảnh đầu vào (tách Color/Shape), bảng top-5 ảnh kết quả + khoảng cách Euclidean |
