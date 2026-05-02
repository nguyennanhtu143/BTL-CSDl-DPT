# Phase 6 — Streamlit UI & Console log

## Mục tiêu
Cho phép upload ảnh truy vấn qua web, hiển thị top-5 ảnh kết quả + khoảng cách Euclidean, đồng thời **in toàn bộ thông tin chi tiết ra console** theo format bắt buộc của đề bài để verify thuật toán bằng mắt.

## Checklist task
- [x] `src/logger.py` với `log_query(filename, vec, results, full_vector)` in console theo format đề bài
- [x] Helper `format_vector_preview` rút gọn vector dài thành `[v0, v1, ..., vn-1, vn]`
- [x] `query_cli.py` (root): script CLI query 1 ảnh không cần browser
- [x] `ui/app.py` Streamlit: upload, preview, button tìm kiếm, grid 5 cột top-5
- [x] Cache CSDL bằng `@st.cache_resource` (load 1 lần khi UI khởi động)
- [x] Sidebar: tuỳ chọn full vector + chỉnh `top-k`
- [x] Expander hiển thị thống kê vector (sum Color/Shape, non-zero bins)
- [x] Xử lý lỗi khi CSDL rỗng / file lỗi
- [x] Smoke test: streamlit khởi động sạch, pipeline end-to-end OK

## Module
- `src/logger.py`
- `query_cli.py` (CLI root)
- `ui/app.py` (Streamlit UI)

## Format console output (bắt buộc)

Mặc định (rút gọn 4 đầu + 4 cuối — dễ đọc):
```
============================================================
[QUERY] Image: 01_thien_nhien_300.jpg
============================================================
[FEATURE VECTOR] Shape: (657,)
  - Color (576 dim): [0.0117, 0.0303, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000, 0.0000]
  - Shape (81 dim) : [0.0036, 0.0012, 0.0041, 0.0027, ..., 0.0013, 0.0005, 0.0004, 0.0005]
------------------------------------------------------------
[TOP 5 RESULTS]
Rank | Filename                     | Distance
  1  | 01_thien_nhien_300.jpg       | 0.0000
  2  | 01_thien_nhien_078.jpg       | 0.0000
  3  | 01_thien_nhien_197.jpg       | 0.1100
  4  | 01_thien_nhien_362.jpg       | 0.1118
  5  | 01_thien_nhien_405.jpg       | 0.1266
============================================================
```

Khi bật `--full` (CLI) hoặc tick "In toàn bộ vector ra console" (UI): in đủ **657 phần tử** dưới dạng `np.array2string(threshold=inf, precision=6)`, tách Color và Shape thành 2 khối.

## Streamlit UI layout

```
┌─────────────────────────────────────────────────────────────┐
│  🌿 CBIR — Tìm kiếm ảnh nền thiên nhiên                     │
│  Vector 657 chiều = Color 576 + Shape 81                    │
├──────────┬──────────────────────────────────────────────────┤
│ Sidebar  │  [File uploader: chọn ảnh .jpg/.png]             │
│ ─────── │                                                   │
│ CSDL: 500│  ┌─────────┐  ┌──────────────────────────────┐   │
│ W_COLOR  │  │  Ảnh    │  │  [🔍 Tìm kiếm Top-K]         │   │
│ = 0.7    │  │ truy    │  │                              │   │
│ W_SHAPE  │  │ vấn     │  │  Top-5 kết quả:              │   │
│ = 0.3    │  │ preview │  │  ┌──┬──┬──┬──┬──┐            │   │
│ TOP_K=5  │  │         │  │  │1 │2 │3 │4 │5 │            │   │
│ ☐ Full   │  │ name.jpg│  │  └──┴──┴──┴──┴──┘            │   │
│ vector   │  │  XX KB  │  │  + Expander vector stats     │   │
└──────────┴──┴─────────┴──┴──────────────────────────────┘   │
                                                              │
   Console (terminal nơi chạy `streamlit run`):                │
   ============================================================│
   [QUERY] Image: ...                                          │
```

## Cách chạy

### Web UI
```bash
streamlit run ui/app.py
# Mặc định mở http://localhost:8501
```

Khi đã build CSDL:
1. Chọn file .jpg/.png ở khung upload.
2. Bấm **🔍 Tìm kiếm Top-K**.
3. Web hiển thị top-5 (ảnh + tên + distance).
4. Console (terminal nơi chạy streamlit) in đầy đủ vector + bảng kết quả.

### CLI (không cần browser)
```bash
python query_cli.py path/to/image.jpg                # rút gọn vector
python query_cli.py path/to/image.jpg --full         # in đủ 657 phần tử
python query_cli.py path/to/image.jpg --k 10         # top-10
```

## Tốc độ thực đo
- Khởi động Streamlit: ~2s (cache `load_database` 5 ms cho 500 vector).
- Mỗi query: extract feature ~30 ms + match top-k vài ms ≈ **~35 ms total**.
- CLI query (lần đầu, gồm import + load): ~700 ms; nếu giữ DB trong RAM: ~35 ms.

## Triết lý thiết kế

### Vì sao tách CLI khỏi Streamlit?
- CLI dùng để **debug & demo nhanh** mà không cần browser; phù hợp khi chấm bài qua SSH hoặc khi giảng viên muốn xem console trực tiếp.
- Streamlit cung cấp **UX trực quan** cho người dùng cuối: thấy ảnh, thấy distance, click thay ảnh.
- **Cùng dùng `src.logger.log_query`** để format console giống nhau ở cả hai ngả.

### Vì sao cache CSDL bằng `@st.cache_resource`?
Streamlit rerun toàn bộ script khi user tương tác. Nếu không cache, mỗi lần upload sẽ load lại 500 vector từ SQLite (5ms × ổ đĩa). `cache_resource` giữ object trong RAM toàn session — tốc độ ~0 ms.

### Vì sao dùng `getvalue()` thay vì `read()` của uploaded file?
`Image.open()` consume buffer; sau đó `st.image(uploaded)` sẽ hiển thị rỗng. Dùng `getvalue()` lấy bytes một lần, dùng nhiều nơi.

## Hạn chế đã biết
- UI chưa hỗ trợ thay đổi `W_COLOR / W_SHAPE` realtime (vì DB lưu vector đã pre-multiply trọng số). Muốn tune trọng số phải sửa `src/config.py` và rebuild — sẽ làm ở Phase 7 nếu cần so sánh cấu hình.
- Khi kết quả top-k chứa file trùng byte (78 cặp duplicate trong dataset), 2 ô đầu hiển thị 2 ảnh giống nhau với d=0. Đây là phản ánh đúng dataset, không phải bug.
