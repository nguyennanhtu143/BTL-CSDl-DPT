# Hướng dẫn chạy ứng dụng CBIR — Tìm kiếm ảnh nền thiên nhiên

Tài liệu này hướng dẫn 3 việc chính:
1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Khởi tạo CSDL đặc trưng](#2-khởi-tạo-csdl-đặc-trưng)
3. [Chạy ứng dụng (CLI và Web UI)](#3-chạy-ứng-dụng)
4. [Xem dữ liệu trong CSDL](#4-xem-dữ-liệu-trong-csdl)

---

## 1. Cài đặt môi trường

### Yêu cầu
- **Python 3.10+** (đã test trên 3.12)
- Hệ điều hành: Windows / macOS / Linux
- Dataset 500 ảnh đã có sẵn ở `Images_Dataset/01_thien_nhien/`

### Cài thư viện
Mở terminal ở thư mục dự án (`D:\BTL-CSDL-DPT`):

```bash
# (khuyến nghị) tạo virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# cài dependencies
pip install -r requirements.txt
```

`requirements.txt` cài các package: `pillow`, `numpy`, `streamlit`, `matplotlib`. SQLite có sẵn trong Python (built-in `sqlite3`), không cần cài thêm.

### Verify cài đặt
```bash
python -c "import streamlit, numpy, PIL; print('OK')"
```

---

## 2. Khởi tạo CSDL đặc trưng

CSDL là file SQLite tại `data/features.db`, chứa 500 vector đặc trưng (mỗi vector 657 chiều = 576 màu + 81 hình dạng).

### Build lần đầu
```bash
python build_database.py
```

Output mẫu:
```
[build] Dataset    : D:\BTL-CSDL-DPT\Images_Dataset\01_thien_nhien
[build] CSDL       : D:\BTL-CSDL-DPT\data\features.db
[build] Tổng ảnh    : 500
[build] Cần xử lý  : 500
------------------------------------------------------------
  [  50/500] avg   23.4 ms/ảnh, elapsed    1.2s, ETA  10.5s
  [ 100/500] avg   22.5 ms/ảnh, elapsed    2.2s, ETA   9.0s
  ...
  [ 500/500] avg   20.3 ms/ảnh, elapsed   10.2s, ETA   0.0s
------------------------------------------------------------
[build] Hoàn tất xử lý 500 ảnh trong 10.16s
[build] Tổng record trong CSDL: 500
[build] Kích thước file       : 2052.0 KB
```

Quá trình mất **~10 giây** trên máy thông thường.

### Các tùy chọn

| Lệnh | Mục đích |
|------|----------|
| `python build_database.py` | Build incremental (skip ảnh đã có trong DB) |
| `python build_database.py --rebuild` | **Xóa DB cũ** rồi build lại từ đầu |
| `python build_database.py --limit 50` | Chỉ build 50 ảnh đầu (debug nhanh) |
| `python build_database.py --dataset PATH` | Đổi thư mục dataset |
| `python build_database.py --db PATH` | Đổi đường dẫn file DB |

### Khi nào cần `--rebuild`?
- Khi thay đổi config (`IMG_WIDTH`, `COLOR_BINS`, `W_COLOR`, ...) — feature trong DB sẽ không khớp với code mới.
- Khi cập nhật code trong `src/color_features.py`, `src/gradient_features.py`, `src/feature_extractor.py`.
- Khi nghi ngờ DB bị hỏng.

### Verify build thành công
```bash
python inspect_db.py
```
Kết quả phải hiển thị `Số record: 500`.

---

## 3. Chạy ứng dụng

Có 2 cách chạy: **CLI** (terminal, không cần browser) và **Web UI** (Streamlit).

### Cách 1 — CLI (nhanh, nhẹ)

```bash
# Truy vấn 1 ảnh, in vector rút gọn ra console
python query_cli.py Images_Dataset/01_thien_nhien/01_thien_nhien_001.jpg
```

Output mẫu:
```
============================================================
[QUERY] Image: 01_thien_nhien_001.jpg
============================================================
[FEATURE VECTOR] Shape: (657,)
  - Color (576 dim): [0.0000, 0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000, 0.0000]
  - Shape (81 dim) : [0.0051, 0.0011, 0.0011, 0.0019, ..., 0.0074, 0.0033, 0.0021, 0.0020]
------------------------------------------------------------
[TOP 5 RESULTS]
Rank | Filename                     | Distance
  1  | 01_thien_nhien_099.jpg       | 0.0000
  2  | 01_thien_nhien_001.jpg       | 0.0000
  3  | 01_thien_nhien_441.jpg       | 0.0871
  4  | 01_thien_nhien_387.jpg       | 0.1049
  5  | 01_thien_nhien_361.jpg       | 0.1062
============================================================
[timing] load_db = 5.5 ms, query = 35.1 ms
```

Tùy chọn:
- `--full` : in **toàn bộ 657 phần tử** thay vì rút gọn
- `--k 10` : trả về top-10 thay vì top-5
- `--db PATH` : dùng file DB khác

```bash
python query_cli.py path/to/image.jpg --full --k 10
```

### Cách 2 — Web UI (Streamlit)

```bash
streamlit run ui/app.py
```

Lần đầu chạy, Streamlit hỏi email — bấm Enter để bỏ qua.

Sau đó mở browser tại **http://localhost:8501**:

1. **Sidebar (trái)**: hiển thị thông tin CSDL, cho phép tích chọn `In toàn bộ vector ra console`, chỉnh top-k.
2. **Khu vực chính**:
   - Bấm **Browse files** để chọn ảnh `.jpg / .jpeg / .png`.
   - Ảnh được preview ở cột trái.
   - Bấm **🔍 Tìm kiếm Top-K**.
3. **Cột phải**: hiển thị top-5 ảnh kết quả, mỗi ô có thumbnail + tên file + khoảng cách Euclidean (`d = X.XXXX`).
4. **Expander "Thông tin vector đặc trưng"**: hiển thị sum Color/Shape, số bin non-zero, thời gian extract/match.

Quan trọng: **mỗi lần bấm "Tìm kiếm" sẽ in toàn bộ vector + bảng top-5 ra terminal** nơi bạn chạy `streamlit run` (theo yêu cầu đề bài). Để dừng app: Ctrl+C trong terminal.

### Test bằng ảnh khác (không có trong dataset)
Bạn có thể tải ảnh thiên nhiên bất kỳ trên mạng (`.jpg/.png`), upload vào UI hoặc truyền vào CLI:

```bash
python query_cli.py "C:/Users/manhh/Desktop/my_landscape.jpg"
```

Hệ thống sẽ resize về 256×144, extract feature, so với 500 ảnh trong DB, trả top-5.

---

## 4. Xem dữ liệu trong CSDL

Có 3 cách xem nội dung file `data/features.db`:

### Cách A — Script tiện ích `inspect_db.py` ⭐ (khuyến nghị)

Đây là cách **đầy đủ nhất**, deserialize đúng vector từ BLOB và hiển thị thân thiện:

```bash
# Tổng quan + 5 record đầu (mặc định)
python inspect_db.py

# Hiển thị thêm SQL schema
python inspect_db.py --schema

# 20 record đầu
python inspect_db.py --limit 20

# Chi tiết 1 ảnh: metadata + vector preview
python inspect_db.py --filename 01_thien_nhien_001.jpg

# In TOÀN BỘ 657 phần tử của vector
python inspect_db.py --filename 01_thien_nhien_001.jpg --full
```

Output mẫu (tổng quan):
```
======================================================================
TỔNG QUAN CSDL
======================================================================
Đường dẫn       : D:\BTL-CSDL-DPT\data\features.db
Kích thước file : 2052.0 KB
Số record       : 500
ID range        : 1 - 500
Width (gốc)     : [640]
Height (gốc)    : [360]
file_size (byte): min=10975, max=130743, avg=46678
Vector dim      : 657 (Color 576 + Shape 81)

======================================================================
DANH SÁCH 5 RECORD ĐẦU
======================================================================
ID   Filename                       WxH            KB   BLOB Created
----------------------------------------------------------------------
1    01_thien_nhien_001.jpg         640x360     43.3   2628 2026-04-27 03:44:42
2    01_thien_nhien_002.jpg         640x360     63.6   2628 2026-04-27 03:44:42
...
```

Output mẫu (chi tiết 1 ảnh):
```
======================================================================
CHI TIẾT RECORD: 01_thien_nhien_250.jpg
======================================================================
ID          : 250
Width × Height : 640 × 360
File size   : 31856 byte (31.1 KB)
BLOB length : 2628 byte (657 × float32)
Created at  : 2026-04-27 03:44:47
----------------------------------------------------------------------
VECTOR ĐẶC TRƯNG (657 chiều)
  Tổng sum  : 1.000000  (kì vọng ~ 1.0)
  Color sum : 0.700000  (kì vọng = 0.7)
  Shape sum : 0.300000  (kì vọng = 0.3)
  Min / Max : 0.000000 / 0.059947
  Non-zero  : Color 111/576, Shape 81/81

  Color preview: [0.0017, 0.0599, 0.0000, ..., 0.0000]
  Shape preview: [0.0017, 0.0008, 0.0013, ..., 0.0004]
```

### Cách B — Dòng lệnh `sqlite3` (cho người quen SQL)

Nếu máy có cài SQLite CLI (Windows: tải tại https://sqlite.org/download.html):

```bash
sqlite3 data/features.db
```

Trong shell SQLite:
```sql
-- Bật chế độ hiển thị đẹp
.mode column
.headers on

-- Xem schema
.schema images

-- Đếm record
SELECT COUNT(*) FROM images;

-- Xem 5 record đầu (KHÔNG hiển thị BLOB - sẽ rất dài)
SELECT id, filename, width, height, file_size, length(feature_vector), created_at
FROM images LIMIT 5;

-- Tìm ảnh theo tên
SELECT id, filename, file_size FROM images WHERE filename LIKE '%_250%';

-- Thoát
.quit
```

⚠️ Lưu ý: cột `feature_vector` là **BLOB nhị phân**, in trực tiếp sẽ ra ký tự rác. Luôn dùng `length(feature_vector)` (= 2628 byte) thay vì select trực tiếp.

### Cách C — GUI: DB Browser for SQLite

Dành cho người thích click chuột:

1. Tải DB Browser for SQLite: https://sqlitebrowser.org/dl/ (free, đa nền tảng).
2. Cài đặt, mở app.
3. **File → Open Database** → chọn `D:\BTL-CSDL-DPT\data\features.db`.
4. Tab **Browse Data** → chọn bảng `images` để xem dạng bảng tính.
5. Cột `feature_vector` hiển thị `<BLOB>` — click vào ô để xem hex/byte (không có ý nghĩa khi đọc thô vì là binary float32).

### Cách D — Inline Python (debug nhanh)

```python
# Mở Python REPL
python

>>> from src.database import load_database
>>> db = load_database()
>>> len(db)
500
>>> db.vectors.shape
(500, 657)
>>> db.filenames[:3]
['01_thien_nhien_001.jpg', '01_thien_nhien_002.jpg', '01_thien_nhien_003.jpg']
>>> db.vectors[0].sum()        # tổng vector phải ~ 1.0
1.0
>>> db.vectors[0, :576].sum()  # phần Color phải = 0.7
0.7
```

---

## Tóm tắt lệnh thường dùng

```bash
# Lần đầu setup
pip install -r requirements.txt
python build_database.py --rebuild

# Truy vấn 1 ảnh nhanh
python query_cli.py path/to/image.jpg

# Mở web UI
streamlit run ui/app.py

# Xem CSDL
python inspect_db.py
python inspect_db.py --filename 01_thien_nhien_001.jpg --full

# Chạy test (verify tất cả các phase)
python notebooks/test_preprocessing.py
python notebooks/test_color_features.py
python notebooks/test_gradient_features.py
python notebooks/test_integration.py
python notebooks/test_database.py
```

## Troubleshooting

| Lỗi | Nguyên nhân & cách xử lý |
|-----|--------------------------|
| `CSDL rỗng. Hãy chạy build_database.py` | Chưa build DB lần nào → chạy `python build_database.py` |
| `BLOB sai shape: (X,) mong đợi (657,)` | DB cũ build với config khác → `python build_database.py --rebuild` |
| `UnicodeEncodeError` ở console Windows | Đã handle ở các script qua `sys.stdout.reconfigure(encoding="utf-8")`. Nếu lỗi ở chỗ khác, set biến môi trường `PYTHONIOENCODING=utf-8` |
| `streamlit: command not found` | Chưa activate venv hoặc chưa cài → `pip install streamlit` |
| Streamlit mở browser nhưng không kết nối được | Đổi port: `streamlit run ui/app.py --server.port 8888` |
| Top-1 trả về ảnh khác chính nó với d=0 | **Không phải lỗi** — dataset có 78 cặp file trùng byte. Xem chi tiết tại `docs/phase_5_database.md` |
