# Phase 5 — CSDL đặc trưng (SQLite)

## Mục tiêu
Tiền xử lý toàn bộ 500 ảnh, lưu vector đặc trưng 657 chiều vào SQLite (file `data/features.db`) để query nhanh ở Phase 6 mà không phải tính lại feature mỗi lần.

## Checklist task
- [x] Schema bảng `images` (`id, filename UNIQUE, width, height, file_size, feature_vector BLOB, created_at`)
- [x] `init_db / reset_db` quản lý vòng đời CSDL
- [x] `serialize_vector / deserialize_vector` chuyển `np.float32` ↔ BLOB
- [x] `insert_image / upsert_image` ghi 1 record (upsert dùng cho build incremental)
- [x] `count_images / existing_filenames / get_image_meta` truy vấn metadata
- [x] `load_database()` đọc toàn bộ về `DatabaseSnapshot(ids, filenames, vectors)`
- [x] `query(image_path, db, k)` end-to-end: load ảnh → extract → top-k
- [x] Script CLI `build_database.py` với `--rebuild`, `--limit`, progress mỗi 50 ảnh
- [x] Build CSDL trên 500 ảnh thật
- [x] Test serialize roundtrip, count, load shape, query in-DB, đo tốc độ

## Module
- `src/database.py` (logic CSDL)
- `build_database.py` (script CLI ở root)

## Schema CSDL

```sql
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    feature_vector BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_filename ON images(filename);
```

| Cột | Kiểu | Mục đích |
|-----|------|----------|
| `id` | INTEGER PK | Khóa nội bộ, auto-increment |
| `filename` | TEXT UNIQUE | Tên file (dùng để dedupe khi build incremental) |
| `width, height` | INTEGER | Kích thước gốc, log để verify dataset đồng nhất |
| `file_size` | INTEGER | Hiển thị metadata trong UI |
| `feature_vector` | BLOB | `np.float32(657,).tobytes()` = **2628 byte** |
| `created_at` | TIMESTAMP | Audit trail khi rebuild |

## Tại sao SQLite + BLOB?

| Tiêu chí | SQLite | CSV/JSON | Pickle |
|----------|--------|----------|--------|
| Built-in Python | ✓ | ✓ | ✓ |
| Query metadata theo điều kiện | ✓ | tự parse | ✗ |
| Lưu vector số nhị phân hiệu quả | ✓ (BLOB) | text base64 cồng kềnh | ✓ |
| An toàn ghi/đọc đồng thời | ✓ (transaction) | ✗ | ✗ |
| Đáp ứng yêu cầu "CSDL" của môn học | ✓ | ✗ | ✗ |

**BLOB vs lưu từng phần tử thành 657 cột:** BLOB đơn giản hơn nhiều, nhanh hơn, và lượng data nhỏ — không có lý do chuẩn hoá.

## Kích thước

- Vector: `657 × 4 byte = 2628 byte/ảnh`
- 500 ảnh: `500 × 2628 ≈ 1.3 MB BLOB` + metadata + index ≈ **2.0 MB** thực tế.

## Build pipeline

```
build_database.py
  └─> reset_db (nếu --rebuild)
  └─> for each *.jpg in dataset:
         load_image -> resize -> extract_features (657 dim)
         read_image_meta (width, height, file_size)
         upsert_image
      progress mỗi 50 ảnh, commit theo batch
```

## Tốc độ thực đo

### Build (lần đầu trên 500 ảnh)
```
[ 50/500] avg 23.4 ms/ảnh
[100/500] avg 22.5 ms/ảnh
...
[500/500] avg 20.3 ms/ảnh, elapsed 10.2s
```

- **Tổng thời gian: 10.16s** (rất nhanh so với ước lượng <5 phút trong plan).
- Trung bình **~20 ms/ảnh** = load JPG + bilinear resize + 576-dim color hist + Sobel + 81-dim shape hist + insert SQLite.
- File CSDL cuối: **2052 KB**.

### Query (DB đã load vào RAM)
- `load_database()`: **5.3 ms** cho 500 vector (deserialize 500 BLOB).
- `query(path, db, k=5)`: **~13 ms/query** (chủ yếu là extract feature ảnh truy vấn; tính khoảng cách + top-k chỉ vài ms).
- **Lần đầu** chậm hơn (~30 ms) do JIT/cache CPU; các lần sau ổn định.

## Phát hiện bất ngờ về dataset

Khi test `query` với `01_thien_nhien_001.jpg`, top-1 trả về `01_thien_nhien_099.jpg` với `d = 0.000000`. Kiểm tra MD5 toàn bộ dataset:

```
Total: 500, unique hashes: 422, duplicate pairs: 78
  001.jpg ≡ 099.jpg
  044.jpg ≡ 093.jpg
  017.jpg ≡ 101.jpg
  ... (tổng 78 cặp)
```

**Dataset chỉ có 422 ảnh duy nhất, 78 ảnh là bản sao byte-identical.** Điều này:
1. Không phải lỗi của thuật toán — feature extraction đúng (vector trùng nhau khi file trùng nhau).
2. Cần lưu ý ở Phase 7 khi đánh giá: xếp hạng top-5 có thể chứa ảnh trùng → khi đo recall/precision phải khử trùng trước.
3. Đã điều chỉnh test: thay vì assert "top-1 = self", assert "self phải có trong top-k và top-1 distance ~ 0".

## Hàm tiện ích cho Phase 6

```python
from src.database import load_database, query

db = load_database()              # cache 1 lần khi UI khởi động
q_vec, results = query(image_path, db=db, k=5)
# results = [('01_thien_nhien_127.jpg', 0.0234), ...]
```

`q_vec` (657,) sẽ dùng để in console theo format yêu cầu (Color 576 + Shape 81).

## Cách chạy

```bash
python build_database.py                    # build incremental
python build_database.py --rebuild          # xoá và build lại
python build_database.py --limit 50         # chỉ build 50 ảnh đầu (debug)
python notebooks/test_database.py           # chạy test
```
