# Hướng dẫn sử dụng phiên bản Compact6

Tài liệu này dành riêng cho phiên bản đặc trưng 6 chiều:

`[mean, stddev, skewness, coarseness, contrast, directionality]`

Pipeline compact6 chạy **song song** với pipeline chính và dùng DB riêng.

## 1) Build CSDL compact6

```bash
python build_database_compact6.py --rebuild
```

Tùy chọn:

- `--limit 50`: build nhanh 50 ảnh đầu
- `--dataset <path>`: đổi thư mục ảnh
- `--db <path>`: đổi file DB (mặc định `data/features_compact6.db`)

## 2) Query compact6

```bash
python query_cli_compact6.py Images_Dataset/01_thien_nhien/01_thien_nhien_0001.jpg --k 5
```

Tùy chọn:

- `--db <path>`: dùng DB compact6 khác
- `--k 10`: trả top-10

## 3) Xem dữ liệu trong DB compact6

Schema bảng:

- `images_compact6`
  - metadata: `filename, width, height, file_size`
  - scalar: `mean, stddev, skewness, coarseness, contrast, directionality`
  - `feature6_vector` (BLOB float32, dim=6)

Xem nhanh bằng Python:

```bash
python -c "import sqlite3; c=sqlite3.connect('data/features_compact6.db'); print([ (r[1], r[2]) for r in c.execute('PRAGMA table_info(images_compact6)') ]); print(c.execute('SELECT filename,mean,stddev,skewness,coarseness,contrast,directionality FROM images_compact6 LIMIT 5').fetchall())"
```

## 4) Ghi chú

- Phiên bản compact6 **không thay đổi** pipeline chính hiện tại.
- Hai phiên bản có thể đánh giá song song trên cùng tập query để so sánh.
