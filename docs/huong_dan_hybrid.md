# Hướng dẫn Hybrid: Color Histogram + Compact6

Pipeline hybrid kết hợp:

- Nhánh màu: `color histogram` (576 chiều)
- Nhánh bố cục gọn: `compact6` (6 chiều)

Điểm tương đồng:

`d_hybrid = w_hist * d_color_hist_norm + w_compact * d_compact6_norm`

Mặc định:

- `w_hist = 0.65`
- `w_compact = 0.35`

## 1) Build CSDL hybrid

```bash
python build_database_hybrid.py --rebuild
```

Tùy chọn:

- `--limit 50`: build nhanh 50 ảnh
- `--dataset <path>`: đổi dataset
- `--db <path>`: đổi DB (mặc định `data/features_hybrid.db`)

## 2) Query hybrid

```bash
python query_cli_hybrid.py Images_Dataset/01_thien_nhien/01_thien_nhien_0001.jpg --k 5
```

Tùy chỉnh trọng số:

```bash
python query_cli_hybrid.py Images_Dataset/01_thien_nhien/01_thien_nhien_0001.jpg --w-hist 0.75 --w-compact 0.25
```

Gợi ý:

- Nếu muốn nhạy màu hơn: tăng `--w-hist`
- Nếu muốn nhạy bố cục hơn: tăng `--w-compact`

## 3) Schema DB hybrid

DB mặc định: `data/features_hybrid.db`  
Bảng: `images_hybrid`

- Metadata: `filename, width, height, file_size`
- Scalar compact6: `mean, stddev, skewness, coarseness, contrast, directionality`
- Vector:
  - `color_hist_vector` (576 dim)
  - `compact6_vector` (6 dim)

## 4) So sánh với các pipeline khác

- Legacy: nhạy màu tốt, ít thông tin bố cục cao cấp.
- Compact6: gọn, nhạy bố cục/hình dạng hơn.
- Hybrid: cân bằng màu + bố cục, phù hợp trường hợp dễ nhầm giữa ảnh bố cục giống nhưng màu khác.
