# Hướng dẫn Hybrid-3: Color + Gradient + Compact6

Pipeline `hybrid-3` kết hợp 3 nhánh:

- `color histogram` (nhạy màu)
- `gradient histogram` (nhạy biên/cấu trúc)
- `compact6` (nhạy bố cục/texture tổng quát)

Khoảng cách tổng hợp:

`d = w_hist * d_hist_norm + w_grad * d_grad_norm + w_compact * d_compact_norm`

Mặc định:

- `w_hist = 0.45`
- `w_grad = 0.30`
- `w_compact = 0.25`

## Build CSDL

```bash
python build_database_hybrid3.py --rebuild
```

Tuỳ chọn:

- `--limit 50`
- `--dataset <path>`
- `--db <path>` (mặc định: `data/features_hybrid3.db`)

## Query

```bash
python query_cli_hybrid3.py Images_Dataset/01_thien_nhien/01_thien_nhien_0001.jpg --k 5
```

Tinh chỉnh trọng số:

```bash
python query_cli_hybrid3.py Images_Dataset/01_thien_nhien/01_thien_nhien_0001.jpg --w-hist 0.5 --w-grad 0.3 --w-compact 0.2
```

## Schema

Bảng `images_hybrid3` lưu:

- metadata (`filename,width,height,file_size`)
- scalar compact6 (`mean,stddev,skewness,coarseness,contrast,directionality`)
- `color_hist_vector` (576 chiều)
- `grad_vector` (81 chiều)
- `compact6_vector` (6 chiều)
