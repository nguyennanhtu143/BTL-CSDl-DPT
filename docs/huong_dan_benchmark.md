# Hướng dẫn benchmark tổng quan

Script benchmark giúp bạn quan sát:

- **Số liệu**: thời gian query, self-hit@1, overlap top-k giữa pipeline
- **Thị giác**: ảnh query và top-k của từng pipeline đặt cạnh nhau

## 1) Chạy benchmark

```bash
python benchmark_pipelines.py --query-limit 10 --k 5
```

Tuỳ chọn query cụ thể:

```bash
python benchmark_pipelines.py --queries Images_Dataset/01_thien_nhien/01_thien_nhien_0001.jpg Images_Dataset/test/test_01.jpg --k 5
```

## 2) Kết quả xuất ra

Trong thư mục `reports/` sẽ có:

- `benchmark_..._summary.json`: tổng hợp chỉ số
- `benchmark_..._details.csv`: bảng chi tiết theo từng query/pipeline
- `benchmark_..._report.html`: gallery trực quan để so sánh ảnh

## 3) Quan sát tổng quan hiệu quả

- Mở `summary.json` để xem:
  - `avg_query_ms` (pipeline nào nhanh hơn)
  - `self_at_1_rate` (độ ổn định khi query ảnh đã có trong DB)
  - `avg_overlap_topk_jaccard` (mức khác biệt kết quả top-k giữa pipeline)
- Mở `details.csv` để lọc theo từng query bị khó.

## 4) Quan sát khác biệt thị giác

- Mở `report.html` bằng trình duyệt.
- Mỗi block gồm:
  - 1 ảnh query
  - 4 cột kết quả (`legacy`, `compact6`, `hybrid`, `hybrid3`)
  - mỗi cột có top-k và distance tương ứng
- Dùng các query “dễ nhầm” (rừng/biển/sa mạc) để thấy pipeline nào cân bằng màu + bố cục tốt hơn.
