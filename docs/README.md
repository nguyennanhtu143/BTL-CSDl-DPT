# Tài liệu triển khai theo Phase

📖 **Người dùng cuối:** đọc [`huong_dan_su_dung.md`](huong_dan_su_dung.md) để cài đặt, build CSDL, chạy CLI/Web UI và xem dữ liệu CSDL.
🧭 **Tổng quan luồng hệ thống:** xem [`system_flow.md`](system_flow.md) để nắm pipeline khởi tạo metadata và truy vấn top-5.

Mỗi file phase mô tả một phase trong [`implementation_plan.md`](../implementation_plan.md): mục tiêu, checklist task, thuật toán, tham số, và kết quả test.

| Phase | Trạng thái | Tài liệu | Module chính |
|-------|-----------|----------|--------------|
| 0 — Khởi tạo dự án | ✅ Done | (cấu hình trong `src/config.py`) | — |
| 1 — Tiền xử lý ảnh | ✅ Done | [phase_1_preprocessing.md](phase_1_preprocessing.md) | `src/preprocessing.py` |
| 2 — Color Histogram | ✅ Done | [phase_2_color_features.md](phase_2_color_features.md) | `src/color_features.py` |
| 3 — Gradient Histogram | ✅ Done | [phase_3_gradient_features.md](phase_3_gradient_features.md) | `src/gradient_features.py` |
| 4 — Tích hợp & So sánh | ✅ Done | [phase_4_integration.md](phase_4_integration.md) | `src/feature_extractor.py`, `src/matcher.py` |
| 5 — CSDL SQLite | ✅ Done | [phase_5_database.md](phase_5_database.md) | `src/database.py`, `build_database.py` |
| 6 — UI Streamlit & Console | ✅ Done | [phase_6_ui.md](phase_6_ui.md) | `src/logger.py`, `query_cli.py`, `ui/app.py` |
| 7 — Đánh giá & Báo cáo | ⏳ Pending | — | `report.md` |

## Chạy test nhanh
```bash
python notebooks/test_preprocessing.py
python notebooks/test_color_features.py
python notebooks/test_gradient_features.py
python notebooks/test_integration.py
python notebooks/test_database.py
```

## Build CSDL
```bash
python build_database.py --rebuild       # 500 ảnh, ~10s
```

## Query
```bash
# CLI (không cần browser)
python query_cli.py path/to/image.jpg
python query_cli.py path/to/image.jpg --full

# Web UI
streamlit run ui/app.py
```

## Vector đặc trưng cuối cùng
- Color: **576 chiều** (lưới 3×3 × RGB 4×4×4)
- Shape: **81 chiều** (lưới 3×3 × 9 bin gradient)
- Tổng: **657 chiều** với trọng số 0.7 · Color + 0.3 · Shape
