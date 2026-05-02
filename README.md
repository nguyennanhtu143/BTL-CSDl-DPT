# Hệ thống tìm kiếm ảnh nền thiên nhiên (CBIR)

Tìm kiếm 5 ảnh tương đồng nhất từ tập 500 ảnh nền thiên nhiên dựa trên đặc trưng màu sắc (Grid-based RGB Color Histogram) và hình dạng (Sobel Gradient Histogram).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
BTL-CSDL-DPT/
├── Images_Dataset/01_thien_nhien/   500 ảnh nền 640x360
├── src/
│   ├── config.py                    Hằng số toàn cục
│   ├── preprocessing.py             Load, resize, grayscale (Phase 1)
│   ├── color_features.py            Color histogram (Phase 2)
│   ├── gradient_features.py         Gradient histogram (Phase 3)
│   ├── feature_extractor.py         Tích hợp đặc trưng (Phase 4)
│   ├── matcher.py                   Khoảng cách + top-K (Phase 4)
│   └── database.py                  SQLite (Phase 5)
├── data/features.db                 CSDL đặc trưng (sinh ở Phase 5)
├── ui/app.py                        Streamlit UI (Phase 6)
├── main.py                          Entry point CLI
├── implementation_plan.md           Kế hoạch chi tiết theo phase
└── requirements.txt
```

## Sử dụng

Build CSDL đặc trưng (chạy 1 lần):
```bash
python main.py build
```

Chạy giao diện web:
```bash
streamlit run ui/app.py
```
