"""Phase 6 (CLI): query 1 ảnh không cần browser, in console theo format đề bài.

Cách dùng:
    python query_cli.py path/to/image.jpg
    python query_cli.py path/to/image.jpg --k 10 --full
    python query_cli.py path/to/image.jpg --full      # in toàn bộ 657 phần tử
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.config import DB_PATH, TOP_K, W_COLOR, W_FREQ, W_SHAPE
from src.database import count_images, load_database, query
from src.feature_extractor import extract_components_from_path
from src.logger import log_query
from src.retrieval_accel import AccelRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="CBIR query CLI")
    parser.add_argument("image", type=Path, help="Đường dẫn ảnh truy vấn (.jpg/.png)")
    parser.add_argument("--k", type=int, default=TOP_K, help="Số kết quả top-k")
    parser.add_argument("--full", action="store_true", help="In toàn bộ vector ra console")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument(
        "--no-accel",
        action="store_true",
        help="Tắt truy vấn tăng tốc PCA+IVF, dùng full-scan",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Không tìm thấy ảnh: {args.image}")
    if count_images(args.db) == 0:
        raise SystemExit(f"CSDL rỗng: {args.db}. Chạy `python build_database.py` trước.")

    t0 = time.perf_counter()
    db = load_database(args.db)
    t_load = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if args.no_accel:
        q_vec, results = query(args.image, db=db, k=args.k)
    else:
        try:
            q_color, q_shape, q_freq = extract_components_from_path(args.image)
            q_vec = np.concatenate([W_COLOR * q_color, W_SHAPE * q_shape, W_FREQ * q_freq]).astype(
                np.float32
            )
            accel = AccelRetriever(db=db)
            results = accel.query(q_color=q_color, q_shape=q_shape, q_freq=q_freq, k=args.k)
        except Exception:
            q_vec, results = query(args.image, db=db, k=args.k)
    t_query = (time.perf_counter() - t0) * 1000

    log_query(args.image.name, q_vec, results, full_vector=args.full)
    print(f"[timing] load_db = {t_load:.1f} ms, query = {t_query:.1f} ms")


if __name__ == "__main__":
    main()
