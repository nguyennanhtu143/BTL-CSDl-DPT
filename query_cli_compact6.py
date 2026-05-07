from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR
from src.database_compact6 import count_images, load_database, query

DB_COMPACT6_DEFAULT = DATA_DIR / "features_compact6.db"


def fmt_vec(v: list[float]) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description="CBIR compact6 query CLI")
    parser.add_argument("image", type=Path, help="Đường dẫn ảnh truy vấn")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=DB_COMPACT6_DEFAULT)
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Không tìm thấy ảnh: {args.image}")
    if count_images(args.db) == 0:
        raise SystemExit(f"CSDL compact6 rỗng: {args.db}. Chạy `python build_database_compact6.py` trước.")

    t0 = time.perf_counter()
    db = load_database(args.db)
    t_load = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    q_vec, results = query(args.image, db=db, k=args.k)
    t_query = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print(f"[QUERY] Image: {args.image.name}")
    print("=" * 60)
    print(f"[FEATURE VECTOR] Shape: {q_vec.shape}")
    print(f"  - Compact6: {fmt_vec(q_vec.tolist())}")
    print("-" * 60)
    print(f"[TOP {len(results)} RESULTS]")
    print(f"Rank | {'Filename':<28} | Distance")
    for i, (name, dist) in enumerate(results, 1):
        print(f"  {i:<2} | {name:<28} | {dist:.4f}")
    print("=" * 60)
    print(f"[timing] load_db = {t_load:.1f} ms, query = {t_query:.1f} ms")


if __name__ == "__main__":
    main()
