from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR
from src.database_hybrid3 import count_images, load_database, query

DB_HYBRID3_DEFAULT = DATA_DIR / "features_hybrid3.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid-3 query CLI (color + gradient + compact6)")
    parser.add_argument("image", type=Path, help="Đường dẫn ảnh truy vấn")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=DB_HYBRID3_DEFAULT)
    parser.add_argument("--w-hist", type=float, default=0.45)
    parser.add_argument("--w-grad", type=float, default=0.30)
    parser.add_argument("--w-compact", type=float, default=0.25)
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Không tìm thấy ảnh: {args.image}")
    if count_images(args.db) == 0:
        raise SystemExit(f"CSDL hybrid-3 rỗng: {args.db}. Chạy `python build_database_hybrid3.py` trước.")

    t0 = time.perf_counter()
    db = load_database(args.db)
    t_load = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    q_hist, q_grad, q_compact, results = query(
        image_path=args.image,
        db=db,
        k=args.k,
        w_hist=args.w_hist,
        w_grad=args.w_grad,
        w_compact=args.w_compact,
    )
    t_query = (time.perf_counter() - t0) * 1000

    print("=" * 70)
    print(f"[QUERY] Image: {args.image.name}")
    print("=" * 70)
    print(
        f"[FEATURE] ColorHist: {q_hist.shape}, Gradient: {q_grad.shape}, Compact6: {q_compact.shape}"
    )
    print(
        f"[WEIGHT] w_hist={args.w_hist:.2f}, w_grad={args.w_grad:.2f}, w_compact={args.w_compact:.2f}"
    )
    print("-" * 70)
    print(f"[TOP {len(results)} RESULTS]")
    print(f"Rank | {'Filename':<28} | Distance")
    for i, (name, dist) in enumerate(results, 1):
        print(f"  {i:<2} | {name:<28} | {dist:.4f}")
    print("=" * 70)
    print(f"[timing] load_db = {t_load:.1f} ms, query = {t_query:.1f} ms")


if __name__ == "__main__":
    main()
