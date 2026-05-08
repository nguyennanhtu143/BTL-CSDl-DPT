"""Tiện ích xem nội dung CSDL features_hybrid3.db.

Cách dùng:
    python inspect_db.py                                          # tổng quan + 5 record đầu
    python inspect_db.py --limit 20                               # 20 record đầu
    python inspect_db.py --filename 01_thien_nhien_0001.jpg       # chi tiết 1 ảnh
    python inspect_db.py --filename ... --full                    # in toàn bộ vector
    python inspect_db.py --schema                                 # in schema bảng
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.config import DATA_DIR, GRAD_DIM
from src.database_hybrid3 import COMPACT6_DIM, HIST_DIM, connect, deserialize_vec

DB_DEFAULT = DATA_DIR / "features_hybrid3.db"


def show_overview(db_path: Path) -> None:
    if not db_path.exists():
        raise SystemExit(
            f"Không tìm thấy CSDL: {db_path}\nChạy `python build_database_hybrid3.py` trước."
        )

    size_kb = db_path.stat().st_size / 1024
    with connect(db_path) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM images_hybrid3").fetchone()
        min_id, max_id = conn.execute("SELECT MIN(id), MAX(id) FROM images_hybrid3").fetchone()
        min_size, max_size, avg_size = conn.execute(
            "SELECT MIN(file_size), MAX(file_size), AVG(file_size) FROM images_hybrid3"
        ).fetchone()

    print("=" * 70)
    print("TỔNG QUAN CSDL HYBRID3")
    print("=" * 70)
    print(f"Đường dẫn       : {db_path}")
    print(f"Kích thước file : {size_kb:.1f} KB")
    print(f"Số record       : {n}")
    print(f"ID range        : {min_id} - {max_id}")
    if avg_size is not None:
        print(f"file_size (byte): min={min_size}, max={max_size}, avg={avg_size:.0f}")
    print(f"Vector dims     : Hist {HIST_DIM} + Grad {GRAD_DIM} + Compact6 {COMPACT6_DIM}")


def show_schema(db_path: Path) -> None:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table', 'index') "
            "AND tbl_name = 'images_hybrid3'"
        ).fetchall()
    print("\n" + "=" * 70)
    print("SCHEMA")
    print("=" * 70)
    for (sql,) in rows:
        if sql:
            print(sql + ";")


def show_samples(db_path: Path, limit: int) -> None:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, filename, width, height, file_size, "
            "length(color_hist_vector), length(grad_vector), length(compact6_vector), created_at "
            "FROM images_hybrid3 ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()

    print("\n" + "=" * 70)
    print(f"DANH SÁCH {limit} RECORD ĐẦU")
    print("=" * 70)
    print(
        f"{'ID':<4} {'Filename':<30} {'WxH':<11} {'KB':>5} "
        f"{'Hist':>6} {'Grad':>5} {'Cmp6':>5}"
    )
    print("-" * 70)
    for id_, fn, w, h, fs, hist_b, grad_b, cmp_b, _ in rows:
        print(
            f"{id_:<4} {fn:<30} {w}x{h:<6} {fs/1024:>5.1f} "
            f"{hist_b:>6} {grad_b:>5} {cmp_b:>5}"
        )


def show_filename(db_path: Path, filename: str, full: bool = False) -> None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, filename, width, height, file_size, "
            "color_hist_vector, grad_vector, compact6_vector, "
            "mean, stddev, skewness, coarseness, contrast, directionality, created_at "
            "FROM images_hybrid3 WHERE filename = ?",
            (filename,),
        ).fetchone()

    if not row:
        raise SystemExit(f"Không tìm thấy ảnh '{filename}' trong CSDL.")

    (id_, fn, w, h, fs, hist_blob, grad_blob, cmp_blob,
     mean, std, skew, coarse, contrast, direc, ts) = row
    hist = deserialize_vec(hist_blob, HIST_DIM)
    grad = deserialize_vec(grad_blob, GRAD_DIM)
    cmp6 = deserialize_vec(cmp_blob, COMPACT6_DIM)

    print("\n" + "=" * 70)
    print(f"CHI TIẾT RECORD: {filename}")
    print("=" * 70)
    print(f"ID            : {id_}")
    print(f"Width × Height: {w} × {h}")
    print(f"File size     : {fs} byte ({fs/1024:.1f} KB)")
    print(f"Created at    : {ts}")
    print("-" * 70)
    print("COMPACT6 SCALARS:")
    print(f"  mean         = {mean:.4f}    coarseness     = {coarse:.4f}")
    print(f"  stddev       = {std:.4f}    contrast       = {contrast:.4f}")
    print(f"  skewness     = {skew:.4f}    directionality = {direc:.4f}")
    print("-" * 70)
    print("VECTOR ĐẶC TRƯNG:")
    print(f"  Color hist ({HIST_DIM}-d): sum={hist.sum():.4f}, non-zero {(hist>0).sum()}/{HIST_DIM}")
    print(f"  Gradient   ({GRAD_DIM}-d) : sum={grad.sum():.4f}, non-zero {(grad>0).sum()}/{GRAD_DIM}")
    print(f"  Compact6   ({COMPACT6_DIM}-d)  : {np.array2string(cmp6, precision=4)}")

    if full:
        with np.printoptions(threshold=np.inf, precision=6, linewidth=120, suppress=True):
            print(f"\n  Color hist:")
            print(np.array2string(hist, separator=", ", prefix="    "))
            print(f"\n  Gradient:")
            print(np.array2string(grad, separator=", ", prefix="    "))


def main() -> None:
    parser = argparse.ArgumentParser(description="Xem nội dung CSDL features_hybrid3.db")
    parser.add_argument("--db", type=Path, default=DB_DEFAULT)
    parser.add_argument("--schema", action="store_true", help="Hiển thị schema")
    parser.add_argument("--limit", type=int, default=5, help="Số record đầu in ra")
    parser.add_argument("--filename", type=str, help="Xem chi tiết 1 ảnh theo tên file")
    parser.add_argument("--full", action="store_true", help="In toàn bộ vector (kết hợp --filename)")
    args = parser.parse_args()

    show_overview(args.db)
    if args.schema:
        show_schema(args.db)

    if args.filename:
        show_filename(args.db, args.filename, full=args.full)
    else:
        show_samples(args.db, args.limit)


if __name__ == "__main__":
    main()
