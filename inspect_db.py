"""Tiện ích xem nội dung CSDL features.db.

Cách dùng:
    python inspect_db.py                          # tổng quan + 5 record đầu
    python inspect_db.py --limit 20               # 20 record đầu
    python inspect_db.py --filename 01_thien_nhien_001.jpg   # chi tiết 1 ảnh
    python inspect_db.py --schema                 # in schema bảng
    python inspect_db.py --vector 01_thien_nhien_001.jpg --full   # in toàn bộ vector
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.config import COLOR_DIM, DB_PATH, GRAD_DIM, TOTAL_DIM
from src.database import connect, deserialize_vector
from src.logger import format_vector_preview


def show_overview(db_path: Path) -> None:
    if not db_path.exists():
        raise SystemExit(f"Không tìm thấy CSDL: {db_path}\nChạy `python build_database.py` trước.")

    size_kb = db_path.stat().st_size / 1024
    with connect(db_path) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM images").fetchone()
        (min_id, max_id) = conn.execute("SELECT MIN(id), MAX(id) FROM images").fetchone()
        widths = conn.execute("SELECT DISTINCT width FROM images").fetchall()
        heights = conn.execute("SELECT DISTINCT height FROM images").fetchall()
        (min_size, max_size, avg_size) = conn.execute(
            "SELECT MIN(file_size), MAX(file_size), AVG(file_size) FROM images"
        ).fetchone()

    print("=" * 70)
    print("TỔNG QUAN CSDL")
    print("=" * 70)
    print(f"Đường dẫn       : {db_path}")
    print(f"Kích thước file : {size_kb:.1f} KB")
    print(f"Số record       : {n}")
    print(f"ID range        : {min_id} - {max_id}")
    print(f"Width (gốc)     : {[w[0] for w in widths]}")
    print(f"Height (gốc)    : {[h[0] for h in heights]}")
    if avg_size is not None:
        print(f"file_size (byte): min={min_size}, max={max_size}, avg={avg_size:.0f}")
    print(f"Vector dim      : {TOTAL_DIM} (Color {COLOR_DIM} + Shape {GRAD_DIM})")


def show_schema(db_path: Path) -> None:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table', 'index') AND tbl_name = 'images'"
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
            "SELECT id, filename, width, height, file_size, length(feature_vector), created_at "
            "FROM images ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()

    print("\n" + "=" * 70)
    print(f"DANH SÁCH {limit} RECORD ĐẦU")
    print("=" * 70)
    print(f"{'ID':<4} {'Filename':<30} {'WxH':<11} {'KB':>5} {'BLOB':>6} {'Created'}")
    print("-" * 70)
    for id_, fn, w, h, fs, blen, ts in rows:
        print(f"{id_:<4} {fn:<30} {w}x{h:<6} {fs/1024:>5.1f} {blen:>6} {ts}")


def show_filename(db_path: Path, filename: str, full_vector: bool = False) -> None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, filename, width, height, file_size, feature_vector, created_at "
            "FROM images WHERE filename = ?",
            (filename,),
        ).fetchone()

    if not row:
        raise SystemExit(f"Không tìm thấy ảnh '{filename}' trong CSDL.")

    id_, fn, w, h, fs, blob, ts = row
    vec = deserialize_vector(blob)
    color = vec[:COLOR_DIM]
    shape = vec[COLOR_DIM:]

    print("\n" + "=" * 70)
    print(f"CHI TIẾT RECORD: {filename}")
    print("=" * 70)
    print(f"ID          : {id_}")
    print(f"Width × Height : {w} × {h}")
    print(f"File size   : {fs} byte ({fs/1024:.1f} KB)")
    print(f"BLOB length : {len(blob)} byte ({TOTAL_DIM} × float32)")
    print(f"Created at  : {ts}")
    print("-" * 70)
    print(f"VECTOR ĐẶC TRƯNG ({TOTAL_DIM} chiều)")
    print(f"  Tổng sum  : {vec.sum():.6f}  (kì vọng ~ 1.0)")
    print(f"  Color sum : {color.sum():.6f}  (kì vọng = 0.7)")
    print(f"  Shape sum : {shape.sum():.6f}  (kì vọng = 0.3)")
    print(f"  Min / Max : {vec.min():.6f} / {vec.max():.6f}")
    print(f"  Non-zero  : Color {(color > 0).sum()}/{COLOR_DIM}, Shape {(shape > 0).sum()}/{GRAD_DIM}")

    if full_vector:
        with np.printoptions(threshold=np.inf, precision=6, linewidth=120, suppress=True):
            print(f"\n  Color ({COLOR_DIM} dim):")
            print(np.array2string(color, separator=", ", prefix="    "))
            print(f"\n  Shape ({GRAD_DIM} dim):")
            print(np.array2string(shape, separator=", ", prefix="    "))
    else:
        print(f"\n  Color preview: {format_vector_preview(color, n_head=8, n_tail=4)}")
        print(f"  Shape preview: {format_vector_preview(shape, n_head=8, n_tail=4)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Xem nội dung CSDL features.db")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--schema", action="store_true", help="Hiển thị schema")
    parser.add_argument("--limit", type=int, default=5, help="Số record đầu in ra (default 5)")
    parser.add_argument("--filename", type=str, help="Xem chi tiết 1 ảnh theo tên file")
    parser.add_argument("--full", action="store_true", help="In toàn bộ vector (kết hợp với --filename)")
    args = parser.parse_args()

    show_overview(args.db)
    if args.schema:
        show_schema(args.db)

    if args.filename:
        show_filename(args.db, args.filename, full_vector=args.full)
    else:
        show_samples(args.db, args.limit)


if __name__ == "__main__":
    main()
