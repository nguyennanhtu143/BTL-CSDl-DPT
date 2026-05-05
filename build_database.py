"""Phase 5: Script build CSDL đặc trưng cho 500 ảnh thiên nhiên.

Cách dùng:
    python build_database.py                  # build incremental (skip ảnh đã có)
    python build_database.py --rebuild        # xoá CSDL và build lại từ đầu
    python build_database.py --limit 50       # chỉ xử lý 50 ảnh đầu (test nhanh)

Output:
    data/features.db (SQLite)
    Console log: progress mỗi 50 ảnh, thời gian/ảnh, tổng thời gian

Lưu ý: đổi USE_LAB_COLOR_HISTOGRAM trong src/config.py -> phải build lại CSDL (--rebuild)
        vì vector 657 chiều (phần màu) không còn tương thích với DB cũ.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATASET_DIR, DB_PATH, USE_LAB_COLOR_HISTOGRAM
from src.database import (
    connect,
    count_images,
    existing_filenames,
    init_db,
    read_image_meta,
    reset_db,
    upsert_image,
)
from src.feature_extractor import extract_from_path

PROGRESS_EVERY = 50


def build(
    dataset_dir: Path = DATASET_DIR,
    db_path: Path = DB_PATH,
    rebuild: bool = False,
    limit: int | None = None,
) -> None:
    if not dataset_dir.exists():
        raise SystemExit(f"Không tìm thấy dataset: {dataset_dir}")

    files = sorted(dataset_dir.glob("*.jpg"))
    if limit is not None:
        files = files[:limit]
    total = len(files)
    if total == 0:
        raise SystemExit(f"Không có file .jpg trong {dataset_dir}")

    if rebuild:
        print(f"[build] --rebuild: xoá CSDL cũ tại {db_path}")
        reset_db(db_path)
    else:
        init_db(db_path)

    skip = existing_filenames(db_path)
    to_process = [f for f in files if f.name not in skip]

    print(f"[build] Dataset    : {dataset_dir}")
    print(f"[build] CSDL       : {db_path}")
    print(f"[build] Histogram màu: {'LAB (CIE D65)' if USE_LAB_COLOR_HISTOGRAM else 'RGB'}")
    print(f"[build] Tổng ảnh    : {total}")
    print(f"[build] Đã có      : {len(skip)} (skip)")
    print(f"[build] Cần xử lý  : {len(to_process)}")
    print("-" * 60)

    if not to_process:
        print("[build] Không có ảnh mới. Done.")
        return

    t_start = time.perf_counter()
    times: list[float] = []
    failures: list[tuple[str, str]] = []

    with connect(db_path) as conn:
        for i, path in enumerate(to_process, start=1):
            t0 = time.perf_counter()
            try:
                width, height, file_size = read_image_meta(path)
                vector = extract_from_path(path)
                upsert_image(
                    conn,
                    filename=path.name,
                    width=width,
                    height=height,
                    file_size=file_size,
                    vector=vector,
                )
            except Exception as exc:
                failures.append((path.name, str(exc)))
                continue

            dt = time.perf_counter() - t0
            times.append(dt)

            if i % PROGRESS_EVERY == 0 or i == len(to_process):
                avg = sum(times) / len(times)
                elapsed = time.perf_counter() - t_start
                remaining = avg * (len(to_process) - i)
                print(
                    f"  [{i:>4}/{len(to_process)}] "
                    f"avg {avg*1000:6.1f} ms/ảnh, "
                    f"elapsed {elapsed:6.1f}s, ETA {remaining:5.1f}s"
                )
                conn.commit()

        conn.commit()

    total_time = time.perf_counter() - t_start
    print("-" * 60)
    print(f"[build] Hoàn tất xử lý {len(to_process)} ảnh trong {total_time:.2f}s")
    if times:
        print(f"[build] Trung bình: {sum(times)/len(times)*1000:.1f} ms/ảnh")
    if failures:
        print(f"[build] {len(failures)} ảnh lỗi:")
        for name, err in failures[:10]:
            print(f"    - {name}: {err}")

    final = count_images(db_path)
    db_size_kb = db_path.stat().st_size / 1024 if db_path.exists() else 0
    print(f"[build] Tổng record trong CSDL: {final}")
    print(f"[build] Kích thước file       : {db_size_kb:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSDL đặc trưng CBIR")
    parser.add_argument("--rebuild", action="store_true", help="Xoá CSDL cũ trước khi build")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N ảnh đầu (debug)")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    build(
        dataset_dir=args.dataset,
        db_path=args.db,
        rebuild=args.rebuild,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
