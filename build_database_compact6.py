from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.compact6_features import extract_compact6_from_path
from src.config import DATASET_DIR, DATA_DIR
from src.database_compact6 import (
    connect,
    count_images,
    existing_filenames,
    init_db,
    read_image_meta,
    reset_db,
    upsert_image,
)

DB_COMPACT6_DEFAULT = DATA_DIR / "features_compact6.db"
PROGRESS_EVERY = 50


def build(dataset_dir: Path, db_path: Path, rebuild: bool = False, limit: int | None = None) -> None:
    if not dataset_dir.exists():
        raise SystemExit(f"Không tìm thấy dataset: {dataset_dir}")
    files = sorted(dataset_dir.glob("*.jpg"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise SystemExit(f"Không có file .jpg trong {dataset_dir}")

    if rebuild:
        print(f"[build-compact6] --rebuild: xoá CSDL cũ tại {db_path}")
        reset_db(db_path)
    else:
        init_db(db_path)

    skip = existing_filenames(db_path)
    to_process = [f for f in files if f.name not in skip]

    print(f"[build-compact6] Dataset   : {dataset_dir}")
    print(f"[build-compact6] CSDL      : {db_path}")
    print(f"[build-compact6] Tổng ảnh   : {len(files)}")
    print(f"[build-compact6] Đã có     : {len(skip)} (skip)")
    print(f"[build-compact6] Cần xử lý : {len(to_process)}")
    print("-" * 60)

    if not to_process:
        print("[build-compact6] Không có ảnh mới. Done.")
        return

    t_start = time.perf_counter()
    times: list[float] = []
    fails: list[tuple[str, str]] = []

    with connect(db_path) as conn:
        for i, p in enumerate(to_process, start=1):
            t0 = time.perf_counter()
            try:
                width, height, file_size = read_image_meta(p)
                vec6, scalars = extract_compact6_from_path(p)
                upsert_image(
                    conn,
                    filename=p.name,
                    width=width,
                    height=height,
                    file_size=file_size,
                    vec6=vec6,
                    scalars=scalars,
                )
            except Exception as exc:  # noqa: BLE001
                fails.append((p.name, str(exc)))
                continue
            dt = time.perf_counter() - t0
            times.append(dt)
            if i % PROGRESS_EVERY == 0 or i == len(to_process):
                avg = sum(times) / len(times)
                elapsed = time.perf_counter() - t_start
                remaining = avg * (len(to_process) - i)
                print(f"  [{i:>4}/{len(to_process)}] avg {avg*1000:6.1f} ms/ảnh, elapsed {elapsed:6.1f}s, ETA {remaining:5.1f}s")
                conn.commit()
        conn.commit()

    print("-" * 60)
    print(f"[build-compact6] Hoàn tất xử lý {len(to_process)} ảnh trong {time.perf_counter() - t_start:.2f}s")
    if fails:
        print(f"[build-compact6] {len(fails)} ảnh lỗi")
    print(f"[build-compact6] Tổng record: {count_images(db_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSDL compact6")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--db", type=Path, default=DB_COMPACT6_DEFAULT)
    args = parser.parse_args()
    build(args.dataset, args.db, rebuild=args.rebuild, limit=args.limit)


if __name__ == "__main__":
    main()
