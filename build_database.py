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

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATASET_DIR, DB_PATH, USE_LAB_COLOR_HISTOGRAM, W_COLOR, W_FREQ, W_SHAPE
from src.database import (
    connect,
    count_images,
    existing_filenames,
    init_db,
    load_database,
    read_image_meta,
    reset_db,
    upsert_image,
)
from src.feature_extractor import extract_components_from_path
from src.ivf_index import build_ivf, save_ivf
from src.pca_utils import fit_pca, save_pca_bundle, transform_pca

PROGRESS_EVERY = 50


def build(
    dataset_dir: Path = DATASET_DIR,
    db_path: Path = DB_PATH,
    rebuild: bool = False,
    limit: int | None = None,
    fit_pca_enabled: bool = True,
    build_index: bool = True,
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
                color_vec, shape_vec, freq_vec = extract_components_from_path(path)
                vector = np.concatenate([W_COLOR * color_vec, W_SHAPE * shape_vec, W_FREQ * freq_vec]).astype(
                    np.float32
                )
                upsert_image(
                    conn,
                    filename=path.name,
                    width=width,
                    height=height,
                    file_size=file_size,
                    vector=vector,
                    color_vector=color_vec,
                    shape_vector=shape_vec,
                    freq_vector=freq_vec,
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

    if fit_pca_enabled:
        from src.config import (
            IVF_INDEX_PATH,
            IVF_NLIST,
            PCA_COLOR_DIM,
            PCA_FREQ_DIM,
            PCA_MODEL_PATH,
            PCA_SHAPE_DIM,
        )

        db = load_database(db_path)
        if len(db) > 0:
            print("[pca] Fitting PCA cho color/shape/freq...")
            c_mean, c_comp = fit_pca(db.color_vectors, PCA_COLOR_DIM)
            s_mean, s_comp = fit_pca(db.shape_vectors, PCA_SHAPE_DIM)
            f_mean, f_comp = fit_pca(db.freq_vectors, PCA_FREQ_DIM)
            save_pca_bundle(
                PCA_MODEL_PATH,
                color_mean=c_mean,
                color_components=c_comp,
                shape_mean=s_mean,
                shape_components=s_comp,
                freq_mean=f_mean,
                freq_components=f_comp,
            )
            print(f"[pca] Saved: {PCA_MODEL_PATH}")

            if build_index:
                print("[ivf] Building IVF index trên vector PCA...")
                c_pca = transform_pca(db.color_vectors, c_mean, c_comp)
                s_pca = transform_pca(db.shape_vectors, s_mean, s_comp)
                f_pca = transform_pca(db.freq_vectors, f_mean, f_comp)
                joined = np.concatenate([W_COLOR * c_pca, W_SHAPE * s_pca, W_FREQ * f_pca], axis=1).astype(
                    np.float32
                )
                ivf = build_ivf(joined, nlist=IVF_NLIST, iters=25, seed=42)
                save_ivf(IVF_INDEX_PATH, ivf)
                print(f"[ivf] Saved: {IVF_INDEX_PATH} (nlist={ivf['centroids'].shape[0]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSDL đặc trưng CBIR")
    parser.add_argument("--rebuild", action="store_true", help="Xoá CSDL cũ trước khi build")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N ảnh đầu (debug)")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Không fit và lưu PCA sau khi build DB",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Không build IVF index (chỉ có hiệu lực khi vẫn fit PCA)",
    )
    args = parser.parse_args()

    build(
        dataset_dir=args.dataset,
        db_path=args.db,
        rebuild=args.rebuild,
        limit=args.limit,
        fit_pca_enabled=not args.no_pca,
        build_index=not args.no_index,
    )


if __name__ == "__main__":
    main()
