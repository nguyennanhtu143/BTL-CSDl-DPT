"""Test cho Phase 5: kiểm tra schema, load, query end-to-end trên features.db."""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import DATASET_DIR, DB_PATH, TOP_K, TOTAL_DIM
from src.database import (
    count_images,
    deserialize_vector,
    get_image_meta,
    load_database,
    query,
    serialize_vector,
)


def test_serialize_roundtrip() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(size=TOTAL_DIM).astype(np.float32)
    blob = serialize_vector(v)
    assert len(blob) == TOTAL_DIM * 4, f"BLOB phải {TOTAL_DIM*4} byte, nhận {len(blob)}"
    v2 = deserialize_vector(blob)
    assert np.array_equal(v, v2), "Roundtrip không khớp"
    print(f"[blob] {TOTAL_DIM}*float32 = {len(blob)} byte, roundtrip OK")


def test_db_count() -> None:
    n = count_images(DB_PATH)
    assert n == 500, f"CSDL phải có 500 record, nhận {n}"
    print(f"[count] DB có {n} ảnh")


def test_load_database_shape() -> None:
    t0 = time.perf_counter()
    db = load_database(DB_PATH)
    dt = time.perf_counter() - t0
    assert len(db) == 500
    assert db.vectors.shape == (500, TOTAL_DIM), f"Shape sai: {db.vectors.shape}"
    assert db.vectors.dtype == np.float32
    assert db.ids.shape == (500,)
    assert len(db.filenames) == 500

    sums = db.vectors.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), (
        f"Mỗi vector phải có sum ~ 1.0, được min={sums.min()} max={sums.max()}"
    )
    print(f"[load] (500, {TOTAL_DIM}) trong {dt*1000:.1f} ms, mọi vector sum ~ 1.0")


def test_get_image_meta() -> None:
    meta = get_image_meta("01_thien_nhien_001.jpg", DB_PATH)
    assert meta is not None
    assert meta["filename"] == "01_thien_nhien_001.jpg"
    assert meta["width"] == 640 and meta["height"] == 360
    assert meta["file_size"] > 0
    print(f"[meta] {meta}")

    assert get_image_meta("not_exist.jpg", DB_PATH) is None


def test_query_in_db() -> None:
    """Query 1 ảnh có sẵn trong DB -> chính nó phải có trong top-k với d ~ 0.

    Lưu ý: dataset có ~78 cặp file trùng byte, nên top-1 có thể là một file
    duplicate khác (cũng d=0). Invariant đúng là: ảnh truy vấn nằm trong top-k
    và top-1 distance gần 0.
    """
    db = load_database(DB_PATH)
    files = sorted(DATASET_DIR.glob("*.jpg"))
    samples = [files[0], files[100], files[250], files[499]]

    for path in samples:
        t0 = time.perf_counter()
        q_vec, results = query(path, db=db, k=TOP_K)
        dt = time.perf_counter() - t0
        assert q_vec.shape == (TOTAL_DIM,)
        assert results[0][1] < 1e-5, f"Top-1 distance phải ~ 0, được {results[0][1]}"
        names = [r[0] for r in results]
        assert path.name in names, f"{path.name} không có trong top-{TOP_K}: {names}"

        self_d = next(d for n, d in results if n == path.name)
        print(f"\n  Query: {path.name}  ({dt*1000:.1f} ms, self_d={self_d:.6f})")
        for rank, (name, d) in enumerate(results, 1):
            tag = " <- self" if name == path.name else ""
            print(f"    {rank}. {name:<35} d={d:.6f}{tag}")


def test_query_speed() -> None:
    """Đo tốc độ query trung bình trên DB đã load (kỳ vọng < 10ms/query)."""
    db = load_database(DB_PATH)
    files = sorted(DATASET_DIR.glob("*.jpg"))[:20]

    t0 = time.perf_counter()
    for f in files:
        query(f, db=db, k=TOP_K)
    dt = time.perf_counter() - t0
    avg_ms = dt / len(files) * 1000
    print(f"\n[speed] {len(files)} query trong {dt*1000:.1f} ms, avg {avg_ms:.1f} ms/query")


def main() -> None:
    test_serialize_roundtrip()
    test_db_count()
    test_load_database_shape()
    test_get_image_meta()
    test_query_in_db()
    test_query_speed()
    print("\nTat ca assertion da pass.")


if __name__ == "__main__":
    main()
