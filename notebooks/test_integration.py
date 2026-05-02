"""Test cho Phase 4: tích hợp feature 657 chiều + Euclidean + top-k."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import (
    COLOR_DIM,
    DATASET_DIR,
    GRAD_DIM,
    TOP_K,
    TOTAL_DIM,
    W_COLOR,
    W_SHAPE,
)
from src.feature_extractor import extract_features, extract_from_path, split_color_shape
from src.matcher import euclidean_distance, euclidean_distance_batch, find_top_k


def test_extract_features_shape_and_weights() -> None:
    files = sorted(DATASET_DIR.glob("*.jpg"))[:2]
    feat = extract_from_path(files[0])
    assert feat.shape == (TOTAL_DIM,), f"Shape sai: {feat.shape}"
    assert feat.dtype == np.float32

    color, shape = split_color_shape(feat)
    assert color.shape == (COLOR_DIM,) and shape.shape == (GRAD_DIM,)
    assert abs(color.sum() - W_COLOR) < 1e-5, f"Color sum phải = {W_COLOR}, được {color.sum()}"
    assert abs(shape.sum() - W_SHAPE) < 1e-5, f"Shape sum phải = {W_SHAPE}, được {shape.sum()}"
    assert abs(feat.sum() - 1.0) < 1e-5, f"Tổng vector tích hợp phải = 1.0, được {feat.sum()}"
    print(f"[extract] {files[0].name}: shape={feat.shape}, color_sum={color.sum():.4f}, shape_sum={shape.sum():.4f}")


def test_euclidean_correctness() -> None:
    rng = np.random.default_rng(42)
    v1 = rng.normal(size=657).astype(np.float32)
    v2 = rng.normal(size=657).astype(np.float32)

    d_custom = euclidean_distance(v1, v2)
    d_numpy = float(np.linalg.norm(v1.astype(np.float64) - v2.astype(np.float64)))
    assert abs(d_custom - d_numpy) < 1e-9, f"Lệch numpy: {d_custom} vs {d_numpy}"

    d_self = euclidean_distance(v1, v1)
    assert d_self == 0.0
    print(f"[euclidean] custom={d_custom:.6f} ≈ numpy={d_numpy:.6f}, self=0")


def test_euclidean_batch() -> None:
    rng = np.random.default_rng(1)
    db = rng.normal(size=(10, 657)).astype(np.float32)
    q = rng.normal(size=657).astype(np.float32)

    batch = euclidean_distance_batch(q, db)
    one_by_one = np.array([euclidean_distance(q, db[i]) for i in range(10)])
    assert np.allclose(batch, one_by_one, atol=1e-9), "Batch lệch single"
    print(f"[batch] 10 vector OK, max diff {np.max(np.abs(batch - one_by_one)):.2e}")


def test_find_top_k_synthetic() -> None:
    db = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [1.0, 1.0],
        [10.0, 0.0],
    ], dtype=np.float32)
    q = np.array([0.0, 0.0], dtype=np.float32)
    ids = ["a", "b", "c", "d"]

    top = find_top_k(q, db, k=3, ids=ids)
    assert [r[0] for r in top] == ["a", "c", "b"], f"Sai thứ tự: {top}"
    assert top[0][1] == 0.0
    assert abs(top[1][1] - np.sqrt(2)) < 1e-6
    assert top[2][1] == 5.0
    print(f"[top_k] {top}")


def test_self_match_zero() -> None:
    """Query 1 ảnh chống lại CSDL chứa chính nó -> distance phải = 0 ở vị trí top-1."""
    files = sorted(DATASET_DIR.glob("*.jpg"))[:5]
    feats = np.stack([extract_from_path(f) for f in files])
    names = [f.name for f in files]

    for i, q in enumerate(feats):
        top = find_top_k(q, feats, k=TOP_K, ids=names)
        assert top[0][0] == names[i], f"Top-1 phải là chính nó, được {top[0][0]}"
        assert top[0][1] < 1e-6, f"Self-distance phải = 0, được {top[0][1]}"
        print(f"\n  Query: {names[i]}")
        for rank, (name, d) in enumerate(top, 1):
            print(f"    {rank}. {name:<35} d={d:.6f}")


def main() -> None:
    test_extract_features_shape_and_weights()
    test_euclidean_correctness()
    test_euclidean_batch()
    test_find_top_k_synthetic()
    test_self_match_zero()
    print("\nTat ca assertion da pass.")


if __name__ == "__main__":
    main()
