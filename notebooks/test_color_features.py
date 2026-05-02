"""Test cho Phase 2: kiểm tra grid split, cell histogram, và pipeline tổng."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.color_features import (
    cell_histogram,
    color_histogram,
    extract_color_feature,
    normalize_l1,
    split_into_grid,
)
from src.config import COLOR_BINS, COLOR_DIM, DATASET_DIR, GRID, IMG_HEIGHT, IMG_WIDTH
from src.preprocessing import load_image, resize_image


def test_split_into_grid() -> None:
    img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    cells = split_into_grid(img)
    assert len(cells) == GRID * GRID, f"Phải có 9 ô, nhận {len(cells)}"
    total_pixels = sum(c.shape[0] * c.shape[1] for c in cells)
    assert total_pixels == IMG_HEIGHT * IMG_WIDTH, "Tổng pixel của 9 ô khác ảnh gốc"
    sizes = [(c.shape[0], c.shape[1]) for c in cells]
    print(f"[grid] 9 ô shape (h, w): {sizes}")


def test_cell_histogram_synthetic() -> None:
    cell = np.zeros((10, 10, 3), dtype=np.uint8)
    hist = cell_histogram(cell)
    assert hist.shape == (COLOR_BINS ** 3,), f"Shape sai: {hist.shape}"
    assert hist[0] == 100, f"Pixel (0,0,0) phải vào bin 0, được {hist[0]}"
    assert hist.sum() == 100

    cell2 = np.full((5, 5, 3), 255, dtype=np.uint8)
    hist2 = cell_histogram(cell2)
    last_bin = COLOR_BINS ** 3 - 1
    assert hist2[last_bin] == 25, f"Pixel (255,255,255) phải vào bin cuối, được {hist2[last_bin]}"

    cell3 = np.array([[[100, 200, 50]]], dtype=np.uint8)
    hist3 = cell_histogram(cell3)
    expected_bin = (100 // 64) * 16 + (200 // 64) * 4 + (50 // 64)
    assert hist3[expected_bin] == 1, f"Pixel (100,200,50) phải vào bin {expected_bin}"
    print(f"[cell_hist] synthetic ok: pixel (100,200,50) -> bin {expected_bin}")


def test_normalize_l1() -> None:
    v = np.array([1, 2, 3, 4], dtype=np.float32)
    n = normalize_l1(v)
    assert abs(n.sum() - 1.0) < 1e-6, f"Tổng sau normalize: {n.sum()}"

    zero = np.zeros(5, dtype=np.float32)
    nz = normalize_l1(zero)
    assert nz.sum() == 0
    print("[normalize] ok")


def test_pipeline_real_image() -> None:
    files = sorted(DATASET_DIR.glob("*.jpg"))[:3]
    for f in files:
        rgb = resize_image(load_image(f))
        feat = extract_color_feature(rgb)

        assert feat.shape == (COLOR_DIM,), f"Shape sai: {feat.shape}"
        assert feat.dtype == np.float32
        assert abs(feat.sum() - 1.0) < 1e-5, f"Sum khác 1: {feat.sum()}"
        assert feat.min() >= 0

        raw = color_histogram(rgb)
        assert raw.sum() == IMG_HEIGHT * IMG_WIDTH, (
            f"Tổng count phải = số pixel ({IMG_HEIGHT * IMG_WIDTH}), nhận {raw.sum()}"
        )

        non_zero = (feat > 0).sum()
        top5 = np.argsort(feat)[-5:][::-1]
        print(f"\n{f.name}")
        print(f"  shape={feat.shape}, sum={feat.sum():.6f}, non_zero_bins={non_zero}/{COLOR_DIM}")
        print(f"  top5 bins (idx, prob): {[(int(i), round(float(feat[i]), 4)) for i in top5]}")


def main() -> None:
    test_split_into_grid()
    test_cell_histogram_synthetic()
    test_normalize_l1()
    test_pipeline_real_image()
    print("\nTat ca assertion da pass.")


if __name__ == "__main__":
    main()
