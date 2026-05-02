"""Test cho Phase 3: kiểm tra Sobel convolution, gradient, histogram pipeline."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import DATASET_DIR, GRAD_BINS, GRAD_DIM, GRID, IMG_HEIGHT, IMG_WIDTH
from src.gradient_features import (
    SOBEL_GX,
    SOBEL_GY,
    cell_gradient_histogram,
    compute_gradient,
    convolve2d,
    extract_gradient_feature,
    gradient_histogram,
)
from src.preprocessing import load_image, resize_image, to_grayscale


def test_convolve_identity_and_shape() -> None:
    img = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)

    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    out = convolve2d(img, identity)
    assert out.shape == img.shape, f"Shape sai: {out.shape}"
    assert np.allclose(out, img), "Convolve với identity phải trả về ảnh gốc"

    out_gx = convolve2d(img, SOBEL_GX)
    assert out_gx.shape == img.shape
    print(f"[convolve] identity ok, Sobel Gx output shape {out_gx.shape}")


def test_sobel_on_synthetic_edge() -> None:
    img = np.zeros((10, 10), dtype=np.float32)
    img[:, 5:] = 255.0

    dx = convolve2d(img, SOBEL_GX)
    dy = convolve2d(img, SOBEL_GY)

    assert dx[5, 4] > 0, f"Gradient ngang phải dương ở cạnh, được {dx[5, 4]}"
    assert abs(dy[5, 4]) < 1e-6, f"Gradient dọc phải ~0, được {dy[5, 4]}"

    img_h = np.zeros((10, 10), dtype=np.float32)
    img_h[5:, :] = 255.0
    dy_h = convolve2d(img_h, SOBEL_GY)
    dx_h = convolve2d(img_h, SOBEL_GX)
    assert dy_h[4, 5] > 0
    assert abs(dx_h[4, 5]) < 1e-6
    print(f"[sobel] cạnh dọc dx[5,4]={dx[5,4]:.0f}, cạnh ngang dy[4,5]={dy_h[4,5]:.0f}")


def test_compute_gradient_angle_range() -> None:
    rng = np.random.default_rng(0)
    gray = rng.uniform(0, 255, size=(IMG_HEIGHT, IMG_WIDTH)).astype(np.float32)
    mag, ang = compute_gradient(gray)
    assert mag.shape == gray.shape and ang.shape == gray.shape
    assert mag.min() >= 0
    assert ang.min() >= 0 and ang.max() < 180.0, f"Góc phải trong [0,180), được [{ang.min()},{ang.max()}]"
    print(f"[gradient] random: mag [{mag.min():.1f}, {mag.max():.1f}], ang [{ang.min():.2f}, {ang.max():.2f})")


def test_cell_histogram_synthetic() -> None:
    mag = np.ones((4, 4), dtype=np.float32)
    ang = np.full((4, 4), 30.0, dtype=np.float32)
    h = cell_gradient_histogram(mag, ang, bins=GRAD_BINS)
    assert h.shape == (GRAD_BINS,)
    assert h[1] == 16.0, f"Bin 1 (20-40°) phải = 16, được {h[1]}"
    assert h.sum() == 16.0

    ang2 = np.array([[10.0, 25.0], [50.0, 175.0]], dtype=np.float32)
    mag2 = np.array([[2.0, 3.0], [1.0, 4.0]], dtype=np.float32)
    h2 = cell_gradient_histogram(mag2, ang2, bins=GRAD_BINS)
    assert h2[0] == 2.0 and h2[1] == 3.0 and h2[2] == 1.0 and h2[8] == 4.0
    print(f"[cell_hist] synthetic ok: {h2.tolist()}")


def test_pipeline_real_image() -> None:
    files = sorted(DATASET_DIR.glob("*.jpg"))[:3]
    for f in files:
        rgb = resize_image(load_image(f))
        gray = to_grayscale(rgb)
        feat = extract_gradient_feature(gray)

        assert feat.shape == (GRAD_DIM,), f"Shape sai: {feat.shape}"
        assert feat.dtype == np.float32
        assert abs(feat.sum() - 1.0) < 1e-5, f"Sum khác 1: {feat.sum()}"
        assert feat.min() >= 0

        raw = gradient_histogram(gray)
        assert raw.shape == (GRAD_DIM,)
        assert raw.sum() > 0

        non_zero = (feat > 0).sum()
        top3 = np.argsort(feat)[-3:][::-1]
        print(f"\n{f.name}")
        print(f"  shape={feat.shape}, sum={feat.sum():.6f}, non_zero_bins={non_zero}/{GRAD_DIM}")
        print(f"  top3 bins (idx, prob): {[(int(i), round(float(feat[i]), 4)) for i in top3]}")


def main() -> None:
    test_convolve_identity_and_shape()
    test_sobel_on_synthetic_edge()
    test_compute_gradient_angle_range()
    test_cell_histogram_synthetic()
    test_pipeline_real_image()
    print("\nTat ca assertion da pass.")


if __name__ == "__main__":
    main()
