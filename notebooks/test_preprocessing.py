"""Test nhanh cho Phase 1: kiểm tra shape, dtype, range giá trị."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import DATASET_DIR, IMG_HEIGHT, IMG_WIDTH
from src.preprocessing import load_image, preprocess, resize_image, to_grayscale


def main() -> None:
    files = sorted(DATASET_DIR.glob("*.jpg"))[:5]
    if not files:
        raise SystemExit(f"Không tìm thấy ảnh trong {DATASET_DIR}")

    print(f"Test trên {len(files)} ảnh đầu tiên trong {DATASET_DIR}\n")
    print(f"{'File':<32} {'Original':>15} {'Resized':>15} {'Gray':>13}")
    print("-" * 80)

    for f in files:
        rgb_orig = load_image(f)
        rgb_resized = resize_image(rgb_orig)
        gray = to_grayscale(rgb_resized)

        assert rgb_resized.shape == (IMG_HEIGHT, IMG_WIDTH, 3), (
            f"Shape sai: {rgb_resized.shape}, mong đợi ({IMG_HEIGHT}, {IMG_WIDTH}, 3)"
        )
        assert rgb_resized.dtype == np.uint8
        assert gray.shape == (IMG_HEIGHT, IMG_WIDTH)
        assert gray.dtype == np.float32
        assert 0 <= rgb_resized.min() and rgb_resized.max() <= 255
        assert 0 <= gray.min() and gray.max() <= 255

        print(
            f"{f.name:<32} "
            f"{str(rgb_orig.shape):>15} "
            f"{str(rgb_resized.shape):>15} "
            f"{str(gray.shape):>13}"
        )

    rgb, gray = preprocess(files[0])
    print(f"\npreprocess() pipeline -> rgb {rgb.shape} {rgb.dtype}, gray {gray.shape} {gray.dtype}")
    print(f"  RGB sample [0,0]   = {rgb[0, 0].tolist()}")
    print(f"  Gray sample [0,0]  = {gray[0, 0]:.2f}")
    print(f"  Gray min/mean/max  = {gray.min():.1f} / {gray.mean():.1f} / {gray.max():.1f}")
    print("\nTat ca assertion da pass.")


if __name__ == "__main__":
    main()
