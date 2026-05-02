"""Phase 1: Tiền xử lý ảnh - load, resize bilinear, chuyển grayscale."""
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import IMG_HEIGHT, IMG_WIDTH


def load_image(path: str | Path) -> np.ndarray:
    """Đọc ảnh JPG bằng Pillow, trả mảng (H, W, 3) uint8 RGB."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def resize_image(
    img: np.ndarray,
    target_w: int = IMG_WIDTH,
    target_h: int = IMG_HEIGHT,
) -> np.ndarray:
    """Resize ảnh bằng bilinear interpolation (tự code, không dùng PIL.resize).

    Với mỗi pixel output (x_out, y_out), ánh xạ về toạ độ source:
        x_src = (x_out + 0.5) * src_w / target_w - 0.5
    rồi nội suy 2 chiều từ 4 neighbor gần nhất.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Yêu cầu ảnh RGB shape (H, W, 3), nhận {img.shape}")

    src_h, src_w = img.shape[:2]

    x_out = np.arange(target_w, dtype=np.float32)
    y_out = np.arange(target_h, dtype=np.float32)
    x_src = (x_out + 0.5) * (src_w / target_w) - 0.5
    y_src = (y_out + 0.5) * (src_h / target_h) - 0.5

    x_src = np.clip(x_src, 0, src_w - 1)
    y_src = np.clip(y_src, 0, src_h - 1)

    x0 = np.floor(x_src).astype(np.int32)
    y0 = np.floor(y_src).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)

    fx = (x_src - x0).reshape(1, target_w, 1)
    fy = (y_src - y0).reshape(target_h, 1, 1)

    img_f = img.astype(np.float32)
    p00 = img_f[y0[:, None], x0[None, :]]
    p01 = img_f[y0[:, None], x1[None, :]]
    p10 = img_f[y1[:, None], x0[None, :]]
    p11 = img_f[y1[:, None], x1[None, :]]

    out = (
        (1 - fx) * (1 - fy) * p00
        + fx * (1 - fy) * p01
        + (1 - fx) * fy * p10
        + fx * fy * p11
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Chuyển RGB -> grayscale theo công thức luminance: Y = 0.299R + 0.587G + 0.114B.

    Trả về float32 để thuận tiện cho phép tích chập Sobel ở Phase 3.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Yêu cầu ảnh RGB shape (H, W, 3), nhận {img.shape}")
    img_f = img.astype(np.float32)
    return 0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]


def preprocess(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline đầy đủ: load -> resize -> trả (rgb_resized, gray_resized)."""
    rgb = resize_image(load_image(path))
    gray = to_grayscale(rgb)
    return rgb, gray
