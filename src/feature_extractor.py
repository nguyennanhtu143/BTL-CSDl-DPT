"""Phase 4: Tích hợp đặc trưng - ghép color (576) + shape (81) thành vector 657 chiều.

Vector cuối cùng:
    feature = concat(W_COLOR * v_color, W_SHAPE * v_shape)
           = concat(0.7 * v_color_576, 0.3 * v_shape_81)
           shape (657,)

Trọng số đã áp trước khi concat -> không normalize lại để giữ ảnh hưởng của trọng số
khi tính khoảng cách Euclidean ở matcher.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.color_features import extract_color_feature
from src.config import COLOR_DIM, GRAD_DIM, TOTAL_DIM, W_COLOR, W_SHAPE
from src.gradient_features import extract_gradient_feature
from src.preprocessing import load_image, resize_image, to_grayscale


def extract_features(
    rgb: np.ndarray,
    gray: np.ndarray | None = None,
    w_color: float = W_COLOR,
    w_shape: float = W_SHAPE,
) -> np.ndarray:
    """Trích xuất vector đặc trưng đầy đủ 657 chiều từ ảnh RGB đã resize.

    Tham số:
        rgb: ảnh `(IMG_HEIGHT, IMG_WIDTH, 3)` uint8.
        gray: ảnh xám float32 nếu đã có sẵn (tránh tính lại); nếu None sẽ tự tính.
        w_color, w_shape: trọng số (mặc định 0.7 / 0.3).

    Trả về vector float32 shape `(TOTAL_DIM,)` với:
        feature[:COLOR_DIM]  = w_color * color_hist (đã L1 normalize -> sum = w_color)
        feature[COLOR_DIM:]  = w_shape * grad_hist  (đã L1 normalize -> sum = w_shape)
    """
    if gray is None:
        gray = to_grayscale(rgb)

    v_color = extract_color_feature(rgb)
    v_shape = extract_gradient_feature(gray)

    assert v_color.shape == (COLOR_DIM,), f"Color dim sai: {v_color.shape}"
    assert v_shape.shape == (GRAD_DIM,), f"Shape dim sai: {v_shape.shape}"

    feature = np.concatenate([w_color * v_color, w_shape * v_shape]).astype(np.float32)
    return feature


def extract_from_path(path: str | Path) -> np.ndarray:
    """Pipeline đầu-cuối từ file: load -> resize -> features (657,)."""
    rgb = resize_image(load_image(path))
    gray = to_grayscale(rgb)
    return extract_features(rgb, gray)


def split_color_shape(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tách vector 657 chiều thành 2 phần (color 576, shape 81) để in console ở Phase 6."""
    if feature.shape != (TOTAL_DIM,):
        raise ValueError(f"Yêu cầu vector ({TOTAL_DIM},), nhận {feature.shape}")
    return feature[:COLOR_DIM], feature[COLOR_DIM:]
