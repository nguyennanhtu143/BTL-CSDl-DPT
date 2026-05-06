"""Phase 4: Tích hợp đặc trưng - ghép color + shape + frequency thành vector cuối.

Histogram màu: theo `USE_LAB_COLOR_HISTOGRAM` trong `config` (LAB khuyến nghị) hoặc RGB.

Vector cuối cùng:
    feature = concat(W_COLOR * v_color, W_SHAPE * v_shape, W_FREQ * v_freq)

Trọng số đã áp trước khi concat -> không normalize lại để giữ ảnh hưởng của trọng số
khi tính khoảng cách Euclidean ở matcher.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.color_features import extract_color_feature
from src.config import COLOR_DIM, FREQ_DIM, GRAD_DIM, TOTAL_DIM, W_COLOR, W_FREQ, W_SHAPE
from src.frequency_features import extract_frequency_feature
from src.gradient_features import extract_gradient_feature
from src.layout_features import extract_layout_scalars
from src.preprocessing import load_image, resize_image, to_grayscale


def extract_features(
    rgb: np.ndarray,
    gray: np.ndarray | None = None,
    w_color: float = W_COLOR,
    w_shape: float = W_SHAPE,
    w_freq: float = W_FREQ,
) -> np.ndarray:
    """Trích xuất vector đặc trưng đầy đủ từ ảnh RGB đã resize.

    Tham số:
        rgb: ảnh `(IMG_HEIGHT, IMG_WIDTH, 3)` uint8.
        gray: ảnh xám float32 nếu đã có sẵn (tránh tính lại); nếu None sẽ tự tính.
        w_color, w_shape: trọng số (mặc định 0.7 / 0.3).

    Trả về vector float32 shape `(TOTAL_DIM,)` với 3 nhánh đã nhân trọng số.
    """
    if gray is None:
        gray = to_grayscale(rgb)

    v_color = extract_color_feature(rgb)
    v_shape = extract_gradient_feature(gray)
    v_freq = extract_frequency_feature(gray)

    assert v_color.shape == (COLOR_DIM,), f"Color dim sai: {v_color.shape}"
    assert v_shape.shape == (GRAD_DIM,), f"Shape dim sai: {v_shape.shape}"
    assert v_freq.shape == (FREQ_DIM,), f"Freq dim sai: {v_freq.shape}"

    feature = np.concatenate([w_color * v_color, w_shape * v_shape, w_freq * v_freq]).astype(
        np.float32
    )
    return feature


def extract_feature_components(
    rgb: np.ndarray,
    gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả riêng 3 vector thành phần đã chuẩn hóa L1: (color, shape, freq)."""
    if gray is None:
        gray = to_grayscale(rgb)
    v_color = extract_color_feature(rgb)
    v_shape = extract_gradient_feature(gray)
    v_freq = extract_frequency_feature(gray)
    assert v_color.shape == (COLOR_DIM,), f"Color dim sai: {v_color.shape}"
    assert v_shape.shape == (GRAD_DIM,), f"Shape dim sai: {v_shape.shape}"
    assert v_freq.shape == (FREQ_DIM,), f"Freq dim sai: {v_freq.shape}"
    return v_color.astype(np.float32), v_shape.astype(np.float32), v_freq.astype(np.float32)


def extract_from_path(path: str | Path) -> np.ndarray:
    """Pipeline đầu-cuối từ file: load -> resize -> features (657,)."""
    rgb = resize_image(load_image(path))
    gray = to_grayscale(rgb)
    return extract_features(rgb, gray)


def extract_components_from_path(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pipeline đầu-cuối từ file: load -> resize -> trả riêng (color, shape, freq)."""
    rgb = resize_image(load_image(path))
    gray = to_grayscale(rgb)
    return extract_feature_components(rgb, gray)


def extract_layout_from_path(path: str | Path) -> np.ndarray:
    """Pipeline đầu-cuối từ file: load -> resize -> trả 4 scalar layout."""
    rgb = resize_image(load_image(path))
    gray = to_grayscale(rgb)
    return np.asarray(extract_layout_scalars(gray), dtype=np.float32)


def split_feature_parts(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tách vector đầy đủ thành 3 phần (color, shape, freq)."""
    if feature.shape != (TOTAL_DIM,):
        raise ValueError(f"Yêu cầu vector ({TOTAL_DIM},), nhận {feature.shape}")
    color = feature[:COLOR_DIM]
    shape = feature[COLOR_DIM:COLOR_DIM + GRAD_DIM]
    freq = feature[COLOR_DIM + GRAD_DIM:]
    return color, shape, freq
