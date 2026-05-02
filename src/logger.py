"""Phase 6: In console kết quả query theo format bắt buộc của đề bài.

Mỗi lần query in ra terminal:
- Tên ảnh truy vấn
- Vector đặc trưng đầy đủ (tách Color 576 + Shape 81)
- Bảng top-K kết quả với khoảng cách Euclidean
"""
from __future__ import annotations

import numpy as np

from src.config import COLOR_DIM, GRAD_DIM

SEPARATOR = "=" * 60
SUB_SEPARATOR = "-" * 60


def format_vector_preview(
    vec: np.ndarray,
    n_head: int = 4,
    n_tail: int = 4,
    precision: int = 4,
) -> str:
    """Rút gọn vector dài thành '[v0, v1, ..., vn-1, vn]' để dễ đọc."""
    arr = np.asarray(vec).ravel()
    if arr.size <= n_head + n_tail:
        items = [f"{x:.{precision}f}" for x in arr]
    else:
        head = [f"{x:.{precision}f}" for x in arr[:n_head]]
        tail = [f"{x:.{precision}f}" for x in arr[-n_tail:]]
        items = head + ["..."] + tail
    return "[" + ", ".join(items) + "]"


def log_query(
    filename: str,
    query_vec: np.ndarray,
    results: list[tuple[str, float]],
    full_vector: bool = False,
) -> None:
    """In ra terminal theo format đề bài.

    Tham số:
        filename   : tên ảnh truy vấn
        query_vec  : vector đặc trưng (657,)
        results    : list (filename, distance) top-K đã sort tăng dần
        full_vector: True => in toàn bộ 657 phần tử; False => rút gọn 4+4
    """
    color = query_vec[:COLOR_DIM]
    shape = query_vec[COLOR_DIM:]

    print(SEPARATOR)
    print(f"[QUERY] Image: {filename}")
    print(SEPARATOR)
    print(f"[FEATURE VECTOR] Shape: {tuple(query_vec.shape)}")

    if full_vector:
        with np.printoptions(threshold=np.inf, precision=6, linewidth=120, suppress=True):
            print(f"  - Color ({COLOR_DIM} dim):")
            print(np.array2string(color, separator=", ", prefix="    "))
            print(f"  - Shape ({GRAD_DIM} dim):")
            print(np.array2string(shape, separator=", ", prefix="    "))
    else:
        print(f"  - Color ({COLOR_DIM} dim): {format_vector_preview(color)}")
        print(f"  - Shape ({GRAD_DIM} dim) : {format_vector_preview(shape)}")

    print(SUB_SEPARATOR)
    print("[TOP {} RESULTS]".format(len(results)))
    print(f"Rank | {'Filename':<28} | Distance")
    for rank, (name, dist) in enumerate(results, 1):
        print(f"  {rank}  | {name:<28} | {dist:.4f}")
    print(SEPARATOR)
