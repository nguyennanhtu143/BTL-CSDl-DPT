"""Phase 6: Streamlit UI cho hệ thống CBIR ảnh nền thiên nhiên.

Cách chạy:
    streamlit run ui/app.py

Layout:
    - Sidebar: tuỳ chọn in toàn bộ vector ra console + thống kê CSDL
    - Cột trái: upload ảnh + preview
    - Cột phải: button "Tìm kiếm" -> grid 5 cột top-5 (ảnh, tên, distance)
    - Expander: thống kê vector (sum Color/Shape, non-zero bins)

Mỗi lần query in ra terminal nơi chạy streamlit theo format bắt buộc.
"""
from __future__ import annotations

import sys
import time
from io import BytesIO
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from PIL import Image

from src.config import COLOR_DIM, DATASET_DIR, GRAD_DIM, TOP_K, TOTAL_DIM, W_COLOR, W_SHAPE
from src.database import load_database
from src.feature_extractor import extract_features
from src.logger import log_query
from src.matcher import find_top_k
from src.preprocessing import resize_image, to_grayscale

st.set_page_config(
    page_title="CBIR - Tìm kiếm ảnh nền thiên nhiên",
    page_icon="🌿",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang nạp CSDL đặc trưng...")
def get_db():
    """Cache CSDL trong RAM cho toàn bộ session - tránh đọc lại mỗi lần rerun."""
    return load_database()


def extract_from_uploaded_bytes(img_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Đọc bytes -> RGB gốc + vector đặc trưng 657 chiều."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    rgb_orig = np.asarray(img, dtype=np.uint8)
    rgb = resize_image(rgb_orig)
    gray = to_grayscale(rgb)
    return rgb_orig, extract_features(rgb, gray)


def render_results(results: list[tuple[str, float]]) -> None:
    """Grid 5 cột: thumbnail + filename + distance."""
    cols = st.columns(len(results))
    for col, (name, dist) in zip(cols, results):
        with col:
            path = DATASET_DIR / name
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning(f"Không tìm thấy {name}")
            st.markdown(f"**{name}**")
            st.code(f"d = {dist:.4f}")


def main() -> None:
    st.title("🌿 CBIR — Tìm kiếm ảnh nền thiên nhiên")
    st.caption(
        f"Vector đặc trưng **{TOTAL_DIM}** chiều = "
        f"Color **{COLOR_DIM}** (lưới 3×3 × RGB 4×4×4) + "
        f"Shape **{GRAD_DIM}** (lưới 3×3 × 9 bin gradient)"
    )

    db = get_db()
    if len(db) == 0:
        st.error("CSDL rỗng. Hãy chạy `python build_database.py` trước.")
        st.stop()

    with st.sidebar:
        st.header("Cấu hình")
        st.success(f"CSDL: {len(db)} ảnh")
        st.text(f"W_COLOR = {W_COLOR}\nW_SHAPE = {W_SHAPE}\nTOP_K   = {TOP_K}")
        full_vector = st.checkbox(
            "In toàn bộ vector ra console",
            value=False,
            help="Mặc định in 4 đầu + 4 cuối. Bật để in đầy đủ 657 phần tử.",
        )
        k = st.number_input("Số kết quả (top-k)", min_value=1, max_value=20, value=TOP_K)

    uploaded = st.file_uploader(
        "Chọn ảnh truy vấn (.jpg / .jpeg / .png)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("👆 Upload một ảnh để bắt đầu tìm kiếm.")
        st.stop()

    img_bytes = uploaded.getvalue()
    file_size_kb = len(img_bytes) / 1024

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Ảnh truy vấn")
        st.image(img_bytes, use_container_width=True)
        st.text(f"Tên file: {uploaded.name}")
        st.text(f"Kích thước: {file_size_kb:.1f} KB")
        search = st.button("🔍 Tìm kiếm Top-K", type="primary", use_container_width=True)

    with col_right:
        if not search:
            st.info("Bấm **Tìm kiếm Top-K** để chạy truy vấn.")
            return

        try:
            t0 = time.perf_counter()
            _, q_vec = extract_from_uploaded_bytes(img_bytes)
            t_extract = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            top = find_top_k(q_vec, db.vectors, k=int(k), ids=db.filenames)
            results = [(str(name), float(dist)) for name, dist in top]
            t_match = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            return

        log_query(uploaded.name, q_vec, results, full_vector=full_vector)

        st.subheader(f"Top-{len(results)} kết quả")
        render_results(results)

        with st.expander("Thông tin vector đặc trưng (657 chiều)"):
            color = q_vec[:COLOR_DIM]
            shape = q_vec[COLOR_DIM:]
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Color (576 dim)",
                f"sum = {color.sum():.4f}",
                f"non-zero {(color > 0).sum()}/{COLOR_DIM}",
            )
            c2.metric(
                "Shape (81 dim)",
                f"sum = {shape.sum():.4f}",
                f"non-zero {(shape > 0).sum()}/{GRAD_DIM}",
            )
            c3.metric(
                "Tổng",
                f"sum = {q_vec.sum():.4f}",
                f"shape = {q_vec.shape}",
            )
            st.caption(
                f"Extract: {t_extract:.1f} ms · Match (top-k trong {len(db)} ảnh): {t_match:.1f} ms"
            )
            st.info("Vector đầy đủ đã được in ra terminal nơi đang chạy `streamlit run`.")


if __name__ == "__main__":
    main()
