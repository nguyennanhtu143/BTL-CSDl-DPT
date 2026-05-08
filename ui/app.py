"""Streamlit UI cho CBIR hybrid3 (color hist + gradient + compact6 scalar).

Cách chạy:
    streamlit run ui/app.py
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

from src.color_features import extract_color_feature
from src.compact6_features import extract_compact6
from src.config import DATA_DIR, DATASET_DIR, TOP_K
from src.database_hybrid3 import count_images, load_database
from src.gradient_features import extract_gradient_feature
from src.matcher_hybrid3 import find_top_k_hybrid3, find_top_k_hybrid3_two_stage
from src.preprocessing import resize_image, to_grayscale

DB_PATH = DATA_DIR / "features_hybrid3.db"

st.set_page_config(
    page_title="CBIR Hybrid3 - Tìm kiếm ảnh nền thiên nhiên",
    page_icon="🌿",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang nạp CSDL hybrid3...")
def get_db():
    return load_database(DB_PATH)


def _read_uploaded_bytes(uploaded) -> bytes:
    """Đọc bytes từ Streamlit UploadedFile, an toàn với mọi phiên bản streamlit.

    `getvalue()` đôi khi fail nếu cursor đã bị di chuyển. Fallback sang seek+read.
    """
    if uploaded is None:
        return b""
    try:
        data = uploaded.getvalue()
    except (AttributeError, ValueError):
        try:
            uploaded.seek(0)
        except Exception:
            pass
        data = uploaded.read()
    if data is None:
        return b""
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    return bytes(data)


def _load_image_bytes(path: Path) -> bytes | None:
    """Đọc file ảnh từ disk thành bytes (để pass trực tiếp cho st.image).

    Đáng tin cậy hơn st.image(str(path)) vì không phụ thuộc cách Streamlit
    serve file path đến client.
    """
    try:
        return path.read_bytes()
    except (OSError, FileNotFoundError):
        return None


def extract_from_uploaded(img_bytes: bytes):
    """Đọc bytes -> RGB gốc + (color hist 576-d, gradient 81-d, compact6 6-d)."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    rgb_orig = np.asarray(img, dtype=np.uint8)
    rgb = resize_image(rgb_orig)
    gray = to_grayscale(rgb)
    hist = extract_color_feature(rgb).astype(np.float32)
    grad = extract_gradient_feature(gray).astype(np.float32)
    compact, _ = extract_compact6(rgb, gray)
    return rgb_orig, hist, grad, compact.astype(np.float32)


def render_results(results: list[tuple[str, float]], query_name: str | None = None) -> None:
    cols = st.columns(len(results))
    for col, (name, dist) in zip(cols, results):
        with col:
            path = DATASET_DIR / name
            img_bytes = _load_image_bytes(path)
            if img_bytes is not None:
                st.image(img_bytes, use_container_width=True)
            else:
                st.warning(f"Không tìm thấy {name}")
            st.markdown(f"**{name}**")
            tag = ""
            if query_name and name == query_name and dist < 1e-5:
                tag = "  ← self"
            elif dist < 1e-5:
                tag = "  ← duplicate"
            st.code(f"d = {dist:.4f}{tag}")


def main() -> None:
    st.title("🌿 CBIR Hybrid3 — Tìm kiếm ảnh nền thiên nhiên")
    st.caption("Vector đặc trưng = Color hist (576-d) + Gradient (81-d) + Compact6 (6-d) · L2 distance")

    if count_images(DB_PATH) == 0:
        st.error(f"CSDL rỗng tại {DB_PATH}. Hãy chạy `python build_database_hybrid3.py` trước.")
        st.stop()

    db = get_db()

    with st.sidebar:
        st.header("Cấu hình")
        st.success(f"CSDL: {len(db)} ảnh")
        k = st.number_input("Số kết quả top-k", min_value=1, max_value=20, value=TOP_K)
        st.markdown("**Trọng số fusion** (mean-normalize per branch)")
        w_hist = st.slider("w_hist (color)", 0.0, 1.0, 0.45, step=0.05)
        w_grad = st.slider("w_grad (gradient)", 0.0, 1.0, 0.30, step=0.05)
        w_compact = st.slider("w_compact (compact6)", 0.0, 1.0, 0.25, step=0.05)
        st.markdown("**Two-stage retrieval**")
        coarse_top = st.number_input(
            "coarse_top (0 = single-stage)",
            min_value=0,
            max_value=len(db),
            value=0,
            help="compact6 lọc thô top-N candidates trước khi rank đầy đủ",
        )

    uploaded = st.file_uploader(
        "Chọn ảnh truy vấn (.jpg / .jpeg / .png)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("👆 Upload một ảnh để bắt đầu tìm kiếm.")
        st.stop()

    img_bytes = _read_uploaded_bytes(uploaded)
    if not img_bytes:
        st.error("Không đọc được nội dung ảnh upload. Thử lại với file khác.")
        st.stop()
    file_size_kb = len(img_bytes) / 1024

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Ảnh truy vấn")
        st.image(img_bytes, use_container_width=True)
        st.text(f"Tên: {uploaded.name}")
        st.text(f"Size: {file_size_kb:.1f} KB")
        search = st.button("🔍 Tìm kiếm Top-K", type="primary", use_container_width=True)

    with col_right:
        if not search:
            st.info("Bấm **Tìm kiếm Top-K** để chạy truy vấn.")
            return

        try:
            t0 = time.perf_counter()
            _, q_hist, q_grad, q_compact = extract_from_uploaded(img_bytes)
            t_extract = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            common = dict(
                q_hist=q_hist,
                q_grad=q_grad,
                q_compact=q_compact,
                db_hist=db.hist_vectors,
                db_grad=db.grad_vectors,
                db_compact=db.compact_vectors,
                ids=db.filenames,
                k=int(k),
                w_hist=w_hist,
                w_grad=w_grad,
                w_compact=w_compact,
            )
            if coarse_top > 0 and coarse_top < len(db):
                top = find_top_k_hybrid3_two_stage(coarse_top=int(coarse_top), **common)
                stage_label = f"two-stage (coarse_top={coarse_top})"
            else:
                top = find_top_k_hybrid3(**common)
                stage_label = "single-stage (full scan)"
            results = [(str(name), float(dist)) for name, dist in top]
            t_match = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            st.exception(exc)
            return

        st.subheader(f"Top-{len(results)} kết quả")
        render_results(results, query_name=uploaded.name)

        with st.expander("Thông tin vector & timing"):
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Color hist (576-d)",
                f"sum = {q_hist.sum():.4f}",
                f"non-zero {(q_hist > 0).sum()}/576",
            )
            c2.metric(
                "Gradient (81-d)",
                f"sum = {q_grad.sum():.4f}",
                f"non-zero {(q_grad > 0).sum()}/81",
            )
            c3.metric(
                "Compact6 (6-d)",
                f"min/max = {q_compact.min():.3f}/{q_compact.max():.3f}",
                f"mean = {q_compact.mean():.3f}",
            )
            st.caption(
                f"Stage: {stage_label} · Extract {t_extract:.1f} ms · "
                f"Match {t_match:.1f} ms ({len(db)} ảnh)"
            )


if __name__ == "__main__":
    main()
