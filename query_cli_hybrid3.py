from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR
from src.database_hybrid3 import count_images, load_database, query

DB_HYBRID3_DEFAULT = DATA_DIR / "features_hybrid3.db"


def _format_vector_preview(
    vec: np.ndarray,
    n_head: int = 6,
    n_tail: int = 4,
    precision: int = 4,
) -> str:
    """Rút gọn vector dài thành '[v0, v1, ..., vn-1, vn]' để dễ đọc trên 1 dòng."""
    arr = np.asarray(vec).ravel()
    if arr.size <= n_head + n_tail:
        items = [f"{x:.{precision}f}" for x in arr]
    else:
        head = [f"{x:.{precision}f}" for x in arr[:n_head]]
        tail = [f"{x:.{precision}f}" for x in arr[-n_tail:]]
        items = head + ["..."] + tail
    return "[" + ", ".join(items) + "]"


def _print_vectors(
    q_hist: np.ndarray,
    q_grad: np.ndarray,
    q_compact: np.ndarray,
    full: bool = False,
) -> None:
    """In 3 vector đặc trưng của ảnh truy vấn.

    Mặc định: preview head+tail. `full=True`: in toàn bộ giá trị.
    Compact6 luôn in đầy đủ vì chỉ 6 chiều.
    """
    print(f"[VECTOR]  Color hist ({q_hist.size}-d): "
          f"sum={q_hist.sum():.4f}, non-zero={(q_hist > 0).sum()}/{q_hist.size}")
    if full:
        with np.printoptions(threshold=np.inf, precision=6, linewidth=120, suppress=True):
            print("          " + np.array2string(q_hist, separator=", ", prefix="          "))
    else:
        print(f"          {_format_vector_preview(q_hist)}")

    print(f"[VECTOR]  Gradient   ({q_grad.size}-d) : "
          f"sum={q_grad.sum():.4f}, non-zero={(q_grad > 0).sum()}/{q_grad.size}")
    if full:
        with np.printoptions(threshold=np.inf, precision=6, linewidth=120, suppress=True):
            print("          " + np.array2string(q_grad, separator=", ", prefix="          "))
    else:
        print(f"          {_format_vector_preview(q_grad)}")

    labels = ["mean_rgb", "stddev", "skewness", "coarseness", "contrast", "directionality"]
    print(f"[VECTOR]  Compact6   ({q_compact.size}-d)  :")
    for label, val in zip(labels, q_compact):
        print(f"          {label:<15} = {val:.4f}")


def _format_text_output(
    args: argparse.Namespace,
    db_size: int,
    q_hist: np.ndarray,
    q_grad: np.ndarray,
    q_compact: np.ndarray,
    breakdown: list[dict],
    t_load_ms: float,
    t_query_ms: float,
) -> None:
    print("=" * 78)
    print(f"[QUERY]   Image: {args.image.name}")
    print("=" * 78)
    print(
        f"[FEATURE] ColorHist: {q_hist.shape}, Gradient: {q_grad.shape}, "
        f"Compact6: {q_compact.shape}"
    )
    print(
        f"[WEIGHT]  w_hist={args.w_hist:.2f}, w_grad={args.w_grad:.2f}, "
        f"w_compact={args.w_compact:.2f}"
    )
    print("[METRIC]  Euclidean (L2) trên cả 3 nhánh, mean-normalize per-branch")
    if args.coarse_top and args.coarse_top < db_size:
        print(
            f"[STAGE]   two-stage: compact6 → top {args.coarse_top} candidates "
            f"→ full rank"
        )
    else:
        print(f"[STAGE]   single-stage: full scan trên N={db_size} ảnh")
    print("-" * 78)
    _print_vectors(q_hist, q_grad, q_compact, full=args.full_vectors)
    print("-" * 78)

    if args.breakdown:
        print(f"[TOP {len(breakdown)} RESULTS] - contribution = w * (d_branch / mean_d_branch)")
        print(
            f"{'Rk':<3}{'Filename':<30}{'Total':>9}{'Color':>9}{'Grad':>9}{'Cmp6':>9}  Tag"
        )
        for i, item in enumerate(breakdown, 1):
            tag = _result_tag(item, args.image.name)
            print(
                f"{i:<3}{item['filename']:<30}"
                f"{item['total']:>9.4f}"
                f"{item['color_contribution']:>9.4f}"
                f"{item['grad_contribution']:>9.4f}"
                f"{item['compact_contribution']:>9.4f}  {tag}"
            )
    else:
        print(f"[TOP {len(breakdown)} RESULTS]")
        print(f"{'Rk':<3}{'Filename':<30}{'Distance':>10}  Tag")
        for i, item in enumerate(breakdown, 1):
            tag = _result_tag(item, args.image.name)
            print(f"{i:<3}{item['filename']:<30}{item['total']:>10.4f}  {tag}")

    print("=" * 78)
    print(f"[timing]  load_db = {t_load_ms:.1f} ms, query = {t_query_ms:.1f} ms")


def _result_tag(item: dict, query_name: str) -> str:
    if item["filename"] == query_name and item["total"] < 1e-5:
        return "← self"
    if item["total"] < 1e-5:
        return "← duplicate (d≈0)"
    if item["filename"] == query_name:
        return "← self"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid-3 query CLI (color hist + gradient + compact6, L2 distance)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="Đường dẫn ảnh truy vấn")
    parser.add_argument("--k", type=int, default=5, help="Số kết quả top-k")
    parser.add_argument("--db", type=Path, default=DB_HYBRID3_DEFAULT)
    parser.add_argument("--w-hist", type=float, default=0.45, help="Trọng số color histogram")
    parser.add_argument("--w-grad", type=float, default=0.30, help="Trọng số gradient")
    parser.add_argument("--w-compact", type=float, default=0.25, help="Trọng số compact6")
    parser.add_argument(
        "--coarse-top",
        type=int,
        default=None,
        help="Two-stage retrieval: compact6 lọc thô giữ N candidates rồi mới rank đầy đủ",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="In thêm contribution per-branch (color/grad/compact) cho mỗi kết quả",
    )
    parser.add_argument(
        "--full-vectors",
        action="store_true",
        help="In toàn bộ giá trị vector đặc trưng (mặc định: chỉ preview head+tail)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output dạng JSON thay vì text (cho automation)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Không tìm thấy ảnh: {args.image}")
    if count_images(args.db) == 0:
        raise SystemExit(
            f"CSDL hybrid-3 rỗng: {args.db}. Chạy `python build_database_hybrid3.py` trước."
        )

    w_sum = args.w_hist + args.w_grad + args.w_compact
    if abs(w_sum - 1.0) > 0.01 and not args.json:
        print(
            f"[warn] Tổng trọng số = {w_sum:.3f} ≠ 1.0 "
            f"(ranking vẫn đúng nhờ mean-normalize, nhưng giá trị distance bị scale)"
        )

    t0 = time.perf_counter()
    db = load_database(args.db)
    t_load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    q_hist, q_grad, q_compact, _, breakdown = query(
        image_path=args.image,
        db=db,
        k=args.k,
        w_hist=args.w_hist,
        w_grad=args.w_grad,
        w_compact=args.w_compact,
        coarse_top=args.coarse_top,
        return_breakdown=True,
    )
    t_query_ms = (time.perf_counter() - t0) * 1000

    if args.json:
        out = {
            "query": str(args.image),
            "params": {
                "k": args.k,
                "weights": {
                    "hist": args.w_hist,
                    "grad": args.w_grad,
                    "compact": args.w_compact,
                },
                "coarse_top": args.coarse_top,
            },
            "db_size": len(db),
            "feature_shapes": {
                "hist": list(q_hist.shape),
                "grad": list(q_grad.shape),
                "compact": list(q_compact.shape),
            },
            "feature_vectors": {
                "hist": q_hist.tolist(),
                "grad": q_grad.tolist(),
                "compact": q_compact.tolist(),
            },
            "timing_ms": {"load_db": t_load_ms, "query": t_query_ms},
            "results": breakdown,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    _format_text_output(
        args=args,
        db_size=len(db),
        q_hist=q_hist,
        q_grad=q_grad,
        q_compact=q_compact,
        breakdown=breakdown,
        t_load_ms=t_load_ms,
        t_query_ms=t_query_ms,
    )


if __name__ == "__main__":
    main()
