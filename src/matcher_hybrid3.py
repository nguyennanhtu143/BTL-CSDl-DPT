from __future__ import annotations

import numpy as np

from src.distances import euclidean_distance_batch, get_hist_metric


def hybrid3_distance_components(
    q_hist: np.ndarray,
    q_grad: np.ndarray,
    q_compact: np.ndarray,
    db_hist: np.ndarray,
    db_grad: np.ndarray,
    db_compact: np.ndarray,
    hist_metric: str = "l2",
) -> dict[str, np.ndarray | float]:
    """Tính raw distance per-branch và scale-factor để mean-normalize.

    Cho phép caller (find_top_k_*) tự cộng có trọng số, đồng thời reuse các
    nhánh để in breakdown phân tích. Tách ra để hai-stage retrieval không
    tính lại compact6.
    """
    hist_fn = get_hist_metric(hist_metric)
    d_hist = hist_fn(q_hist, db_hist)
    d_grad = euclidean_distance_batch(q_grad, db_grad)
    d_comp = euclidean_distance_batch(q_compact, db_compact)

    eps = 1e-12
    return {
        "d_hist": d_hist,
        "d_grad": d_grad,
        "d_compact": d_comp,
        "s_hist": max(float(np.mean(d_hist)), eps),
        "s_grad": max(float(np.mean(d_grad)), eps),
        "s_compact": max(float(np.mean(d_comp)), eps),
    }


def combine_distances(
    components: dict[str, np.ndarray | float],
    w_hist: float,
    w_grad: float,
    w_compact: float,
) -> np.ndarray:
    """Cộng có trọng số các nhánh đã được mean-normalize."""
    return (
        w_hist * (components["d_hist"] / components["s_hist"])
        + w_grad * (components["d_grad"] / components["s_grad"])
        + w_compact * (components["d_compact"] / components["s_compact"])
    )


def hybrid3_distance_batch(
    q_hist: np.ndarray,
    q_grad: np.ndarray,
    q_compact: np.ndarray,
    db_hist: np.ndarray,
    db_grad: np.ndarray,
    db_compact: np.ndarray,
    w_hist: float = 0.45,
    w_grad: float = 0.30,
    w_compact: float = 0.25,
    hist_metric: str = "l2",
) -> np.ndarray:
    """Backward-compatible: trả tổng distance dạng mảng (N,)."""
    components = hybrid3_distance_components(
        q_hist=q_hist,
        q_grad=q_grad,
        q_compact=q_compact,
        db_hist=db_hist,
        db_grad=db_grad,
        db_compact=db_compact,
        hist_metric=hist_metric,
    )
    return combine_distances(components, w_hist, w_grad, w_compact)


def _build_breakdown(
    components: dict[str, np.ndarray | float],
    order: np.ndarray,
    ids: list[str],
    distances: np.ndarray,
    w_hist: float,
    w_grad: float,
    w_compact: float,
) -> list[dict]:
    """Tạo list chi tiết per-branch contribution cho top-k đã rank."""
    out = []
    for i in order:
        d_hist_norm = components["d_hist"][i] / components["s_hist"]
        d_grad_norm = components["d_grad"][i] / components["s_grad"]
        d_comp_norm = components["d_compact"][i] / components["s_compact"]
        out.append(
            {
                "filename": ids[i],
                "total": float(distances[i]),
                "color_contribution": float(w_hist * d_hist_norm),
                "grad_contribution": float(w_grad * d_grad_norm),
                "compact_contribution": float(w_compact * d_comp_norm),
                "color_raw": float(components["d_hist"][i]),
                "grad_raw": float(components["d_grad"][i]),
                "compact_raw": float(components["d_compact"][i]),
            }
        )
    return out


def find_top_k_hybrid3(
    q_hist: np.ndarray,
    q_grad: np.ndarray,
    q_compact: np.ndarray,
    db_hist: np.ndarray,
    db_grad: np.ndarray,
    db_compact: np.ndarray,
    ids: list[str],
    k: int,
    w_hist: float = 0.45,
    w_grad: float = 0.30,
    w_compact: float = 0.25,
    hist_metric: str = "l2",
    return_breakdown: bool = False,
):
    """Single-stage: full scan rồi rank theo combined distance."""
    n = db_hist.shape[0]
    if n == 0:
        return ([], []) if return_breakdown else []
    if db_grad.shape[0] != n or db_compact.shape[0] != n:
        raise ValueError("db_hist, db_grad, db_compact phải cùng số dòng")
    if len(ids) != n:
        raise ValueError(f"len(ids)={len(ids)} khác N={n}")

    k = min(k, n)
    components = hybrid3_distance_components(
        q_hist=q_hist,
        q_grad=q_grad,
        q_compact=q_compact,
        db_hist=db_hist,
        db_grad=db_grad,
        db_compact=db_compact,
        hist_metric=hist_metric,
    )
    d = combine_distances(components, w_hist, w_grad, w_compact)
    order = np.argsort(d)[:k]
    results = [(ids[i], float(d[i])) for i in order]

    if return_breakdown:
        breakdown = _build_breakdown(components, order, ids, d, w_hist, w_grad, w_compact)
        return results, breakdown
    return results


def find_top_k_hybrid3_two_stage(
    q_hist: np.ndarray,
    q_grad: np.ndarray,
    q_compact: np.ndarray,
    db_hist: np.ndarray,
    db_grad: np.ndarray,
    db_compact: np.ndarray,
    ids: list[str],
    k: int,
    coarse_top: int,
    w_hist: float = 0.45,
    w_grad: float = 0.30,
    w_compact: float = 0.25,
    hist_metric: str = "l2",
    return_breakdown: bool = False,
):
    """Two-stage retrieval theo lý thuyết CSDL Đa phương tiện.

    Stage 1: lọc thô bằng compact6 (chỉ 6-dim, rất nhanh) -> giữ M = coarse_top candidates.
    Stage 2: tính color hist + gradient + compact6 đầy đủ trên M ảnh -> top-k cuối.

    Khi coarse_top >= N hoặc <= 0: rơi về single-stage (find_top_k_hybrid3).
    """
    n = db_hist.shape[0]
    if n == 0:
        return ([], []) if return_breakdown else []
    if coarse_top <= 0 or coarse_top >= n:
        return find_top_k_hybrid3(
            q_hist=q_hist,
            q_grad=q_grad,
            q_compact=q_compact,
            db_hist=db_hist,
            db_grad=db_grad,
            db_compact=db_compact,
            ids=ids,
            k=k,
            w_hist=w_hist,
            w_grad=w_grad,
            w_compact=w_compact,
            hist_metric=hist_metric,
            return_breakdown=return_breakdown,
        )

    d_compact_all = euclidean_distance_batch(q_compact, db_compact)
    cand_idx = np.argsort(d_compact_all)[:coarse_top]
    sub_ids = [ids[i] for i in cand_idx]

    return find_top_k_hybrid3(
        q_hist=q_hist,
        q_grad=q_grad,
        q_compact=q_compact,
        db_hist=db_hist[cand_idx],
        db_grad=db_grad[cand_idx],
        db_compact=db_compact[cand_idx],
        ids=sub_ids,
        k=k,
        w_hist=w_hist,
        w_grad=w_grad,
        w_compact=w_compact,
        hist_metric=hist_metric,
        return_breakdown=return_breakdown,
    )
