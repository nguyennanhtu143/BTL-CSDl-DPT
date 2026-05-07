from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATASET_DIR, DATA_DIR
from src.database import count_images as legacy_count_images
from src.database import load_database as legacy_load_database
from src.database import query as legacy_query
from src.database_compact6 import count_images as compact_count_images
from src.database_compact6 import load_database as compact_load_database
from src.database_compact6 import query as compact_query
from src.database_hybrid import count_images as hybrid_count_images
from src.database_hybrid import load_database as hybrid_load_database
from src.database_hybrid import query as hybrid_query
from src.database_hybrid3 import count_images as hybrid3_count_images
from src.database_hybrid3 import load_database as hybrid3_load_database
from src.database_hybrid3 import query as hybrid3_query


def _validate_db_non_empty(path: Path, counter_fn, name: str) -> None:
    if counter_fn(path) == 0:
        raise SystemExit(f"CSDL `{name}` rỗng hoặc chưa tồn tại: {path}")


def _build_filename_index(dataset_root: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in dataset_root.rglob("*.jpg"):
        idx[p.name] = p
    return idx


def _normalize_query_list(queries: list[Path], dataset_root: Path, limit: int | None) -> list[Path]:
    if queries:
        resolved = [q.resolve() for q in queries if q.exists()]
    else:
        resolved = sorted(dataset_root.glob("*.jpg"))
        if limit is not None:
            resolved = resolved[:limit]
    if not resolved:
        raise SystemExit("Không tìm thấy ảnh query hợp lệ.")
    return resolved


def _run_benchmark(
    query_paths: list[Path],
    k: int,
    db_legacy: Path,
    db_compact: Path,
    db_hybrid: Path,
    db_hybrid3: Path,
) -> tuple[list[dict], dict]:
    _validate_db_non_empty(db_legacy, legacy_count_images, "legacy")
    _validate_db_non_empty(db_compact, compact_count_images, "compact6")
    _validate_db_non_empty(db_hybrid, hybrid_count_images, "hybrid")
    _validate_db_non_empty(db_hybrid3, hybrid3_count_images, "hybrid3")

    legacy_db = legacy_load_database(db_legacy)
    compact_db = compact_load_database(db_compact)
    hybrid_db = hybrid_load_database(db_hybrid)
    hybrid3_db = hybrid3_load_database(db_hybrid3)

    details: list[dict] = []
    stats = {
        "legacy": {"query_ms": [], "self_at_1": 0},
        "compact6": {"query_ms": [], "self_at_1": 0},
        "hybrid": {"query_ms": [], "self_at_1": 0},
        "hybrid3": {"query_ms": [], "self_at_1": 0},
        "overlap_topk": defaultdict(list),
    }

    for q in query_paths:
        qname = q.name

        t0 = time.perf_counter()
        _, top_legacy = legacy_query(q, db=legacy_db, k=k)
        t_legacy = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _, top_compact = compact_query(q, db=compact_db, k=k)
        t_compact = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _, _, top_hybrid = hybrid_query(q, db=hybrid_db, k=k)
        t_hybrid = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _, _, _, top_hybrid3 = hybrid3_query(q, db=hybrid3_db, k=k)
        t_hybrid3 = (time.perf_counter() - t0) * 1000

        packs = {
            "legacy": top_legacy,
            "compact6": top_compact,
            "hybrid": top_hybrid,
            "hybrid3": top_hybrid3,
        }
        times = {
            "legacy": t_legacy,
            "compact6": t_compact,
            "hybrid": t_hybrid,
            "hybrid3": t_hybrid3,
        }

        for name in ("legacy", "compact6", "hybrid", "hybrid3"):
            stats[name]["query_ms"].append(times[name])
            if packs[name] and packs[name][0][0] == qname:
                stats[name]["self_at_1"] += 1

        sets = {n: {x[0] for x in packs[n]} for n in packs}
        pairs = [("legacy", "compact6"), ("legacy", "hybrid"), ("legacy", "hybrid3"), ("hybrid", "hybrid3")]
        for a, b in pairs:
            union = sets[a] | sets[b]
            jac = len(sets[a] & sets[b]) / len(union) if union else 0.0
            stats["overlap_topk"][f"{a}_vs_{b}"].append(jac)

        details.append(
            {
                "query_path": str(q),
                "query_name": qname,
                "legacy_ms": t_legacy,
                "compact6_ms": t_compact,
                "hybrid_ms": t_hybrid,
                "hybrid3_ms": t_hybrid3,
                "legacy_topk": packs["legacy"],
                "compact6_topk": packs["compact6"],
                "hybrid_topk": packs["hybrid"],
                "hybrid3_topk": packs["hybrid3"],
            }
        )

    n = len(query_paths)
    summary = {
        "n_queries": n,
        "k": k,
        "pipelines": {
            name: {
                "avg_query_ms": float(sum(stats[name]["query_ms"]) / max(len(stats[name]["query_ms"]), 1)),
                "self_at_1": int(stats[name]["self_at_1"]),
                "self_at_1_rate": float(stats[name]["self_at_1"] / n),
            }
            for name in ("legacy", "compact6", "hybrid", "hybrid3")
        },
        "avg_overlap_topk_jaccard": {
            key: float(sum(vals) / max(len(vals), 1)) for key, vals in stats["overlap_topk"].items()
        },
    }
    return details, summary


def _write_csv(details: list[dict], out_csv: Path, k: int) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["query_name", "pipeline", "query_ms"] + [f"rank{i}_name" for i in range(1, k + 1)] + [
            f"rank{i}_dist" for i in range(1, k + 1)
        ]
        writer.writerow(header)

        for row in details:
            for pipeline in ("legacy", "compact6", "hybrid", "hybrid3"):
                topk = row[f"{pipeline}_topk"]
                names = [x[0] for x in topk] + [""] * (k - len(topk))
                dists = [f"{x[1]:.6f}" for x in topk] + [""] * (k - len(topk))
                writer.writerow([row["query_name"], pipeline, f"{row[f'{pipeline}_ms']:.4f}", *names[:k], *dists[:k]])


def _img_tag(img_path: Path, width: int = 200) -> str:
    if not img_path.exists():
        return "<div>missing</div>"
    ext = img_path.suffix.lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    raw = img_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    src = f"data:{mime};base64,{b64}"
    return f'<img src="{src}" width="{width}" loading="lazy" />'


def _write_html(details: list[dict], out_html: Path, filename_to_path: dict[str, Path], k: int) -> None:
    parts: list[str] = []
    parts.append(
        """
<!doctype html>
<html><head><meta charset="utf-8" />
<title>CBIR Benchmark Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
h1, h2 { margin: 8px 0; }
.query-block { border: 1px solid #ddd; padding: 12px; margin-bottom: 18px; border-radius: 8px; }
.row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }
.card { border: 1px solid #ccc; border-radius: 6px; padding: 6px; width: 220px; }
.label { font-size: 12px; color: #444; margin: 4px 0; }
.small { font-size: 12px; color: #666; }
</style></head><body>
<h1>Benchmark 4 Pipelines</h1>
<p class="small">Legacy vs Compact6 vs Hybrid vs Hybrid3</p>
"""
    )
    for row in details:
        q_path = Path(row["query_path"])
        parts.append(f'<div class="query-block"><h2>Query: {html.escape(row["query_name"])}</h2>')
        parts.append('<div class="row">')
        parts.append(f'<div class="card"><div class="label">Query image</div>{_img_tag(q_path)}</div>')

        for pipeline in ("legacy", "compact6", "hybrid", "hybrid3"):
            parts.append('<div class="card">')
            parts.append(f'<div class="label">{pipeline} | query {row[f"{pipeline}_ms"]:.1f} ms</div>')
            for rank, (name, dist) in enumerate(row[f"{pipeline}_topk"][:k], start=1):
                rpath = filename_to_path.get(name, Path(name))
                parts.append(f'<div class="small">#{rank} {html.escape(name)} | d={dist:.4f}</div>')
                parts.append(_img_tag(rpath, width=180))
            parts.append("</div>")
        parts.append("</div></div>")
    parts.append("</body></html>")
    out_html.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark nhiều pipeline CBIR và xuất report tổng quan.")
    parser.add_argument("--queries", nargs="*", type=Path, default=[])
    parser.add_argument("--query-limit", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR.parent)
    parser.add_argument("--db-legacy", type=Path, default=DATA_DIR / "features.db")
    parser.add_argument("--db-compact6", type=Path, default=DATA_DIR / "features_compact6.db")
    parser.add_argument("--db-hybrid", type=Path, default=DATA_DIR / "features_hybrid.db")
    parser.add_argument("--db-hybrid3", type=Path, default=DATA_DIR / "features_hybrid3.db")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    query_paths = _normalize_query_list(args.queries, DATASET_DIR, args.query_limit)
    filename_to_path = _build_filename_index(args.dataset_root)

    t0 = time.perf_counter()
    details, summary = _run_benchmark(
        query_paths=query_paths,
        k=args.k,
        db_legacy=args.db_legacy,
        db_compact=args.db_compact6,
        db_hybrid=args.db_hybrid,
        db_hybrid3=args.db_hybrid3,
    )
    elapsed = time.perf_counter() - t0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"benchmark_{ts}_summary.json"
    out_csv = out_dir / f"benchmark_{ts}_details.csv"
    out_html = out_dir / f"benchmark_{ts}_report.html"

    _write_csv(details, out_csv, args.k)
    _write_html(details, out_html, filename_to_path, args.k)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] n_queries={len(query_paths)}, k={args.k}, elapsed={elapsed:.2f}s")
    print(f"[output] summary: {out_json}")
    print(f"[output] details: {out_csv}")
    print(f"[output] report : {out_html}")


if __name__ == "__main__":
    main()
