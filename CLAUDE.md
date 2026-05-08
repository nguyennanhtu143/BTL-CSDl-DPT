# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Content-Based Image Retrieval (CBIR) system for nature background images. Uses the **hybrid3 pipeline** — a single retrieval pipeline combining color histogram, gradient histogram, and compact6 scalar features. Searches a 500-image dataset using feature vectors stored in SQLite.

## Setup

```bash
pip install -r requirements.txt
```

Uses a `.venv` local virtual environment. Activate with `.venv\Scripts\activate` (Windows).

## Common Commands

**Build feature database (run once, or after dataset changes):**
```bash
python build_database_hybrid3.py                  # incremental
python build_database_hybrid3.py --rebuild        # force rebuild from scratch
python build_database_hybrid3.py --limit 50       # quick test with 50 images
```

**Query via CLI:**
```bash
python query_cli_hybrid3.py path/to/image.jpg
python query_cli_hybrid3.py img.jpg --k 10 --breakdown
python query_cli_hybrid3.py img.jpg --coarse-top 50    # two-stage retrieval
python query_cli_hybrid3.py img.jpg --json             # JSON output
```

**Run the web UI:**
```bash
streamlit run ui/app.py
```

**Inspect the database:**
```bash
python inspect_db.py
python inspect_db.py --filename 01_thien_nhien_0001.jpg --full
python inspect_db.py --schema
```

## Architecture

### Single Pipeline: Hybrid3 (663-dim)

| Branch | Module | Dim | Description |
|---|---|---|---|
| **Color** | `src/color_features.py` | 576 | RGB histogram on 3×3 grid × 4³ bins, L1-normalized |
| **Gradient** | `src/gradient_features.py` | 81 | Sobel HOG on 3×3 grid × 9 unsigned-orientation bins, L1-normalized |
| **Compact6** | `src/compact6_features.py` | 6 | Scalar features: mean_rgb, stddev_rgb, skewness_rgb, coarseness, contrast, directionality |

Compact6 vector: `[mean_rgb, stddev_rgb, skewness_rgb, coarseness, contrast, directionality]`.

### Distance Computation (L2 only)

`src/distances.py` provides `euclidean_distance_batch(query, db)`. The matcher computes Euclidean distance per branch, mean-normalizes each branch (divides by the per-query mean distance across the DB so all branches have equal amplitude), then combines with weights:

```
total = w_hist * (d_hist / mean_d_hist)
      + w_grad * (d_grad / mean_d_grad)
      + w_compact * (d_compact / mean_d_compact)
```

Default weights (in `src/matcher_hybrid3.py`): `w_hist=0.45, w_grad=0.30, w_compact=0.25`.

### Two-Stage Retrieval (optional)

`src/matcher_hybrid3.py:find_top_k_hybrid3_two_stage` uses compact6 (6-dim, very cheap) to coarse-filter top-M candidates, then ranks them with the full hybrid3 distance. Triggered via `--coarse-top N` in the CLI.

### Data Flow

1. Image loaded via `src/preprocessing.py` (load → resize to 256×144 → grayscale)
2. Three feature branches extracted independently
3. Stored as float32 BLOBs in SQLite via `src/database_hybrid3.py`
4. At query time, the full DB is loaded into a `Hybrid3Snapshot` dataclass (all vectors in RAM as numpy arrays)
5. Vectorized batch L2 distance computed against all N=500 vectors in one numpy broadcast
6. Top-k returned sorted by ascending combined distance

### Key Config (`src/config.py`)

- `DATASET_DIR`: `Images_Dataset/01_thien_nhien/`
- `DATA_DIR`: `data/`
- `IMG_WIDTH=256`, `IMG_HEIGHT=144` (resize target)
- `GRID=3`, `COLOR_BINS=4`, `GRAD_BINS=9`
- `TOP_K=5`

### Streamlit UI (`ui/app.py`)

Loads `data/features_hybrid3.db` once into `@st.cache_resource`. On search, extracts features from uploaded image bytes (in-memory, no temp file), runs `find_top_k_hybrid3` (or `_two_stage`), displays results as a grid. Sidebar exposes weights, top-k, and `coarse_top` for two-stage retrieval.
