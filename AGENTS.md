# Repository Guidelines

## Project Structure & Module Organization

This repository implements a Python content-based image retrieval system for nature images using the hybrid3 pipeline. Core reusable logic lives in `src/`: preprocessing, color features, gradient features, compact scalar features, distance calculations, SQLite persistence, and matching. Entry-point scripts are at the repository root: `build_database_hybrid3.py`, `query_cli_hybrid3.py`, and `inspect_db.py`. The Streamlit interface is in `ui/app.py`. Dataset images are expected under `Images_Dataset/01_thien_nhien/`; generated feature databases live in `data/`; benchmark outputs are stored in `reports/`; design and usage notes are in `docs/`.

## Build, Test, and Development Commands

Set up dependencies from a virtual environment:

```bash
pip install -r requirements.txt
```

Build or refresh the feature database:

```bash
python build_database_hybrid3.py
python build_database_hybrid3.py --rebuild
python build_database_hybrid3.py --limit 50
```

Run queries and inspect results:

```bash
python query_cli_hybrid3.py path/to/image.jpg --k 10 --breakdown
python query_cli_hybrid3.py path/to/image.jpg --coarse-top 50 --json
python inspect_db.py --schema
```

Start the web UI with:

```bash
streamlit run ui/app.py
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and clear snake_case names for functions, variables, and modules. Keep feature extraction and distance math vectorized with NumPy where practical. Prefer small, focused functions in `src/` and keep CLI argument handling in root scripts. Configuration constants belong in `src/config.py` and should use uppercase names, such as `TOP_K` or `DATASET_DIR`.

## Testing Guidelines

There is no committed automated test suite yet. For changes, run a quick build with `--limit`, query at least one known image, and inspect the database schema when persistence changes. If adding tests, place them under `tests/`, name files `test_*.py`, and prefer deterministic fixtures over reading the full image dataset.

## Commit & Pull Request Guidelines

Recent history uses short subjects such as `feat: hybrid search`, `feat: hybrid 3`, and `clean code`. Prefer concise imperative commits, optionally with a conventional prefix (`feat:`, `fix:`, `docs:`). Pull requests should describe the retrieval behavior affected, list commands run for verification, link related issues or phase docs, and include screenshots when changing `ui/app.py`.

## Security & Configuration Tips

Do not commit generated databases, virtual environments, or large benchmark artifacts unless intentionally updating shared results. Treat dataset paths as local configuration, and avoid hard-coding absolute machine-specific paths outside `src/config.py`.
