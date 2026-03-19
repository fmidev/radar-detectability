# detectability — Copilot Instructions

Python package for computing radar detectability (beam overshooting/filling) products. This is a modern Python reimplementation/replacement of the C/shell legacy implementation in `legacy/`. To be used in Apache Airflow pipelines for operational radar data processing at the Finnish Meteorological Institute (FMI).

**Core values**: Modern tools and standards, code readability and maintainability, reuse over reimplementation. Where applicable, leverage `scipy`, `xarray`, `wradlib`, and other established libraries for mathematical and radar-specific operations instead of reimplementing algorithms. **Do it the pythonic way.** This is not strictly a rewrite
project, but rather a replacement with comparable functionality and results.

## Project layout

```
src/detectability/   # main package (src layout)
tests/               # pytest tests
legacy/              # reference C implementation (read-only reference)
```

## Build & tooling

Uses **Hatch** as the project manager and build backend (`hatchling` + `hatch-vcs` for version from git tags).

No `setup.py` or `requirements.txt` — all metadata lives in `pyproject.toml`.

## Airflow Integration

This package is deployed as a containerized service in FMI's Airflow radar production system. The integration pattern:

- **Deployment**: Docker container with this package installed (built via [Containerfile](Containerfile), image `quay.io/fmi/radar-detectability:vx.y.z`).
- **Airflow tasks**: Use `@task.docker` decorator to invoke Python API
- **No DAGs in this repo**: Workflow orchestration lives in the separate Airflow radar production repository
- **Robustness**: Handle missing/corrupted input files and edge cases gracefully, log processing steps

## Conventions

- **Python ≥ 3.12** — use modern syntax freely (e.g. `match`, `type` aliases, `X | Y` unions).
- **src layout** — package root is `src/detectability/`, not the repo root.
- **SPDX headers** — every new source file must start with:
  ```python
  # SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
  #
  # SPDX-License-Identifier: MIT
  ```

### Style
- Follow Black formatting
- Naming, comments, etc. in English
- Mention corresponding legacy names for key variables in comments/docstrings if helpful
- It's better to briefly quote legacy code than to refer to line numbers
- Use `logging` module for debug/info/warning/error messages
- Type hinting for all functions
- Succinct, to the point documentation
- Avoid repeating bad practices from legacy code

## Domain knowledge

The package computes **radar detectability fields** — polar-coordinate grids that describe at what range a precipitating cloud becomes visible to a weather radar beam.

Key concepts (from legacy docs):
- **Echo-top (TOP)**: the altitude of the highest radar echo above a dBZ threshold (typically −10 dBZ).
- **Azimuthal smoothing**: raw TOPs are smoothed over a configurable azimuthal sector (e.g. 60°).
- **Detectability range**: derived from the lowest radar beam geometry — full detectability when the beam top is below TOP, zero detectability when the beam bottom is above TOP.

The legacy C programs (`legacy/src/`) are the authoritative reference for algorithms. When implementing Python equivalents, cross-check against them for numerical consistency.

## Testing

Tests live in `tests/`. Use the agent hooks to run tests.
