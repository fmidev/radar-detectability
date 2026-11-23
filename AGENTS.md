# detectability — Agent Instructions

Python package for computing radar detectability (beam overshooting/filling) products. This is a modern Python reimplementation/replacement of the C/shell legacy implementation in `legacy/`. To be used in Apache Airflow pipelines for operational radar data processing at the Finnish Meteorological Institute (FMI).

**Core values**: Modern tools and standards, code readability and maintainability, reuse over reimplementation. Where applicable, leverage `scipy`, `xarray`, `wradlib`, and other established libraries for mathematical and radar-specific operations instead of reimplementing algorithms. **Do it the pythonic way.** This is not strictly a rewrite project, but rather a replacement with comparable functionality and results.

## Project layout

```
src/detectability/   # main package (src layout)
src/scripts/         # data exploration scripts (not packaged)
tests/               # pytest tests
docs/                # release guide
legacy/              # reference C implementation (read-only reference, gitignored)
```

The PyPI distribution is `radar-detectability` (`detectability` is taken on
PyPI); the import package is `detectability`.

## Setup and checks

The environment is managed with [uv](https://docs.astral.sh/uv/) from
[pyproject.toml](pyproject.toml) and the committed `uv.lock`; the Python version
comes from [.python-version](.python-version). Run commands from the repository
root with `uv run`, which creates or syncs `.venv` as needed; do not rely on shell
activation. Without uv: `python -m pip install -e . --group dev` (pip 25.1+).
The build backend is `hatchling`, with `hatch-vcs` deriving the version from Git
tags.

- Run the relevant tests first, e.g. `uv run pytest tests/test_detection.py`.
  Full suite: `uv run pytest tests/`.
- Lint and format touched Python files with Ruff (configured in
  [pyproject.toml](pyproject.toml)): `uv run ruff check --fix <files>` and
  `uv run ruff format <files>`.
- Type checking: `uv run mypy src/detectability tests` must stay clean.
- [CI](.github/workflows/ci.yml) runs Ruff, mypy, the full test suite and a
  wheel build on the whole tree, and repeats the tests with the oldest runtime
  dependency versions allowed by `pyproject.toml`.
- Wheel build, for packaging changes: `uv build`.
  [Containerfile](Containerfile) defines the production image: dependencies
  come from `uv.lock`, and the base image tag fixes the Python minor version
  and Debian release.
- Add or update dependencies with `uv add` (`uv add --dev` for development
  tools) so `uv.lock` stays in sync; commit both files. Give runtime
  dependencies a lower bound that passes the oldest-dependency CI job.
- Releases are Git tags; the version comes from hatch-vcs, never from the
  source. Follow the [release guide](docs/releasing.md); do not tag or publish
  unless asked.
- Report checks run and any failures. Documentation-only changes do not require
  Python tests or builds.

## Airflow Integration

This package is deployed as a containerized service in FMI's Airflow radar production system. You may check the current deployment over ssh at dev.tutka:/airflow/dev/dags/radar_detectability. For changing the deployment, write an easy to copy-paste handover document describing the changes for an agent running on dev.tutka.

The integration pattern:

- **Deployment**: Docker container with this package installed (built via [Containerfile](Containerfile), image `quay.io/fmi/radar_detectability:vx.y.z`).
- **Airflow tasks**: Use `@task.docker` decorator to invoke Python API
- **No DAGs in this repo**: Workflow orchestration lives in the separate Airflow radar production repository
- **Robustness**: Handle missing/corrupted input files and edge cases gracefully, log processing steps

## Conventions

- **Python ≥ 3.14** — use modern syntax freely (e.g. `match`, `type` aliases, `X | Y` unions).
- **src layout** — package root is `src/detectability/`, not the repo root.
- **SPDX headers** — every new source file must start with:
  ```python
  # SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
  #
  # SPDX-License-Identifier: MIT
  ```

### Style

- Ruff formatting and lint rules (see above)
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

`legacy/` is gitignored, so a fresh clone does not have it. **Do not modify
anything in it.** If a change needs the legacy source to resolve algorithm
intent and it is missing, ask for it rather than guess. Reproduce the legacy
algorithm's meaningful behavior, not its accidents: where legacy behavior is a
bug, implement what is physically meaningful, explain the deviation in a
comment and cover it with a test.

## Testing

Tests live in `tests/`, with input fixtures in `tests/data/`; see the
[test guide](tests/README.md). Add regression tests in the corresponding test
module when behavior changes. Tests must not depend on `legacy/`.
Do not relax tolerances or rewrite expected values merely to make a change pass.
