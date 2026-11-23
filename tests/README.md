# Tests

Run from the repository root:

```bash
uv run pytest tests/                                   # full suite
uv run pytest tests/test_detection.py                  # one module
uv run pytest tests/ --cov=detectability --cov-report=term-missing
```

## Organization

Each module tests the corresponding module in `src/detectability/`:

| Test file           | Module         | Covers                                                     |
| ------------------- | -------------- | ---------------------------------------------------------- |
| `test_io.py`        | `io.py`        | ODIM echo-top reading, decoding, nominal time, elevation   |
| `test_filtering.py` | `filtering.py` | Azimuthal spike filter                                     |
| `test_analysis.py`  | `analysis.py`  | Ray TOP selection and sector smoothing                     |
| `test_beam.py`      | `beam.py`      | Beam geometry                                              |
| `test_detection.py` | `detection.py` | Detection ranges and beam overshoot encoding               |
| `test_georef.py`    | `georef.py`    | Polar-to-projected regridding and COG output               |
| `test_state.py`     | `state.py`     | Background TOP state, aging and climatology fallback       |
| `test_composite.py` | `composite.py` | Minimum composite and provenance tags                      |
| `test_pipeline.py`  | `pipeline.py`  | End-to-end `process()` on synthetic and real inputs        |

Most tests use synthetic arrays or small ODIM HDF5 files written to
`tmp_path`, so expected values can be derived analytically. Real files check
that operational inputs are read and processed correctly.

## Test data

FMI ODIM HDF5 files in `data/`, 360 rays × 500 bins:

| File                                                   | Radar           | Use                                        |
| ------------------------------------------------------ | --------------- | ------------------------------------------ |
| `202603191120_fivim_etop_-10_dbzh_polar_qc.h5`         | Vimpeli         | Main real echo-top input (I/O, pipeline)   |
| `202611051710_{fikor,filuo,fivih}_etop_-10_dbzh_polar_qc.h5` | Korppoo, Luosto, Vihti | Lowest elevations 0.5°, 0.1° and 0.3° |
| `202603191120_fivim_volume_ABCDEF_qc.h5`               | Vimpeli         | Volume scan for `src/scripts/explore_data.py` only; not in the sdist |

## Legacy reference

The legacy C sources in `legacy/` are gitignored and may be missing. Tests
must not require them: cross-checks against legacy behavior restate the legacy
formula in the test (e.g. `bindist()` in `test_beam.py`) and quote it briefly.
