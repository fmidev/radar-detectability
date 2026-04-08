# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for the top-level processing pipeline (detectability.pipeline)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

import detectability
from detectability.pipeline import process

DATA_DIR = Path(__file__).parent / "data"
REAL_ECHOTOP = DATA_DIR / "202603191120_fivim_etop_-10_dbzh_polar_qc.h5"


# ---------------------------------------------------------------------------
# Integration: real file
# ---------------------------------------------------------------------------


class TestProcessRealFile:
    def test_output_file_created(self, tmp_path: Path) -> None:
        out = tmp_path / "detectability.tif"
        process(REAL_ECHOTOP, out, lowest_elevation=0.3)
        assert out.exists()

    def test_output_is_readable_raster(self, tmp_path: Path) -> None:
        import rioxarray  # noqa: F401

        out = tmp_path / "detectability.tif"
        process(REAL_ECHOTOP, out, lowest_elevation=0.3)
        import rasterio  # type: ignore[import-untyped]

        with rasterio.open(str(out)) as src:
            assert src.count == 1
            assert src.dtypes[0] == "uint8"

    def test_output_values_in_range(self, tmp_path: Path) -> None:
        import rasterio  # type: ignore[import-untyped]

        out = tmp_path / "detectability.tif"
        process(REAL_ECHOTOP, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            data = src.read(1)
        assert data.min() >= 0
        assert data.max() <= 255

    def test_explicit_beamwidth_accepted(self, tmp_path: Path) -> None:
        out = tmp_path / "detectability.tif"
        process(REAL_ECHOTOP, out, lowest_elevation=0.3, beamwidth=0.97)
        assert out.exists()

    def test_output_parent_created(self, tmp_path: Path) -> None:
        """Output directory is created if it does not exist."""
        out = tmp_path / "nested" / "deep" / "out.tif"
        process(REAL_ECHOTOP, out, lowest_elevation=0.3)
        assert out.exists()

    def test_top_level_import(self, tmp_path: Path) -> None:
        """process is importable directly from detectability."""
        out = tmp_path / "detectability.tif"
        detectability.process(REAL_ECHOTOP, out, lowest_elevation=0.3)
        assert out.exists()


# ---------------------------------------------------------------------------
# Unit: parameter validation
# ---------------------------------------------------------------------------


class TestProcessValidation:
    def _make_ds_no_beamwidth(self) -> xr.Dataset:
        """Minimal xradar-like Dataset without beamwidth attrs."""
        nrays, nbins = 360, 500
        return xr.Dataset(
            {"HGHT": xr.DataArray(
                np.zeros((nrays, nbins), dtype=np.float64),
                dims=["azimuth", "range"],
            )},
            coords={
                "longitude": 23.82,
                "latitude": 63.10,
                "altitude": 200.0,
                "azimuth": np.arange(nrays, dtype=np.float64) + 0.5,
                "range": (np.arange(nbins) + 0.5) * 500.0,
            },
            # No beamwidth_h attr
        )

    def test_missing_beamwidth_raises(self, tmp_path: Path) -> None:
        ds_no_bw = self._make_ds_no_beamwidth()
        with patch("detectability.pipeline.read_echotop", return_value=ds_no_bw):
            with pytest.raises(ValueError, match="beamwidth"):
                process("dummy.h5", tmp_path / "out.tif", lowest_elevation=0.3)

    def test_explicit_beamwidth_bypasses_file(self, tmp_path: Path) -> None:
        """Explicit beamwidth= should succeed even if file has none."""
        ds_no_bw = self._make_ds_no_beamwidth()
        out = tmp_path / "out.tif"
        with patch("detectability.pipeline.read_echotop", return_value=ds_no_bw):
            # Should not raise
            process("dummy.h5", out, lowest_elevation=0.3, beamwidth=1.0)
        assert out.exists()
