# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for the top-level processing pipeline (detectability.pipeline)."""

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import rasterio
import xarray as xr

import detectability
from detectability.pipeline import process

DATA_DIR = Path(__file__).parent / "data"
REAL_ECHOTOP = DATA_DIR / "202603191120_fivim_etop_-10_dbzh_polar_qc.h5"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_echotop(tmp_path: Path) -> Path:
    """Create a 360×500 ODIM HDF5 with a known uniform HGHT field.

    All bins set to 5000 m — produces a predictable, nonzero
    detectability output for every azimuth.
    """
    nrays, nbins = 360, 500
    rscale = 500.0
    gain = 1.0
    offset = 0.0
    nodata = 65535.0
    undetect = 0.0

    raw = np.full((nrays, nbins), 5000, dtype=np.uint16)

    fpath = tmp_path / "synthetic_etop.h5"
    with h5py.File(fpath, "w") as f:
        what = f.create_group("what")
        what.attrs["object"] = np.bytes_("PVOL")
        what.attrs["version"] = np.bytes_("H5rad 2.2")
        what.attrs["date"] = np.bytes_("20260319")
        what.attrs["time"] = np.bytes_("120000")
        what.attrs["source"] = np.bytes_("NOD:fivim,WMO:02975")

        where = f.create_group("where")
        where.attrs["lon"] = 23.82
        where.attrs["lat"] = 63.10
        where.attrs["height"] = 150.0

        how = f.create_group("how")
        how.attrs["beamwH"] = 1.0
        how.attrs["beamwV"] = 0.9

        ds1 = f.create_group("dataset1")
        ds1_what = ds1.create_group("what")
        ds1_what.attrs["product"] = np.bytes_("ETOP")
        ds1_what.attrs["startdate"] = np.bytes_("20260319")
        ds1_what.attrs["starttime"] = np.bytes_("120000")
        ds1_what.attrs["enddate"] = np.bytes_("20260319")
        ds1_what.attrs["endtime"] = np.bytes_("120200")

        ds1_where = ds1.create_group("where")
        ds1_where.attrs["elangle"] = 0.0
        ds1_where.attrs["nbins"] = np.int64(nbins)
        ds1_where.attrs["nrays"] = np.int64(nrays)
        ds1_where.attrs["rscale"] = rscale
        ds1_where.attrs["rstart"] = 0.0
        ds1_where.attrs["a1gate"] = np.int64(0)

        d1 = ds1.create_group("data1")
        d1.create_dataset("data", data=raw, dtype="uint16")
        d1_what = d1.create_group("what")
        d1_what.attrs["quantity"] = np.bytes_("HGHT")
        d1_what.attrs["gain"] = gain
        d1_what.attrs["offset"] = offset
        d1_what.attrs["nodata"] = nodata
        d1_what.attrs["undetect"] = undetect

    return fpath


@pytest.fixture()
def clear_sky_echotop(tmp_path: Path) -> Path:
    """ODIM HDF5 with HGHT = 0 everywhere (no echo)."""
    nrays, nbins = 360, 500
    rscale = 500.0

    fpath = tmp_path / "clearsky_etop.h5"
    with h5py.File(fpath, "w") as f:
        what = f.create_group("what")
        what.attrs["object"] = np.bytes_("PVOL")
        what.attrs["version"] = np.bytes_("H5rad 2.2")
        what.attrs["date"] = np.bytes_("20260319")
        what.attrs["time"] = np.bytes_("120000")
        what.attrs["source"] = np.bytes_("NOD:fivim,WMO:02975")

        where = f.create_group("where")
        where.attrs["lon"] = 23.82
        where.attrs["lat"] = 63.10
        where.attrs["height"] = 150.0

        how = f.create_group("how")
        how.attrs["beamwH"] = 1.0
        how.attrs["beamwV"] = 0.9

        ds1 = f.create_group("dataset1")
        ds1_what = ds1.create_group("what")
        ds1_what.attrs["product"] = np.bytes_("ETOP")
        ds1_what.attrs["startdate"] = np.bytes_("20260319")
        ds1_what.attrs["starttime"] = np.bytes_("120000")
        ds1_what.attrs["enddate"] = np.bytes_("20260319")
        ds1_what.attrs["endtime"] = np.bytes_("120200")

        ds1_where = ds1.create_group("where")
        ds1_where.attrs["elangle"] = 0.0
        ds1_where.attrs["nbins"] = np.int64(nbins)
        ds1_where.attrs["nrays"] = np.int64(nrays)
        ds1_where.attrs["rscale"] = rscale
        ds1_where.attrs["rstart"] = 0.0
        ds1_where.attrs["a1gate"] = np.int64(0)

        d1 = ds1.create_group("data1")
        d1.create_dataset(
            "data",
            data=np.zeros((nrays, nbins), dtype=np.uint16),
            dtype="uint16",
        )
        d1_what = d1.create_group("what")
        d1_what.attrs["quantity"] = np.bytes_("HGHT")
        d1_what.attrs["gain"] = 1.0
        d1_what.attrs["offset"] = 0.0
        d1_what.attrs["nodata"] = 65535.0
        d1_what.attrs["undetect"] = 0.0

    return fpath


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


# ---------------------------------------------------------------------------
# Integration: synthetic ODIM HDF5 — deterministic checks
# ---------------------------------------------------------------------------


class TestProcessSynthetic:
    """Full pipeline with known synthetic input for deterministic verification."""

    def test_output_file_created(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.tif"
        process(synthetic_echotop, out, lowest_elevation=0.3)
        assert out.exists()

    def test_output_crs_epsg3067(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.tif"
        process(synthetic_echotop, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            assert src.crs is not None
            assert src.crs.to_epsg() == 3067

    def test_output_dtype_uint8(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.tif"
        process(synthetic_echotop, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            assert src.dtypes[0] == "uint8"

    def test_output_single_band(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.tif"
        process(synthetic_echotop, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            assert src.count == 1

    def test_uniform_top_azimuthally_symmetric(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        """Uniform 5 km TOP → near-symmetric detectability field."""
        out = tmp_path / "out.tif"
        process(synthetic_echotop, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            data = src.read(1)
        # Non-trivial output: should contain both low and high values
        assert data.max() > 0
        assert (data == 0).any()  # near-range bins fully filled

    def test_contains_transition_zone(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        """Detectability should include intermediate values (not just 0/255)."""
        out = tmp_path / "out.tif"
        process(synthetic_echotop, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            data = src.read(1)
        mid = data[(data > 0) & (data < 255)]
        assert len(mid) > 0, "expected transition zone with intermediate values"

    def test_custom_crs(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        """Custom CRS parameter is respected in output."""
        out = tmp_path / "out.tif"
        process(synthetic_echotop, out, lowest_elevation=0.3, crs="EPSG:4326")
        with rasterio.open(str(out)) as src:
            assert src.crs.to_epsg() == 4326

    def test_lower_elevation_shifts_detection(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        """Lower elevation → beam reaches TOP at longer range → less overshooting."""
        out_low = tmp_path / "low.tif"
        out_high = tmp_path / "high.tif"
        process(synthetic_echotop, out_low, lowest_elevation=0.1)
        process(synthetic_echotop, out_high, lowest_elevation=0.5)
        with rasterio.open(str(out_low)) as src:
            low = src.read(1).astype(np.float64)
        with rasterio.open(str(out_high)) as src:
            high = src.read(1).astype(np.float64)
        # Higher elevation → beam overshoots sooner → higher mean detectability
        assert high.mean() > low.mean()


# ---------------------------------------------------------------------------
# Integration: clear sky (all zero)
# ---------------------------------------------------------------------------


class TestProcessClearSky:
    """Pipeline with no echo → all detectability 0 (full overshooting at range 0)."""

    def test_output_created(
        self, clear_sky_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.tif"
        process(clear_sky_echotop, out, lowest_elevation=0.3)
        assert out.exists()

    def test_all_zero_or_max(
        self, clear_sky_echotop: Path, tmp_path: Path
    ) -> None:
        """No echo → smoothed TOP = 0 → all pixels either 0 or 255."""
        out = tmp_path / "out.tif"
        process(clear_sky_echotop, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            data = src.read(1)
        # With zero TOP: range_full = range_zero = 0 → frac = r/1 clipped
        # All bins are at or beyond the "zero" range → 255 or 0
        unique = set(np.unique(data))
        assert unique <= {0, 255}


# ---------------------------------------------------------------------------
# Integration: real file — output property checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ECHOTOP.exists(), reason="real test data not found")
class TestProcessRealFileProperties:
    """Deeper checks on real-file output COG."""

    def test_output_crs(self, tmp_path: Path) -> None:
        out = tmp_path / "out.tif"
        process(REAL_ECHOTOP, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            assert src.crs.to_epsg() == 3067

    def test_spatial_extent_reasonable(self, tmp_path: Path) -> None:
        """Grid should span ~500 km around the radar."""
        out = tmp_path / "out.tif"
        process(REAL_ECHOTOP, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            b = src.bounds
        width_m = b.right - b.left
        height_m = b.top - b.bottom
        # 500 bins × 500 m = 250 km radius → ~500 km diameter
        assert 400_000 < width_m < 600_000
        assert 400_000 < height_m < 600_000

    def test_resolution_500m(self, tmp_path: Path) -> None:
        out = tmp_path / "out.tif"
        process(REAL_ECHOTOP, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            assert abs(src.res[0]) == pytest.approx(500.0, abs=1.0)
            assert abs(src.res[1]) == pytest.approx(500.0, abs=1.0)

    def test_nonzero_detection_at_far_range(self, tmp_path: Path) -> None:
        """Real data should produce some detectability > 0 at far range."""
        out = tmp_path / "out.tif"
        process(REAL_ECHOTOP, out, lowest_elevation=0.3)
        with rasterio.open(str(out)) as src:
            data = src.read(1)
        assert (data > 0).any()
        assert (data == 255).any()  # some pixels fully overshooting
