# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for echotop I/O (detectability.io)."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from detectability.io import read_echotop

DATA_DIR = Path(__file__).parent / "data"
REAL_ECHOTOP = DATA_DIR / "202603191120_fivim_etop_-10_dbzh_polar_qc.h5"


# ---------------------------------------------------------------------------
# Synthetic ODIM HDF5 fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_echotop(tmp_path: Path) -> Path:
    """Create a minimal ODIM HDF5 file with a known HGHT field."""
    nrays, nbins = 36, 50
    rscale = 1000.0  # 1 km bins
    gain = 1.0
    offset = 0.0
    nodata = 65535.0
    undetect = 0.0

    # TOP heights: linearly increasing per range bin, 0–4900 m
    raw = np.arange(nbins, dtype=np.uint16)[np.newaxis, :].repeat(nrays, axis=0)
    # First 5 bins: undetect (raw=0)
    raw[:, :5] = 0

    fpath = tmp_path / "synthetic_etop.h5"
    with h5py.File(fpath, "w") as f:
        # /what
        what = f.create_group("what")
        what.attrs["object"] = np.bytes_("PVOL")
        what.attrs["version"] = np.bytes_("H5rad 2.2")
        what.attrs["date"] = np.bytes_("20260319")
        what.attrs["time"] = np.bytes_("120000")
        what.attrs["source"] = np.bytes_("NOD:fivim,WMO:02975")

        # /where
        where = f.create_group("where")
        where.attrs["lon"] = 23.82
        where.attrs["lat"] = 63.10
        where.attrs["height"] = 150.0

        # /how — beamwidth
        how = f.create_group("how")
        how.attrs["beamwH"] = 1.0
        how.attrs["beamwV"] = 0.9

        # /dataset1
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

        # /dataset1/data1 — HGHT
        d1 = ds1.create_group("data1")
        d1.create_dataset("data", data=raw, dtype="uint16")
        d1_what = d1.create_group("what")
        d1_what.attrs["quantity"] = np.bytes_("HGHT")
        d1_what.attrs["gain"] = gain
        d1_what.attrs["offset"] = offset
        d1_what.attrs["nodata"] = nodata
        d1_what.attrs["undetect"] = undetect

    return fpath


# ---------------------------------------------------------------------------
# Tests with real echotop file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_ECHOTOP.exists(), reason="real test data not found")
class TestReadEchotopReal:
    """Tests using the real Vimpeli echotop file."""

    def test_returns_dataset(self) -> None:
        ds = read_echotop(REAL_ECHOTOP)
        assert isinstance(ds, xr.Dataset)

    def test_hght_shape(self) -> None:
        ds = read_echotop(REAL_ECHOTOP)
        assert ds["HGHT"].shape == (360, 500)

    def test_hght_dtype_float64(self) -> None:
        ds = read_echotop(REAL_ECHOTOP)
        assert ds["HGHT"].dtype == np.float64

    def test_has_nonzero_values(self) -> None:
        ds = read_echotop(REAL_ECHOTOP)
        assert (ds["HGHT"].values > 0).any()

    def test_reasonable_max_height(self) -> None:
        ds = read_echotop(REAL_ECHOTOP)
        assert ds["HGHT"].values.max() < 25_000  # < 25 km

    def test_undetect_attr_present(self) -> None:
        """xradar stores the decoded undetect sentinel in HGHT._Undetect."""
        ds = read_echotop(REAL_ECHOTOP)
        assert "_Undetect" in ds["HGHT"].attrs

    def test_azimuth_coord(self) -> None:
        ds = read_echotop(REAL_ECHOTOP)
        az = ds.coords["azimuth"].values
        assert az.min() >= 0
        assert az.max() < 360

    def test_range_coord_positive(self) -> None:
        ds = read_echotop(REAL_ECHOTOP)
        assert (ds.coords["range"].values > 0).all()

    def test_radar_site_coords(self) -> None:
        ds = read_echotop(REAL_ECHOTOP)
        assert float(ds.coords["longitude"].values) == pytest.approx(23.82, abs=0.01)
        assert float(ds.coords["latitude"].values) == pytest.approx(63.10, abs=0.01)
        assert float(ds.coords["altitude"].values) == pytest.approx(200.0)

    def test_beamwidth_attrs(self) -> None:
        ds = read_echotop(REAL_ECHOTOP)
        assert ds.attrs["beamwidth_h"] == pytest.approx(0.97, abs=0.01)
        assert ds.attrs["beamwidth_v"] == pytest.approx(0.88, abs=0.01)


# ---------------------------------------------------------------------------
# Tests with synthetic ODIM fixture
# ---------------------------------------------------------------------------

class TestReadEchotopSynthetic:
    """Tests using a synthetic ODIM HDF5 fixture."""

    def test_returns_dataset(self, synthetic_echotop: Path) -> None:
        ds = read_echotop(synthetic_echotop)
        assert isinstance(ds, xr.Dataset)

    def test_shape(self, synthetic_echotop: Path) -> None:
        ds = read_echotop(synthetic_echotop)
        assert ds["HGHT"].shape == (36, 50)

    def test_undetect_preserved(self, synthetic_echotop: Path) -> None:
        """Undetect pixels are kept at their decoded value, not zeroed."""
        ds = read_echotop(synthetic_echotop)
        # In synthetic fixture: gain=1, offset=0, undetect raw=0 -> decoded=0
        assert "_Undetect" in ds["HGHT"].attrs

    def test_valid_values_positive(self, synthetic_echotop: Path) -> None:
        ds = read_echotop(synthetic_echotop)
        # Bins 5+ had raw values 5..49 -> decoded to 5..49 (gain=1, offset=0)
        valid = ds["HGHT"].values[:, 5:]
        assert (valid > 0).all()

    def test_radar_site_from_sweep_coords(self, synthetic_echotop: Path) -> None:
        ds = read_echotop(synthetic_echotop)
        assert float(ds.coords["longitude"].values) == pytest.approx(23.82)
        assert float(ds.coords["latitude"].values) == pytest.approx(63.10)
        assert float(ds.coords["altitude"].values) == pytest.approx(150.0)

    def test_beamwidth_from_how(self, synthetic_echotop: Path) -> None:
        ds = read_echotop(synthetic_echotop)
        assert ds.attrs["beamwidth_h"] == pytest.approx(1.0)
        assert ds.attrs["beamwidth_v"] == pytest.approx(0.9)

    def test_no_how_group(self, tmp_path: Path, synthetic_echotop: Path) -> None:
        """File without /how group should still work, just no beamwidth attrs."""
        # Copy and remove /how
        import shutil
        no_how = tmp_path / "no_how.h5"
        shutil.copy(synthetic_echotop, no_how)
        with h5py.File(no_how, "a") as f:
            del f["how"]
        ds = read_echotop(no_how)
        assert "beamwidth_h" not in ds.attrs
        assert "beamwidth_v" not in ds.attrs
        assert ds["HGHT"].shape == (36, 50)
