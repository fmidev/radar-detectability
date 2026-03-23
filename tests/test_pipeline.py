# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Integration tests for the detectability pipeline."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio

from detectability.pipeline import process
from detectability.state import CLIMATOLOGICAL_TOP_M, BackgroundState, save_state

# Vimpeli radar parameters
RADAR_LAT = 63.1048
RADAR_LON = 23.8209
RADAR_ALT = 200.0
BEAMW_H = 0.97
BEAMW_V = 0.88
LOWEST_ELEV = 0.3


@pytest.fixture()
def synthetic_echotop(tmp_path: Path) -> Path:
    """Create a minimal ODIM HDF5 echotop product."""
    path = tmp_path / "echotop.h5"
    nrays, nbins = 360, 100
    rscale = 500.0

    # Dome pattern: TOP decreasing with range (5 km → 0)
    rng = np.arange(nbins)
    top_m = np.maximum(5000.0 - rng * 50.0, 0.0)
    top_2d = np.broadcast_to(top_m[np.newaxis, :], (nrays, nbins)).copy()

    # Encode: raw = (physical - offset) / gain
    gain, offset = 1.0, 0.0
    raw = top_2d.astype(np.uint16)

    with h5py.File(path, "w") as f:
        f.create_group("what").attrs["object"] = "PVOL"

        where = f.create_group("where")
        where.attrs["lat"] = RADAR_LAT
        where.attrs["lon"] = RADAR_LON
        where.attrs["height"] = RADAR_ALT

        how = f.create_group("how")
        how.attrs["beamwH"] = BEAMW_H
        how.attrs["beamwV"] = BEAMW_V

        ds = f.create_group("dataset1")
        ds_where = ds.create_group("where")
        ds_where.attrs["elangle"] = 0.0
        ds_where.attrs["nbins"] = nbins
        ds_where.attrs["nrays"] = nrays
        ds_where.attrs["rscale"] = rscale
        ds_where.attrs["rstart"] = 0.0

        data_group = ds.create_group("data1")
        data_what = data_group.create_group("what")
        data_what.attrs["quantity"] = np.bytes_(b"HGHT")
        data_what.attrs["gain"] = gain
        data_what.attrs["offset"] = offset
        data_what.attrs["nodata"] = 65535.0
        data_what.attrs["undetect"] = 0.0

        data_group.create_dataset("data", data=raw)

    return path


class TestProcessIntegration:
    """End-to-end pipeline integration tests."""

    def test_produces_cog(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "det.tif"
        process(
            synthetic_echotop,
            out,
            lowest_elevation=LOWEST_ELEV,
            nbins=100,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_cog_crs(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "det.tif"
        process(
            synthetic_echotop,
            out,
            lowest_elevation=LOWEST_ELEV,
            nbins=100,
        )
        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 3067

    def test_cog_dtype_uint8(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "det.tif"
        process(
            synthetic_echotop,
            out,
            lowest_elevation=LOWEST_ELEV,
            nbins=100,
        )
        with rasterio.open(out) as src:
            assert src.dtypes[0] == "uint8"

    def test_returns_polar_dataset(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "det.tif"
        ds = process(
            synthetic_echotop,
            out,
            lowest_elevation=LOWEST_ELEV,
            nbins=100,
        )
        assert "detectability" in ds
        assert ds["detectability"].dtype == np.uint8
        assert ds["detectability"].shape == (360, 100)

    def test_value_range(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "det.tif"
        ds = process(
            synthetic_echotop,
            out,
            lowest_elevation=LOWEST_ELEV,
            nbins=100,
        )
        assert ds["detectability"].values.min() >= 0
        assert ds["detectability"].values.max() <= 255

    def test_state_created(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "det.tif"
        state_path = tmp_path / "state.json"
        process(
            synthetic_echotop,
            out,
            lowest_elevation=LOWEST_ELEV,
            state_path=state_path,
            nbins=100,
        )
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert "top_m" in data
        assert data["top_m"] > 0

    def test_state_used_on_second_run(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out1 = tmp_path / "det1.tif"
        out2 = tmp_path / "det2.tif"
        state_path = tmp_path / "state.json"
        t1 = datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(minutes=15)

        process(
            synthetic_echotop,
            out1,
            lowest_elevation=LOWEST_ELEV,
            state_path=state_path,
            nbins=100,
            current_time=t1,
        )
        process(
            synthetic_echotop,
            out2,
            lowest_elevation=LOWEST_ELEV,
            state_path=state_path,
            nbins=100,
            current_time=t2,
        )
        data = json.loads(state_path.read_text())
        assert data["timestamp"] == t2.isoformat()

    def test_no_state_uses_climatology(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        """Without state_path, pipeline still runs using climatological default."""
        out = tmp_path / "det.tif"
        ds = process(
            synthetic_echotop,
            out,
            lowest_elevation=LOWEST_ELEV,
            nbins=100,
        )
        assert ds["detectability"].shape == (360, 100)

    def test_custom_crs(
        self, synthetic_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "det.tif"
        process(
            synthetic_echotop,
            out,
            lowest_elevation=LOWEST_ELEV,
            nbins=100,
            crs="EPSG:3035",
        )
        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 3035


class TestProcessWithRealData:
    """Tests using the real echotop test data (skipped if unavailable)."""

    DATA_PATH = Path(__file__).parent / "data" / "202603191120_fivim_etop_-10_dbzh_polar_qc.h5"

    @pytest.fixture()
    def real_echotop(self) -> Path:
        if not self.DATA_PATH.exists():
            pytest.skip("Real echotop data not available")
        return self.DATA_PATH

    def test_real_data_pipeline(
        self, real_echotop: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "det_real.tif"
        ds = process(
            real_echotop,
            out,
            lowest_elevation=LOWEST_ELEV,
        )
        assert out.exists()
        assert ds["detectability"].shape == (360, 500)
        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 3067
            assert src.dtypes[0] == "uint8"
