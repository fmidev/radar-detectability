# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for composite detectability (detectability.composite)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from detectability.composite import NODATA, composite_min


def _write_test_cog(
    path: Path,
    data: np.ndarray,
    bounds: tuple[float, float, float, float],
    nodata: int = NODATA,
    tags: dict[str, str] | None = None,
) -> None:
    """Write a minimal uint8 COG for testing."""
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:3067",
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.uint8), 1)
        if tags:
            dst.update_tags(**tags)


class TestCompositeMin:
    """Tests for composite_min."""

    def test_single_input_passthrough(self) -> None:
        """Single input should produce equivalent output."""
        bounds = (100000.0, 6600000.0, 110000.0, 6610000.0)
        data = np.full((20, 20), 100, dtype=np.uint8)
        data[5:15, 5:15] = 50

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "radar1.tif"
            out = Path(tmpdir) / "composite.tif"
            _write_test_cog(inp, data, bounds)
            composite_min(
                [inp], out, bounds=bounds, resolution=500.0
            )
            with rasterio.open(out) as src:
                result = src.read(1)
                assert src.nodata == NODATA
                assert src.crs.to_epsg() == 3067
                assert src.descriptions == ("beam overshoot",)
                assert src.units == ("%",)
                # Centre region should have value 50
                ny, nx = result.shape
                assert result[ny // 2, nx // 2] == 50

    def test_min_aggregation_overlapping(self) -> None:
        """Overlapping areas should take the minimum value."""
        bounds = (100000.0, 6600000.0, 110000.0, 6610000.0)
        # Radar 1: uniform 90
        data1 = np.full((20, 20), 90, dtype=np.uint8)
        # Radar 2: uniform 80
        data2 = np.full((20, 20), 80, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp1 = Path(tmpdir) / "radar1.tif"
            inp2 = Path(tmpdir) / "radar2.tif"
            out = Path(tmpdir) / "composite.tif"
            _write_test_cog(inp1, data1, bounds)
            _write_test_cog(inp2, data2, bounds)
            composite_min(
                [inp1, inp2], out, bounds=bounds, resolution=500.0
            )
            with rasterio.open(out) as src:
                result = src.read(1)
                # All pixels should be min(90, 80) = 80
                valid = result[result != NODATA]
                assert np.all(valid == 80)

    def test_nodata_transparent_in_merge(self) -> None:
        """Nodata cells from one radar should not override valid data."""
        bounds = (100000.0, 6600000.0, 110000.0, 6610000.0)
        # Radar 1: left half valid (100), right half nodata
        data1 = np.full((20, 20), NODATA, dtype=np.uint8)
        data1[:, :10] = 100
        # Radar 2: right half valid (50), left half nodata
        data2 = np.full((20, 20), NODATA, dtype=np.uint8)
        data2[:, 10:] = 50

        with tempfile.TemporaryDirectory() as tmpdir:
            inp1 = Path(tmpdir) / "radar1.tif"
            inp2 = Path(tmpdir) / "radar2.tif"
            out = Path(tmpdir) / "composite.tif"
            _write_test_cog(inp1, data1, bounds)
            _write_test_cog(inp2, data2, bounds)
            composite_min(
                [inp1, inp2], out, bounds=bounds, resolution=500.0
            )
            with rasterio.open(out) as src:
                result = src.read(1)
                # Left side should have radar1's value (100)
                assert np.all(result[:, :10] == 100)
                # Right side should have radar2's value (50)
                assert np.all(result[:, 10:] == 50)

    def test_no_coverage_is_nodata(self) -> None:
        """Cells not covered by any radar should be nodata."""
        # Composite bounds larger than input coverage
        composite_bounds = (100000.0, 6600000.0, 120000.0, 6620000.0)
        # Input only covers bottom-left quarter
        input_bounds = (100000.0, 6600000.0, 110000.0, 6610000.0)
        data = np.full((20, 20), 50, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "radar1.tif"
            out = Path(tmpdir) / "composite.tif"
            _write_test_cog(inp, data, input_bounds)
            composite_min(
                [inp], out, bounds=composite_bounds, resolution=500.0
            )
            with rasterio.open(out) as src:
                result = src.read(1)
                # Top-right corner should be nodata
                assert result[0, -1] == NODATA
                # Bottom-left should have valid data
                assert result[-1, 0] == 50

    def test_empty_input_raises(self) -> None:
        """Empty input list should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "composite.tif"
            with pytest.raises(ValueError, match="No input paths"):
                composite_min([], out)

    def test_output_creates_parent_dirs(self) -> None:
        """Output path parent directories should be created."""
        bounds = (100000.0, 6600000.0, 110000.0, 6610000.0)
        data = np.full((20, 20), 100, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "radar1.tif"
            out = Path(tmpdir) / "sub" / "dir" / "composite.tif"
            _write_test_cog(inp, data, bounds)
            composite_min(
                [inp], out, bounds=bounds, resolution=500.0
            )
            assert out.exists()

    def test_output_dtype_uint8(self) -> None:
        """Output COG should be uint8."""
        bounds = (100000.0, 6600000.0, 110000.0, 6610000.0)
        data = np.full((20, 20), 100, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "radar1.tif"
            out = Path(tmpdir) / "composite.tif"
            _write_test_cog(inp, data, bounds)
            composite_min(
                [inp], out, bounds=bounds, resolution=500.0
            )
            with rasterio.open(out) as src:
                assert src.dtypes[0] == "uint8"


class TestCompositeProvenance:
    """Composite GeoTIFF tags derived from the input tags."""

    BOUNDS = (100000.0, 6600000.0, 110000.0, 6610000.0)
    NOMINAL_TIME = "2026-03-19T11:20:00+00:00"

    def _composite_tags(
        self, tmp_path: Path, input_tags: list[dict[str, str]]
    ) -> dict[str, str]:
        data = np.full((20, 20), 50, dtype=np.uint8)
        inputs = []
        for i, tags in enumerate(input_tags):
            inp = tmp_path / f"radar{i}.tif"
            _write_test_cog(inp, data, self.BOUNDS, tags=tags)
            inputs.append(inp)
        out = tmp_path / "composite.tif"
        composite_min(inputs, out, bounds=self.BOUNDS, resolution=500.0)
        with rasterio.open(out) as src:
            return src.tags()

    def test_radars_and_nominal_time(self, tmp_path: Path) -> None:
        tags = self._composite_tags(
            tmp_path,
            [
                {"radar": "fivim", "nominal_time": self.NOMINAL_TIME},
                {"radar": "fikor", "nominal_time": self.NOMINAL_TIME},
            ],
        )
        assert tags["radars"] == "fikor,fivim"
        assert tags["n_radars"] == "2"
        assert tags["nominal_time"] == self.NOMINAL_TIME
        assert tags["composite_method"] == "min"

    def test_missing_radar_tag_excluded(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        tags = self._composite_tags(
            tmp_path,
            [
                {"radar": "fivim", "nominal_time": self.NOMINAL_TIME},
                {"nominal_time": self.NOMINAL_TIME},
            ],
        )
        assert tags["radars"] == "fivim"
        assert tags["n_radars"] == "2"
        assert "No radar tag" in caplog.text

    def test_missing_nominal_time_omitted(self, tmp_path: Path) -> None:
        tags = self._composite_tags(tmp_path, [{"radar": "fivim"}])
        assert tags["radars"] == "fivim"
        assert "nominal_time" not in tags

    def test_differing_nominal_times_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="differing nominal times"):
            self._composite_tags(
                tmp_path,
                [
                    {"radar": "fivim", "nominal_time": self.NOMINAL_TIME},
                    {"radar": "fikor", "nominal_time": "2026-03-19T11:15:00+00:00"},
                ],
            )
