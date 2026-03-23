# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for georeferencing and COG output (detectability.georef)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

from detectability.detection import compute_detection_ranges
from detectability.georef import polar_to_projected, write_cog

# Vimpeli radar parameters
RADAR_LAT = 63.104840
RADAR_LON = 23.820860
RADAR_ALT = 200.0
BEAMWIDTH = 0.97
LOWEST_ELEV = 0.3
SITEALT = RADAR_ALT


def _make_detection_ds(
    top_m: float = 5000.0, nrays: int = 360, nbins: int = 500
) -> xr.Dataset:
    """Create a detection range dataset with uniform TOP."""
    top = np.full(nrays, top_m)
    return compute_detection_ranges(
        top,
        lowest_elevation=LOWEST_ELEV,
        beamwidth=BEAMWIDTH,
        sitealt=SITEALT,
        nbins=nbins,
    )


class TestPolarToProjected:
    """Tests for polar_to_projected."""

    def test_output_is_dataarray(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        assert isinstance(da, xr.DataArray)

    def test_output_dtype_uint8(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        assert da.dtype == np.uint8

    def test_value_range_0_255(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        assert da.values.min() >= 0
        assert da.values.max() <= 255

    def test_has_crs(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        assert da.rio.crs is not None
        assert da.rio.crs.to_epsg() == 3067

    def test_custom_crs(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
            crs="EPSG:3035",
        )
        assert da.rio.crs.to_epsg() == 3035

    def test_dims_y_x(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        assert da.dims == ("y", "x")

    def test_y_descending(self) -> None:
        """Y coordinates should be descending (north-up raster)."""
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        y = da.coords["y"].values
        assert np.all(y[:-1] >= y[1:])

    def test_grid_resolution(self) -> None:
        ds = _make_detection_ds(nbins=50)
        resolution = 1000.0
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
            resolution_m=resolution,
        )
        dx = np.diff(da.coords["x"].values)
        dy = np.abs(np.diff(da.coords["y"].values))
        np.testing.assert_allclose(dx, resolution, rtol=1e-10)
        np.testing.assert_allclose(dy, resolution, rtol=1e-10)

    def test_grid_covers_polar_extent(self) -> None:
        """Projected grid should cover full polar extent."""
        nbins = 50
        rng_max = (nbins - 0.5) * 500  # max bin centre
        ds = _make_detection_ds(nbins=nbins)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        x_span = da.coords["x"].values.max() - da.coords["x"].values.min()
        y_span = da.coords["y"].values.max() - da.coords["y"].values.min()
        # Each span should be at least as big as the polar diameter
        # (2 * max range), minus some grid-edge tolerance
        assert x_span >= rng_max * 1.5
        assert y_span >= rng_max * 1.5

    def test_attrs_preserved(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        assert da.attrs["beamwidth_deg"] == BEAMWIDTH

    def test_uniform_top_symmetry(self) -> None:
        """With uniform TOP, the pattern should be roughly symmetric."""
        # Low TOP so detection transition occurs within the grid extent
        ds = _make_detection_ds(top_m=500.0, nbins=100)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        vals = da.values.astype(np.float64)
        ny, nx = vals.shape
        cy, cx = ny // 2, nx // 2
        # Quadrant sums should be roughly similar for uniform TOP
        q1 = vals[:cy, :cx].sum()
        q2 = vals[:cy, cx:].sum()
        q3 = vals[cy:, :cx].sum()
        q4 = vals[cy:, cx:].sum()
        total = q1 + q2 + q3 + q4
        assert total > 0, "Total sum should be nonzero"
        for q in [q1, q2, q3, q4]:
            assert q / total > 0.15, "Quadrant sums too unbalanced"


class TestWriteCog:
    """Tests for write_cog."""

    def test_writes_file(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tif"
            write_cog(da, path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_cog_has_correct_crs(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tif"
            write_cog(da, path)
            with rasterio.open(path) as src:
                assert src.crs.to_epsg() == 3067

    def test_cog_shape_matches(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tif"
            write_cog(da, path)
            with rasterio.open(path) as src:
                assert src.width == da.sizes["x"]
                assert src.height == da.sizes["y"]

    def test_cog_creates_parent_dirs(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "output.tif"
            write_cog(da, path)
            assert path.exists()

    def test_cog_dtype_uint8(self) -> None:
        ds = _make_detection_ds(nbins=50)
        da = polar_to_projected(
            ds,
            radar_lon=RADAR_LON,
            radar_lat=RADAR_LAT,
            radar_alt=RADAR_ALT,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tif"
            write_cog(da, path)
            with rasterio.open(path) as src:
                assert src.dtypes[0] == "uint8"
