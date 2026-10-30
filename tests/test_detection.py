# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for detection range computation (detectability.detection)."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from wradlib.georef import bin_altitude

from detectability.beam import slant_range_from_height
from detectability.detection import NODATA, compute_detection_ranges

# Vimpeli radar parameters
SITEALT = 200.0
BEAMWIDTH = 0.97  # degrees
LOWEST_ELEV = 0.3  # degrees


class TestComputeDetectionRanges:
    """Tests for compute_detection_ranges."""

    def test_output_shape(self) -> None:
        top = np.full(360, 5000.0)  # 5 km uniform TOP
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        assert ds["detectability"].shape == (360, 500)
        assert ds["range_full"].shape == (360,)
        assert ds["range_zero"].shape == (360,)

    def test_output_dtype(self) -> None:
        top = np.full(360, 5000.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        assert ds["detectability"].dtype == np.uint8

    def test_value_range(self) -> None:
        top = np.full(360, 5000.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        grid = ds["detectability"].values
        assert grid.min() >= 0
        assert grid.max() <= 100

    def test_close_range_is_zero(self) -> None:
        """Bins well within detection range should be 0 (full filling)."""
        top = np.full(360, 5000.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        # First bin center is at 250m — should be 0 at all azimuths
        assert_array_equal(ds["detectability"].values[:, 0], 0)

    def test_far_range_is_100(self) -> None:
        """Bins well beyond detection range should be 100 (total overshooting)."""
        top = np.full(360, 3000.0)  # low TOP → short detection range
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        # Last bin (~250 km) should definitely be 100
        assert_array_equal(ds["detectability"].values[:, -1], 100)

    def test_monotonic_increase_along_range(self) -> None:
        """Detectability values should be non-decreasing along range."""
        top = np.full(360, 5000.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        grid = ds["detectability"].values.astype(np.int16)
        diffs = np.diff(grid, axis=1)
        assert np.all(diffs >= 0)

    def test_higher_top_extends_detection(self) -> None:
        """Higher echo-top should push detection range farther out."""
        top_low = np.full(360, 3000.0)
        top_high = np.full(360, 8000.0)
        ds_low = compute_detection_ranges(
            top_low,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        ds_high = compute_detection_ranges(
            top_high,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        assert ds_high["range_full"].values.mean() > ds_low["range_full"].values.mean()
        assert ds_high["range_zero"].values.mean() > ds_low["range_zero"].values.mean()

    def test_range_full_less_than_range_zero(self) -> None:
        """Beam top hits TOP before beam bottom does."""
        top = np.full(360, 5000.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        assert np.all(ds["range_full"].values <= ds["range_zero"].values)

    def test_range_consistency_with_beam(self) -> None:
        """range_full and range_zero should agree with slant_range_from_height."""
        top_m = 5000.0
        top = np.full(360, top_m)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        expected_full = slant_range_from_height(
            top_m, LOWEST_ELEV + BEAMWIDTH / 2, SITEALT
        )
        expected_zero = slant_range_from_height(
            top_m, LOWEST_ELEV - BEAMWIDTH / 2, SITEALT
        )
        assert ds["range_full"].values[0] == pytest.approx(float(expected_full), rel=1e-6)
        assert ds["range_zero"].values[0] == pytest.approx(float(expected_zero), rel=1e-6)

    def test_uniform_across_azimuths_for_uniform_top(self) -> None:
        """Uniform TOP should produce identical detectability for all azimuths."""
        top = np.full(360, 5000.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        grid = ds["detectability"].values
        # All rows should be identical
        assert_array_equal(grid[0], grid[180])

    def test_variable_top_per_azimuth(self) -> None:
        """Different TOPs per azimuth should produce different detection ranges."""
        top = np.linspace(3000, 8000, 360)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        # Higher TOP (last azimuths) should have larger range_zero
        assert ds["range_zero"].values[-1] > ds["range_zero"].values[0]

    def test_custom_resolution_and_nbins(self) -> None:
        """Custom range_resolution and nbins should be reflected in output."""
        top = np.full(10, 5000.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
            range_resolution=1000.0,
            nbins=250,
        )
        assert ds["detectability"].shape == (10, 250)
        assert ds["range"].values[0] == pytest.approx(500.0)  # first bin centre
        assert ds["range"].values[-1] == pytest.approx(249_500.0)

    def test_attributes_stored(self) -> None:
        top = np.full(360, 5000.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        assert ds.attrs["lowest_elevation_deg"] == LOWEST_ELEV
        assert ds.attrs["beamwidth_deg"] == BEAMWIDTH
        assert ds.attrs["sitealt_m"] == SITEALT

    def test_top_below_site_gives_all_100(self) -> None:
        """TOP below the radar site means total overshooting everywhere."""
        top = np.full(360, 0.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        assert_array_equal(ds["detectability"].values, 100)

    def test_nan_top_gives_nodata(self) -> None:
        top = np.array([5000.0, np.nan])
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        grid = ds["detectability"].values
        assert_array_equal(grid[1], NODATA)
        assert (grid[0] <= 100).all()

    def test_geometric_overshoot_fraction(self) -> None:
        """Values equal the fraction of the beam above TOP, in percent."""
        top_m = 5000.0
        ds = compute_detection_ranges(
            np.full(1, top_m),
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        r = ds["range"].values
        h_top = bin_altitude(r, LOWEST_ELEV + BEAMWIDTH / 2, SITEALT)
        h_bot = bin_altitude(r, LOWEST_ELEV - BEAMWIDTH / 2, SITEALT)
        expected = np.clip((h_top - top_m) / (h_top - h_bot), 0, 1) * 100
        np.testing.assert_allclose(
            ds["detectability"].values[0], expected, atol=0.5
        )

    def test_endpoints_match_detection_ranges(self) -> None:
        """0 % up to range_full, 100 % beyond range_zero."""
        ds = compute_detection_ranges(
            np.full(1, 3000.0),
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        r = ds["range"].values
        grid = ds["detectability"].values[0]
        assert (grid[r < ds["range_full"].values[0]] == 0).all()
        assert (grid[r > ds["range_zero"].values[0]] == 100).all()

    def test_not_below_legacy_linear_ramp(self) -> None:
        """Geometric fraction is ≥ the legacy linear-in-range ramp.

        Legacy: k = 255/(lowbeam - highbeam);
                outbyte = (B/2.0 - highbeam) * k
        """
        ds = compute_detection_ranges(
            np.full(1, 5000.0),
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        r = ds["range"].values
        rf = ds["range_full"].values[0]
        rz = ds["range_zero"].values[0]
        linear = np.clip((r - rf) / (rz - rf), 0, 1) * 100
        grid = ds["detectability"].values[0]
        assert (grid >= np.floor(linear)).all()
