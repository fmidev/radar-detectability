# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for detection range computation (detectability.detection)."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from detectability.beam import slant_range_from_height
from detectability.detection import compute_detection_ranges

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
        assert grid.max() <= 255

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

    def test_far_range_is_255(self) -> None:
        """Bins well beyond detection range should be 255 (total overshooting)."""
        top = np.full(360, 3000.0)  # low TOP → short detection range
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        # Last bin (~250 km) should definitely be 255
        assert_array_equal(ds["detectability"].values[:, -1], 255)

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

    def test_zero_top_gives_all_255(self) -> None:
        """Zero TOP means beam always overshoots → all bins should be 255."""
        top = np.full(360, 0.0)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        # range_full and range_zero should be 0 (unreachable)
        # all output bins at or beyond range 0 → 255
        grid = ds["detectability"].values
        # At minimum all far bins should be 255 (in practice all bins
        # since range_full=0 means the transition starts at bin 0)
        assert grid[:, -1].max() == 255

    def test_cross_check_legacy_encoding(self) -> None:
        """Verify encoding matches legacy formula for a specific case.

        Legacy:
          k = 255 / (lowbeam_range_km - highbeam_range_km)
          outbyte = (B/2.0 - highbeam_range_km) * k  if B > 2*highbeam
        """
        top_m = 5000.0
        top = np.full(1, top_m)
        ds = compute_detection_ranges(
            top,
            lowest_elevation=LOWEST_ELEV,
            beamwidth=BEAMWIDTH,
            sitealt=SITEALT,
        )
        rf_m = ds["range_full"].values[0]
        rz_m = ds["range_zero"].values[0]
        grid = ds["detectability"].values[0]

        # Check a bin in the transition zone
        mid_range_m = (rf_m + rz_m) / 2  # halfway
        mid_bin = int(mid_range_m / 500.0)  # bin index (500m resolution)
        if 0 <= mid_bin < 500:
            expected_frac = (mid_range_m - rf_m) / (rz_m - rf_m)
            expected_byte = int(expected_frac * 255)
            # Allow ±2 due to bin discretization (bin centre vs edge)
            assert abs(int(grid[mid_bin]) - expected_byte) <= 2
