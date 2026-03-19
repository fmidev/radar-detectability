# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for radar beam geometry (detectability.beam)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from detectability.beam import beam_height, ground_distance, slant_range_from_height

# Vimpeli radar site altitude
SITEALT = 200.0


class TestBeamHeight:
    """Tests for beam_height (wradlib.georef.bin_altitude wrapper)."""

    def test_zero_range_returns_sitealt(self) -> None:
        assert beam_height(0.0, 0.3, SITEALT) == pytest.approx(SITEALT)

    def test_known_value_low_elevation(self) -> None:
        # 100 km slant range, 0.3° elevation, 200 m site
        h = beam_height(100_000.0, 0.3, SITEALT)
        # Expected ~1312 m (verified against legacy C and wradlib)
        assert h == pytest.approx(1312, abs=5)

    def test_higher_elevation_gives_higher_beam(self) -> None:
        h_low = beam_height(100_000.0, 0.3, SITEALT)
        h_high = beam_height(100_000.0, 5.0, SITEALT)
        assert h_high > h_low

    def test_vectorized(self) -> None:
        ranges = np.array([50_000.0, 100_000.0, 200_000.0])
        h = beam_height(ranges, 0.3, SITEALT)
        assert h.shape == (3,)
        # Heights should increase with range
        assert np.all(np.diff(h) > 0)

    def test_broadcasting(self) -> None:
        ranges = np.array([50_000.0, 100_000.0, 150_000.0])
        elevations = np.array([0.3, 1.5, 5.0])
        h = beam_height(ranges[:, None], elevations[None, :], SITEALT)
        assert h.shape == (3, 3)


class TestGroundDistance:
    """Tests for ground_distance (wradlib.georef.bin_distance wrapper)."""

    def test_zero_range(self) -> None:
        d = ground_distance(0.0, 0.3, SITEALT)
        assert d == pytest.approx(0.0, abs=1)

    def test_ground_distance_less_than_slant_range(self) -> None:
        r = 100_000.0
        d = ground_distance(r, 5.0, SITEALT)
        assert d < r


class TestSlantRangeFromHeight:
    """Tests for the analytical inverse: height → slant range."""

    def test_roundtrip_single(self) -> None:
        """beam_height → slant_range_from_height recovers original range."""
        r_orig = 100_000.0
        elev = 0.3
        h = beam_height(r_orig, elev, SITEALT)
        r_back = slant_range_from_height(h, elev, SITEALT)
        assert r_back == pytest.approx(r_orig, abs=3)

    @pytest.mark.parametrize("r_km", [50, 100, 150, 200, 250])
    @pytest.mark.parametrize("elev", [0.3, 0.7, 1.5, 5.0, 9.0])
    def test_roundtrip_parametrized(self, r_km: int, elev: float) -> None:
        r_m = r_km * 1000.0
        h = beam_height(r_m, elev, SITEALT)
        r_back = slant_range_from_height(h, elev, SITEALT)
        assert r_back == pytest.approx(r_m, abs=3)

    def test_vectorized(self) -> None:
        heights = np.array([1000.0, 3000.0, 5000.0])
        r = slant_range_from_height(heights, 0.3, SITEALT)
        assert r.shape == (3,)
        # Farther range for higher target
        assert np.all(np.diff(r) > 0)

    def test_unreachable_height_returns_zero(self) -> None:
        # Height below site altitude is unreachable at positive elevation
        r = slant_range_from_height(0.0, 5.0, SITEALT)
        assert r == pytest.approx(0.0)

    def test_broadcasting(self) -> None:
        heights = np.array([1000.0, 3000.0, 5000.0])
        elevations = np.array([0.3, 1.5])
        r = slant_range_from_height(heights[:, None], elevations[None, :], SITEALT)
        assert r.shape == (3, 2)

    def test_matches_legacy_formula(self) -> None:
        """Cross-check against the legacy C bindist() formula directly."""
        # Legacy: r = cos(e)*ER*(sqrt(sin(e)^2 + 0.002*(h-h0)/ER) - sin(e))
        # where ER=8495.0506667 (km), h and h0 in m, e in radians, result in km
        ER_km = 8495.0506667
        h, h0, e_deg = 5000.0, SITEALT, 0.3
        e_rad = np.radians(e_deg)
        r_legacy_km = (
            np.cos(e_rad) * ER_km
            * (np.sqrt(np.sin(e_rad) ** 2 + 0.002 * (h - h0) / ER_km) - np.sin(e_rad))
        )
        r_ours = slant_range_from_height(h, e_deg, h0) / 1000  # m → km
        # The two models differ slightly (different ER, slant vs ground range)
        # but should agree to within ~1 km for practical ranges
        assert r_ours == pytest.approx(r_legacy_km, rel=0.01)
