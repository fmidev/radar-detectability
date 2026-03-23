# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for radar beam geometry (detectability.beam)."""

import numpy as np
import pytest
from wradlib.georef import bin_altitude

from detectability.beam import slant_range_from_height

# Vimpeli radar site altitude
SITEALT = 200.0


class TestSlantRangeFromHeight:
    """Tests for the analytical inverse: height → slant range."""

    def test_roundtrip_single(self) -> None:
        """bin_altitude → slant_range_from_height recovers original range."""
        r_orig = 100_000.0
        elev = 0.3
        h = bin_altitude(r_orig, elev, SITEALT)
        r_back = slant_range_from_height(h, elev, SITEALT)
        assert r_back == pytest.approx(r_orig, abs=3)

    @pytest.mark.parametrize("r_km", [50, 100, 150, 200, 250])
    @pytest.mark.parametrize("elev", [0.3, 0.7, 1.5, 5.0, 9.0])
    def test_roundtrip_parametrized(self, r_km: int, elev: float) -> None:
        r_m = r_km * 1000.0
        h = bin_altitude(r_m, elev, SITEALT)
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
