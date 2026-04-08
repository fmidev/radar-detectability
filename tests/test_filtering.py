# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for azimuthal spike filter (detectability.filtering)."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from detectability.filtering import azimuthal_filter


def _uniform(value: float, nrays: int = 8, nbins: int = 4) -> np.ndarray:
    return np.full((nrays, nbins), value)


class TestAzimuthalFilter:
    def test_returns_ndarray(self) -> None:
        assert isinstance(azimuthal_filter(_uniform(1000.0)), np.ndarray)

    def test_dtype_float64(self) -> None:
        assert azimuthal_filter(_uniform(1000.0)).dtype == np.float64

    def test_shape_preserved(self) -> None:
        arr = np.ones((12, 6))
        assert azimuthal_filter(arr).shape == (12, 6)

    def test_uniform_field_unchanged(self) -> None:
        """Uniform non-zero field: no bin exceeds sum of neighbours."""
        arr = _uniform(5000.0)
        out = azimuthal_filter(arr)
        assert_array_equal(out, arr)

    def test_all_zero_unchanged(self) -> None:
        arr = _uniform(0.0)
        assert_array_equal(azimuthal_filter(arr), arr)

    def test_undetect_negatives_zeroed(self) -> None:
        """Values ≤ 0 (xradar undetect) are treated as no-echo and zeroed."""
        arr = np.full((4, 4), -0.305)
        out = azimuthal_filter(arr)
        assert (out == 0).all()

    def test_isolated_spike_removed(self) -> None:
        """A single non-zero ray surrounded by all-zero rays is zeroed."""
        arr = np.zeros((8, 4))
        arr[3, :] = 5000.0  # azimuth 3 is the only non-zero ray
        out = azimuthal_filter(arr)
        # Both neighbours (rays 2 and 4) are zero → spike removed
        assert (out[3, :] == 0).all()

    def test_non_isolated_spike_kept(self) -> None:
        """A non-zero ray with at least one non-zero neighbour is kept."""
        arr = np.zeros((8, 4))
        arr[3, :] = 5000.0
        arr[4, :] = 5000.0  # ray 4 is the next neighbour of ray 3
        out = azimuthal_filter(arr)
        # Ray 3: prev=0, next=5000 → not both-zero; 5000 <= 0+5000 → kept
        assert (out[3, :] == 5000.0).all()

    def test_exceeds_sum_removed(self) -> None:
        """Bin value greater than sum of neighbours is zeroed."""
        arr = np.zeros((8, 4))
        arr[3, :] = 1000.0
        arr[2, :] = 300.0
        arr[4, :] = 300.0
        # 1000 > 300 + 300 = 600 → should be zeroed
        out = azimuthal_filter(arr)
        assert (out[3, :] == 0).all()

    def test_exactly_equal_to_sum_kept(self) -> None:
        """Bin value equal to sum of neighbours is kept (legacy: strict >)."""
        arr = np.zeros((8, 4))
        arr[3, :] = 600.0
        arr[2, :] = 300.0
        arr[4, :] = 300.0
        # 600 == 300 + 300 → not strictly greater → kept
        out = azimuthal_filter(arr)
        assert (out[3, :] == 600.0).all()

    def test_less_than_sum_kept(self) -> None:
        arr = np.zeros((8, 4))
        arr[3, :] = 500.0
        arr[2, :] = 300.0
        arr[4, :] = 300.0
        # 500 < 600 → kept
        out = azimuthal_filter(arr)
        assert (out[3, :] == 500.0).all()

    def test_circular_wrap_first_ray(self) -> None:
        """Ray 0's previous neighbour wraps to the last ray (N-1)."""
        arr = np.zeros((8, 4))
        arr[0, :] = 5000.0  # ray 0 isolated: neighbours are ray 7 and ray 1
        out = azimuthal_filter(arr)
        # Ray 7 = 0, ray 1 = 0 → both neighbours zero → removed
        assert (out[0, :] == 0).all()

    def test_circular_wrap_last_ray(self) -> None:
        """Last ray's next neighbour wraps to ray 0."""
        nrays = 8
        arr = np.zeros((nrays, 4))
        arr[nrays - 1, :] = 5000.0
        out = azimuthal_filter(arr)
        assert (out[nrays - 1, :] == 0).all()

    def test_circular_wrap_preserves_non_isolated(self) -> None:
        """Ray 0 kept when its wrap-around neighbour (last ray) is non-zero."""
        nrays = 8
        arr = np.zeros((nrays, 4))
        arr[0, :] = 3000.0
        arr[nrays - 1, :] = 3000.0  # previous neighbour (wrap)
        out = azimuthal_filter(arr)
        # prev=3000, next=0 → not both-zero; 3000 <= 3000+0 → kept
        assert (out[0, :] == 3000.0).all()

    def test_per_bin_independence(self) -> None:
        """Spike at one range bin does not affect other bins in same ray."""
        arr = np.zeros((8, 6))
        arr[3, 2] = 5000.0   # only this bin is non-zero in its ray
        out = azimuthal_filter(arr)
        assert out[3, 2] == 0.0       # isolated spike removed
        assert (out[:, 3] == 0).all() # adjacent bins unaffected

    def test_realistic_360_ray_field(self) -> None:
        """Smoke test on a 360×500 field similar to the real echotop product."""
        rng = np.random.default_rng(42)
        arr = rng.uniform(0, 8000, (360, 500))
        out = azimuthal_filter(arr)
        assert out.shape == (360, 500)
        assert out.dtype == np.float64
        assert (out >= 0).all()
