# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for azimuthal filtering (detectability.filtering)."""

import numpy as np
from numpy.testing import assert_array_equal

from detectability.filtering import azimuthal_filter


class TestAzimuthalFilter:
    """Tests for azimuthal_filter."""

    def test_all_zeros_unchanged(self) -> None:
        data = np.zeros((10, 5), dtype=np.uint8)
        assert_array_equal(azimuthal_filter(data), data)

    def test_uniform_nonzero_unchanged(self) -> None:
        """Uniform field: no bin exceeds sum of neighbors."""
        data = np.full((10, 5), 100, dtype=np.uint8)
        assert_array_equal(azimuthal_filter(data), data)

    def test_single_ray_spike_removed(self) -> None:
        """A single ray with high values between zero neighbors is zeroed."""
        data = np.zeros((10, 5), dtype=np.uint8)
        data[3, :] = 200
        result = azimuthal_filter(data)
        assert_array_equal(result[3, :], 0)

    def test_supported_ray_kept(self) -> None:
        """A ray supported by at least one equal neighbor stays."""
        data = np.zeros((10, 5), dtype=np.uint8)
        data[3, :] = 100
        data[4, :] = 100
        result = azimuthal_filter(data)
        # data[4] has neighbor data[3]=100 and data[5]=0, sum=100 ≥ 100
        assert_array_equal(result[4, :], 100)

    def test_exceeds_sum_zeroed(self) -> None:
        """Centre > sum of neighbors → zeroed."""
        data = np.zeros((10, 5), dtype=np.uint8)
        data[4, :] = 50
        data[5, :] = 200  # 200 > 50+0
        data[6, :] = 0
        result = azimuthal_filter(data)
        assert_array_equal(result[5, :], 0)

    def test_circular_wrap(self) -> None:
        """Filter wraps: ray 0 sees ray 359 and ray 1."""
        data = np.zeros((360, 3), dtype=np.uint8)
        data[0, :] = 200  # neighbors: ray 359=0, ray 1=0
        result = azimuthal_filter(data)
        assert_array_equal(result[0, :], 0)

    def test_preserves_dtype(self) -> None:
        data = np.full((10, 5), 50, dtype=np.uint8)
        result = azimuthal_filter(data)
        assert result.dtype == np.uint8

    def test_preserves_shape(self) -> None:
        data = np.full((360, 500), 100, dtype=np.uint8)
        result = azimuthal_filter(data)
        assert result.shape == (360, 500)

    def test_gradual_field_mostly_preserved(self) -> None:
        """Gradually varying field should be mostly unchanged."""
        nrays = 360
        # Create smooth azimuthal gradient
        az = np.linspace(0, 255, nrays, dtype=np.uint8)
        data = np.broadcast_to(az[:, np.newaxis], (nrays, 50)).copy()
        result = azimuthal_filter(data)
        changed = np.count_nonzero(result != data)
        # Very few pixels should change in a smooth field
        assert changed < nrays * 50 * 0.05
