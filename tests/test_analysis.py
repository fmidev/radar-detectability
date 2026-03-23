# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for analysis module (detectability.analysis)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from detectability.analysis import pick_ray_tops, sector_smooth


class TestPickRayTops:
    """Tests for pick_ray_tops."""

    def test_output_shapes(self) -> None:
        top = np.random.default_rng(42).uniform(0, 5000, (360, 500))
        ray_top, ray_weight = pick_ray_tops(top)
        assert ray_top.shape == (360,)
        assert ray_weight.shape == (360,)

    def test_weight_range(self) -> None:
        top = np.random.default_rng(42).uniform(0, 5000, (360, 500))
        _, ray_weight = pick_ray_tops(top)
        assert ray_weight.min() >= 0.0
        assert ray_weight.max() <= 1.0

    def test_all_zero_gives_zero(self) -> None:
        top = np.zeros((10, 50))
        ray_top, ray_weight = pick_ray_tops(top)
        assert_allclose(ray_top, 0.0)
        assert_allclose(ray_weight, 0.0)

    def test_uniform_row(self) -> None:
        """All bins at same height → TOP = that height, weight = 1."""
        top = np.full((4, 100), 3000.0)
        ray_top, ray_weight = pick_ray_tops(top, highpart=0.1)
        assert_allclose(ray_top, 3000.0)
        assert_allclose(ray_weight, 1.0)

    def test_samplepoint_0_picks_highest(self) -> None:
        """samplepoint=0 should select the highest value."""
        top = np.zeros((1, 100))
        top[0, :10] = np.arange(1, 11) * 100  # 100..1000
        ray_top, _ = pick_ray_tops(top, highpart=0.2, samplepoint=0.0)
        assert ray_top[0] == 1000.0

    def test_higher_highpart_includes_more(self) -> None:
        """Larger highpart → more bins considered → lower pick possible."""
        top = np.zeros((1, 100))
        top[0, :50] = np.linspace(5000, 100, 50)
        top_narrow, _ = pick_ray_tops(top, highpart=0.05, samplepoint=0.5)
        top_wide, _ = pick_ray_tops(top, highpart=0.5, samplepoint=0.5)
        assert top_narrow[0] >= top_wide[0]

    def test_sparse_data_low_weight(self) -> None:
        """Few valid bins → low weight."""
        top = np.zeros((1, 1000))
        top[0, :5] = 5000.0  # only 5 valid bins
        _, ray_weight = pick_ray_tops(top, highpart=0.1)
        assert ray_weight[0] == pytest.approx(5.0 / 100.0)


class TestSectorSmooth:
    """Tests for sector_smooth."""

    def test_output_shape(self) -> None:
        ray_top = np.full(360, 5000.0)
        ray_weight = np.ones(360)
        result = sector_smooth(ray_top, ray_weight, 5500.0)
        assert result.shape == (360,)

    def test_uniform_input_unchanged(self) -> None:
        """Uniform input should remain uniform after smoothing."""
        ray_top = np.full(360, 3000.0)
        ray_weight = np.ones(360)
        result = sector_smooth(ray_top, ray_weight, 5500.0)
        assert_allclose(result, 3000.0, atol=0.1)

    def test_zero_weight_uses_background(self) -> None:
        """With zero weight everywhere, result equals background."""
        ray_top = np.full(360, 8000.0)
        ray_weight = np.zeros(360)
        result = sector_smooth(ray_top, ray_weight, 4000.0)
        assert_allclose(result, 4000.0, atol=0.1)

    def test_smooths_spike(self) -> None:
        """A single-ray spike should be attenuated."""
        ray_top = np.full(360, 5000.0)
        ray_top[180] = 10000.0
        ray_weight = np.ones(360)
        result = sector_smooth(ray_top, ray_weight, 5000.0, sector_width=60)
        # Spike should be reduced
        assert result[180] < 10000.0
        # But still above uniform background
        assert result[180] > 5000.0

    def test_circular_wrap(self) -> None:
        """Smoothing should wrap around azimuth 0/359 boundary."""
        ray_top = np.full(360, 5000.0)
        ray_top[0] = 10000.0
        ray_weight = np.ones(360)
        result = sector_smooth(ray_top, ray_weight, 5000.0, sector_width=60)
        # Neighbors near 0/360 boundary should be affected
        assert result[359] > 5000.0
        assert result[1] > 5000.0

    def test_sector_width_1_no_smoothing(self) -> None:
        """Sector width of 1 should be a no-op (only centre ray)."""
        ray_top = np.arange(360, dtype=np.float64) * 10
        ray_weight = np.ones(360)
        result = sector_smooth(ray_top, ray_weight, 0.0, sector_width=1)
        assert_allclose(result, ray_top, atol=0.01)
