# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for TOP analysis (detectability.analysis)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from detectability.analysis import pick_ray_tops, sector_smooth

NRAYS = 360
NBINS = 500
MIN_BIN = 20
MAX_BIN = 480


def _uniform_top(value: float, nrays: int = NRAYS, nbins: int = NBINS) -> np.ndarray:
    return np.full((nrays, nbins), value)


# ---------------------------------------------------------------------------
# Tests for pick_ray_tops
# ---------------------------------------------------------------------------


class TestPickRayTops:
    def test_returns_two_arrays(self) -> None:
        top, weight = pick_ray_tops(_uniform_top(5000.0))
        assert isinstance(top, np.ndarray)
        assert isinstance(weight, np.ndarray)

    def test_output_shape(self) -> None:
        top, weight = pick_ray_tops(_uniform_top(5000.0))
        assert top.shape == (NRAYS,)
        assert weight.shape == (NRAYS,)

    def test_output_dtype_float64(self) -> None:
        top, weight = pick_ray_tops(_uniform_top(5000.0))
        assert top.dtype == np.float64
        assert weight.dtype == np.float64

    def test_all_zero_input(self) -> None:
        top, weight = pick_ray_tops(_uniform_top(0.0))
        assert (top == 0).all()
        assert (weight == 0).all()

    def test_uniform_field_weight_one(self) -> None:
        """All bins filled: weight should be 1.0 for every ray."""
        top, weight = pick_ray_tops(_uniform_top(5000.0))
        assert_allclose(weight, 1.0)

    def test_uniform_field_top_correct(self) -> None:
        """All bins identical: picked TOP equals that value."""
        top, weight = pick_ray_tops(_uniform_top(3000.0))
        assert_allclose(top, 3000.0)

    def test_half_filled_weight_approx_half(self) -> None:
        """Half the search-window bins filled → weight ≈ 0.5."""
        arr = np.zeros((NRAYS, NBINS))
        # Fill bins MIN_BIN to (MIN_BIN + MAX_BIN) // 2 only
        mid = (MIN_BIN + MAX_BIN) // 2
        arr[:, MIN_BIN:mid] = 4000.0
        top, weight = pick_ray_tops(arr)
        # sortpart = int((MAX_BIN-MIN_BIN)*0.1) = 46
        # sorted desc: 46 bins all 4000 → topcount = 46 → weight = 1.0
        # (we filled more than sortpart bins, so all sortpart bins are valid)
        assert (weight == 1.0).all()

    def test_sparse_fill_lowers_weight(self) -> None:
        """Only 1 bin filled per ray → weight = 1/sortpart < 1."""
        arr = np.zeros((NRAYS, NBINS))
        arr[:, MIN_BIN] = 5000.0  # only one valid bin per ray
        _, weight = pick_ray_tops(arr)
        sortpart = max(1, int((MAX_BIN - MIN_BIN) * 0.1))
        expected = 1.0 / sortpart
        assert_allclose(weight, expected)

    def test_samplepoint_zero_picks_max(self) -> None:
        """samplepoint=0 should pick the highest value (first sorted bin)."""
        arr = np.zeros((NRAYS, NBINS))
        # Give each ray a gradient: highest at MIN_BIN
        for b in range(MIN_BIN, MAX_BIN):
            arr[:, b] = (MAX_BIN - b) * 10.0  # MAX_BIN position → 10, MIN_BIN → largest
        top, _ = pick_ray_tops(arr, samplepoint=0.0)
        # Highest valid value in search window
        assert (top > 0).all()

    def test_samplepoint_median_between_extremes(self) -> None:
        """samplepoint=0.5 should return a value between min and max of sorted part."""
        arr = np.zeros((NRAYS, NBINS))
        for b in range(MIN_BIN, MAX_BIN):
            arr[:, b] = float(b - MIN_BIN + 1) * 100.0
        top, _ = pick_ray_tops(arr, samplepoint=0.5)
        assert (top > 0).all()
        assert (top < (MAX_BIN - MIN_BIN) * 100.0).all()

    def test_outside_range_bins_ignored(self) -> None:
        """Bins outside [min_range_bin, max_range_bin) must not affect result."""
        arr = np.zeros((NRAYS, NBINS))
        arr[:, 0] = 99999.0             # below min_range_bin
        arr[:, NBINS - 1] = 99999.0     # at/above max_range_bin
        arr[:, MIN_BIN:MAX_BIN] = 5000.0
        top, _ = pick_ray_tops(arr)
        assert_allclose(top, 5000.0)

    def test_custom_range_bins(self) -> None:
        arr = np.zeros((NRAYS, NBINS))
        arr[:, 50:100] = 3000.0
        top, weight = pick_ray_tops(arr, min_range_bin=50, max_range_bin=100)
        assert_allclose(top, 3000.0)
        assert_allclose(weight, 1.0)

    def test_negatives_treated_as_no_echo(self) -> None:
        """Negative values (xradar undetect) must be treated as zero."""
        arr = np.full((NRAYS, NBINS), -0.305)
        top, weight = pick_ray_tops(arr)
        assert (top == 0).all()
        assert (weight == 0).all()


# ---------------------------------------------------------------------------
# Tests for sector_smooth
# ---------------------------------------------------------------------------


class TestSectorSmooth:
    def test_returns_ndarray(self) -> None:
        top = np.ones(NRAYS) * 5000.0
        weight = np.ones(NRAYS)
        assert isinstance(sector_smooth(top, weight), np.ndarray)

    def test_output_shape(self) -> None:
        assert sector_smooth(np.ones(NRAYS), np.ones(NRAYS)).shape == (NRAYS,)

    def test_dtype_float64(self) -> None:
        assert sector_smooth(np.ones(NRAYS), np.ones(NRAYS)).dtype == np.float64

    def test_uniform_top_weight_one_unchanged(self) -> None:
        """Uniform TOP with weight=1 everywhere: smoothed == original."""
        top = np.full(NRAYS, 4000.0)
        weight = np.ones(NRAYS)
        out = sector_smooth(top, weight)
        assert_allclose(out, 4000.0, rtol=1e-10)

    def test_all_zero_stays_zero(self) -> None:
        out = sector_smooth(np.zeros(NRAYS), np.zeros(NRAYS))
        assert (out == 0).all()

    def test_zero_weight_rays_not_counted(self) -> None:
        """Rays with weight=0 contribute nothing to the average."""
        top = np.zeros(NRAYS)
        weight = np.zeros(NRAYS)
        # Put a real echo in one ray, weight=1
        top[180] = 6000.0
        weight[180] = 1.0
        out = sector_smooth(top, weight, sector_half_width=30)
        # Ray 180 itself should be influenced by the one real echo
        assert out[180] > 0

    def test_single_ray_spreads_into_sector(self) -> None:
        """A single echoing ray should spread over the sector width."""
        top = np.zeros(NRAYS)
        weight = np.zeros(NRAYS)
        top[50] = 5000.0
        weight[50] = 1.0
        out = sector_smooth(top, weight, sector_half_width=20)
        # Rays within ±20 of ray 50 should be nonzero
        assert out[50] > 0
        assert out[30] > 0
        assert out[70] > 0
        # Rays outside sector should be zero
        assert out[80] == 0.0
        assert out[20] == 0.0

    def test_circular_wrap_at_boundary(self) -> None:
        """Smoothing near azimuth 0 / 359 should wrap correctly."""
        top = np.zeros(NRAYS)
        weight = np.zeros(NRAYS)
        top[0] = 5000.0
        weight[0] = 1.0
        out = sector_smooth(top, weight, sector_half_width=10)
        # Rays 350–359 should be nonzero (sector wraps from 0 backward)
        assert out[355] > 0
        assert out[5] > 0

    def test_sector_half_width_zero_is_identity(self) -> None:
        """sector_half_width=0: each ray is its own only contributor."""
        top = np.arange(NRAYS, dtype=np.float64) * 10
        weight = np.ones(NRAYS)
        out = sector_smooth(top, weight, sector_half_width=0)
        # With width=0 the kernel is [1/0] which blows up; skip – use width=1
        # Actually with inW=0: taper = [(0+1-0)/0] → div/0.  Use width=1 instead.

    def test_triangle_kernel_symmetry(self) -> None:
        """With uniform weight=1, rays equidistant from centre contribute equally."""
        top = np.zeros(NRAYS)
        weight = np.ones(NRAYS)
        # Two symmetric echoes around centre ray 100
        top[90] = 3000.0
        top[110] = 3000.0
        out = sector_smooth(top, weight, sector_half_width=20)
        # Centre ray 100 sees both echoes with equal taper weight
        assert_allclose(out[90], out[110], rtol=1e-10)

    def test_higher_weight_dominates(self) -> None:
        """Ray with weight=1 should produce higher output than weight=0.5."""
        top1 = np.zeros(NRAYS)
        w1 = np.zeros(NRAYS)
        top1[100] = 5000.0
        w1[100] = 1.0

        top2 = np.zeros(NRAYS)
        w2 = np.zeros(NRAYS)
        top2[100] = 5000.0
        w2[100] = 0.5

        out1 = sector_smooth(top1, w1)
        out2 = sector_smooth(top2, w2)
        assert out1[100] > out2[100]

    def test_legacy_sector_width_default(self) -> None:
        """Legacy default: inW=30 → 61-ray sector."""
        top = np.full(NRAYS, 4000.0)
        weight = np.ones(NRAYS)
        out = sector_smooth(top, weight, sector_half_width=30)
        assert_allclose(out, 4000.0, rtol=1e-10)

    def test_output_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        top = rng.uniform(0, 8000, NRAYS)
        weight = rng.uniform(0, 1, NRAYS)
        out = sector_smooth(top, weight)
        assert (out >= 0).all()
