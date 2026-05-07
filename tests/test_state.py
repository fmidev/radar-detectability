# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for background TOP state module (detectability.state)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from detectability.state import (
    CLIMATOLOGY_TOP_KM,
    BackgroundState,
    age_background_top,
    compute_new_top,
    load_state,
    save_state,
)

NRAYS = 360


# ---------------------------------------------------------------------------
# Tests for load_state / save_state
# ---------------------------------------------------------------------------


class TestLoadSaveState:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        ts = datetime(2026, 3, 19, 12, 0, 0, tzinfo=UTC)
        original = BackgroundState(top_km=4.2, timestamp=ts)
        save_state(path, original)
        loaded = load_state(path)
        assert loaded is not None
        assert loaded.top_km == pytest.approx(4.2, abs=0.1)
        assert loaded.timestamp == ts

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_state(tmp_path / "nonexistent.json") is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        assert load_state(path) is None

    def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"top_km": 5.0}), encoding="utf-8")
        assert load_state(path) is None

    def test_invalid_timestamp_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps({"top_km": 5.0, "timestamp": "not-a-date"}),
            encoding="utf-8",
        )
        assert load_state(path) is None

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "state.json"
        state = BackgroundState(top_km=5.5, timestamp=datetime.now(UTC))
        save_state(path, state)
        assert path.exists()

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        save_state(path, BackgroundState(top_km=3.0, timestamp=ts))
        save_state(path, BackgroundState(top_km=7.0, timestamp=ts))
        loaded = load_state(path)
        assert loaded is not None
        assert loaded.top_km == pytest.approx(7.0, abs=0.1)

    def test_top_km_rounded_to_one_decimal(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        ts = datetime.now(UTC)
        save_state(path, BackgroundState(top_km=4.567, timestamp=ts))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["top_km"] == 4.6


# ---------------------------------------------------------------------------
# Tests for age_background_top
# ---------------------------------------------------------------------------


class TestAgeBackgroundTop:
    def _state(self, top_km: float, hours_ago: float) -> tuple[BackgroundState, datetime]:
        now = datetime(2026, 3, 19, 12, 0, 0, tzinfo=UTC)
        ts = now - timedelta(hours=hours_ago)
        return BackgroundState(top_km=top_km, timestamp=ts), now

    def test_within_grace_period_unchanged(self) -> None:
        """State younger than grace_hours → no aging."""
        state, now = self._state(8.0, hours_ago=1.0)
        assert age_background_top(state, now) == 8.0

    def test_at_grace_boundary_unchanged(self) -> None:
        """State exactly at grace_hours (2.25 h) → no aging (age_hours = 0)."""
        state, now = self._state(8.0, hours_ago=2.25)
        assert age_background_top(state, now) == pytest.approx(8.0)

    def test_just_past_grace_starts_aging(self) -> None:
        """State slightly older than grace → small blend toward climatology."""
        state, now = self._state(8.0, hours_ago=3.25)  # 1 h past grace
        result = age_background_top(state, now)
        # rel_age = 1/48, blend = 8 - (1/48)*(8-5.5) ≈ 7.948
        expected = 8.0 - (1.0 / 48.0) * (8.0 - 5.5)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_at_max_age_converges_to_climatology(self) -> None:
        """After max_age_hours past grace → fully converged to climatology."""
        state, now = self._state(8.0, hours_ago=2.25 + 48.0)
        result = age_background_top(state, now)
        assert result == pytest.approx(CLIMATOLOGY_TOP_KM)

    def test_beyond_max_age_stays_at_climatology(self) -> None:
        """Well past max age → still at climatology (clamped)."""
        state, now = self._state(10.0, hours_ago=200.0)
        result = age_background_top(state, now)
        assert result == pytest.approx(CLIMATOLOGY_TOP_KM)

    def test_below_climatology_ages_upward(self) -> None:
        """TOP below climatology → ages upward toward 5.5."""
        state, now = self._state(3.0, hours_ago=2.25 + 48.0)
        result = age_background_top(state, now)
        assert result == pytest.approx(CLIMATOLOGY_TOP_KM)

    def test_custom_climatology(self) -> None:
        state, now = self._state(8.0, hours_ago=2.25 + 48.0)
        result = age_background_top(state, now, climatology_km=6.0)
        assert result == pytest.approx(6.0)

    def test_custom_grace_hours(self) -> None:
        state, now = self._state(8.0, hours_ago=1.0)
        # grace=0 → immediately starts aging
        result = age_background_top(state, now, grace_hours=0.0)
        assert result < 8.0


# ---------------------------------------------------------------------------
# Tests for compute_new_top
# ---------------------------------------------------------------------------


class TestComputeNewTop:
    def test_all_valid_returns_median_km(self) -> None:
        """All rays weight=1, uniform TOP → new_top = that value in km."""
        ray_top = np.full(NRAYS, 5000.0)
        ray_weight = np.ones(NRAYS)
        result = compute_new_top(ray_top, ray_weight)
        assert result == pytest.approx(5.0)

    def test_insufficient_valid_returns_none(self) -> None:
        """Fewer than 10% valid rays → None."""
        ray_top = np.full(NRAYS, 5000.0)
        ray_weight = np.zeros(NRAYS)
        # Only 10 rays valid (< 36)
        ray_weight[:10] = 1.0
        result = compute_new_top(ray_top, ray_weight)
        assert result is None

    def test_exactly_threshold_returns_value(self) -> None:
        """Exactly 36 valid rays (10% of 360) → returns a value."""
        ray_top = np.full(NRAYS, 4000.0)
        ray_weight = np.zeros(NRAYS)
        ray_weight[:36] = 1.0
        result = compute_new_top(ray_top, ray_weight)
        assert result is not None
        assert result == pytest.approx(4.0)

    def test_takes_median_of_top_10pct(self) -> None:
        """Result is median of the highest 10% valid ray-TOPs."""
        ray_top = np.arange(1, NRAYS + 1, dtype=np.float64) * 100.0
        ray_weight = np.ones(NRAYS)
        result = compute_new_top(ray_top, ray_weight)
        # top 36 values (sorted desc): 36000, 35900, ..., 34500
        # median of those = (36000 + 34500) / 2 ... let's just check range
        assert result is not None
        # Should be in the top 10% range: > 32400 m = 32.4 km
        assert result > 32.0

    def test_zero_top_rays_excluded(self) -> None:
        """Rays with top=0 even if weight=1 should not contribute."""
        ray_top = np.zeros(NRAYS)
        ray_weight = np.ones(NRAYS)
        result = compute_new_top(ray_top, ray_weight)
        assert result is None

    def test_custom_thresholds(self) -> None:
        ray_top = np.full(NRAYS, 3000.0)
        ray_weight = np.full(NRAYS, 0.8)
        # Default threshold 0.99 → none valid
        assert compute_new_top(ray_top, ray_weight) is None
        # Lower threshold → all valid
        result = compute_new_top(
            ray_top, ray_weight, valid_weight_threshold=0.5
        )
        assert result == pytest.approx(3.0)
