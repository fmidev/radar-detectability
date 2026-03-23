# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Tests for background state management (detectability.state)."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from detectability.state import (
    CLIMATOLOGICAL_TOP_M,
    DECAY_PERIOD_H,
    GRACE_PERIOD_H,
    BackgroundState,
    age_background,
    load_state,
    save_state,
)


def _make_state(top_m: float = 8000.0, hours_ago: float = 0.0) -> BackgroundState:
    t = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return BackgroundState(top_m=top_m, timestamp=t.isoformat(), valid_ray_count=200)


class TestSaveLoadState:
    """Tests for save_state / load_state round-trip."""

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        state = _make_state(top_m=7000.0)
        save_state(state, path)
        loaded = load_state(path)
        assert loaded is not None
        assert loaded.top_m == 7000.0
        assert loaded.valid_ray_count == state.valid_ray_count

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_state(tmp_path / "nope.json") is None

    def test_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json{{{")
        assert load_state(path) is None

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "state.json"
        save_state(_make_state(), path)
        assert path.exists()

    def test_json_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_state(_make_state(top_m=6000.0, hours_ago=1.0), path)
        data = json.loads(path.read_text())
        assert "top_m" in data
        assert "timestamp" in data
        assert data["top_m"] == 6000.0


class TestAgeBackground:
    """Tests for age_background."""

    def test_within_grace_period_no_change(self) -> None:
        state = _make_state(top_m=8000.0, hours_ago=1.0)
        now = datetime.now(timezone.utc)
        assert age_background(state, now) == pytest.approx(8000.0)

    def test_just_after_grace_period(self) -> None:
        state = _make_state(top_m=8000.0, hours_ago=GRACE_PERIOD_H + 1.0)
        now = datetime.now(timezone.utc)
        aged = age_background(state, now)
        # Should have started decaying toward 5500
        assert aged < 8000.0
        assert aged > CLIMATOLOGICAL_TOP_M

    def test_fully_decayed(self) -> None:
        state = _make_state(top_m=8000.0, hours_ago=GRACE_PERIOD_H + DECAY_PERIOD_H + 1)
        now = datetime.now(timezone.utc)
        aged = age_background(state, now)
        assert_allclose(aged, CLIMATOLOGICAL_TOP_M, atol=1.0)

    def test_climatological_unchanged(self) -> None:
        """If TOP is already at climatology, aging changes nothing."""
        state = _make_state(top_m=CLIMATOLOGICAL_TOP_M, hours_ago=24.0)
        now = datetime.now(timezone.utc)
        aged = age_background(state, now)
        assert_allclose(aged, CLIMATOLOGICAL_TOP_M, atol=1.0)

    def test_below_climatology_ages_upward(self) -> None:
        """TOP below climatology should age upward toward 5500."""
        state = _make_state(top_m=3000.0, hours_ago=GRACE_PERIOD_H + 24.0)
        now = datetime.now(timezone.utc)
        aged = age_background(state, now)
        assert aged > 3000.0
        assert aged <= CLIMATOLOGICAL_TOP_M
