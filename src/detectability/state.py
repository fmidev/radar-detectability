# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Background TOP state management.

Persists the background echo-top height between pipeline runs
as a JSON file, with time-based aging toward climatology.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CLIMATOLOGICAL_TOP_M = 5500.0
"""Climatological echo-top default [m] (5.5 km)."""

GRACE_PERIOD_H = 2.25
"""Hours before background aging starts (legacy: 2.25)."""

DECAY_PERIOD_H = 48.0
"""Hours over which background converges fully to climatology."""


@dataclass
class BackgroundState:
    """Persisted background echo-top state."""

    top_m: float
    timestamp: str  # ISO 8601
    valid_ray_count: int = 0

    @property
    def time(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)


def load_state(path: str | Path) -> BackgroundState | None:
    """Load background state from JSON, or *None* if unavailable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return BackgroundState(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Could not read state from %s", path)
        return None


def save_state(state: BackgroundState, path: str | Path) -> None:
    """Write background state to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2) + "\n")
    logger.info("Saved background state to %s (TOP=%.0f m)", path, state.top_m)


def age_background(state: BackgroundState, current_time: datetime) -> float:
    """Apply time-based aging to background TOP.

    After a grace period the background TOP linearly converges toward
    the climatological default over ``DECAY_PERIOD_H`` hours.

    Legacy logic from ``analyse_top_for_detection_range.c``::

        prevtop_age = (curtime - prevtoptime) / 3600.0 - 2.25;
        rel_age = prevtop_age / 48.0;
        newTOPprev = TOPprev - rel_age * (TOPprev - 5.5);

    Returns
    -------
    float
        Aged background TOP [m].
    """
    age_h = (current_time - state.time).total_seconds() / 3600.0
    effective_age = age_h - GRACE_PERIOD_H

    if effective_age <= 0:
        return state.top_m

    rel_age = min(effective_age / DECAY_PERIOD_H, 1.0)
    return state.top_m - rel_age * (state.top_m - CLIMATOLOGICAL_TOP_M)
