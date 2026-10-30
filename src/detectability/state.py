# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Background TOP state persistence and aging.

Implements the legacy ``TOPprev``/``OLDTOP`` mechanism: a single
background echo-top height value is persisted between pipeline runs,
aged toward climatology during no-echo periods, and used to stabilize
the detectability product in clear-sky or sparse-echo conditions.

Legacy reference: ``analyse_top_for_detection_range.c`` lines 202–232
(aging) and 470–485 (update threshold).
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from detectability.defaults import (
    CLIMATOLOGY_TOP_KM,
    GRACE_HOURS,
    MAX_AGE_HOURS,
    VALID_RAY_WEIGHT,
)

logger = logging.getLogger(__name__)

type BackgroundSource = Literal["state", "aged_state", "climatology"]


@dataclass
class BackgroundState:
    """Persisted background echo-top height."""

    top_km: float
    """Background TOP value [km]."""

    timestamp: datetime
    """Nominal time of the data from which this TOP was last derived (UTC)."""


@dataclass(frozen=True)
class Background:
    """Background TOP resolved for a single pipeline run."""

    top_km: float
    """Effective background TOP [km]."""

    source: BackgroundSource
    """``"state"`` (fresh), ``"aged_state"`` (aged toward climatology),
    or ``"climatology"`` (no usable state)."""

    age_hours: float | None
    """Age of the persisted state [h] relative to the data, ``None`` if
    no state was available."""


def load_state(path: str | Path) -> BackgroundState | None:
    """Load background state from a JSON file.

    Returns ``None`` (and logs a warning) if the file is missing,
    unreadable, or contains invalid data.  Naive timestamps are
    interpreted as UTC.
    """
    path = Path(path)
    if not path.exists():
        # Expected only on the first run; otherwise the state was lost,
        # e.g. removed by a cache cleanup.
        logger.warning("State file %s not found; using climatology", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(data["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return BackgroundState(top_km=float(data["top_km"]), timestamp=timestamp)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Corrupt state file %s: %s", path, exc)
        return None


def save_state(path: str | Path, state: BackgroundState) -> None:
    """Atomically write background state to a JSON file.

    Uses write-to-temp + rename to avoid corruption on crash.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "top_km": round(state.top_km, 1),
        "timestamp": state.timestamp.isoformat(),
    }
    # Write to a temp file in the same directory, then rename (atomic on POSIX)
    fd = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=".state_",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    )
    try:
        json.dump(data, fd)
        fd.close()
        Path(fd.name).replace(path)
    except BaseException:
        Path(fd.name).unlink(missing_ok=True)
        raise


def age_background_top(
    state: BackgroundState,
    now: datetime,
    *,
    climatology_km: float = CLIMATOLOGY_TOP_KM,
    max_age_hours: float = MAX_AGE_HOURS,
    grace_hours: float = GRACE_HOURS,
) -> float:
    """Apply time-based aging of background TOP toward climatology.

    After ``grace_hours`` without update, the background TOP linearly
    converges from its last valid value toward ``climatology_km`` over
    ``max_age_hours``.

    Legacy reference (``analyse_top_for_detection_range.c``)::

        prevtop_age = (curtime - prevtoptime) / 3600.0 - 2.25;
        rel_age = prevtop_age / 48.0;  // clamped [0, 1]
        TOPprev = TOPprev - rel_age * (TOPprev - 5.5);

    Returns
    -------
    float
        Effective background TOP [km] after aging.
    """
    age_hours = (now - state.timestamp).total_seconds() / 3600.0 - grace_hours
    if age_hours <= 0.0:
        return state.top_km
    rel_age = min(age_hours / max_age_hours, 1.0)
    return state.top_km - rel_age * (state.top_km - climatology_km)


def resolve_background(
    state: BackgroundState | None,
    now: datetime,
    *,
    climatology_km: float = CLIMATOLOGY_TOP_KM,
    max_age_hours: float = MAX_AGE_HOURS,
    grace_hours: float = GRACE_HOURS,
) -> Background:
    """Resolve the background TOP to use for the data valid at ``now``.

    Always returns a value: the persisted state (aged toward climatology
    if older than ``grace_hours``), or ``climatology_km`` when no state
    is available.  Unlike the legacy implementation, which starts with
    no background, this keeps clear-sky output meaningful from the very
    first run.

    Parameters
    ----------
    state
        Persisted background state, or ``None``.
    now
        Nominal time of the data being processed (timezone-aware).
    """
    if state is None:
        return Background(top_km=climatology_km, source="climatology", age_hours=None)
    age_hours = (now - state.timestamp).total_seconds() / 3600.0
    if age_hours < 0.0:
        # E.g. backfill of older data after newer runs; use state as is
        logger.info(
            "Background state (%s) is newer than data (%s)",
            state.timestamp.isoformat(),
            now.isoformat(),
        )
    if age_hours <= grace_hours:
        return Background(top_km=state.top_km, source="state", age_hours=age_hours)
    top_km = age_background_top(
        state,
        now,
        climatology_km=climatology_km,
        max_age_hours=max_age_hours,
        grace_hours=grace_hours,
    )
    return Background(top_km=top_km, source="aged_state", age_hours=age_hours)


def compute_new_top(
    ray_top: npt.ArrayLike,
    ray_weight: npt.ArrayLike,
    *,
    valid_weight_threshold: float = VALID_RAY_WEIGHT,
    min_valid_fraction: float = 0.1,
) -> float | None:
    """Derive a new background TOP from the current scan's ray analysis.

    Computes the median of the highest 10% of valid ray-TOPs (rays
    with weight above ``valid_weight_threshold``).  Returns ``None`` if
    fewer than ``min_valid_fraction`` of rays meet the threshold — the
    state file should not be updated in that case.

    Legacy reference (``analyse_top_for_detection_range.c``)::

        if(valid_raytop_count >= 36) fprintf(PREVTOPF, "%.1f\\n", TOPprev);

    Parameters
    ----------
    ray_top
        Representative echo-top height [m] per ray (from ``pick_ray_tops``).
    ray_weight
        Confidence weight per ray in [0, 1].
    valid_weight_threshold
        Minimum ray weight to count as "valid" (legacy: 0.99).
    min_valid_fraction
        Minimum fraction of total rays that must be valid to produce a
        new background TOP (legacy: 36/360 = 0.1).

    Returns
    -------
    float or None
        New background TOP [km], or ``None`` if threshold not met.
    """
    top = np.asarray(ray_top, dtype=np.float64)
    weight = np.asarray(ray_weight, dtype=np.float64)
    nrays = top.shape[0]

    valid_mask = weight >= valid_weight_threshold
    valid_count = int(valid_mask.sum())

    if valid_count < nrays * min_valid_fraction:
        return None

    # Legacy: sort descending, take top 36, pick median
    valid_tops = top[valid_mask]
    valid_tops_sorted = np.sort(valid_tops)[::-1]
    n_top = max(1, int(nrays * min_valid_fraction))
    highest = valid_tops_sorted[:n_top]
    # Only consider nonzero values
    highest_nonzero = highest[highest > 0]
    if len(highest_nonzero) == 0:
        return None
    median_m = float(np.median(highest_nonzero))
    return median_m / 1000.0  # convert m → km
