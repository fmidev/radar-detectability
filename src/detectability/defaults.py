# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Default configuration values for the detectability pipeline.

All operationally relevant tuning parameters are collected here with
documentation.  These defaults match the legacy C implementation and
are suitable for most FMI radars.  They can be overridden by passing
keyword arguments to :func:`~detectability.process`.

In Airflow ``@task.docker`` calls, pass these as keyword arguments::

    @task.docker(image="quay.io/fmi/radar-detectability:v1.0.0")
    def process_radar(input_path, output_path, **kwargs):
        from detectability import process
        process(input_path, output_path, **kwargs)

    process_radar(
        input_path="/data/etop.h5",
        output_path="/data/detect.tif",
        lowest_elevation=0.3,
        state_path="/state/fivim.json",
        climatology_top_km=5.5,
        highpart=0.1,
        samplepoint=0.5,
    )
"""

# ---------------------------------------------------------------------------
# Echo-top ray analysis
# ---------------------------------------------------------------------------

HIGHPART: float = 0.1
"""Fraction of sorted range bins (highest first) to consider per ray.

A value of 0.1 means the top 10% of bins (by echo-top height) are used
to derive the representative ray TOP.  Lower values focus on the very
highest echoes; higher values include more of the ray.

Legacy name: ``HIGHPART`` / ``sortage``.  Typical range: 0.05–0.2.
"""

SAMPLEPOINT: float = 0.5
"""Quantile position within valid sorted bins to pick as ray TOP.

0.0 picks the absolute maximum, 0.5 picks the median of the selected
fraction, 1.0 would pick the lowest.  Noisier radars (e.g. Korppoo)
may use lower values like 0.15 to favour higher TOPs.

Legacy name: ``SAMPLEPOINT``.
"""

# ---------------------------------------------------------------------------
# Azimuthal smoothing
# ---------------------------------------------------------------------------

SECTOR_HALF_WIDTH: int = 30
"""Half-width of the azimuthal smoothing sector [rays].

Full sector size is ``2 * SECTOR_HALF_WIDTH + 1`` rays.  For a scan
with 1° azimuth resolution, default 30 gives a 61° smoothing sector.

Legacy name: ``AVERAGING_SECTOR`` (full width); legacy default 60°.
"""

# ---------------------------------------------------------------------------
# Background state / clear-sky fallback
# ---------------------------------------------------------------------------

CLIMATOLOGY_TOP_KM: float = 5.5
"""Climatological echo-top height [km] used as the aging target.

When no valid echoes are observed for an extended period, the
background TOP converges toward this value.  Represents a "typical"
precipitating cloud-top height for the Finnish climate.

Legacy: hard-coded 5.5 in ``analyse_top_for_detection_range.c``.
Could be adjusted seasonally or regionally if needed.
"""

MAX_AGE_HOURS: float = 48.0
"""Time [hours] for full convergence from last valid TOP to climatology.

After this many hours past the grace period without a valid update,
the background TOP equals ``CLIMATOLOGY_TOP_KM``.
"""

GRACE_HOURS: float = 2.25
"""Grace period [hours] before aging begins.

The background TOP remains at its last valid value for this duration
after the last update before aging toward climatology starts.  Matches
the legacy offset of 2.25 hours (roughly one volume scan cycle plus
margin).
"""

# ---------------------------------------------------------------------------
# Range and resolution
# ---------------------------------------------------------------------------

MIN_RANGE_KM: float = 10.0
"""Minimum radial range [km] for ray TOP analysis.

Bins closer than this are excluded (ground clutter zone).
"""

MAX_RANGE_KM: float = 240.0
"""Maximum radial range [km] for ray TOP analysis."""

RANGE_RESOLUTION: float = 500.0
"""Range bin size [m] of the output detectability grid."""
