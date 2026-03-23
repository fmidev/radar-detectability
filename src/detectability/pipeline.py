# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Pipeline orchestration: echotop → detectability COG.

Ties together I/O, analysis, detection, filtering, georeferencing,
and background state management into a single ``process`` call.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from detectability.analysis import pick_ray_tops, sector_smooth
from detectability.detection import compute_detection_ranges
from detectability.filtering import azimuthal_filter
from detectability.georef import polar_to_projected, write_cog
from detectability.io import read_echotop
from detectability.state import (
    CLIMATOLOGICAL_TOP_M,
    BackgroundState,
    age_background,
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)

MIN_VALID_RAYS = 36
"""Minimum valid rays to update the background state."""


def process(
    input_path: str | Path,
    output_path: str | Path,
    *,
    lowest_elevation: float,
    state_path: str | Path | None = None,
    highpart: float = 0.1,
    samplepoint: float = 0.5,
    sector_width: int = 60,
    range_resolution: float = 500.0,
    nbins: int = 500,
    crs: str = "EPSG:3067",
    grid_resolution: float = 500.0,
    current_time: datetime | None = None,
) -> xr.Dataset:
    """Run the full detectability pipeline.

    Reads a pre-computed polar echotop product, analyses echo-top
    heights, computes detection ranges, and writes a georeferenced
    COG output.

    Parameters
    ----------
    input_path
        Path to ODIM HDF5 polar echotop product.
    output_path
        Path for output COG file.
    lowest_elevation
        Elevation angle [degrees] of the lowest radar sweep.
    state_path
        Path to JSON background state file.  If *None*, no state
        persistence (uses climatological default).
    highpart
        Fraction of bins for per-ray TOP picking (legacy: sortage).
    samplepoint
        Percentile position for TOP picking (0 = highest, 0.5 = median).
    sector_width
        Azimuthal smoothing sector width [degrees].
    range_resolution
        Detection range grid bin size [m].
    nbins
        Number of range bins in detection grid.
    crs
        Target CRS for the output COG.
    grid_resolution
        Projected grid resolution [m].
    current_time
        Processing timestamp (default: now UTC).

    Returns
    -------
    xr.Dataset
        Polar detection range dataset (before georeferencing).
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)

    # 1. Read input
    top_da, meta = read_echotop(input_path)
    logger.info(
        "Radar: lat=%.4f lon=%.4f alt=%.0f m, beamwidth=%.2f°",
        meta.latitude,
        meta.longitude,
        meta.height,
        meta.beamwidth_h,
    )

    # 2. Per-ray TOP picking
    ray_top, ray_weight = pick_ray_tops(
        top_da.values, highpart=highpart, samplepoint=samplepoint
    )
    valid_rays = int(np.count_nonzero(ray_weight > 0))
    logger.info(
        "TOP picking: %d/%d rays with valid TOPs", valid_rays, len(ray_top)
    )

    # 3. Load and age background
    background_top = CLIMATOLOGICAL_TOP_M
    if state_path is not None:
        prev_state = load_state(state_path)
        if prev_state is not None:
            background_top = age_background(prev_state, current_time)
            logger.info(
                "Background TOP: %.0f m (aged from %.0f m)",
                background_top,
                prev_state.top_m,
            )

    # 4. Sector smoothing with background blending
    smoothed_top = sector_smooth(
        ray_top, ray_weight, background_top, sector_width=sector_width
    )

    # 5. Compute detection ranges
    ds = compute_detection_ranges(
        smoothed_top,
        lowest_elevation=lowest_elevation,
        beamwidth=meta.beamwidth_h,
        sitealt=meta.height,
        range_resolution=range_resolution,
        nbins=nbins,
    )

    # 6. Azimuthal filter
    ds["detectability"].values[:] = azimuthal_filter(ds["detectability"].values)

    # 7. Update background state
    if state_path is not None and valid_rays >= MIN_VALID_RAYS:
        positive = smoothed_top[smoothed_top > 0]
        new_top = float(np.mean(positive)) if len(positive) > 0 else background_top
        save_state(
            BackgroundState(
                top_m=new_top,
                timestamp=current_time.isoformat(),
                valid_ray_count=valid_rays,
            ),
            state_path,
        )

    # 8. Georeference and write COG
    da_proj = polar_to_projected(
        ds,
        radar_lon=meta.longitude,
        radar_lat=meta.latitude,
        radar_alt=meta.height,
        crs=crs,
        resolution_m=grid_resolution,
    )
    write_cog(da_proj, output_path)

    return ds
