# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Top-level processing pipeline for radar detectability products.

Chains I/O, filtering, analysis, detection range computation, and
georeferencing into a single callable entry point.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from detectability.analysis import pick_ray_tops, sector_smooth
from detectability.defaults import (
    CLIMATOLOGY_TOP_KM,
    HIGHPART,
    MAX_RANGE_KM,
    MIN_RANGE_KM,
    RANGE_RESOLUTION,
    SAMPLEPOINT,
    SECTOR_HALF_WIDTH,
)
from detectability.detection import compute_detection_ranges
from detectability.filtering import azimuthal_filter
from detectability.georef import polar_to_projected, write_cog
from detectability.io import read_echotop
from detectability.logs import streamlogger_setup
from detectability.state import (
    BackgroundState,
    age_background_top,
    compute_new_top,
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)
streamlogger_setup(logger)


def process(
    input_path: str | Path,
    output_path: str | Path,
    *,
    lowest_elevation: float,
    beamwidth: float | None = None,
    min_range_km: float = MIN_RANGE_KM,
    max_range_km: float = MAX_RANGE_KM,
    range_resolution: float = RANGE_RESOLUTION,
    sector_half_width: int = SECTOR_HALF_WIDTH,
    highpart: float = HIGHPART,
    samplepoint: float = SAMPLEPOINT,
    climatology_top_km: float = CLIMATOLOGY_TOP_KM,
    crs: str = "EPSG:3067",
    state_path: str | Path | None = None,
) -> None:
    """Run the full detectability product pipeline.

    Reads a polar echotop ODIM HDF5 file, computes detectability ranges
    and writes the result as a Cloud Optimized GeoTIFF.

    Parameters
    ----------
    input_path
        Path to the input ODIM HDF5 echotop file (HGHT quantity).
    output_path
        Path for the output COG (.tif).  Parent directory is created
        if it does not exist.
    lowest_elevation
        Elevation angle of the lowest radar sweep used to determine
        detection ranges [degrees].
    beamwidth
        Radar 3-dB beamwidth [degrees].  If ``None``, read from the
        file's ``/how`` group (``beamwH``).
    min_range_km, max_range_km
        Radial range window for ray TOP analysis [km].  Defaults match
        the legacy pipeline (10–240 km).
    range_resolution
        Range bin size [m] of the product (default 500 m).
    sector_half_width
        Half-width of the azimuthal smoothing sector in rays.  Full
        sector size is ``2 * sector_half_width + 1`` rays.  Default 30
        gives a 61-ray sector for 1° azimuthal sampling.
    highpart
        Fraction of sorted range bins (highest first) to consider per
        ray.  Default 0.1 = top 10%.  Lower values focus on the very
        highest echoes.  Legacy name: ``HIGHPART`` / ``sortage``.
    samplepoint
        Quantile position within valid sorted bins to pick as ray TOP.
        0.0 = maximum, 0.5 = median.  Noisier radars may use ~0.15.
        Legacy name: ``SAMPLEPOINT``.
    climatology_top_km
        Climatological echo-top height [km] used as the aging target
        when no valid echoes are observed.  Default 5.5 km.
    crs
        Target CRS for the output COG (default ``EPSG:3067``).
    state_path
        Path to a JSON file for persisting background echo-top state
        between runs.  When provided, enables background blending for
        low-confidence rays and time-based aging toward climatology.
        The file is created on first run and updated when sufficient
        valid rays are present.  When ``None`` (default), no background
        blending is applied.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info("Processing %s", input_path.name)

    # --- I/O ----------------------------------------------------------------
    ds = read_echotop(input_path)

    # Resolve beamwidth: caller-supplied takes priority, then file metadata
    if beamwidth is None:
        if "beamwidth_h" not in ds.attrs:
            raise ValueError(
                "beamwidth not supplied and not found in file /how (beamwH). "
                "Pass beamwidth= explicitly."
            )
        beamwidth = float(ds.attrs["beamwidth_h"])
        logger.debug("Using beamwidth_h from file: %.3f°", beamwidth)

    radar_lon = float(ds.coords["longitude"].values)
    radar_lat = float(ds.coords["latitude"].values)
    radar_alt = float(ds.coords["altitude"].values)
    nbins = ds.sizes["range"]

    # --- Filtering ----------------------------------------------------------
    hght = ds["HGHT"].values  # (nrays, nbins), float64 decoded by xradar
    filtered = azimuthal_filter(hght)

    # --- Ray TOP analysis ---------------------------------------------------
    # Convert km to bin indices: bin_i = range_km * 1000 / range_resolution
    # Legacy: StartBin = int(minrange * 2.0) for 500 m bins
    min_bin = int(min_range_km * 1000.0 / range_resolution)
    max_bin = int(max_range_km * 1000.0 / range_resolution)

    ray_top, ray_weight = pick_ray_tops(
        filtered,
        min_range_bin=min_bin,
        max_range_bin=max_bin,
        highpart=highpart,
        samplepoint=samplepoint,
    )

    # --- Background state ---------------------------------------------------
    background_top_m: float | None = None
    if state_path is not None:
        state = load_state(state_path)
        if state is not None:
            now = datetime.now(UTC)
            effective_top_km = age_background_top(
                state, now, climatology_km=climatology_top_km
            )
            background_top_m = effective_top_km * 1000.0
            logger.info(
                "Background TOP: %.1f km (aged from %.1f km)",
                effective_top_km,
                state.top_km,
            )

    # --- Sector smoothing ---------------------------------------------------
    smoothed_top = sector_smooth(
        ray_top,
        ray_weight,
        sector_half_width=sector_half_width,
        background_top_m=background_top_m,
    )

    # --- Detection range computation ----------------------------------------
    ds_polar = compute_detection_ranges(
        smoothed_top,
        lowest_elevation=lowest_elevation,
        beamwidth=beamwidth,
        sitealt=radar_alt,
        range_resolution=range_resolution,
        nbins=nbins,
    )

    # --- Georeferencing & output --------------------------------------------
    da_proj = polar_to_projected(
        ds_polar,
        radar_lon=radar_lon,
        radar_lat=radar_lat,
        radar_alt=radar_alt,
        crs=crs,
        max_range_m=nbins * range_resolution,
    )

    write_cog(da_proj, output_path)

    # --- State update -------------------------------------------------------
    if state_path is not None:
        new_top_km = compute_new_top(ray_top, ray_weight)
        if new_top_km is not None:
            new_state = BackgroundState(
                top_km=new_top_km, timestamp=datetime.now(UTC)
            )
            save_state(state_path, new_state)
            logger.info("Updated background state: %.1f km", new_top_km)
        else:
            logger.info(
                "Insufficient valid rays; background state not updated"
            )

    logger.info("Done: %s", output_path)
