# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Top-level processing pipeline for radar detectability products.

Chains I/O, filtering, analysis, detection range computation, and
georeferencing into a single callable entry point.
"""

import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from detectability.analysis import pick_ray_tops, sector_smooth
from detectability.defaults import (
    CLIMATOLOGY_TOP_KM,
    HIGHPART,
    MAX_RANGE_KM,
    MIN_RANGE_KM,
    RANGE_RESOLUTION,
    SAMPLEPOINT,
    SECTOR_HALF_WIDTH,
    VALID_RAY_WEIGHT,
)
from detectability.detection import compute_detection_ranges
from detectability.filtering import azimuthal_filter
from detectability.georef import polar_to_projected, write_cog
from detectability.io import read_echotop
from detectability.logs import streamlogger_setup
from detectability.state import (
    Background,
    BackgroundState,
    compute_new_top,
    load_state,
    resolve_background,
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
    and writes the result as a Cloud Optimized GeoTIFF.  Provenance
    (radar, nominal time, background TOP and its source, echo coverage)
    is written as GeoTIFF tags.

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
        Range bin size [m] of the polar product grid (default 500 m).
        Independent of the input bin size; the product extends to the
        input's maximum range.
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
        Climatological echo-top height [km], used as the background TOP
        when no state is available and as the aging target when no
        valid echoes are observed.  Default 5.5 km.
    crs
        Target CRS for the output COG (default ``EPSG:3067``).
    state_path
        Path to a JSON file for persisting background echo-top state
        between runs.  Low-confidence rays are always blended with a
        background TOP; with a state file it follows the latest valid
        observations and ages toward climatology during no-echo
        periods.  The file is created on first run and updated when
        sufficient valid rays are present, unless it already holds
        state newer than the data.  Ages are measured from the ODIM
        nominal time, so reruns and backfills are deterministic.
        When ``None`` (default), the climatological value is used.
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

    nominal_time: datetime | None = ds.attrs.get("nominal_time")
    if nominal_time is None:
        nominal_time = datetime.now(UTC)
        logger.warning(
            "Nominal time not found in %s; using current time %s",
            input_path.name,
            nominal_time.isoformat(),
        )

    if "radar" not in ds.attrs:
        logger.warning("Radar NOD not found in %s /what source", input_path.name)

    radar_lon = float(ds.coords["longitude"].values)
    radar_lat = float(ds.coords["latitude"].values)
    radar_alt = float(ds.coords["altitude"].values)
    max_range_m = _max_range_m(ds["range"])
    out_nbins = math.ceil(max_range_m / range_resolution)

    # --- Filtering ----------------------------------------------------------
    hght = ds["HGHT"].values  # (nrays, nbins), float64 decoded by xradar
    filtered = azimuthal_filter(hght)

    # --- Ray TOP analysis ---------------------------------------------------
    # Input bins with centres in [min_range, max_range); input bin size
    # need not match range_resolution.
    # Legacy assumed 500 m bins: StartBin = int(minrange * 2.0)
    min_bin, max_bin = np.searchsorted(
        ds["range"].values, [min_range_km * 1000.0, max_range_km * 1000.0]
    ).tolist()

    ray_top, ray_weight = pick_ray_tops(
        filtered,
        min_range_bin=min_bin,
        max_range_bin=max_bin,
        highpart=highpart,
        samplepoint=samplepoint,
    )

    # --- Background state ---------------------------------------------------
    state = load_state(state_path) if state_path is not None else None
    background = resolve_background(
        state, nominal_time, climatology_km=climatology_top_km
    )
    logger.info(
        "Background TOP: %.1f km (source: %s)", background.top_km, background.source
    )

    # --- Sector smoothing ---------------------------------------------------
    smoothed_top = sector_smooth(
        ray_top,
        ray_weight,
        sector_half_width=sector_half_width,
        background_top_m=background.top_km * 1000.0,
    )

    # --- Detection range computation ----------------------------------------
    ds_polar = compute_detection_ranges(
        smoothed_top,
        lowest_elevation=lowest_elevation,
        beamwidth=beamwidth,
        sitealt=radar_alt,
        range_resolution=range_resolution,
        nbins=out_nbins,
    )
    ds_polar.attrs.update(_provenance_attrs(nominal_time, background, ray_weight))
    # ODIM source identification, used e.g. for listing composite inputs
    ds_polar.attrs.update(
        {k: ds.attrs[k] for k in ("radar", "source") if k in ds.attrs}
    )

    # --- Georeferencing & output --------------------------------------------
    da_proj = polar_to_projected(
        ds_polar,
        radar_lon=radar_lon,
        radar_lat=radar_lat,
        radar_alt=radar_alt,
        crs=crs,
        max_range_m=max_range_m,
    )

    write_cog(da_proj, output_path)

    # --- State update -------------------------------------------------------
    if state_path is not None:
        _update_state(state_path, state, nominal_time, ray_top, ray_weight)

    logger.info("Done: %s", output_path)


def _max_range_m(range_coord: xr.DataArray) -> float:
    """Far edge [m] of the last bin of an xradar ``range`` coordinate."""
    centres = range_coord.values
    gate = range_coord.attrs.get("meters_between_gates")
    if gate is None:
        gate = centres[-1] - centres[-2]
    return float(centres[-1] + float(gate) / 2.0)


def _provenance_attrs(
    nominal_time: datetime, background: Background, ray_weight: np.ndarray
) -> dict[str, str | float]:
    """Output metadata describing how the product was derived.

    ``echo_coverage`` is the mean ray weight (legacy ``TOP_PERCENTAGE``
    / 100); ``valid_ray_fraction`` the fraction of fully echo-covered
    rays.  Low values mean the product relies mostly on the background.
    """
    attrs: dict[str, str | float] = {
        "nominal_time": nominal_time.isoformat(),
        "background_top_km": round(background.top_km, 2),
        "background_source": background.source,
        "echo_coverage": round(float(np.mean(ray_weight)), 3),
        "valid_ray_fraction": round(float(np.mean(ray_weight >= VALID_RAY_WEIGHT)), 3),
    }
    if background.age_hours is not None:
        attrs["background_age_hours"] = round(background.age_hours, 2)
    return attrs


def _update_state(
    state_path: str | Path,
    state: BackgroundState | None,
    nominal_time: datetime,
    ray_top: np.ndarray,
    ray_weight: np.ndarray,
) -> None:
    """Persist a new background TOP, unless the existing state is newer."""
    if state is not None and nominal_time < state.timestamp:
        logger.info(
            "Background state (%s) is newer than data (%s); not updated",
            state.timestamp.isoformat(),
            nominal_time.isoformat(),
        )
        return
    new_top_km = compute_new_top(ray_top, ray_weight)
    if new_top_km is None:
        logger.info("Insufficient valid rays; background state not updated")
        return
    save_state(state_path, BackgroundState(top_km=new_top_km, timestamp=nominal_time))
    logger.info("Updated background state: %.1f km", new_top_km)
