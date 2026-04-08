# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Top-level processing pipeline for radar detectability products.

Chains I/O, filtering, analysis, detection range computation, and
georeferencing into a single callable entry point.
"""

import logging
from pathlib import Path

from detectability.analysis import pick_ray_tops, sector_smooth
from detectability.detection import compute_detection_ranges
from detectability.filtering import azimuthal_filter
from detectability.georef import polar_to_projected, write_cog
from detectability.io import read_echotop

logger = logging.getLogger(__name__)


def process(
    input_path: str | Path,
    output_path: str | Path,
    *,
    lowest_elevation: float,
    beamwidth: float | None = None,
    min_range_km: float = 10.0,
    max_range_km: float = 240.0,
    range_resolution: float = 500.0,
    sector_half_width: int = 30,
    crs: str = "EPSG:3067",
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
    crs
        Target CRS for the output COG (default ``EPSG:3067``).
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
    )

    # --- Sector smoothing ---------------------------------------------------
    smoothed_top = sector_smooth(
        ray_top,
        ray_weight,
        sector_half_width=sector_half_width,
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
    )

    write_cog(da_proj, output_path)
    logger.info("Done: %s", output_path)
