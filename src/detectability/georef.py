# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Georeferencing and COG output for polar detectability fields.

Reprojects polar (azimuth × range) detectability grids to a regular
projected grid (default EPSG:3067 / ETRS-TM35FIN) and writes the
result as a Cloud Optimized GeoTIFF.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import NearestNDInterpolator
from wradlib.georef import spherical_to_proj

logger = logging.getLogger(__name__)


def polar_to_projected(
    ds_polar: xr.Dataset,
    *,
    radar_lon: float,
    radar_lat: float,
    radar_alt: float = 0.0,
    crs: str = "EPSG:3067",
    resolution_m: float = 500.0,
) -> xr.DataArray:
    """Reproject polar detectability field to a regular projected grid.

    Uses :func:`wradlib.georef.spherical_to_proj` for coordinate
    transformation and nearest-neighbour interpolation for regridding.

    Parameters
    ----------
    ds_polar
        Dataset from :func:`~detectability.detection.compute_detection_ranges`,
        with ``detectability`` variable on (azimuth, range) dims.
    radar_lon, radar_lat
        Radar site longitude and latitude [degrees].
    radar_alt
        Radar site altitude [m] above sea level.
    crs
        Target CRS (default EPSG:3067).
    resolution_m
        Target grid cell size [m].

    Returns
    -------
    xarray.DataArray
        Projected detectability grid (y × x) with CRS attached via
        rioxarray. Values are uint8, 0–255.
    """
    import rioxarray  # noqa: F401 — registers .rio accessor

    det = ds_polar["detectability"].values  # (nrays, nbins), uint8
    az = ds_polar["azimuth"].values  # degrees
    rng = ds_polar["range"].values  # meters, bin centres

    # Meshgrids for spherical_to_proj: shape (nrays, nbins)
    r_mesh, az_mesh = np.meshgrid(rng, az)
    # Elevation = 0 for product-level projection
    elev = np.float64(ds_polar.attrs.get("lowest_elevation_deg", 0.0))
    theta_mesh = np.full_like(r_mesh, elev)

    site = (radar_lon, radar_lat, radar_alt)
    coords_proj = spherical_to_proj(r_mesh, az_mesh, theta_mesh, site, crs=crs)
    x_polar = coords_proj[..., 0]
    y_polar = coords_proj[..., 1]

    # Build regular target grid
    x_min = np.floor(x_polar.min() / resolution_m) * resolution_m
    x_max = np.ceil(x_polar.max() / resolution_m) * resolution_m
    y_min = np.floor(y_polar.min() / resolution_m) * resolution_m
    y_max = np.ceil(y_polar.max() / resolution_m) * resolution_m

    x_grid = np.arange(x_min, x_max, resolution_m) + resolution_m / 2
    y_grid = np.arange(y_min, y_max, resolution_m) + resolution_m / 2

    logger.debug(
        "Target grid %d×%d, resolution %.0f m", len(y_grid), len(x_grid), resolution_m
    )

    # Nearest-neighbour regridding
    src_points = np.column_stack([x_polar.ravel(), y_polar.ravel()])
    interp = NearestNDInterpolator(src_points, det.ravel())
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = interp(xx, yy).astype(np.uint8)

    # y axis descending (north-up raster convention)
    y_sorted = np.sort(y_grid)[::-1]
    if not np.array_equal(y_grid, y_sorted):
        grid = grid[::-1]
        y_grid = y_sorted

    da = xr.DataArray(
        grid,
        dims=["y", "x"],
        coords={"y": y_grid, "x": x_grid},
        attrs=ds_polar.attrs,
    )
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform()
    return da


def write_cog(da: xr.DataArray, path: str | Path) -> None:
    """Write a projected DataArray to Cloud Optimized GeoTIFF.

    Parameters
    ----------
    da
        Projected DataArray with CRS attached (via rioxarray).
    path
        Output file path.
    """
    import rioxarray  # noqa: F401

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(str(path), driver="COG")
    logger.info("Wrote COG: %s", path)
