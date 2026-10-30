# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Georeferencing and COG output for polar detectability fields.

Reprojects polar (azimuth × range) detectability grids to a regular
projected grid (default EPSG:3067 / ETRS-TM35FIN) and writes the
result as a Cloud Optimized GeoTIFF.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import xarray as xr
from rasterio.transform import Affine
from scipy.interpolate import NearestNDInterpolator
from wradlib.georef import spherical_to_proj

from detectability.detection import BAND_DESCRIPTION, BAND_UNIT, NODATA

logger = logging.getLogger(__name__)


def polar_to_projected(
    ds_polar: xr.Dataset,
    *,
    radar_lon: float,
    radar_lat: float,
    radar_alt: float = 0.0,
    crs: str = "EPSG:3067",
    resolution_m: float = 500.0,
    max_range_m: float | None = None,
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
    max_range_m
        Maximum radar range [m].  Pixels beyond this distance from the
        radar site are masked to :data:`~detectability.detection.NODATA`.
        When ``None``, no
        masking is applied.

    Returns
    -------
    xarray.DataArray
        Projected detectability grid (y × x) with CRS attached via
        rioxarray.  Values are uint8 beam overshoot [%], 0–100.  Nodata
        is set when ``max_range_m`` is provided.
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

    if max_range_m is not None:
        # Compute radar centre in projected coordinates
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        cx, cy = transformer.transform(radar_lon, radar_lat)
        # Distance from radar centre for each grid cell
        xs = da.coords["x"].values
        ys = da.coords["y"].values
        gx, gy = np.meshgrid(xs, ys)
        dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        da.values[dist > max_range_m] = NODATA
        da = da.rio.write_nodata(NODATA)

    return da


def write_cog(da: xr.DataArray, path: str | Path) -> None:
    """Write a projected DataArray to Cloud Optimized GeoTIFF.

    Attributes are written as GeoTIFF tags.

    Parameters
    ----------
    da
        Projected DataArray with CRS attached (via rioxarray).
    path
        Output file path.
    """
    import rioxarray  # noqa: F401

    tags = {k: v for k, v in da.attrs.items() if not k.startswith("_")}
    write_detectability_cog(
        path,
        da.values,
        crs=da.rio.crs,
        transform=da.rio.transform(),
        nodata=da.rio.nodata,
        tags=tags,
    )
    logger.info("Wrote COG: %s", path)


def write_detectability_cog(
    path: str | Path,
    data: npt.NDArray[np.uint8],
    *,
    crs: str | rasterio.crs.CRS,
    transform: Affine,
    nodata: int | None = NODATA,
    tags: Mapping[str, object] | None = None,
) -> None:
    """Write a single-band detectability grid as COG.

    Sets the band description and unit.  Written with rasterio directly
    because rioxarray does not support band units.

    Parameters
    ----------
    path
        Output file path.  Parent directories are created.
    data
        2-D uint8 beam overshoot grid (y × x), north-up.
    crs, transform
        Georeferencing of *data*.
    nodata
        Nodata value, or ``None`` for none.
    tags
        Dataset-level GeoTIFF tags.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="COG",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.uint8), 1)
        dst.set_band_description(1, BAND_DESCRIPTION)
        dst.set_band_unit(1, BAND_UNIT)
        if tags:
            dst.update_tags(**tags)
