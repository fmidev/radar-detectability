# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Composite detectability from multiple radars.

Merges single-radar detectability COGs into a national composite using
pixel-wise minimum of beam overshoot (best detectability wins).
"""

import logging
from pathlib import Path

import rasterio
from rasterio.merge import merge

from detectability.defaults import COMPOSITE_BOUNDS_EPSG3067, RANGE_RESOLUTION
from detectability.detection import NODATA
from detectability.georef import write_detectability_cog

logger = logging.getLogger(__name__)

__all__ = ["NODATA", "composite_min"]


def composite_min(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    bounds: tuple[float, float, float, float] = COMPOSITE_BOUNDS_EPSG3067,
    resolution: float = RANGE_RESOLUTION,
    crs: str = "EPSG:3067",
) -> None:
    """Create a composite detectability field from single-radar COGs.

    Merges all input rasters onto a fixed national grid using pixel-wise
    minimum.  Cells not covered by any radar are set to :data:`NODATA`.

    Parameters
    ----------
    input_paths
        Paths to single-radar detectability COGs (uint8, nodata
        :data:`NODATA`).
    output_path
        Path for the output composite COG.
    bounds
        Fixed grid extent (xmin, ymin, xmax, ymax) in the target CRS.
        Default covers Finland's radar network in EPSG:3067.
    resolution
        Grid cell size [m].  Default 500 m.
    crs
        Target CRS.  Default EPSG:3067.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_paths:
        raise ValueError("No input paths provided for compositing.")

    datasets = [rasterio.open(p) for p in input_paths]
    try:
        mosaic, transform = merge(
            datasets,
            bounds=bounds,
            res=resolution,
            nodata=NODATA,
            method="min",
        )
    finally:
        for ds in datasets:
            ds.close()

    # mosaic shape: (bands, height, width)
    height, width = mosaic.shape[1], mosaic.shape[2]
    write_detectability_cog(
        output_path, mosaic[0], crs=crs, transform=transform, nodata=NODATA
    )

    logger.info(
        "Wrote composite COG: %s (%d inputs, %dx%d)",
        output_path,
        len(input_paths),
        width,
        height,
    )
