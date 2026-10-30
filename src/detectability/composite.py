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

    Provenance is written as GeoTIFF tags: ``radars`` (comma-separated
    NOD identifiers of the inputs, from their ``radar`` tags),
    ``n_radars``, ``nominal_time`` and ``composite_method``.

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

    Raises
    ------
    ValueError
        If *input_paths* is empty or the inputs have differing nominal
        times.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_paths:
        raise ValueError("No input paths provided for compositing.")

    datasets = [rasterio.open(p) for p in input_paths]
    try:
        tags = _provenance_tags([ds.tags() for ds in datasets], input_paths)
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
        output_path,
        mosaic[0],
        crs=crs,
        transform=transform,
        nodata=NODATA,
        tags=tags,
    )

    logger.info(
        "Wrote composite COG: %s (%d inputs: %s, %dx%d)",
        output_path,
        len(input_paths),
        tags["radars"],
        width,
        height,
    )


def _provenance_tags(
    input_tags: list[dict[str, str]], input_paths: list[str | Path]
) -> dict[str, str | int]:
    """Composite tags derived from the single-radar input tags."""
    radars = []
    for tags, path in zip(input_tags, input_paths, strict=True):
        if "radar" in tags:
            radars.append(tags["radar"])
        else:
            logger.warning("No radar tag in composite input %s", path)

    nominal_times = {t["nominal_time"] for t in input_tags if "nominal_time" in t}
    if len(nominal_times) > 1:
        raise ValueError(
            f"Composite inputs have differing nominal times: {sorted(nominal_times)}"
        )

    result: dict[str, str | int] = {
        "radars": ",".join(sorted(radars)),
        "n_radars": len(input_tags),
        "composite_method": "min",
    }
    if nominal_times:
        result["nominal_time"] = nominal_times.pop()
    else:
        logger.warning("No nominal_time tag in any composite input")
    return result
