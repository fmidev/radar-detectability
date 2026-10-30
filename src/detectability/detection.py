# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Detection range computation from smoothed echo-top heights.

Corresponds to the detection range encoding section of
``analyse_top_for_detection_range.c``.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr
from wradlib.georef import bin_altitude

from detectability.beam import slant_range_from_height

#: Nodata value of the uint8 product (outside radar coverage).
#: Values 101–254 are reserved.
NODATA: int = 255
#: Band description of the output rasters.
BAND_DESCRIPTION: str = "beam overshoot"
#: Unit of the output rasters.
BAND_UNIT: str = "%"


def compute_detection_ranges(
    smoothed_top: npt.ArrayLike,
    *,
    lowest_elevation: float,
    beamwidth: float,
    sitealt: float = 0.0,
    range_resolution: float = 500.0,
    nbins: int = 500,
) -> xr.Dataset:
    """Compute radar detection range field from smoothed TOP heights.

    Each bin holds the beam overshoot: the percentage of the 3 dB beam
    cross-section above the echo top,

    .. math::

        100 \\cdot \\frac{h_{top}(r) - TOP}{h_{top}(r) - h_{bot}(r)}

    clipped to 0–100 and rounded, where :math:`h_{top}` and
    :math:`h_{bot}` are the heights of the upper and lower beam edges.
    0 = full beam filling, 100 = total overshooting.

    Legacy encoded a linear ramp in *range* from 0 at ``range_full`` to
    255 at ``range_zero``; the endpoints are the same but the geometric
    fraction is slightly higher in between.

    Parameters
    ----------
    smoothed_top
        Smoothed echo-top height per azimuth [m].  1-D array of length
        *nrays* (typically 360).
    lowest_elevation
        Elevation angle of the lowest radar sweep [degrees].
    beamwidth
        Radar 3-dB beamwidth [degrees].
    sitealt
        Radar site altitude [m] above sea level.
    range_resolution
        Range bin size [m] of the output grid (default 500 m, matching
        legacy).
    nbins
        Number of range bins in the output grid.

    Returns
    -------
    xarray.Dataset
        Dataset with:

        - ``detectability`` — ``uint8`` polar grid (azimuth × range) of
          beam overshoot [%], 0 = full beam filling, 100 = total
          overshooting, :data:`NODATA` where TOP is undefined.
        - ``range_full`` — slant range [m] per azimuth where beam top
          reaches TOP (full filling limit).
        - ``range_zero`` — slant range [m] per azimuth where beam
          bottom reaches TOP (zero filling limit).
    """
    top = np.asarray(smoothed_top, dtype=np.float64)
    nrays = top.shape[0]
    half_bw = beamwidth / 2.0

    # Slant range at which the beam *top* edge (elev + half_bw) reaches TOP.
    # Within this range the beam is fully filled.
    # Legacy: maxR_analyzed_highbeam = bindist(TOP*1000, lowest_elev+half_bw, 0)
    range_full = slant_range_from_height(
        top, lowest_elevation + half_bw, sitealt
    )

    # Slant range at which the beam *bottom* edge (elev - half_bw) reaches TOP.
    # Beyond this range the beam completely overshoots.
    # Legacy: maxR_analyzed_lowbeam = bindist(TOP*1000, lowest_elev-half_bw, 0)
    range_zero = slant_range_from_height(
        top, lowest_elevation - half_bw, sitealt
    )

    range_m = (np.arange(nbins) + 0.5) * range_resolution  # bin centres

    # Beam edge heights per range bin, shape (nbins,)
    h_top = bin_altitude(range_m, lowest_elevation + half_bw, sitealt)
    h_bot = bin_altitude(range_m, lowest_elevation - half_bw, sitealt)

    # Fraction of the beam above TOP, shape (nrays, nbins).
    # Replaces the legacy linear-in-range ramp:
    #   k = 255/(lowbeam - highbeam); outbyte = (B/2.0 - highbeam) * k
    # h_top > h_bot for all bin centres (r > 0), so no division by zero.
    frac = (h_top - top[:, np.newaxis]) / (h_top - h_bot)
    grid = np.where(
        np.isfinite(frac), np.rint(np.clip(frac, 0.0, 1.0) * 100.0), NODATA
    ).astype(np.uint8)

    # Coordinates
    azimuth = np.arange(nrays, dtype=np.float64)
    range_coord = range_m

    ds = xr.Dataset(
        {
            "detectability": (["azimuth", "range"], grid),
            "range_full": (["azimuth"], range_full),
            "range_zero": (["azimuth"], range_zero),
        },
        coords={
            "azimuth": azimuth,
            "range": range_coord,
        },
        attrs={
            "lowest_elevation_deg": lowest_elevation,
            "beamwidth_deg": beamwidth,
            "sitealt_m": sitealt,
            "range_resolution_m": range_resolution,
        },
    )
    return ds
