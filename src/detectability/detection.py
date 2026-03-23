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

from detectability.beam import slant_range_from_height


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

    For each azimuth, determines the slant-range interval where
    detectability transitions from full beam filling (value 0) to total
    beam overshooting (value 255).

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

        - ``detectability`` — ``uint8`` polar grid (azimuth × range),
          0 = full beam filling, 255 = total overshooting.
        - ``range_full`` — slant range [m] per azimuth where beam top
          reaches TOP (full filling limit).
        - ``range_zero`` — slant range [m] per azimuth where beam
          bottom reaches TOP (zero filling limit).
    """
    top = np.asarray(smoothed_top, dtype=np.float64)
    nrays = top.shape[0]
    half_bw = beamwidth / 2.0

    # Slant range at which the beam *top* edge (elev + half_bw) reaches TOP.
    # Beyond this range the beam is fully filled.
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

    # Build output polar grid
    # Legacy: B in 0..499, range = B * 500m, so B/2.0 = range in km
    range_m = (np.arange(nbins) + 0.5) * range_resolution  # bin centres
    # shape: (nrays, nbins) via broadcasting
    r = range_m[np.newaxis, :]  # (1, nbins)
    rf = range_full[:, np.newaxis]  # (nrays, 1)
    rz = range_zero[:, np.newaxis]  # (nrays, 1)

    # Linear ramp from 0 at range_full to 255 at range_zero
    # Legacy: k = 255/(lowbeam - highbeam);
    #         outbyte = (B/2.0 - highbeam) * k  if B > 2*highbeam
    span = rz - rf
    # Avoid division by zero when TOP is very low (ranges collapse)
    safe_span = np.where(span > 0, span, 1.0)
    frac = (r - rf) / safe_span
    grid = np.clip(frac * 255.0, 0, 255).astype(np.uint8)

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
