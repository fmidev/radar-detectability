# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Per-ray TOP picking and azimuthal sector smoothing.

Implements the ray-analysis section of ``analyse_top_for_detection_range.c``,
without background blending (deferred).
"""

import numpy as np
import numpy.typing as npt


def pick_ray_tops(
    top_2d: npt.ArrayLike,
    *,
    min_range_bin: int = 20,
    max_range_bin: int = 480,
    highpart: float = 0.1,
    samplepoint: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a representative echo-top height and confidence weight per ray.

    For each azimuth ray:

    1. Extract bins in ``[min_range_bin, max_range_bin)``.
    2. Sort descending, take the top ``highpart`` fraction (``sortpart``).
    3. Count nonzero bins in that fraction → ``topcount``.
    4. Pick the value at position ``int(topcount * samplepoint)`` within
       the nonzero sorted values → ``ray_top[A]``.
    5. ``ray_weight[A] = topcount / sortpart``.

    Corresponds to the ray-sorting block in
    ``analyse_top_for_detection_range.c`` (lines ~308–345).

    Parameters
    ----------
    top_2d
        2-D array of echo-top heights [m], shape (nrays, nbins).
        Zero and negative values are treated as "no echo".
    min_range_bin, max_range_bin
        Range-bin slice ``[min_range_bin, max_range_bin)`` to analyse.
        Legacy defaults correspond to 10–240 km at 500 m resolution
        (bins 20–480).
    highpart
        Fraction of sorted bins to consider (default 0.1 = top 10%).
        Legacy: ``sortage``.
    samplepoint
        Quantile position within the nonzero sorted fraction to pick
        (default 0.5 = median).  Legacy: ``samplepoint``.

    Returns
    -------
    ray_top : numpy.ndarray, shape (nrays,)
        Representative echo-top height [m] per ray.  0 where no valid
        echo was found.
    ray_weight : numpy.ndarray, shape (nrays,)
        Confidence weight in [0, 1]: fraction of ``sortpart`` bins that
        had a valid echo.
    """
    arr = np.asarray(top_2d, dtype=np.float64)
    nrays = arr.shape[0]
    sector = arr[:, min_range_bin:max_range_bin]
    nbins = sector.shape[1]
    sortpart = max(1, int(nbins * highpart))

    ray_top = np.zeros(nrays, dtype=np.float64)
    ray_weight = np.zeros(nrays, dtype=np.float64)

    for A in range(nrays):
        # Sort descending; take top sortpart fraction
        row = sector[A]
        sorted_desc = np.sort(row)[::-1][:sortpart]
        valid = sorted_desc[sorted_desc > 0]
        topcount = len(valid)
        if topcount > 0:
            picbin = int(topcount * samplepoint)
            # Clamp to valid index (samplepoint=1.0 would overshoot)
            picbin = min(picbin, topcount - 1)
            ray_top[A] = valid[picbin]
        ray_weight[A] = topcount / sortpart

    return ray_top, ray_weight


def sector_smooth(
    ray_top: npt.ArrayLike,
    ray_weight: npt.ArrayLike,
    *,
    sector_half_width: int = 30,
) -> np.ndarray:
    """Azimuthal sector-weighted smoothing of per-ray echo-top heights.

    For each output azimuth A, computes a weighted average over the
    sector ``[A - sector_half_width, A + sector_half_width]`` (inclusive)
    using a triangle (linear taper) kernel.  The contribution of each
    ray is further scaled by its ``ray_weight``, so low-confidence rays
    contribute less.

    Corresponds to the sector-weighting block in
    ``analyse_top_for_detection_range.c`` (lines ~356–408).

    Parameters
    ----------
    ray_top
        Representative echo-top height [m] per ray, shape (nrays,).
    ray_weight
        Confidence weight per ray in [0, 1], shape (nrays,).
    sector_half_width
        Half-width of the sector in rays.  Full sector size is
        ``2 * sector_half_width + 1``.  Legacy argument ``inW``
        (default 30, i.e. 61-ray sector for a 1°-sampled scan).

    Returns
    -------
    numpy.ndarray, shape (nrays,)
        Smoothed echo-top height [m] per azimuth.  Rays with no valid
        echo in the entire sector return 0.
    """
    top = np.asarray(ray_top, dtype=np.float64)
    weight = np.asarray(ray_weight, dtype=np.float64)
    nrays = top.shape[0]
    inW = sector_half_width

    # Triangle kernel: values 1/inW … inW/inW … 1/inW (centre = 1.0)
    # Legacy: wg = (inW+1) - abs(inW-i); weightarr[i] = wg / inW
    i_idx = np.arange(2 * inW + 1)
    taper = ((inW + 1) - np.abs(inW - i_idx)) / inW  # shape (sector_size,)
    Wsecsum = taper.sum()

    smoothed = np.zeros(nrays, dtype=np.float64)
    for A in range(nrays):
        # Ray indices for the sector, with circular wrap
        offsets = np.arange(-inW, inW + 1)
        ray_indices = (A + offsets) % nrays
        w_ray = weight[ray_indices]
        h_ray = top[ray_indices]
        # Legacy: Rayval = Wray*highTOP + (1-Wray)*TOPprev
        # Background blending deferred → TOPprev term omitted;
        # rays with weight=0 contribute 0.
        secsum = np.sum(taper * w_ray * h_ray)
        # Normalise by the same Wsecsum used in the legacy code
        smoothed[A] = secsum / Wsecsum

    return smoothed
