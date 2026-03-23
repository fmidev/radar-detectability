# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Echo-top analysis: per-ray picking and sector smoothing.

Implements the TOP analysis logic from ``analyse_top_for_detection_range.c``.
"""

import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve1d


def pick_ray_tops(
    top_2d: npt.ArrayLike,
    *,
    highpart: float = 0.1,
    samplepoint: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Select representative TOP height per azimuth ray.

    For each ray, sorts bins descending, takes the top *highpart*
    fraction, and picks the value at *samplepoint* position within
    the valid (>0) sub-set.

    Corresponds to the per-ray sorting and picking in
    ``analyse_top_for_detection_range.c``::

        qsort(ray, BinCount, sizeof(unsigned char), sort_uchar_desc);
        sortpart_ray = (int)((double)BinCount * sortage);
        picbin = (int)((double)topcount[A] * samplepoint);

    Parameters
    ----------
    top_2d
        Echo-top heights [m], shape (nrays, nbins).
    highpart
        Fraction of bins to consider (legacy: ``sortage``).
    samplepoint
        Position within valid high bins to pick (0 = highest,
        0.5 = median of high part).

    Returns
    -------
    ray_top : ndarray
        Representative TOP height [m] per azimuth, shape (nrays,).
    ray_weight : ndarray
        Confidence weight [0–1] per azimuth, shape (nrays,).
    """
    top = np.asarray(top_2d, dtype=np.float64)
    nrays, nbins = top.shape
    sortpart = max(1, int(nbins * highpart))

    # Sort each ray descending; valid (>0) values come first
    sorted_desc = np.sort(top, axis=1)[:, ::-1]
    high_bins = sorted_desc[:, :sortpart]

    valid_count = np.count_nonzero(high_bins > 0, axis=1)

    # Pick index: samplepoint position within valid bins
    pick_idx = np.where(
        valid_count > 0,
        np.minimum((valid_count * samplepoint).astype(int), valid_count - 1),
        0,
    )

    ray_top = high_bins[np.arange(nrays), pick_idx]
    ray_top[valid_count == 0] = 0.0

    ray_weight = np.clip(valid_count / sortpart, 0.0, 1.0)
    return ray_top, ray_weight


def sector_smooth(
    ray_top: npt.ArrayLike,
    ray_weight: npt.ArrayLike,
    background_top: float,
    *,
    sector_width: int = 60,
) -> np.ndarray:
    """Triangle-weighted circular sector smoothing with background blending.

    For each azimuth, blends the ray TOP with the background based on
    confidence, then applies a normalised triangle-weighted circular
    average over the sector.

    Corresponds to the sector smoothing in
    ``analyse_top_for_detection_range.c``::

        Rayval = Wray*highTOP + (1.0-Wray)*TOPprev;
        Secval = Wsec * Rayval;

    Parameters
    ----------
    ray_top
        Per-ray TOP [m], shape (nrays,).
    ray_weight
        Per-ray confidence [0–1], shape (nrays,).
    background_top
        Background TOP [m] for blending.
    sector_width
        Smoothing sector width [degrees/rays].

    Returns
    -------
    ndarray
        Smoothed TOP [m] per azimuth, shape (nrays,).
    """
    ray_top = np.asarray(ray_top, dtype=np.float64)
    ray_weight = np.asarray(ray_weight, dtype=np.float64)

    # Blend each ray with background based on confidence weight
    blended = ray_weight * ray_top + (1.0 - ray_weight) * background_top

    # Normalised triangle kernel
    half_w = sector_width // 2
    offsets = np.arange(-half_w, half_w + 1)
    kernel = 1.0 - np.abs(offsets) / (half_w + 1)
    kernel /= kernel.sum()

    return convolve1d(blended, kernel, mode="wrap")
