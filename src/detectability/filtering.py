# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Azimuthal spike filtering for polar detectability grids.

Implements the post-encoding filter from ``azimuthal_filter.c``.
"""

import numpy as np
import numpy.typing as npt


def azimuthal_filter(data: npt.ArrayLike) -> np.ndarray:
    """Remove isolated azimuthal spikes from a polar grid.

    Zeroes out bins where the centre value exceeds the sum of its
    circular azimuthal neighbours.

    Corresponds to ``azimuthal_filter.c``::

        if(b > (ob+nb)) outarr[A][B] = 0;

    Parameters
    ----------
    data
        Polar grid (azimuth × range), typically uint8.

    Returns
    -------
    ndarray
        Filtered grid, same shape and dtype.
    """
    data = np.asarray(data)
    out = data.copy()

    prev = np.roll(data, 1, axis=0)
    next_ = np.roll(data, -1, axis=0)

    # Use int16 to avoid uint8 overflow in addition
    neighbor_sum = prev.astype(np.int16) + next_.astype(np.int16)
    exceeds = data.astype(np.int16) > neighbor_sum

    out[exceeds] = 0
    return out
