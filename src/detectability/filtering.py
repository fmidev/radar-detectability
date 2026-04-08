# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Azimuthal spike filter for polar echo-top fields.

Corresponds to ``azimuthal_filter.c`` in the legacy pipeline, applied
here to the TOP height field before detection range computation rather
than to the byte-encoded output.
"""

import numpy as np
import numpy.typing as npt


def azimuthal_filter(top: npt.ArrayLike) -> np.ndarray:
    """Remove isolated azimuthal spikes from a polar echo-top field.

    Ports the logic of ``azimuthal_filter.c``: for each bin, zero it if
    both azimuthal neighbours are zero, or if its value exceeds the sum
    of its neighbours.  Wraps circularly at azimuth 0 / N-1.

    Undetect values (≤ 0) are treated as "no echo" (0) on input so that
    the filter threshold is computed against valid echoes only.

    Parameters
    ----------
    top
        2-D array of echo-top heights [m], shape (nrays, nbins).
        Typically the ``HGHT`` variable from :func:`~detectability.io.read_echotop`
        with undetect (small negative) values still present.

    Returns
    -------
    numpy.ndarray
        Filtered echo-top heights [m], same shape, dtype float64.
        Undetect and removed spike bins are set to 0.
    """
    arr = np.asarray(top, dtype=np.float64)
    # Treat undetect (≤ 0) as 0; legacy filter operates on values where
    # "zero" means "no echo".
    arr = np.where(arr > 0, arr, 0.0)

    # Circular neighbours along azimuth axis (axis=0)
    prev = np.roll(arr, 1, axis=0)   # arr[A-1, B], wraps 0 → N-1
    nxt = np.roll(arr, -1, axis=0)   # arr[A+1, B], wraps N-1 → 0

    # Legacy: if(!(ob | nb) && b) outarr[A][B]=0;
    both_zero = (prev == 0) & (nxt == 0)
    # Legacy: if(b > (ob+nb)) outarr[A][B]=0;
    exceeds_sum = arr > (prev + nxt)

    return np.where(both_zero | exceeds_sum, 0.0, arr)
