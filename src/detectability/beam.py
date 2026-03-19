# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Radar beam geometry helpers.
"""

import numpy as np
import numpy.typing as npt


def slant_range_from_height(
    height: npt.ArrayLike,
    elevation: npt.ArrayLike,
    sitealt: float,
    *,
    re: float = 6_371_000.0,
    ke: float = 4.0 / 3.0,
) -> np.ndarray:
    """Slant range [m] at which the beam centre reaches *height* [m].

    Analytical inverse of :func:`beam_height`
    (``wradlib.georef.bin_altitude``).

    Solves the Doviak & Zrnić effective-earth-radius equation for *r*:

    .. math::

        h = \\sqrt{r^2 + (k_e r_e)^2 + 2 r\\, k_e r_e \\sin\\theta}
            - k_e r_e + h_0

    giving

    .. math::

        r = -K \\sin\\theta
            + \\sqrt{(h - h_0 + K)^2 - (K \\cos\\theta)^2}

    where :math:`K = k_e r_e`.

    Corresponds to the legacy C function ``bindist()`` in
    ``analyse_top_for_detection_range.c``.

    Parameters
    ----------
    height
        Target altitude(s) [m] above sea level.
    elevation
        Elevation angle(s) [degrees], broadcastable with *height*.
    sitealt
        Radar site altitude [m] above sea level.
    re
        Earth radius [m].
    ke
        Effective earth radius multiplier (default 4/3).

    Returns
    -------
    numpy.ndarray
        Slant range(s) [m].  Returns 0 where the requested height is
        unreachable (discriminant < 0).
    """
    height = np.asarray(height, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=np.float64)
    theta = np.deg2rad(elevation)
    K = ke * re
    H = height - sitealt + K
    discriminant = H**2 - (K * np.cos(theta)) ** 2
    r_raw = -K * np.sin(theta) + np.sqrt(np.maximum(discriminant, 0))
    r = np.where((discriminant > 0) & (r_raw > 0), r_raw, 0.0)
    return r
