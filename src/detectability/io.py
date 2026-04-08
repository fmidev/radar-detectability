# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Read pre-computed polar echotop products in ODIM HDF5 format.

Uses xradar for reading; h5py is used only to extract instrument
metadata (beamwidth) that xradar does not expose.
"""

import logging
from pathlib import Path

import h5py
import xarray as xr
import xradar as xd

logger = logging.getLogger(__name__)

#: ODIM quantity name for echo-top height.
_HGHT_QUANTITY = "HGHT"


def read_echotop(path: str | Path) -> xr.Dataset:
    """Read a pre-computed polar echotop product (ODIM HDF5).

    Returns the xradar sweep Dataset as-is, augmented with instrument
    metadata from the ODIM ``/how`` group that xradar does not expose.

    Parameters
    ----------
    path
        Path to the ODIM HDF5 file containing a polar echotop product
        with the ``HGHT`` quantity.

    Returns
    -------
    xarray.Dataset
        xradar sweep Dataset (``sweep_0``), with ``HGHT`` [m] on
        (azimuth, range) dims.  Radar site coordinates (``longitude``,
        ``latitude``, ``altitude``) are xradar-native sweep coordinates.
        If the file's ``/how`` group contains ``beamwH`` / ``beamwV``,
        they are added as ``beamwidth_h`` / ``beamwidth_v`` attrs [deg].

    Notes
    -----
    Undetect pixels are kept at their decoded value (``HGHT._Undetect``,
    typically a small negative number).  Downstream code is responsible
    for zeroing or masking them.
    """
    path = Path(path)
    dt = xd.io.open_odim_datatree(str(path))
    ds = dt["sweep_0"].ds

    if _HGHT_QUANTITY not in ds:
        raise ValueError(
            f"HGHT quantity not found in {path.name}; "
            f"available variables: {list(ds.data_vars)}"
        )

    how_attrs = _read_how_attrs(path)
    if how_attrs:
        ds = ds.assign_attrs(how_attrs)

    logger.debug(
        "Read echotop %s: %dx%d, site (%.4fE, %.4fN, %.0f m)",
        path.name,
        ds.sizes["azimuth"],
        ds.sizes["range"],
        float(ds.coords["longitude"].values),
        float(ds.coords["latitude"].values),
        float(ds.coords["altitude"].values),
    )
    return ds


def _read_how_attrs(path: Path) -> dict[str, float]:
    """Extract instrument metadata from the ODIM ``/how`` group.

    Returns a dict that may contain ``beamwidth_h`` and ``beamwidth_v``
    (degrees).  Missing attributes are silently omitted.
    """
    result: dict[str, float] = {}
    try:
        with h5py.File(path, "r") as f:
            how = f.get("how")
            if how is None:
                return result
            for odim_key, our_key in (
                ("beamwH", "beamwidth_h"),
                ("beamwV", "beamwidth_v"),
            ):
                val = how.attrs.get(odim_key)
                if val is not None:
                    result[our_key] = float(val)
    except OSError:
        logger.warning("Could not read /how attrs from %s", path)
    return result
