# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Read pre-computed polar echotop products in ODIM HDF5 format.

Uses xradar for reading; h5py is used only to extract metadata
(nominal time, source, beamwidth) that xradar does not expose.
"""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import h5py
import xarray as xr
import xradar as xd

logger = logging.getLogger(__name__)

#: ODIM quantity name for echo-top height.
_HGHT_QUANTITY = "HGHT"


def read_echotop(path: str | Path) -> xr.Dataset:
    """Read a pre-computed polar echotop product (ODIM HDF5).

    Returns the xradar sweep Dataset as-is, augmented with metadata from
    the ODIM ``/what`` and ``/how`` groups that xradar does not expose.

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
        ``latitude``, ``altitude``) are attached as scalar sweep coordinates.
        If the file's ``/how`` group contains ``beamwH`` / ``beamwV``,
        they are added as ``beamwidth_h`` / ``beamwidth_v`` attrs [deg].
        The ODIM nominal time (``/what`` ``date`` + ``time``) is added as
        a timezone-aware UTC ``datetime`` attr ``nominal_time``, if present.
        The ODIM ``/what`` ``source`` string is added as attr ``source``
        and its ``NOD`` identifier (e.g. ``fivim``) as attr ``radar``.

    Notes
    -----
    Undetect pixels are kept at their decoded value (``HGHT._Undetect``,
    typically a small negative number).  Downstream code is responsible
    for zeroing or masking them.
    """
    path = Path(path)
    dt = xd.open_datatree(path, engine="odim")
    ds = dt["sweep_0"].ds

    if _HGHT_QUANTITY not in ds:
        raise ValueError(
            f"HGHT quantity not found in {path.name}; "
            f"available variables: {list(ds.data_vars)}"
        )

    for coord in ("longitude", "latitude", "altitude"):
        if coord not in ds.coords and coord in dt.ds.coords:
            ds = ds.assign_coords({coord: dt.ds.coords[coord]})

    extra_attrs = _read_odim_attrs(path)
    if extra_attrs:
        ds = ds.assign_attrs(extra_attrs)

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


def _read_odim_attrs(path: Path) -> dict[str, str | float | datetime]:
    """Extract metadata from the ODIM ``/what`` and ``/how`` groups.

    Returns a dict that may contain ``nominal_time`` (UTC datetime),
    ``source`` / ``radar`` (ODIM source string and its NOD) and
    ``beamwidth_h`` / ``beamwidth_v`` (degrees).  Missing or malformed
    attributes are omitted.
    """
    result: dict[str, str | float | datetime] = {}
    try:
        with h5py.File(path, "r") as f:
            what = f.get("what")
            if what is not None:
                nominal_time = _parse_nominal_time(
                    what.attrs.get("date"), what.attrs.get("time")
                )
                if nominal_time is not None:
                    result["nominal_time"] = nominal_time
                if (source_attr := what.attrs.get("source")) is not None:
                    source = result["source"] = _decode(source_attr)
                    nod = _parse_nod(source)
                    if nod is not None:
                        result["radar"] = nod
            how = f.get("how")
            if how is not None:
                for odim_key, our_key in (
                    ("beamwH", "beamwidth_h"),
                    ("beamwV", "beamwidth_v"),
                ):
                    val = how.attrs.get(odim_key)
                    if val is not None:
                        result[our_key] = float(val)
    except OSError:
        logger.warning("Could not read /what and /how attrs from %s", path)
    return result


def _parse_nominal_time(date: object, time: object) -> datetime | None:
    """Parse ODIM ``YYYYMMDD`` + ``HHmmss`` attributes into a UTC datetime."""
    if date is None or time is None:
        return None
    date_str, time_str = _decode(date), _decode(time)
    # Check the format strictly: strptime accepts e.g. "20260319" + "99"
    if re.fullmatch(r"\d{8}", date_str) and re.fullmatch(r"\d{6}", time_str):
        try:
            return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S").replace(
                tzinfo=UTC
            )
        except ValueError:
            pass
    logger.warning("Malformed ODIM nominal time: date=%r time=%r", date, time)
    return None


def _parse_nod(source: str) -> str | None:
    """Extract the ``NOD`` identifier from an ODIM source string.

    E.g. ``"WMO:02925,RAD:FI49,PLC:Vimpeli,NOD:fivim"`` -> ``"fivim"``.
    """
    for item in source.split(","):
        key, _, value = item.partition(":")
        if key.strip() == "NOD" and value.strip():
            return value.strip()
    logger.warning("No NOD identifier in ODIM source: %r", source)
    return None


def _decode(value: object) -> str:
    """HDF5 attribute value as ``str``."""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)
