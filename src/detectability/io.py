# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""I/O for ODIM HDF5 echotop products."""

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RadarMetadata:
    """Radar site and scan metadata from ODIM HDF5."""

    latitude: float  # degrees
    longitude: float  # degrees
    height: float  # m ASL
    beamwidth_h: float  # degrees
    beamwidth_v: float  # degrees


def read_echotop(path: str | Path) -> tuple[xr.DataArray, RadarMetadata]:
    """Read pre-computed polar echotop from ODIM HDF5.

    Extracts the HGHT quantity and radar metadata.

    Parameters
    ----------
    path
        Path to ODIM HDF5 echotop product.

    Returns
    -------
    top : xr.DataArray
        Echo-top heights [m] on (azimuth, range) grid.
        Masked pixels (nodata/undetect) are set to 0.
    metadata : RadarMetadata
        Radar site metadata.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        where = f["where"].attrs
        meta = RadarMetadata(
            latitude=float(where["lat"]),
            longitude=float(where["lon"]),
            height=float(where["height"]),
            beamwidth_h=float(f["how"].attrs["beamwH"]),
            beamwidth_v=float(f["how"].attrs.get("beamwV", f["how"].attrs["beamwH"])),
        )

        ds_where = f["dataset1/where"].attrs
        nbins = int(ds_where["nbins"])
        nrays = int(ds_where["nrays"])
        rscale = float(ds_where["rscale"])
        rstart = float(ds_where.get("rstart", 0.0))

        # Find HGHT data group
        hght_group = _find_quantity(f, "dataset1", "HGHT")
        if hght_group is None:
            raise ValueError(f"No HGHT quantity found in {path}")

        what = f[f"{hght_group}/what"].attrs
        gain = float(what["gain"])
        offset = float(what["offset"])
        nodata = float(what["nodata"])
        undetect = float(what["undetect"])
        raw = f[f"{hght_group}/data"][:]

    # Convert raw → physical, mask nodata/undetect
    raw_f = raw.astype(np.float64)
    mask = (raw_f == nodata) | (raw_f == undetect)
    top_m = raw_f * gain + offset
    top_m[mask] = 0.0

    azimuth = np.arange(nrays, dtype=np.float64)
    range_m = rstart * 1000.0 + (np.arange(nbins) + 0.5) * rscale

    da = xr.DataArray(
        top_m,
        dims=["azimuth", "range"],
        coords={"azimuth": azimuth, "range": range_m},
        attrs={"units": "m", "long_name": "echo_top_height"},
    )
    logger.info(
        "Read echotop %s: %d rays × %d bins, rscale=%.0f m",
        path.name,
        nrays,
        nbins,
        rscale,
    )
    return da, meta


def _find_quantity(f: h5py.File, dataset: str, quantity: str) -> str | None:
    """Find the data group with a given quantity within a dataset."""
    for i in range(1, 100):
        key = f"{dataset}/data{i}"
        if key not in f:
            break
        raw_q = f[f"{key}/what"].attrs.get("quantity", b"")
        q = raw_q.decode() if isinstance(raw_q, bytes) else str(raw_q)
        if q == quantity:
            return key
    return None
