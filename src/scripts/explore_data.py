# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Explore ODIM HDF5 test data: volume scan and echotop product."""

from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"
VOLUME_FILE = DATA_DIR / "202603191120_fivim_volume_ABCDEF_qc.h5"
ETOP_FILE = DATA_DIR / "202603191120_fivim_etop_-10_dbzh_polar_qc.h5"


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def explore_odim_group(f: h5py.File) -> None:
    """Print ODIM root-level metadata."""
    for attr_group in ("what", "where", "how"):
        if attr_group in f:
            grp = f[attr_group]
            print(f"\n  /{attr_group} attributes:")
            for k, v in grp.attrs.items():
                val = v.decode() if isinstance(v, bytes) else v
                print(f"    {k}: {val}")


def explore_datasets(f: h5py.File, max_datasets: int = 30) -> dict:
    """Enumerate dataset groups, print geometry and quantities."""
    datasets = sorted(
        [k for k in f.keys() if k.startswith("dataset")],
        key=lambda x: int(x.replace("dataset", "")),
    )
    print(f"\n  Total dataset groups: {len(datasets)}")

    summary: dict = {"datasets": []}
    for ds_name in datasets[:max_datasets]:
        ds = f[ds_name]
        info: dict = {"name": ds_name}

        # where: geometry
        if "where" in ds:
            w = ds["where"]
            for k in ("elangle", "nbins", "nrays", "rscale", "rstart", "a1gate"):
                if k in w.attrs:
                    val = w.attrs[k]
                    info[k] = float(val) if isinstance(val, (np.floating, float)) else int(val)
            geom_str = ", ".join(f"{k}={info.get(k)}" for k in ("elangle", "nrays", "nbins", "rscale", "rstart") if k in info)
            print(f"\n  {ds_name}: {geom_str}")

        # what
        if "what" in ds:
            for k, v in ds["what"].attrs.items():
                val = v.decode() if isinstance(v, bytes) else v
                info[f"what_{k}"] = val
                print(f"    what/{k}: {val}")

        # data groups: quantities and statistics
        data_groups = sorted([k for k in ds.keys() if k.startswith("data")])
        for dg_name in data_groups:
            dg = ds[dg_name]
            if "what" not in dg:
                continue
            dw = dg["what"]
            quantity = dw.attrs.get("quantity", b"?")
            if isinstance(quantity, bytes):
                quantity = quantity.decode()
            gain = dw.attrs.get("gain", None)
            offset = dw.attrs.get("offset", None)
            nodata = dw.attrs.get("nodata", None)
            undetect = dw.attrs.get("undetect", None)

            # Read data array
            if "data" in dg:
                raw = dg["data"][:]
                dtype = raw.dtype
                shape = raw.shape

                # Compute physical values excluding nodata/undetect
                mask = np.ones(raw.shape, dtype=bool)
                if nodata is not None:
                    mask &= raw != nodata
                if undetect is not None:
                    mask &= raw != undetect
                valid = raw[mask]

                phys_stats = ""
                if gain is not None and offset is not None and valid.size > 0:
                    phys = valid.astype(np.float64) * gain + offset
                    phys_stats = (
                        f", phys range: [{phys.min():.2f}, {phys.max():.2f}], "
                        f"mean={phys.mean():.2f}, std={phys.std():.2f}"
                    )

                coverage = valid.size / raw.size * 100 if raw.size > 0 else 0
                print(
                    f"    {dg_name}: {quantity}, dtype={dtype}, shape={shape}, "
                    f"gain={gain}, offset={offset}, nodata={nodata}, undetect={undetect}"
                )
                print(
                    f"      raw range: [{raw.min()}, {raw.max()}], "
                    f"valid pixels: {valid.size}/{raw.size} ({coverage:.1f}%)"
                    f"{phys_stats}"
                )

                info[f"{dg_name}_quantity"] = quantity
                info[f"{dg_name}_shape"] = shape
                info[f"{dg_name}_coverage_pct"] = coverage

        summary["datasets"].append(info)
    return summary


def assess_echotop_applicability(f: h5py.File) -> None:
    """Assess whether echotop product is usable as input for detection range analysis."""
    print_header("ECHOTOP APPLICABILITY ASSESSMENT")

    datasets = [k for k in f.keys() if k.startswith("dataset")]
    if not datasets:
        print("  No datasets found!")
        return

    ds = f[datasets[0]]
    data_groups = [k for k in ds.keys() if k.startswith("data")]

    for dg_name in data_groups:
        dg = ds[dg_name]
        if "what" not in dg or "data" not in dg:
            continue
        dw = dg["what"]
        quantity = dw.attrs.get("quantity", b"?")
        if isinstance(quantity, bytes):
            quantity = quantity.decode()

        raw = dg["data"][:]
        gain = float(dw.attrs.get("gain", 1))
        offset = float(dw.attrs.get("offset", 0))
        nodata = dw.attrs.get("nodata", None)
        undetect = dw.attrs.get("undetect", None)

        mask = np.ones(raw.shape, dtype=bool)
        if nodata is not None:
            mask &= raw != nodata
        if undetect is not None:
            mask &= raw != undetect
        valid = raw[mask]

        if valid.size == 0:
            print(f"  {quantity}: no valid data")
            continue

        phys = valid.astype(np.float64) * gain + offset
        nrays = raw.shape[0]

        print(f"\n  Quantity: {quantity}")
        print(f"    Grid shape: {raw.shape} (nrays × nbins)")
        print(f"    Physical unit implied by scaling: gain={gain}, offset={offset}")
        print(f"    Value range: [{phys.min():.2f}, {phys.max():.2f}] (physical)")
        print(f"    Data coverage: {valid.size}/{raw.size} ({valid.size/raw.size*100:.1f}%)")

        # Per-ray coverage (fraction of bins with valid TOP)
        ray_valid = mask.sum(axis=1)
        ray_total = raw.shape[1]
        ray_coverage = ray_valid / ray_total
        rays_with_data = (ray_valid > 0).sum()
        print(f"    Rays with any data: {rays_with_data}/{nrays} ({rays_with_data/nrays*100:.1f}%)")
        print(f"    Per-ray coverage: min={ray_coverage.min():.3f}, max={ray_coverage.max():.3f}, "
              f"mean={ray_coverage.mean():.3f}")

        # Check if data looks like TOP height (expecting values in hundreds/thousands of meters)
        if phys.max() > 50:
            print(f"    ✓ Values look like heights (max={phys.max():.0f}), likely meters or 100s of meters")
        else:
            print(f"    ⚠ Values are small (max={phys.max():.2f}), may be in km or scaled units")

        # Percentile distribution
        pcts = np.percentile(phys, [10, 25, 50, 75, 90, 95, 99])
        print(f"    Percentiles [10,25,50,75,90,95,99]: {[f'{p:.1f}' for p in pcts]}")

    # Check geometry from dataset-level or root-level where
    ds = f[datasets[0]]
    where = ds.get("where")
    root_where = f.get("where")

    if where is not None:
        # Polar product: dataset-level geometry
        nbins = int(where.attrs.get("nbins", 0))
        nrays = int(where.attrs.get("nrays", 0))
        rscale = float(where.attrs.get("rscale", 0))
        rstart = float(where.attrs.get("rstart", 0))
        elangle = where.attrs.get("elangle", None)

        print(f"\n  Geometry (polar, dataset-level):")
        print(f"    nrays={nrays}, nbins={nbins}, rscale={rscale}m, rstart={rstart}km")
        print(f"    Max range: {rstart + nbins * rscale / 1000:.1f} km")
        if elangle is not None:
            print(f"    Elevation angle: {float(elangle):.2f}°")
            print("    ⚠ Single elevation → this is a 2D product, not a volume scan")

    if root_where is not None:
        # Cartesian or composite product: root-level geometry
        xsize = root_where.attrs.get("xsize", None)
        ysize = root_where.attrs.get("ysize", None)
        xscale = root_where.attrs.get("xscale", None)
        yscale = root_where.attrs.get("yscale", None)
        projdef = root_where.attrs.get("projdef", None)
        epsg = root_where.attrs.get("EPSG", None)
        if isinstance(projdef, bytes):
            projdef = projdef.decode()

        if xsize is not None:
            print(f"\n  Geometry (root-level, Cartesian product):")
            print(f"    xsize={xsize}, ysize={ysize}, xscale={xscale}m, yscale={yscale}m")
            if projdef:
                print(f"    projdef: {projdef}")
            if epsg is not None:
                print(f"    EPSG: {int(epsg)}")
            ll_lat = root_where.attrs.get("LL_lat", None)
            ll_lon = root_where.attrs.get("LL_lon", None)
            ur_lat = root_where.attrs.get("UR_lat", None)
            ur_lon = root_where.attrs.get("UR_lon", None)
            if ll_lat is not None:
                print(f"    Bounding box: ({ll_lat:.3f}, {ll_lon:.3f}) to ({ur_lat:.3f}, {ur_lon:.3f})")
            bbox_native = root_where.attrs.get("BBOX_native", None)
            if bbox_native is not None:
                print(f"    BBOX_native (projected): {bbox_native}")

    obj_type = f["what"].attrs.get("object", b"").decode() if "what" in f else ""
    product = ""
    if "what" in ds:
        product = ds["what"].attrs.get("product", b"")
        if isinstance(product, bytes):
            product = product.decode()

    print(f"\n  ASSESSMENT:")
    if obj_type in ("COMP", "IMAGE") or (xsize is not None if root_where is not None else False):
        print(f"    This echotop product is a CARTESIAN (Composite/Image) grid, NOT polar.")
        print(f"    Object type: {obj_type}, Product: {product}")
        print(f"    It has already been projected to EPSG:{int(epsg) if epsg else '?'} at {xscale}×{yscale}m resolution.")
        print(f"    ")
        print(f"    For our detection range method, we need polar (azimuth × range) TOP data,")
        print(f"    because the analysis operates per-ray (per-azimuth) and computes ranges")
        print(f"    along the beam. A Cartesian echotop product is NOT directly usable because:")
        print(f"      - The ray/range structure is lost after reprojection")
        print(f"      - Per-ray TOP sampling (HIGHPART/SAMPLEPOINT) needs polar coordinates")
        print(f"      - Detection range encoding references beam geometry along radials")
        print(f"    ")
        print(f"    Options:")
        print(f"      a) Compute echotop from volume scans ourselves (steps 3-4), staying in polar coords")
        print(f"      b) Find/generate polar echotop products (PGM or ODIM polar) upstream")
        print(f"      c) Reverse-project Cartesian→polar (lossy, not recommended)")
    else:
        print(f"    The echotop product is a 2D polar field (azimuth × range) of echo-top heights.")
        print(f"    This is exactly the intermediate product that the legacy pipeline computes")
        print(f"    internally via radial_echotop_h5.c before feeding it to the detection range")
        print(f"    analysis in analyse_top_for_detection_range.c.")
        print(f"    → Can potentially be used DIRECTLY as input to the analysis step (step 6),")
        print(f"      skipping steps 3-4 (volume I/O + echotop computation) when pre-computed")
        print(f"      echotop products are available.")
        print(f"    Key checks needed:")
        print(f"      - Physical units: are values in meters, 100s of meters, or km?")
        print(f"      - Does the encoding match what analyse_top_for_detection_range.c expects?")
        print(f"        (legacy encodes TOP as (pixel−1)×100 meters)")


def main() -> None:
    for filepath, label in [
        (VOLUME_FILE, "VOLUME SCAN"),
        (ETOP_FILE, "ECHOTOP PRODUCT"),
    ]:
        print_header(f"{label}: {filepath.name}")
        if not filepath.exists():
            print(f"  FILE NOT FOUND: {filepath}")
            continue

        with h5py.File(filepath, "r") as f:
            explore_odim_group(f)
            explore_datasets(f)

    # Echotop applicability assessment
    if ETOP_FILE.exists():
        with h5py.File(ETOP_FILE, "r") as f:
            assess_echotop_applicability(f)


if __name__ == "__main__":
    main()
