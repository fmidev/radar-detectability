# detectability

Computes radar precipitation detectability fields from polar echotop (HGHT) ODIM HDF5 data. Outputs a Cloud Optimized GeoTIFF encoding, for each azimuth, the range interval over which the lowest radar sweep transitions from full beam overshoot to full beam filling.

## Algorithm

1. **Filter** — remove point-like azimuthal artefacts (clutter, birds, aircraft) from the raw echotop field.
2. **Ray-top analysis** — for each radial, derive a representative echo-top height from the upper decile of range bins within the analysis window.
3. **Sector smoothing** — azimuthally smooth ray tops with a linearly-weighted sliding window to suppress directional noise.
4. **Detection ranges** — using standard beam propagation geometry, compute the ranges at which the bottom and top edges of the lowest sweep align with the smoothed echo top. These define the partial-detection transition zone.
5. **Georeferencing** — reproject from polar to a projected CRS and write a COG.

## Output

The product is a single-band, 8-bit Cloud Optimized GeoTIFF in a projected
CRS (default EPSG:3067 / ETRS-TM35FIN), regridded from polar coordinates to a
regular grid (default 500 m cells) by nearest-neighbour interpolation.

The pixel value encodes the **degree of beam overshoot** of the lowest sweep
on a linear `uint8` ramp:

| Value          | Meaning                                                               |
| -------------- | --------------------------------------------------------------------- |
| `0`            | Full beam filling — precipitation at this point is fully detected     |
| `1`–`254`      | Partial detection — within the transition zone, increasing overshoot  |
| `255`          | Total overshoot — the beam passes entirely above the echo top         |
| `255` (nodata) | Beyond `max_range_m`, where set, pixels are masked to nodata          |

For each azimuth the ramp runs from `0` at the range where the beam *top* edge
reaches the smoothed echo top (full filling) to `255` at the range where the
beam *bottom* edge reaches it (zero filling). Note that `255` serves both as
the "total overshoot" value and, when `max_range_m` is supplied, as the GeoTIFF
nodata value for out-of-range pixels.

## Usage

```python
from detectability import process

process(
    "vantaa_echotop_202601010000.h5",
    "vantaa_detectability.tif",
    lowest_elevation=0.5,   # degrees
    beamwidth=1.0,          # degrees; read from file if omitted
    crs="EPSG:3067",
)
```

## Install

```bash
pip install .
```

Requires Python ≥ 3.14 and GDAL/GEOS/PROJ system libraries (for rioxarray).

## Container

A two-stage `Containerfile` is provided for use as an Airflow `@task.docker` worker:

```bash
podman build -t detectability .
podman run detectability python -c 'from detectability import process; process(...)'
```

## License

MIT
