# detectability

Computes radar precipitation detectability fields from polar echotop (HGHT) ODIM HDF5 data. Outputs a Cloud Optimized GeoTIFF encoding, for each azimuth, the range interval over which the lowest radar sweep transitions from full beam overshoot to full beam filling.

## Algorithm

1. **Filter** — remove point-like azimuthal artefacts (clutter, birds, aircraft) from the raw echotop field.
2. **Ray-top analysis** — for each radial, derive a representative echo-top height from the upper decile of range bins within the analysis window.
3. **Sector smoothing** — azimuthally smooth ray tops with a linearly-weighted sliding window to suppress directional noise.
4. **Detection ranges** — using standard beam propagation geometry, compute the ranges at which the bottom and top edges of the lowest sweep align with the smoothed echo top. These define the partial-detection transition zone.
5. **Georeferencing** — reproject from polar to a projected CRS and write a COG.

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
