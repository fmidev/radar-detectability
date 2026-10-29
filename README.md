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

### Background echo top and clear sky

Rays with little or no echo are blended with a **background echo top**, so
that clear-sky output stays meaningful. The background is always defined:

- **`state`** — the last valid observed TOP from the state file (`state_path`),
  if it is no older than the grace period (2.25 h).
- **`aged_state`** — an older state, converging linearly toward climatology
  over 48 h.
- **`climatology`** — `climatology_top_km` (default 5.5 km) when there is no
  usable state, including when `state_path` is not given.

State ages are measured from the input's ODIM nominal time (`/what` `date` +
`time`), not wall-clock time, so retries and backfills give reproducible
results. Older data never overwrites the state saved from newer data.

### Metadata tags

The GeoTIFF carries provenance tags:

| Tag                    | Meaning                                                                      |
| ---------------------- | ---------------------------------------------------------------------------- |
| `nominal_time`         | ODIM nominal time of the input (ISO 8601, UTC)                               |
| `background_top_km`    | Background echo top used for blending [km]                                   |
| `background_source`    | `state`, `aged_state` or `climatology`                                       |
| `background_age_hours` | Age of the state relative to `nominal_time` (only if a state was available)  |
| `echo_coverage`        | Mean ray weight, 0–1 (legacy `TOP_PERCENTAGE` / 100)                         |
| `valid_ray_fraction`   | Fraction of rays fully covered by echo (weight ≥ 0.99)                       |
| `lowest_elevation_deg`, `beamwidth_deg`, `sitealt_m`, `range_resolution_m` | Beam geometry used |

A low `echo_coverage` means the product relies mostly on the background TOP.

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
