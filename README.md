# detectability

Computes radar precipitation detectability fields from polar echo-top (HGHT)
ODIM HDF5 data. For each azimuth, the output encodes the range interval over
which the lowest sweep transitions from full beam filling to full beam
overshoot. Single-radar products can be merged into a national composite.

Python replacement for the legacy C/shell implementation in [`legacy/`](legacy/).

## Algorithm

1. **Filter** — remove isolated azimuthal spikes (clutter, birds, aircraft)
   from the echo-top field.
2. **Ray TOP** — per ray, pick a representative echo top from the highest
   bins (`highpart`, default top 10%) within the analysis window
   (`min_range_km`–`max_range_km`, default 10–240 km). The fraction of those
   bins with echo is the ray weight.
3. **Sector smoothing** — azimuthally smooth ray TOPs with a triangular
   kernel (default 61°), blending low-weight rays toward a
   [background echo top](#background-echo-top-and-clear-sky).
4. **Detection ranges** — using 4/3 effective earth radius beam geometry,
   find the ranges where the top and bottom edges of the lowest sweep reach
   the smoothed echo top.
5. **Georeferencing** — regrid from polar to a projected CRS and write a COG.

## Output

A single-band `uint8` Cloud Optimized GeoTIFF in a projected CRS (default
EPSG:3067 / ETRS-TM35FIN), regridded to 500 m cells by nearest neighbour.

The pixel value is the **beam overshoot** of the lowest sweep: the percentage
of the 3 dB beam height lying above the smoothed echo top (band description
`beam overshoot`, unit `%`):

| Value     | Meaning                                                    |
| --------- | ---------------------------------------------------------- |
| `0`       | Full beam filling — precipitation is fully detected        |
| `1`–`99`  | Partial overshoot                                          |
| `100`     | Total overshoot — the whole beam is above the echo top     |
| `255`     | Nodata — outside the radar's coverage                      |

Values `101`–`254` are reserved. Per bin, overshoot is
`(h_top − TOP) / (h_top − h_bot)`, clipped to 0–1, where `h_top` and `h_bot`
are the heights of the beam's upper and lower edges.

The legacy product was a linear-in-range ramp from `0` to `255` between the
same endpoints (beam top / beam bottom reaching TOP); the geometric fraction
is up to ~12 percentage points higher in between.

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

Keep the state file on persistent storage that is not cleaned up sooner than
48 h after the last write. Deleting it earlier skips aging and drops the
background straight to climatology (logged as a warning). Use one state file
per radar, and don't run backfills concurrently with live runs that share it.

### Metadata tags

Single-radar GeoTIFFs carry provenance tags:

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

### Composite

`composite_min` merges single-radar products onto a fixed grid (default: the
Finrad extent in EPSG:3067) by pixel-wise minimum, i.e. the best
detectability wins. Cells covered by no radar are nodata (`255`).

## Usage

```python
from detectability import composite_min, process

process(
    "202603191120_fivim_etop_-10_dbzh_polar_qc.h5",
    "fivim_detectability.tif",
    lowest_elevation=0.3,              # degrees
    state_path="state/fivim.json",     # optional; see above
)

composite_min(["fivim_detectability.tif", ...], "composite.tif")
```

The beamwidth is read from the input file (`/how` `beamwH`) unless given as
`beamwidth=`. Other tuning parameters and their defaults are documented in
[`defaults.py`](src/detectability/defaults.py).

## Install

```bash
pip install .
```

Requires Python ≥ 3.14.

## Development

```bash
hatch run test:run      # pytest
hatch run types:check   # mypy
```

## Container

A two-stage [`Containerfile`](Containerfile) builds the image used by Airflow
`@task.docker` tasks (published as `quay.io/fmi/radar_detectability:vX.Y.Z`).
It has no entrypoint:

```bash
podman build -t detectability .
podman run detectability python -c 'from detectability import process; process(...)'
```

## License

MIT
