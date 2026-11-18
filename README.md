# geohierarchy

A high-performance hierarchical geospatial data management library, built on
Polars and GeoPandas.

`geohierarchy` manages a **graph of spatial levels** (e.g. county → tract →
block group → block, or any other set of nested or overlapping geometries)
and automatically fills in shared attribute columns across every level,
using the correct spatial resampling strategy in each direction (upscale
aggregation like `Sum`/`Mean`/`Max`/`Min`, or downscale broadcast/split).

## Install

```bash
pip install -e .
```

or, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Requires Python >= 3.10.

## Quickstart

```python
from geohierarchy import GeoHierarchy, Sum, Mean

gh = GeoHierarchy(crs="EPSG:4326")

gh.add_level("county", county_gdf, id_col="GEOID", agg=Sum())
gh.add_level("tract", tract_gdf, id_col="GEOID", agg=Sum(), parent="county")
gh.add_level("block", block_gdf, id_col="GEOID", agg=Sum(), parent="tract")

# Bring in an outside dataset that isn't part of the hierarchy.
gh.add_vector_data(streets_gdf, level="tract", agg=Mean(weight_column="population_total"))

gh["tract"]   # GeoDataFrame: geometry + every column, filled in at every level -- no propagate() call needed
```

See [`docs/guide.md`](docs/guide.md) for the full concept guide and API reference
(levels, aggregation strategies, H3/raster ingestion, non-layer data, and
more), and [`examples/geohierarchy_walkthrough.ipynb`](examples/geohierarchy_walkthrough.ipynb)
for a worked, narrated example.

Interactive map rendering (`geohierarchy.maps`) is an optional extra built
on `folium`/`branca` and is not imported by plain `import geohierarchy`.

## Tests

```bash
pytest
```

Test fixtures live in `tests/test_files/`. A few large fixtures
(`accessibility_streets.gpkg`, `block.gpkg`, `blockgroup.gpkg`) are not
committed to the repo -- tests that depend on them are skipped if the files
aren't present locally.

## Development

```bash
uv sync   # installs dev dependencies (pytest, etc.) declared under [tool.uv]
pre-commit install
```

`pre-commit` runs `ruff` (lint + format) and basic hygiene checks
(trailing whitespace, large-file check: see "Large files" below) on every commit.

## Use as a library

geohierarchy has no implicit cache location: the id-mapping cache functions in `geohierarchy.spatial_cache`
(`load_mapping`, `save_mapping`, `get_id_mapping`, ...) take an explicit `cache_dir`, and nothing is written to the
current working directory otherwise. Pass a persistent folder to reuse mappings across runs:

```python
from geohierarchy.spatial_cache import get_id_mapping
mapping = get_id_mapping(..., cache_dir="/data/my_project/cache/geohierarchy")
```

## Large files

A pre-commit hook (`tools/check_large_files.py`, installed with `python3 tools/check_large_files.py --install`) refuses a commit with a file over 10 MB once: run the same commit again to keep it. Files over 50 MB are never committed.

## Local files

This repository is public. API keys (`api_keys.json`, `.env`), notes about a particular machine or its processes (hosting, uploads,
local paths) and upload scripts are never committed: keep notes as `*.local.md` or under `local/`, which `.gitignore` excludes.
