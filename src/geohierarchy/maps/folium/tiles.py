"""Vector tile generation using the fastest available methods.

This module provides the default tile generation implementation:
- freestiler for GeoParquet files (10-100x faster than pure Python)
- PMTiles output for optimal static hosting

For backward compatibility with the old API,
this module wraps the fast implementation from fast_tiles.py.

The old slow implementation (H3J/H3T) is available in h3_slow.py.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import geopandas as gpd
import pandas as pd
import polars as pl

# Import fast tile generation functions
from .fast_tiles import (
    generate_pmtiles_from_geoparquet,
    generate_pmtiles_from_polars,
    generate_xyz_tiles_from_geoparquet,
    build_fast_hierarchy_map,
    HAS_FREESTILER,
)

# Import PMTiles extraction function
from .pmtiles_to_xyz import extract_xyz_from_pmtiles as _extract_xyz_from_pmtiles

# H3J/H3T functions are in h3_slow.py (NOT RECOMMENDED for new code)
# Import them only for backward compatibility
from .h3_slow import (
    generate_h3j_from_h3_indices,
    generate_h3t_tiles_from_h3_indices,
)

# Row-count threshold above which a level is tiled in H3-resolution-4
# chunks (~1,770 km^2 each) instead of a single whole-level freestiler
# call. freestiler's `freestile_file` builds every zoom level for the
# entire input GeoDataFrame in one Rust call, which has repeatedly
# OOM-killed real production builds on our largest levels (e.g. Boston's
# statewide census "block" level, ~103,000 polygons, combined with its
# ~1.6M-row H3 grid, under a 20GB memory cgroup cap). Splitting a large
# level into many small per-chunk `.pmtiles` files keeps each Rust call's
# peak memory bounded by chunk size instead of level size. Tunable; kept
# as a plain module constant rather than a hardcoded literal so it's easy
# to adjust without hunting through the function body.
#
# Small-to-medium levels (the overwhelming common case -- every level in
# every city smaller than Boston) stay well under this threshold and are
# tiled exactly as before, as a single `{level_name}.pmtiles` file.
H3_CHUNK_ROW_THRESHOLD = 150_000

# H3 resolution used to *start* partitioning an oversized level into
# chunks. This is only a starting point, not a guarantee: chunking is now
# adaptive (see `_adaptive_h3_chunk_keys` below) because a FIXED resolution
# does not bound any individual chunk's row count for a geographically
# concentrated dataset. Real 2026-08-30 evidence from Boston's metro H3
# grid (1,133,434 rows, native resolution 11): partitioning by resolution-4
# parent alone produces only 12 chunks, and the densest one (downtown
# Boston) holds 430,653 rows -- nearly 3x `H3_CHUNK_ROW_THRESHOLD` -- so
# `freestile_file` was still being invoked on a chunk far larger than the
# threshold was meant to cap, which is exactly what OOM-killed the tile
# build ("tile pool capped at 1 worker(s)" with no further output). Any
# chunk still over threshold at this resolution is recursively re-split at
# the next finer resolution until it fits (or `H3_CHUNK_MAX_RESOLUTION` is
# reached). For the same Boston data, resolution 5 alone already brings
# the densest chunk down to 86,924 rows (46 chunks total) -- comfortably
# under threshold -- so the recursion in practice only goes one level deep
# for real-world skew.
H3_CHUNK_RESOLUTION = 4

# Hard cap on how deep adaptive chunking will recurse. For H3-grid levels
# (rows carrying an `h3_cell` column) this is moot in practice -- resolution
# is already clamped to the level's own native resolution, at which point
# every "chunk" is a single row. For non-H3 polygon levels (census
# tract/blockgroup/block, chunked by representative-point lat/lng), this
# bounds recursion depth in the pathological case of many rows sharing
# (almost) the same representative point, at the cost of accepting a
# still-oversized chunk rather than recursing forever.
H3_CHUNK_MAX_RESOLUTION = 9


def prep_level_gdf(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    property_cols: Iterable[str],
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """Reproject/trim a level's geometry to the columns needed for tiling.

    This is kept for backward compatibility with the old API.
    For new code, use Polars DataFrames directly with generate_pmtiles_from_polars().
    """
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    keep_cols = [id_col] + [
        c for c in property_cols if c in gdf.columns and c != id_col
    ]
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in gdf.columns]
    work = gdf[[*keep_cols, gdf.geometry.name]].copy()
    if work.geometry.name != "geometry":
        work = work.rename(columns={work.geometry.name: "geometry"}).set_geometry(
            "geometry"
        )

    work = work[work.geometry.notna() & ~work.geometry.is_empty]
    return work, keep_cols


def _h3_keys_at_resolution(gdf: gpd.GeoDataFrame, resolution: int) -> "pl.Series":
    """Return, for every row in ``gdf``, its containing H3 cell at ``resolution``.

    Two cases, matching the two kinds of levels this codebase tiles:

    - H3-grid levels (rows already carry an ``h3_cell`` column, e.g. the
      `h3_9`/`h3_11` levels derived from ``h3_grid.parquet``): the parent
      cell is a cheap direct call, ``h3.cell_to_parent(cell, resolution)``
      -- no spatial op needed. ``resolution`` is clamped to the level's own
      native H3 resolution (can't ask for a "parent" finer than the cell
      itself); every row is then already its own singleton group.
    - Non-H3 polygon levels (census tract/blockgroup/block -- no
      ``h3_cell`` column): each row's representative point is looked up
      with ``h3.latlng_to_cell(lat, lng, resolution)``.

    Returns a string Series of H3 cell ids (hex), aligned with ``gdf``'s
    row order.
    """
    import h3

    if "h3_cell" in gdf.columns:
        cells = gdf["h3_cell"].astype(str).tolist()
        if cells:
            resolution = min(resolution, h3.get_resolution(cells[0]))
        parents = [h3.cell_to_parent(c, resolution) for c in cells]
        return pl.Series("_h3_chunk", parents)

    pts = gdf.geometry.representative_point()
    keys = [h3.latlng_to_cell(pt.y, pt.x, resolution) for pt in pts]
    return pl.Series("_h3_chunk", keys)


def _adaptive_h3_chunk_keys(
    gdf: gpd.GeoDataFrame,
    threshold: int = H3_CHUNK_ROW_THRESHOLD,
    base_resolution: int = H3_CHUNK_RESOLUTION,
    max_resolution: int = H3_CHUNK_MAX_RESOLUTION,
) -> "pd.Series":
    """Assign every row of ``gdf`` a chunk key with at most ``threshold`` rows
    per key, quadtree-style.

    A single fixed H3 resolution (the old behavior) does not bound any
    individual chunk's size for a geographically concentrated dataset --
    e.g. Boston's downtown resolution-4 cell alone held 430,653 rows, ~3x
    `H3_CHUNK_ROW_THRESHOLD`, because the whole metro H3 grid is denser
    there than the fixed partition assumed. This recursively re-splits any
    still-oversized group at the next finer H3 resolution -- only where
    needed -- until every group is at or under ``threshold``, or
    ``max_resolution`` is reached (whichever first; see its docstring for
    why that's an acceptable fallback).

    Returns a pandas Series of string chunk keys (``"res{resolution}_{h3cell}"``,
    filesystem-safe -- these are used directly in output filenames -- with
    the resolution prefix keeping keys from different recursion depths from
    colliding), aligned with ``gdf``'s (integer position) index.
    """
    import pandas as pd

    keys = pd.Series(index=gdf.index, dtype=object)

    def _assign(sub_gdf: gpd.GeoDataFrame, resolution: int) -> None:
        cell_keys = _h3_keys_at_resolution(sub_gdf, resolution).to_pandas()
        cell_keys.index = sub_gdf.index
        for cell, idx in cell_keys.groupby(cell_keys).groups.items():
            if len(idx) > threshold and resolution < max_resolution:
                _assign(sub_gdf.loc[idx], resolution + 1)
            else:
                keys.loc[idx] = f"res{resolution}_{cell}"

    _assign(gdf, base_resolution)
    return keys


def _write_level_tiles_chunked(
    gdf: gpd.GeoDataFrame,
    level_name: str,
    tiles_dir: Path,
    min_zoom: int,
    max_zoom: int,
    use_xyz: bool = False,
    extract_xyz_from_pmtiles: bool = True,
) -> List[Path]:
    """Tile an oversized level as many bounded-size `.pmtiles` chunks.

    Called by `write_level_tiles` once a level's row count exceeds
    `H3_CHUNK_ROW_THRESHOLD`. Each row is assigned to exactly one chunk by
    its containing H3 cell, starting at resolution `H3_CHUNK_RESOLUTION`
    and adaptively recursing to finer resolutions for any cell whose rows
    still exceed the threshold (`_adaptive_h3_chunk_keys`) -- a fixed
    resolution alone does not bound chunk size for a geographically
    concentrated dataset (real Boston data: one resolution-4 cell alone
    held 430,653 rows, ~3x threshold). Each chunk's full, unclipped
    geometry is passed whole into its own `freestile_file` call -- no
    overlap/buffer between chunks is needed because freestiler already
    does its own internal tile-boundary MVT clipping per zoom/tile at
    render time. Output layout:
    ``tiles_dir/{level_name}/chunk_{chunk_key}.pmtiles`` where
    ``chunk_key`` is e.g. ``res5_842a301ffffffff``.

    freestiler exposes no merge/append primitive (`freestile`,
    `freestile_file`, `freestile_h3`, `freestile_layer`, `freestile_query`
    only), so this many-small-files layout is the final on-disk format,
    not an intermediate one merged away later. The MapLibre consumer
    (`geohierarchy.maps.maplibre.render`) registers one source per chunk
    file, all sharing the same `source-layer: level_name`, so they render
    as one seamless layer.
    """
    import gc

    level_dir = tiles_dir / level_name
    if level_dir.exists():
        shutil.rmtree(level_dir)
    level_dir.mkdir(parents=True, exist_ok=True)

    # A rebuild can flip a level between chunked/unchunked (e.g. a study's
    # AOI grows past H3_CHUNK_ROW_THRESHOLD between runs). Clear the
    # unchunked single-file output from a prior build so a stale
    # `{level_name}.pmtiles` doesn't linger alongside the new chunk
    # directory -- harmless to the maplibre consumer (which only looks for
    # one or the other, preferring the chunk directory) but confusing on
    # disk and wasted space otherwise.
    stale_single_file = tiles_dir / f"{level_name}.pmtiles"
    stale_single_file.unlink(missing_ok=True)

    chunk_keys = _adaptive_h3_chunk_keys(gdf)
    gdf = gdf.reset_index(drop=True)
    chunk_keys = chunk_keys.reset_index(drop=True)

    pmtiles_paths: List[Path] = []
    for chunk_key, idx in chunk_keys.groupby(chunk_keys).groups.items():
        chunk_gdf = gdf.iloc[list(idx)]
        pmtiles_path = level_dir / f"chunk_{chunk_key}.pmtiles"
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            chunk_gdf.to_parquet(tmp_path)
            del chunk_gdf
            gc.collect()
            generate_pmtiles_from_geoparquet(
                tmp_path,
                pmtiles_path,
                layer_name=level_name,
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                quiet=True,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        pmtiles_paths.append(pmtiles_path)

    del gdf
    gc.collect()

    if use_xyz and extract_xyz_from_pmtiles:
        # XYZ .pbf extraction from a chunked level: each chunk's PMTiles
        # is extracted into the same combined level XYZ directory tree in
        # turn. This works because XYZ dirs are inherently keyed by
        # z/x/y, and different chunks (H3 res-4 cells) cover disjoint
        # geographic areas, so their z/x/y tile sets don't collide.
        # This path is exercised by the legacy folium/Leaflet.VectorGrid
        # renderer only; production (CS_transitLOS) runs with
        # TRANSITLOS_RENDERER=maplibre, which reads .pmtiles directly and
        # never takes this branch.
        xyz_paths: List[Path] = []
        for pmtiles_path in pmtiles_paths:
            xyz_dir = _extract_xyz_from_pmtiles(
                pmtiles_path,
                tiles_dir,
                layer_name=level_name,
                overwrite=False,
                quiet=True,
            )
            xyz_paths.append(xyz_dir)
        # De-duplicate: every chunk extracts into the same combined dir.
        seen = []
        for p in xyz_paths:
            if p not in seen:
                seen.append(p)
        return seen + pmtiles_paths

    return pmtiles_paths


def write_level_tiles(
    gdf: gpd.GeoDataFrame,
    level_name: str,
    tiles_dir: str,
    min_zoom: int,
    max_zoom: int,
    id_col: str,
    property_cols: Iterable[str],
    buffer_frac: float = 0.02,
    use_xyz: bool = False,
    extract_xyz_from_pmtiles: bool = True,
) -> List[Path]:
    """Generate vector tiles for a GeoDataFrame level using fast methods.

    This is the main entry point for tile generation. It will:
    - Always generate .pmtiles files using freestiler (FASTEST, RECOMMENDED)
    - If use_xyz=True, also extract XYZ .pbf directories from the PMTiles for backward compatibility
    - Fall back to old Python-based method if freestiler not available

    Note: PMTiles (.pmtiles) is now the PRIMARY format. For backward compatibility,
    XYZ PBF directories can be extracted from PMTiles by setting use_xyz=True.
    This ensures the existing Leaflet.VectorGrid code continues to work.

    For H3 cells with WKB geometry, freestiler uses the existing geometry directly.
    For H3 cells without geometry, use generate_pmtiles_from_h3_with_centers() to avoid
    computing polygon boundaries.

    Args:
        gdf: Level geometry (EPSG:4326), typically hierarchy.get_level(name).
        level_name: Name of the level; also used as the MVT layer name and
            the top-level directory under tiles_dir.
        tiles_dir: Root output directory for all levels' tiles.
        min_zoom: Minimum zoom (inclusive) to generate tiles for.
        max_zoom: Maximum zoom (inclusive) to generate tiles for.
        id_col: Feature id column to keep in tile properties.
        property_cols: Additional attribute columns to keep in tile
            properties (styling/popups read these client-side).
        buffer_frac: Tile buffer as a fraction of tile width (default: 2%).
            Note: Not used with freestiler-based generation.
        use_xyz: If True, also extract XYZ .pbf directories from PMTiles for backward compatibility.
                If False (default), only generate .pmtiles files.
        extract_xyz_from_pmtiles: If True (default) and use_xyz=True, extract XYZ from PMTiles.

    Returns:
        List of paths to every generated tile file/directory.
        If use_xyz=True: List with one Path to the layer's XYZ directory
            (per chunk, if chunked -- see below) plus every .pmtiles path.
        If use_xyz=False: List of .pmtiles path(s). Normally a single
            ``{level_name}.pmtiles``; for a level with more rows than
            `H3_CHUNK_ROW_THRESHOLD`, a list of
            ``{level_name}/chunk_{h3res4cell}.pmtiles`` paths instead (see
            module docstring / `H3_CHUNK_ROW_THRESHOLD`) -- freestiler
            exposes no merge/append primitive, so many small per-chunk
            files is the final on-disk format, not an intermediate one.
    """
    tiles_dir = Path(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # For fast implementation, we use freestiler
    if HAS_FREESTILER:
        # Freestiler requires EPSG:4326 geometry
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            print(
                f"  Reprojecting {level_name} from {gdf.crs.to_epsg()} to EPSG:4326..."
            )
            gdf = gdf.to_crs(4326)

        # Keep only the columns we need for tiling
        # id_col and property_cols are the attribute columns, plus geometry
        keep_cols = []
        if id_col in gdf.columns:
            keep_cols.append(id_col)
        keep_cols.extend([c for c in property_cols if c in gdf.columns and c != id_col])
        # Ensure geometry column is included
        geom_col = gdf.geometry.name
        if geom_col not in keep_cols:
            keep_cols.append(geom_col)

        # Filter to only the columns we have. Callers that already pass a
        # pre-trimmed `gdf` (e.g. `HierarchyMap.build()`, via
        # `GeoHierarchy.get_level(name, columns=property_cols)`) skip the
        # reselect + `.copy()` here entirely -- on a metro-scale H3 level
        # (1M+ rows), an unconditional `.copy()` of an already-correctly-
        # shaped frame doubled peak memory for zero benefit.
        present_keep_cols = [c for c in keep_cols if c in gdf.columns]
        if list(gdf.columns) != present_keep_cols:
            gdf = gdf[present_keep_cols].copy()

        # Downcast float64 property columns to float32 before writing to
        # GeoParquet. Vector tile properties are already lossy (geometry is
        # snapped to the tile pixel grid, styling reads coarse-grained
        # buckets/gradients), so float32's ~7 significant digits are more
        # precision than a rendered map can show -- but on a level with many
        # numeric columns (e.g. Boston's 100+ census/derived fields at H3
        # resolution 11, ~1.6M rows), this halves the numeric payload's
        # memory footprint at exactly the point (pre-tiling temp file) where
        # peak memory has repeatedly OOM-killed large tile builds.
        float64_cols = [
            c for c in present_keep_cols if c != geom_col and gdf[c].dtype == "float64"
        ]
        if float64_cols:
            gdf[float64_cols] = gdf[float64_cols].astype("float32")

        if len(gdf) > H3_CHUNK_ROW_THRESHOLD:
            return _write_level_tiles_chunked(
                gdf,
                level_name,
                tiles_dir,
                min_zoom,
                max_zoom,
                use_xyz=use_xyz,
                extract_xyz_from_pmtiles=extract_xyz_from_pmtiles,
            )

        # Symmetric to the chunked branch's stale-single-file cleanup: a
        # rebuild can flip a level from chunked back to unchunked (e.g. the
        # threshold changes, or the AOI shrinks), so clear a stale chunk
        # directory from a prior build before writing the single-file
        # output.
        stale_chunk_dir = tiles_dir / level_name
        if stale_chunk_dir.is_dir():
            shutil.rmtree(stale_chunk_dir)

        # Save to temporary GeoParquet using geopandas
        # This ensures proper geo metadata (CRS) is written to the file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            gdf.to_parquet(tmp_path)

            # `freestile_file` reads `tmp_path` directly in Rust; the
            # in-memory `gdf` (and any lingering copy pandas/geopandas made
            # while writing it) is pure overhead from this point on, and on
            # a metro-scale level (1M+ rows, 100+ columns) can itself be
            # a large chunk of the process's resident memory right as the
            # Rust tiler needs its own headroom to build every zoom level.
            import gc

            del gdf
            gc.collect()

            # Always generate PMTiles first
            pmtiles_path = tiles_dir / f"{level_name}.pmtiles"
            generate_pmtiles_from_geoparquet(
                tmp_path,
                pmtiles_path,
                layer_name=level_name,
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                quiet=True,
            )

            # If XYZ is requested, extract from PMTiles
            if use_xyz and extract_xyz_from_pmtiles:
                xyz_dir = _extract_xyz_from_pmtiles(
                    pmtiles_path,
                    tiles_dir,
                    layer_name=level_name,
                    overwrite=True,
                    quiet=True,
                )
                return [xyz_dir, pmtiles_path]  # Return both
            else:
                return [pmtiles_path]
        finally:
            tmp_path.unlink(missing_ok=True)

    # Fall back to old method if freestiler not available
    # This maintains backward compatibility
    from .tiles_slow import write_level_tiles as write_level_tiles_slow

    return write_level_tiles_slow(
        gdf,
        level_name,
        tiles_dir,
        min_zoom,
        max_zoom,
        id_col,
        property_cols,
        buffer_frac,
    )


def write_level_zoom_tiles(
    work: gpd.GeoDataFrame,
    level_name: str,
    tiles_dir: str,
    z: int,
    keep_cols: List[str],
    buffer_frac: float = 0.02,
) -> List[Path]:
    """Tile one level's already-prepped geometry at a single zoom level.

    For backward compatibility with the old parallel processing API.
    In the fast implementation, we don't use this as we generate all zooms at once.
    """
    # This is a fallback for old code that calls this directly
    # The fast implementation generates all zooms in one call

    # For now, use the old method to avoid issues with single-zoom freestiler calls
    # freestiler is optimized for generating all zooms at once, not one at a time
    from .tiles_slow import write_level_zoom_tiles as write_level_zoom_tiles_slow

    return write_level_zoom_tiles_slow(
        work, level_name, tiles_dir, z, keep_cols, buffer_frac
    )

    # Future: Implement a smarter approach that batches single-zoom calls
    # or modifies HierarchyMap.build() to generate all zooms at once


# Re-export the fast functions for direct use
__all__ = [
    "prep_level_gdf",
    "write_level_tiles",
    "write_level_zoom_tiles",
    "generate_pmtiles_from_geoparquet",
    "generate_pmtiles_from_polars",
    "generate_xyz_tiles_from_geoparquet",
    # H3J/H3T are available but NOT RECOMMENDED
    "generate_h3j_from_h3_indices",
    "generate_h3t_tiles_from_h3_indices",
    "build_fast_hierarchy_map",
    "HAS_FREESTILER",
]
