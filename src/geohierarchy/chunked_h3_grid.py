"""Per-tile GeoParquet output for a large H3 attribute grid, never one big in-memory GeoDataFrame.

Companion to :func:`geohierarchy.raster_resample.raster_to_h3_tiled` -- same
motivation (Shanghai/Boston-scale OOM, one process holding the whole city's
data at once), same partition-by-H3-ancestor strategy, but for a plain
*attribute* table (population/access/census columns keyed by H3 cell id,
e.g. `code.pipeline.read_h3_grid_table`'s output in `CS_transitLOS`) rather
than a raster.

Unlike :func:`raster_to_h3_tiled`, this needs **no halo/buffer**: that
function's halo exists because a raster PIXEL can straddle the true
geometric boundary between two H3 cells, so a tile computing "which cell
does this pixel belong to, and how much of it" needs to see its neighbors'
candidate cells to split correctly. Here, every row already IS one specific,
whole H3 cell (no splitting question), and a cell's parent at any coarser
resolution is a pure id-arithmetic fact (`h3.cell_to_parent`), not a
geometric computation that could disagree near a boundary. Grouping rows by
their `tile_resolution` ancestor is therefore an EXACT partition on its
own -- every fine cell belongs to exactly one tile, full stop, matching the
same reasoning already used (and tested) for the in-memory chunk loops in
`CS_transitLOS.code.pipeline._chunked_h3_grid_join`/`_resample_h3_chunked`.
The one place a real geometric halo genuinely would matter -- a street
segment or census polygon that spans a tile boundary -- belongs to a
DIFFERENT stage (isochrones / census join) than this module, which only
builds hexagon geometry + writes out already-computed attributes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import polars as pl


def partition_h3_table(
    df: pl.DataFrame, tile_resolution: int, h3_col: str = "h3_cell"
) -> dict[str, pl.DataFrame]:
    """Group `df`'s rows by each row's H3 ancestor cell at `tile_resolution`.

    Exact, non-overlapping, non-duplicating partition (see module docstring)
    -- every row of `df` appears in exactly one output value, under its own
    tile id key.
    """
    import h3

    if df.is_empty():
        return {}
    tile_ids = [h3.cell_to_parent(c, tile_resolution) for c in df[h3_col].to_list()]
    tagged = df.with_columns(pl.Series("_tile_id", tile_ids))
    out: dict[str, pl.DataFrame] = {}
    for tile_id in tagged["_tile_id"].unique(maintain_order=True).to_list():
        out[tile_id] = tagged.filter(pl.col("_tile_id") == tile_id).drop("_tile_id")
    return out


def _tile_worker(table_bytes: bytes, tile_id: str, h3_col: str, output_dir: str) -> str:
    """Build hexagon geometry for one tile's rows and write its own GeoParquet.

    Runs in a separate process (see :func:`build_h3_grid_chunked`) -- the
    table is handed over pre-serialized (IPC bytes) rather than as a live
    Polars object so nothing about the parent process's memory is shared or
    pickled implicitly.
    """
    import io

    # Local import so this also works when `UrbanAccessAnalyzer` isn't a hard
    # dependency of every `geohierarchy` install -- only needed inside the
    # worker, only when this chunked path is actually used.
    from UrbanAccessAnalyzer import h3_ops

    table = pl.read_ipc(io.BytesIO(table_bytes))
    gdf = h3_ops.to_gdf(table, h3_column=h3_col)
    out_path = str(Path(output_dir) / f"tile_{tile_id}.parquet")
    gdf.to_parquet(out_path)
    return out_path


def build_h3_grid_chunked(
    df: pl.DataFrame,
    output_dir: str,
    tile_resolution: int = 5,
    h3_col: str = "h3_cell",
    max_workers: Optional[int] = None,
) -> List[str]:
    """Write one hexagon-geometry GeoParquet file PER `tile_resolution` H3 tile.

    This is the chunked replacement for building one whole-city
    `GeoDataFrame` (`code.pipeline._add_h3_grid` / `h3_ops.to_gdf` called
    once on everything) -- at Shanghai's scale (~25.7M resolution-11 cells)
    that single call needs 20GB+ resident simultaneously; this instead
    partitions by each row's `tile_resolution` ancestor (exact, no halo
    needed -- see module docstring) and builds+writes each tile's own
    (small, bounded) geometry in its own worker process, so peak memory is
    bounded by one tile's size times `max_workers`, never the whole city's.

    Args:
        df: Geometry-free H3 attribute table (e.g.
            `code.pipeline.read_h3_grid_table`'s output), one row per H3
            cell, with an `h3_col` id column.
        output_dir: Directory to write `tile_<id>.parquet` files into
            (created if missing). Each file is a standalone GeoParquet with
            hexagon geometry + every attribute column `df` had.
        tile_resolution: Coarser H3 resolution defining the partition
            (default 5, matching `raster_to_h3_tiled`'s default and
            `geohierarchy`'s other tiling machinery).
        h3_col: Name of the cell-id column.
        max_workers: `ProcessPoolExecutor` worker count (default:
            `os.cpu_count()`). Pass 1 to run serially in-process (useful for
            tests / tiny fixtures where pool startup overhead would
            dominate, and for environments where `spawn`-context pools
            aren't available, e.g. some sandboxed CI).

    Returns:
        Paths of every tile file written, one per non-empty tile. Never
        merged -- callers (map tiling, cross-chunk stats) work from this
        list of files directly.
    """
    import io
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    partitions = partition_h3_table(df, tile_resolution, h3_col=h3_col)
    if not partitions:
        return []

    def _to_ipc_bytes(table: pl.DataFrame) -> bytes:
        buf = io.BytesIO()
        table.write_ipc(buf)
        return buf.getvalue()

    workers = max_workers if max_workers is not None else os.cpu_count()
    tile_items = list(partitions.items())

    if workers and workers > 1 and len(tile_items) > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _tile_worker, _to_ipc_bytes(table), tile_id, h3_col, output_dir
                )
                for tile_id, table in tile_items
            ]
            paths = [f.result() for f in futures]
    else:
        paths = [
            _tile_worker(_to_ipc_bytes(table), tile_id, h3_col, output_dir)
            for tile_id, table in tile_items
        ]

    return paths


def read_h3_grid_chunked_columns(
    tile_paths: Sequence[str], columns: Sequence[str]
) -> pl.LazyFrame:
    """Lazily read only `columns` (never geometry) from every per-tile GeoParquet, as one scan.

    The polars-lazy cross-chunk aggregation primitive requested for
    citywide statistics (Distribution/Regression/ANOVA panels, ...): a
    `pl.scan_parquet` glob-style multi-file scan restricted to exactly the
    non-geometry columns a given statistic needs, so a stats computation
    over ALL tiles never materializes any tile's geometry and never holds
    more than one tile's worth of the needed columns in memory at a time
    (`polars`' streaming engine reads/aggregates file-by-file lazily).

    Args:
        tile_paths: Per-tile GeoParquet paths, e.g.
            `build_h3_grid_chunked`'s return value.
        columns: Attribute column names to read (must exclude `geometry`;
            geometry is WKB-encoded and not meaningful to scan this way).

    Returns:
        An unevaluated `polars.LazyFrame` over every tile's `columns` union,
        row-order-independent (callers `.collect()` after any further
        filter/aggregate they need).
    """
    if "geometry" in columns:
        raise ValueError(
            "read_h3_grid_chunked_columns never reads geometry -- exclude it from `columns`."
        )
    if not tile_paths:
        return pl.DataFrame({c: [] for c in columns}).lazy()
    return pl.scan_parquet(list(tile_paths)).select(columns)
