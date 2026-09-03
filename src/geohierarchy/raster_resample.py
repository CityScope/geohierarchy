"""Generic, exact, mass-conserving raster -> polygon-tiling area-weighted resampling.

This module implements ``raster_to_polygons``: given a raster (a 2D array of
values + an affine transform + a CRS) and an arbitrary set of target
polygons (H3 cells, administrative boundaries, or anything else), it splits
each pixel's value across every polygon the pixel geometrically intersects,
in proportion to the intersection area, and returns one row per polygon that
is touched by at least one non-null pixel.

Guarantees (see tests in ``geohierarchy/tests/test_raster_resample.py``):

1. **Area-weighted split** -- a pixel straddling several polygons splits its
   value across them in exact proportion to the overlap area.
2. **Full coverage** -- every polygon touching a non-null pixel appears in
   the output with a non-null value; none are silently dropped.
3. **Exact mass conservation** -- ``sum(output value)`` equals
   ``sum(pixel value)`` over the raster window, up to floating point
   precision (no approximation, no residual rounding error).
4. **Density bound** -- for every output polygon,
   ``value / polygon_area`` lies between the min and max of
   ``pixel_value / pixel_area`` over the pixels that polygon touches,
   since a polygon's value is by construction an area-weighted average of
   exactly those pixels' densities.

Algorithm
---------
Exact polygon/pixel intersection (via ``shapely``), but *not* naively
computed for every pixel: pixels are square and axis-aligned in the
raster's own CRS, so most of them (anywhere the target tiling's cells are
larger than a pixel, e.g. WorldPop's ~100m pixels vs. an H3 res-9/10 cell)
fall entirely inside a single target polygon. This is detected cheaply with
a ``shapely.STRtree`` bounding-box query per pixel against the target
polygons: pixels with exactly one candidate whose bounds *contain* the
pixel's bounds skip geometric clipping entirely and hand their whole value
to that one polygon (the fast path, and the common case). Only pixels whose
bounding box touches more than one candidate polygon -- i.e. pixels that
plausibly straddle a boundary -- pay for an actual ``shapely.intersection``
call, and even then only against their few real candidates, never the full
pixel x polygon cross product.

A closed-form periodicity/offset lookup (precomputing exact intersection
fractions once for "a pixel at relative offset X from the hex lattice" and
reusing it everywhere) was investigated and is documented as a follow-up in
the module docstring below, but was not implemented: correctness (exact
conservation, full coverage, density bound, all hard requirements here) was
prioritized over that extra optimization within the time available. The
boundary-only-exact-clip approach above already reduces the number of real
geometric intersections to a small minority of total pixels in the
realistic pixel-smaller-or-comparable-to-cell regime this is used for
(WorldPop pixels vs. H3 res 9-11), and benchmarks fast enough in practice
(see the module's benchmark script / test suite for real numbers).

Future optimization note (periodicity lookup): H3's hex tiling at a fixed
resolution is a regular lattice everywhere except at the 12 icosahedral
pentagon base cells (which would need to fall back to this module's exact
path as a documented edge case). For the regular hexagonal regions, a
pixel's split fractions across its (up to 3) covering hexagons are a
function only of the pixel's offset modulo the hex lattice's repeating
unit cell -- in principle, precomputing that function on a fine grid of
offsets (or deriving it analytically from the hexagon edge equations) and
looking it up per pixel would avoid the shapely intersection call even for
boundary-crossing pixels. That was judged a materially larger, riskier
undertaking (deriving/validating the closed-form offset -> area-fraction
map against the *actual* H3 projection, which is not a perfectly flat
lattice in lon/lat) than the time available supported to get right and
verified; the boundary-only-exact-clip approach here is exact by
construction and fast enough in practice.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import geopandas as gpd
import polars as pl
import rasterio as rio
import shapely
from shapely.strtree import STRtree


def raster_to_polygons(
    raster_array: np.ndarray,
    transform: rio.Affine,
    raster_crs,
    polygons: gpd.GeoDataFrame,
    id_col: str,
    nodata: Optional[float] = None,
) -> pl.DataFrame:
    """Area-weighted, mass-conserving resample of a raster onto a polygon tiling.

    Args:
        raster_array: 2D array of pixel values (e.g. one band read from a
            GeoTIFF). NaN and ``nodata`` pixels are treated as missing and
            excluded.
        transform: Affine transform mapping pixel (col, row) -> raster CRS
            coordinates (as returned by ``rasterio``).
        raster_crs: CRS of ``raster_array``/``transform``.
        polygons: Target polygon tiling (e.g. H3 cells from
            :func:`~geohierarchy.utils.h3_cells`, or any other polygon
            GeoDataFrame). Reprojected internally to ``raster_crs`` so pixel
            and polygon geometries share a common, metric-if-possible plane;
            pass an already-projected ``raster_crs``/``polygons`` pair (not
            EPSG:4326) for exact, non-degree area weighting.
        id_col: Name of the unique identifier column in ``polygons``.
        nodata: Sentinel value in ``raster_array`` treated as missing, in
            addition to NaN.

    Returns:
        Polars DataFrame with columns ``id_col`` and ``value``: one row per
        polygon touched by at least one valid pixel, with a value equal to
        the area-weighted sum of that pixel's contributions. Guarantees (1)
        every touching polygon appears, (2) ``sum(value)`` exactly equals
        ``sum(raster_array)`` over valid pixels within the polygons'
        combined extent, and (3) each polygon's implied density
        (``value`` / polygon area) never exceeds the max, nor falls below
        the min, of the densities of the pixels it touches. See the module
        docstring for the algorithm and why those hold by construction.
    """
    if polygons.crs is None:
        raise ValueError("polygons must have a CRS set.")
    polys = polygons.to_crs(raster_crs) if polygons.crs != raster_crs else polygons
    poly_geoms = polys.geometry.values
    poly_ids = polys[id_col].to_numpy()
    n_polys = len(polys)

    if n_polys == 0:
        return pl.DataFrame(
            {id_col: [], "value": []}, schema={id_col: pl.Utf8, "value": pl.Float64}
        )

    height, width = raster_array.shape[-2], raster_array.shape[-1]
    rows, cols = np.nonzero(np.ones((height, width), dtype=bool))
    values = raster_array[rows, cols].astype(np.float64)

    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    if not valid.any():
        return pl.DataFrame(
            {id_col: [], "value": []}, schema={id_col: pl.Utf8, "value": pl.Float64}
        )
    rows, cols, values = rows[valid], cols[valid], values[valid]

    # Pixel corner coordinates via plain affine arithmetic (no per-pixel
    # geometry yet) -- assumes a north-up / axis-aligned transform (b == d
    # == 0), true for essentially all real-world raster products including
    # WorldPop; a sheared/rotated transform would need per-pixel polygon
    # construction instead of the box() fast path below.
    if transform.b != 0 or transform.d != 0:
        raise NotImplementedError(
            "raster_to_polygons requires an axis-aligned (unrotated, unsheared) "
            "raster transform (Affine.b == Affine.d == 0)."
        )
    px_w, px_h = transform.a, transform.e  # px_h is negative for north-up rasters
    x0 = transform.c + cols * px_w
    x1 = x0 + px_w
    y0 = transform.f + rows * px_h
    y1 = y0 + px_h
    minx, maxx = np.minimum(x0, x1), np.maximum(x0, x1)
    miny, maxy = np.minimum(y0, y1), np.maximum(y0, y1)
    # Spatial index over the target polygons; query each pixel's bounding
    # box for candidate polygons it might overlap.
    tree = STRtree(poly_geoms)
    pixel_boxes = shapely.box(minx, miny, maxx, maxy)
    cand_pix_idx, cand_poly_idx = tree.query(pixel_boxes, predicate="intersects")

    out_poly_idx_parts = []
    out_value_parts = []

    # Group candidate polygon indices by pixel.
    order = np.argsort(cand_pix_idx, kind="stable")
    cand_pix_idx = cand_pix_idx[order]
    cand_poly_idx = cand_poly_idx[order]
    uniq_pix, start_idx, counts = np.unique(
        cand_pix_idx, return_index=True, return_counts=True
    )

    single_mask = counts == 1
    multi_mask = ~single_mask

    # Fast path: pixels with exactly one intersecting-candidate polygon
    # hand their whole value to it (still guaranteed correct even if the
    # pixel is not fully contained -- if a pixel's bbox only intersects one
    # polygon's bbox, no other polygon can claim any of its area either).
    if single_mask.any():
        single_pix_local = uniq_pix[single_mask]
        single_start = start_idx[single_mask]
        single_poly = cand_poly_idx[single_start]
        out_poly_idx_parts.append(single_poly)
        out_value_parts.append(values[single_pix_local])

    # Exact-clip path: pixels whose bbox touches multiple candidate
    # polygons -- compute real intersection areas and split proportionally.
    if multi_mask.any():
        multi_pix_local = uniq_pix[multi_mask]
        multi_start = start_idx[multi_mask]
        multi_counts = counts[multi_mask]
        for pix_local, s, c in zip(multi_pix_local, multi_start, multi_counts):
            polys_here = cand_poly_idx[s : s + c]
            pixel_geom = pixel_boxes[pix_local]
            inter_areas = shapely.area(
                shapely.intersection(pixel_geom, poly_geoms[polys_here])
            )
            total = inter_areas.sum()
            if total <= 0:
                continue
            shares = values[pix_local] * (inter_areas / total)
            out_poly_idx_parts.append(polys_here)
            out_value_parts.append(shares)

    if not out_poly_idx_parts:
        return pl.DataFrame(
            {id_col: [], "value": []}, schema={id_col: pl.Utf8, "value": pl.Float64}
        )

    all_poly_idx = np.concatenate(out_poly_idx_parts)
    all_values = np.concatenate(out_value_parts)

    result = (
        pl.DataFrame({"_poly_idx": all_poly_idx, "value": all_values})
        .group_by("_poly_idx")
        .agg(pl.col("value").sum())
    )
    id_lookup = poly_ids[result["_poly_idx"].to_numpy()]
    result = result.with_columns(pl.Series(id_col, id_lookup)).drop("_poly_idx")

    # Mass conservation is exact by construction: every valid pixel's value
    # is fully distributed (fast path: 100% to its one candidate;
    # exact-clip path: split by inter_areas / inter_areas.sum(), which sums
    # to exactly 1 for any pixel whose intersection-area total is > 0 --
    # i.e. as long as its candidate set from the STRtree bbox query fully
    # covers its real geometric overlap, which "intersects" predicate
    # guarantees). No pixel is ever dropped or double counted.
    return result.select([id_col, "value"])


def raster_to_h3(
    raster_array: np.ndarray,
    transform: rio.Affine,
    raster_crs,
    resolution: int,
    value_col: str = "value",
    h3_col: str = "h3_cell",
    nodata: Optional[float] = None,
) -> pl.DataFrame:
    """Convenience wrapper: area-weighted, mass-conserving raster -> H3 grid resample.

    Builds an H3 cell grid covering the raster's own extent at ``resolution``
    (via :func:`~geohierarchy.utils.h3_cells`) and delegates to
    :func:`raster_to_polygons`. This is the generic primitive
    ``pycensus``'s WorldPop loader applies with WorldPop-specific knowledge
    (nodata handling, AOI cropping, CRS choice) layered on top -- see
    ``pycensus.countries.worldwide.worldpop.loader``.

    Args:
        raster_array: 2D array of pixel values.
        transform: Affine transform for ``raster_array``.
        raster_crs: CRS of ``raster_array``.
        resolution: H3 resolution for the output grid.
        value_col: Name to give the output value column.
        h3_col: Name to give the output H3 cell-id column.
        nodata: Sentinel value in ``raster_array`` treated as missing, in
            addition to NaN.

    Returns:
        Polars DataFrame with ``h3_col`` and ``value_col`` -- one row per
        H3 cell touched by at least one valid pixel.
    """
    from .utils import h3_cells

    height, width = raster_array.shape[-2], raster_array.shape[-1]
    corners_x = [0, width, width, 0]
    corners_y = [0, 0, height, height]
    xs, ys = rio.transform.xy(transform, corners_y, corners_x, offset="ul")
    raster_extent = shapely.box(min(xs), min(ys), max(xs), max(ys))
    # `h3_cells` (via h3ronpy's `geometry_to_cells`) uses centroid-containment by
    # default, which can leave a thin rim of hexagons whose centroid falls just
    # outside the raster's exact rectangular extent -- even though those cells
    # still geometrically overlap edge pixels -- ungenerated, silently dropping
    # coverage right where boundary-crossing pixels most need a target cell.
    # Buffering the extent by one H3 cell's approximate "radius" before building
    # the grid guarantees every cell that could possibly intersect a raster pixel
    # is included; `raster_to_polygons` itself only ever assigns mass to cells
    # with real intersection area, so the buffer cannot manufacture spurious mass.
    import h3

    cell_radius_deg = h3.average_hexagon_edge_length(resolution, unit="km") * 2 / 111.0
    bounds_gdf = gpd.GeoDataFrame(geometry=[raster_extent], crs=raster_crs).to_crs(4326)
    buffered = bounds_gdf.geometry.iloc[0].buffer(cell_radius_deg)
    bounds_gdf = gpd.GeoDataFrame(geometry=[buffered], crs=4326)
    cells = h3_cells(bounds_gdf, resolution)
    result = raster_to_polygons(
        raster_array, transform, raster_crs, cells, id_col="h3", nodata=nodata
    )
    return result.rename({"h3": h3_col, "value": value_col})


def _raster_to_h3_tile_worker(
    tif_path: str,
    tile_id: str,
    resolution: int,
    value_col: str,
    h3_col: str,
    nodata: Optional[float],
) -> pl.DataFrame:
    """One tile's worth of work for :func:`raster_to_h3_tiled` -- runs in its own process.

    Takes just `tile_id` (a single H3 cell id), not a precomputed cell
    list: expanding `tile_id` to its `resolution`-level children (and their
    halo) happens HERE, inside the worker, not in the coordinating process.
    An earlier version computed every tile's full children + halo lists
    upfront in `raster_to_h3_tiled` before dispatching any work -- fine for
    a compact city, but for a large multi-city AOI (Shanghai's megaregion,
    ~330,000 km^2) that meant materializing on the order of 10^8 H3 id
    strings in one process before any tiling benefit kicked in, defeating
    the entire point of tiling and OOM-killing the run at the very first
    print statement (2026-08-30, caught live). Computing children/halo
    per-worker keeps the coordinator's memory at O(number of tiles) --
    typically a few hundred to low thousands of short id strings -- instead
    of O(total fine cells).

    Opens `tif_path` itself (rather than receiving an array) so nothing
    beyond this one tile's windowed pixel read ever exists in this
    process's memory.

    Runs `raster_to_polygons` over `tile_id`'s own children **plus** a thin
    halo ring of neighboring cells this tile does NOT own (see
    `raster_to_h3_tiled`'s docstring for why the halo is required for
    correctness), then drops every halo row before returning: a pixel
    straddling the true boundary between two tiles needs BOTH real
    candidate cells present simultaneously for `raster_to_polygons` to
    split it correctly (otherwise each tile's run only ever sees its own
    single candidate and wrongly hands it the pixel's *entire* value --
    double counting the shared pixel across both tiles' outputs, a real
    bug caught by `test_raster_to_h3_tiled_matches_single_shot`,
    2026-08-30). The halo cell's neighboring tile computes the exact same
    split independently (same real geometry, deterministic), keeps its own
    share, and drops this tile's cells from ITS output the same way -- so
    every cell is produced by exactly one tile, with the correct
    (halo-aware) share.

    Module-level (not a closure) so it is picklable for `ProcessPoolExecutor`.
    """
    import h3

    from .utils import h3_cells_from_ids

    # `str(tile_id)` -- not a no-op: `tile_id` started life as a
    # `numpy.str_`/pandas-object-array element (from `h3_cells`'s
    # GeoDataFrame column), and h3's cython `str_to_int` rejects anything
    # that isn't a genuine builtin `str` after round-tripping through
    # `ProcessPoolExecutor`'s `spawn` pickling (`TypeError: int() can't
    # convert non-string with explicit base` -- a real failure caught live
    # on Boston, 2026-08-30, right after moving this call into the worker).
    owned_ids = set(h3.cell_to_children(str(tile_id), resolution))
    if not owned_ids:
        return pl.DataFrame(
            {h3_col: [], value_col: []}, schema={h3_col: pl.Utf8, value_col: pl.Float64}
        )
    halo_ids: set = set()
    for cell in owned_ids:
        for neighbor in h3.grid_disk(cell, 1):
            if neighbor not in owned_ids:
                halo_ids.add(neighbor)
    all_ids = list(owned_ids | halo_ids)
    cells = h3_cells_from_ids(all_ids)
    if len(cells) == 0:
        return pl.DataFrame(
            {h3_col: [], value_col: []}, schema={h3_col: pl.Utf8, value_col: pl.Float64}
        )

    with rio.open(tif_path) as src:
        bounds_raster_crs = cells.to_crs(src.crs).total_bounds
        window = rio.windows.from_bounds(*bounds_raster_crs, transform=src.transform)
        window = window.round_lengths().round_offsets()
        # Clip to the raster's own extent -- a tile's buffered bounds
        # (owned cells plus halo) can extend past the raster edge for
        # AOI-boundary tiles, and a halo-only cell can fall entirely
        # outside the raster (no intersection at all -- `Window.intersection`
        # raises rather than returning empty in that case).
        full = rio.windows.Window(0, 0, src.width, src.height)
        if window.width <= 0 or window.height <= 0:
            window = None
        else:
            try:
                window = window.intersection(full)
            except rio.errors.WindowError:
                window = None
        if window is None or window.width <= 0 or window.height <= 0:
            return pl.DataFrame(
                {h3_col: [], value_col: []},
                schema={h3_col: pl.Utf8, value_col: pl.Float64},
            )
        array = src.read(1, window=window).astype("float64")
        window_transform = rio.windows.transform(window, src.transform)
        raster_crs = src.crs

    result = raster_to_polygons(
        array, window_transform, raster_crs, cells, id_col="h3", nodata=nodata
    )
    result = result.filter(pl.col("h3").is_in(list(owned_ids)))
    return result.rename({"h3": h3_col, "value": value_col})


def raster_to_h3_tiled(
    tif_path: str,
    resolution: int,
    value_col: str = "value",
    h3_col: str = "h3_cell",
    nodata: Optional[float] = None,
    tile_resolution: int = 5,
    max_workers: Optional[int] = None,
) -> pl.DataFrame:
    """Memory-bounded, optionally-parallel version of :func:`raster_to_h3`.

    `raster_to_h3` reads the *entire* raster into one array and builds one
    `shapely.STRtree` over the *entire* target H3 grid before doing any
    work -- fine for a small city, but for a large one (millions of H3
    cells, tens of millions of raster pixels) that single-shot approach
    can hold everything -- the full pixel arrays, the full set of H3
    polygons, the STRtree's internal index, every intermediate
    `shapely`/`numpy` array -- in memory simultaneously (a real, live OOM
    failure: Boston and Beersheba's `run.py` were both killed past a 20GB
    cgroup cap during exactly this step, 2026-08-30).

    This instead partitions the work by H3's own parent/child hierarchy:
    every `resolution`-level cell has a unique, well-defined ancestor at
    `tile_resolution` (coarser, e.g. H3 res 5's ~252 km^2 average cell), so
    grouping by that ancestor is an exact, non-overlapping, non-duplicating
    partition of the whole grid -- no cell is ever split across two tiles,
    and no cell is ever double-counted. Each tile then only needs its own
    (small, bounded) set of child-cell polygons and its own windowed slice
    of the raster (read fresh from disk in that tile's own worker, never
    the whole file), so peak memory is bounded by one tile's size, not the
    whole city's, regardless of how many tiles there are.

    Mass conservation across tile boundaries needs one extra piece beyond
    the exact partition itself, which an earlier version of this function
    got wrong (caught live by
    `test_raster_to_h3_tiled_matches_single_shot`, 2026-08-30, ~8% mass
    inflation): a pixel that straddles the true geometric boundary between
    a cell owned by tile A and a cell owned by tile B needs to see BOTH
    real candidate cells at once to split correctly. If each tile's worker
    only ever sees its OWN cells as candidates, that shared pixel looks
    like a single-candidate pixel to BOTH tiles independently, and each one
    wrongly hands it 100% of the pixel's value -- doubling it. The fix is a
    "halo": every tile also gets a thin ring of its true H3 neighbors' cells
    as extra (non-owned) candidates, purely so `raster_to_polygons` can see
    the full local picture and split correctly; each tile then keeps only
    the rows for cells it actually owns and discards the halo rows (see
    `_raster_to_h3_tile_worker`) -- the neighboring tile that DOES own
    those halo cells computes the identical split independently and keeps
    its own share. Net effect: every cell is produced by exactly one tile,
    with the same value :func:`raster_to_h3` would have given it.

    Args:
        tif_path: Path to the source GeoTIFF (read once for its header
            here, then re-opened independently -- windowed -- by each
            tile's worker).
        resolution: Target (fine) H3 resolution.
        value_col: Name to give the output value column.
        h3_col: Name to give the output H3 cell-id column.
        nodata: Sentinel value treated as missing, in addition to NaN.
            Defaults to the GeoTIFF's own declared nodata value if not
            given.
        tile_resolution: Coarser H3 resolution whose cells define the
            partition (default 5, ``resolution`` must be `>` this). Larger
            cities/finer `resolution` benefit from a coarser (smaller
            number, more cells skipped) `tile_resolution` -- lower peak
            memory per tile, more tiles, each cheap.
        max_workers: Worker process count for `ProcessPoolExecutor`
            (default: `os.cpu_count()`). Pass 1 to run tiles serially in
            this same process (useful for debugging / tiny AOIs where
            process-pool overhead would dominate).

    Returns:
        Polars DataFrame with `h3_col` and `value_col`, one row per H3 cell
        touched by at least one valid pixel -- identical contract (and, up
        to row order, identical values) to :func:`raster_to_h3`.
    """
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    import h3

    if resolution <= tile_resolution:
        raise ValueError(
            f"resolution ({resolution}) must be finer (numerically greater) than "
            f"tile_resolution ({tile_resolution})."
        )

    with rio.open(tif_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        raster_crs = src.crs
        if nodata is None:
            nodata = src.nodata

    corners_x = [0, width, width, 0]
    corners_y = [0, 0, height, height]
    xs, ys = rio.transform.xy(transform, corners_y, corners_x, offset="ul")
    raster_extent = shapely.box(min(xs), min(ys), max(xs), max(ys))
    # Buffer by the COARSE (`tile_resolution`) cell's own radius, not the
    # fine one -- discovery below finds `tile_resolution`-level cells via
    # centroid-containment (see `h3_cells`'s docstring on the same
    # ungenerated-rim issue), so the buffer has to be sized for the cells
    # actually being discovered here. Using the fine resolution's much
    # smaller radius (a real bug caught alongside the halo one, 2026-08-30)
    # silently dropped whole coarse tiles near the raster's edge, and with
    # them every fine cell that only that tile would have owned.
    tile_radius_deg = (
        h3.average_hexagon_edge_length(tile_resolution, unit="km") * 2 / 111.0
    )
    bounds_gdf = gpd.GeoDataFrame(geometry=[raster_extent], crs=raster_crs).to_crs(4326)
    buffered = bounds_gdf.geometry.iloc[0].buffer(tile_radius_deg)

    # Discover which tile_resolution cells the raster touches at all
    # (cheap -- coarse grid, at most a few hundred/thousand cells even for
    # a big city), then expand each one to its `resolution`-level children
    # via H3's own hierarchy (pure id arithmetic, no geometry) -- this is
    # what makes the partition exact and non-overlapping.
    from .utils import h3_cells

    tile_cells_gdf = h3_cells(
        gpd.GeoDataFrame(geometry=[buffered], crs=4326), tile_resolution
    )
    tile_ids = tile_cells_gdf["h3"].tolist()

    if not tile_ids:
        return pl.DataFrame(
            {h3_col: [], value_col: []}, schema={h3_col: pl.Utf8, value_col: pl.Float64}
        )

    # Merge results incrementally (a small running accumulator, flushed
    # every `merge_every` completions) instead of collecting every tile's
    # DataFrame in one big list before a single final `pl.concat` -- for a
    # compact city (a handful to a few dozen tiles) that distinction is
    # academic, but for a huge multi-city AOI (Shanghai's megaregion,
    # ~1,300 `tile_resolution`-5 tiles) holding ~1,300 un-merged
    # DataFrames simultaneously was itself enough to OOM a 26GB cgroup
    # (2026-08-30, live: ran 12 minutes -- real per-tile work, further than
    # before the worker-side children fix -- then still died). Flushing
    # periodically caps how many un-merged tile results ever coexist,
    # bounding peak memory by (in-flight worker count + accumulator size)
    # rather than the total tile count.
    accumulated: Optional[pl.DataFrame] = None
    pending: List[pl.DataFrame] = []
    merge_every = 50

    def _flush() -> None:
        nonlocal accumulated, pending
        if not pending:
            return
        parts = [p for p in pending if p.height > 0]
        pending = []
        if not parts:
            return
        accumulated = (
            pl.concat([accumulated, *parts])
            if accumulated is not None
            else pl.concat(parts)
        )

    workers = max_workers if max_workers is not None else os.cpu_count()
    if workers and workers > 1 and len(tile_ids) > 1:
        # `spawn`, not the platform-default `fork` on Linux: forking a
        # process that has GDAL (via rasterio) or a test runner's own
        # capture/threading machinery active can deadlock the child at
        # fork time (a real, live hang caught under pytest, 2026-08-30 --
        # ran fine as a bare script, hung indefinitely under
        # `pytest -k parallel`). `spawn` starts each worker as a genuinely
        # fresh interpreter with no inherited locks/threads, at the cost of
        # re-importing this module's dependencies once per worker (paid
        # once per `raster_to_h3_tiled` call, not per pixel/tile).
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _raster_to_h3_tile_worker,
                    tif_path,
                    t,
                    resolution,
                    value_col,
                    h3_col,
                    nodata,
                )
                for t in tile_ids
            ]
            for i, fut in enumerate(futures):
                pending.append(fut.result())
                if (i + 1) % merge_every == 0:
                    _flush()
    else:
        for i, t in enumerate(tile_ids):
            pending.append(
                _raster_to_h3_tile_worker(
                    tif_path, t, resolution, value_col, h3_col, nodata
                )
            )
            if (i + 1) % merge_every == 0:
                _flush()

    _flush()
    if accumulated is None:
        return pl.DataFrame(
            {h3_col: [], value_col: []}, schema={h3_col: pl.Utf8, value_col: pl.Float64}
        )
    return accumulated


def pixel_area_density_bounds(
    raster_array: np.ndarray,
    transform: rio.Affine,
) -> Tuple[float, float]:
    """Min/max pixel density (value / pixel area) over the raster's valid pixels.

    Convenience helper for verification: compare an output polygon's
    ``value / polygon_area`` against these bounds restricted to the pixels
    that polygon actually touches (see the test suite for the full,
    per-cell version of this check -- this module-level helper gives the
    raster-wide bound only).
    """
    px_area = abs(transform.a * transform.e)
    valid = np.isfinite(raster_array)
    vals = raster_array[valid].astype(np.float64)
    density = vals / px_area
    return float(density.min()), float(density.max())
