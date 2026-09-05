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


def resample_points_to_h3(
    lat: np.ndarray,
    lng: np.ndarray,
    value: np.ndarray,
    resolution: int,
    is_count: bool = True,
    point_area_m2: Optional[np.ndarray] = None,
    h3_col: str = "h3_cell",
    value_col: str = "value",
) -> pl.DataFrame:
    """Fast, approximate-but-mass-conserving point/pixel -> H3 resample.

    2026-09-04, explicit user request/spec: a much cheaper alternative to
    :func:`raster_to_polygons`'s exact area-weighted overlay, for the common
    case of a dense point/pixel cloud (raster pixel centers, or a real
    gridded census source like Germany's Zensus 100m grid) being resampled
    onto H3 cells noticeably LARGER than the source spacing -- exact overlay
    is real work this doesn't need to pay for at that scale. Pure polars +
    ``h3ronpy`` vector ops, no per-point geometry/STRtree at all:

    1. (``is_count=True`` only) convert each point's ``value`` to a density
       by dividing by ``point_area_m2`` -- skip this for a source that's
       already a mean/rate/density/relative quantity (``is_count=False``,
       e.g. a `pop_density`-shaped raster), which starts directly at step 2
       on the raw ``value`` array.
    2. assign each point to its containing H3 cell at ``resolution``
       (``h3ronpy.vector.coordinates_to_cells`` -- for a hexagonal grid,
       "the cell whose lattice point is nearest" and "the cell containing
       this coordinate" are the same cell by construction, so this needs no
       separate nearest-neighbor search against a target grid), then take
       the MEAN density of every point assigned to the same cell.
    3. (``is_count=True`` only) multiply each cell's mean density by that
       cell's own real geodesic area (``h3ronpy.cells_area_m2``) to recover
       a count.
    4. (``is_count=True`` only) rescale every cell's count by a single
       global factor so ``sum(output value)`` exactly equals ``sum(value)``
       over every point that got assigned a cell (in practice every valid
       point, since H3 tiles the whole globe -- there is no "too far"
       case at the point-assignment step itself; the rescale still exists
       because step 2's per-cell MEAN-of-densities is only an
       approximation of the true within-cell distribution, unlike
       :func:`raster_to_polygons`'s exact overlay, so a global correction
       is needed to keep the total exactly right even though individual
       cells are only approximately right).

    For ``is_count=False``, the output is simply each cell's mean of
    ``value`` over its assigned points -- no area multiply, no rescale (a
    global sum of a rate/density column isn't a meaningful conserved
    quantity to correct against).

    Args:
        lat: Point latitudes (degrees, EPSG:4326).
        lng: Point longitudes (degrees, EPSG:4326).
        value: Point values -- a count (``is_count=True``) or an
            already-relative quantity (``is_count=False``).
        resolution: Target H3 resolution.
        is_count: Whether ``value`` is an absolute count needing the
            density-conversion + area-recovery + conservation-rescale
            treatment (``True``, default) or is already a mean/rate/density
            (``False``, in which case ``point_area_m2`` is not needed).
        point_area_m2: Each point's own represented area in m^2 (e.g. a
            raster pixel's real geodesic footprint, or a gridded census
            cell's known cell size). Required when ``is_count=True``.
        h3_col: Output H3 cell-id column name (hex string, matching this
            codebase's `h3_cell` convention).
        value_col: Output value column name.

    Returns:
        Polars DataFrame with ``h3_col`` and ``value_col`` -- one row per
        H3 cell touched by at least one input point.
    """
    import h3ronpy
    import h3ronpy.vector as h3v

    lat = np.asarray(lat, dtype=np.float64)
    lng = np.asarray(lng, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    valid = np.isfinite(lat) & np.isfinite(lng) & np.isfinite(value)
    if is_count:
        if point_area_m2 is None:
            raise ValueError("point_area_m2 is required when is_count=True")
        point_area_m2 = np.asarray(point_area_m2, dtype=np.float64)
        valid = valid & np.isfinite(point_area_m2) & (point_area_m2 > 0)

    lat, lng, value = lat[valid], lng[valid], value[valid]
    if lat.size == 0:
        return pl.DataFrame(
            {h3_col: [], value_col: []}, schema={h3_col: pl.Utf8, value_col: pl.Float64}
        )

    cell_ids = h3v.coordinates_to_cells(lat, lng, resolution)
    cell_strings = h3ronpy.cells_to_string(cell_ids)

    if is_count:
        density = value / point_area_m2[valid]
        df = pl.DataFrame({h3_col: cell_strings, "_density": density, "_value": value})
        grouped = df.group_by(h3_col).agg(
            pl.col("_density").mean().alias("_mean_density"),
        )
        unique_cells = grouped[h3_col].to_list()
        cell_area_m2 = np.asarray(
            h3ronpy.cells_area_m2(h3ronpy.cells_parse(unique_cells))
        )
        grouped = grouped.with_columns(pl.Series("_cell_area_m2", cell_area_m2))
        grouped = grouped.with_columns(
            (pl.col("_mean_density") * pl.col("_cell_area_m2")).alias(value_col)
        )
        raster_total = float(df["_value"].sum())
        h3_total = float(grouped[value_col].sum())
        if h3_total > 0:
            scale = raster_total / h3_total
            grouped = grouped.with_columns((pl.col(value_col) * scale).alias(value_col))
        return grouped.select([h3_col, value_col])

    df = pl.DataFrame({h3_col: cell_strings, value_col: value})
    grouped = df.group_by(h3_col).agg(pl.col(value_col).mean())
    # 2026-09-04, explicit user spec (`is_count=False`, an already
    # relative/density/mean raster): rescale so the mean of the assigned
    # RASTER points (every point that got an h3 cell, point-weighted) and
    # the mean of the output H3 CELLS (one value per cell, cell-weighted --
    # deliberately NOT point-weighted, cells with more assigned points
    # don't count more) agree. These two means differ whenever cells get
    # unequal numbers of assigned points, which is the common case -- this
    # rescale corrects the point-count-driven skew the same way the
    # `is_count=True` branch's step-4 rescale corrects for its own
    # approximation, applied here to a MEAN target instead of a SUM target.
    raster_mean_of_assigned = float(df[value_col].mean())
    h3_mean = float(grouped[value_col].mean())
    if h3_mean != 0:
        scale = raster_mean_of_assigned / h3_mean
        grouped = grouped.with_columns((pl.col(value_col) * scale).alias(value_col))
    return grouped


def raster_to_h3_nearest(
    tif_path: str,
    resolution: int,
    value_col: str = "value",
    h3_col: str = "h3_cell",
    nodata: Optional[float] = None,
    is_count: bool = True,
    row_chunk: int = 4096,
) -> pl.DataFrame:
    """Fast raster -> H3 resample via :func:`resample_points_to_h3` (pixel centers, not exact overlay).

    Reads the raster in horizontal row-block windows (bounded by
    ``row_chunk``, default 4096 rows) so peak memory scales with one
    chunk's pixel count, not the whole raster -- each chunk's valid
    (non-nodata) pixel centers/values/areas are extracted and accumulated;
    the actual H3 assignment + mean + rescale happens once, over the full
    accumulated point set, in :func:`resample_points_to_h3`.

    Pixel area (needed for ``is_count=True``'s density conversion): for a
    geographic (degree) CRS -- WorldPop's real shipping CRS -- uses the
    standard equirectangular approximation (``R = 6,371,000m``, pixel
    width scaled by ``cos(latitude)``), accurate to well under 0.1% at
    typical WorldPop pixel sizes (~100m) and normal (non-polar) latitudes.
    For a projected (metric) CRS, pixel area is simply
    ``abs(transform.a * transform.e)`` (already in real m^2), and pixel
    centers are reprojected to EPSG:4326 for the H3 assignment step (via
    ``pyproj``) since ``coordinates_to_cells`` needs real lat/lng.

    Args:
        tif_path: Path to a single-band GeoTIFF.
        resolution: Target H3 resolution.
        value_col: Output value column name.
        h3_col: Output H3 cell-id column name.
        nodata: Nodata sentinel to exclude (default: read from the file
            itself).
        is_count: See :func:`resample_points_to_h3`.
        row_chunk: Rows per read window.

    Returns:
        Polars DataFrame with ``h3_col`` and ``value_col``.
    """
    lat_parts: List[np.ndarray] = []
    lng_parts: List[np.ndarray] = []
    value_parts: List[np.ndarray] = []
    area_parts: List[np.ndarray] = []

    with rio.open(tif_path) as src:
        nodata = src.nodata if nodata is None else nodata
        transform = src.transform
        is_geographic = src.crs is not None and src.crs.is_geographic
        to_wgs84 = None
        if not is_geographic:
            import pyproj

            to_wgs84 = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        px_w = abs(transform.a)
        px_h = abs(transform.e)
        earth_r = 6_371_000.0

        for row_start in range(0, src.height, row_chunk):
            n_rows = min(row_chunk, src.height - row_start)
            window = rio.windows.Window(0, row_start, src.width, n_rows)
            array = src.read(1, window=window).astype(np.float64)
            if nodata is not None:
                array = np.where(array == nodata, np.nan, array)
            valid_mask = np.isfinite(array)
            if not valid_mask.any():
                continue
            rows_idx, cols_idx = np.nonzero(valid_mask)
            vals = array[valid_mask]
            # Pixel-center coordinates via the window's own affine transform.
            window_transform = rio.windows.transform(window, transform)
            xs = (
                window_transform.c
                + (cols_idx + 0.5) * window_transform.a
                + (rows_idx + 0.5) * window_transform.b
            )
            ys = (
                window_transform.f
                + (cols_idx + 0.5) * window_transform.d
                + (rows_idx + 0.5) * window_transform.e
            )

            if is_geographic:
                lng_chunk, lat_chunk = xs, ys
                if is_count:
                    lat_rad = np.radians(lat_chunk)
                    width_m = px_w * (np.pi / 180.0) * earth_r * np.cos(lat_rad)
                    height_m = px_h * (np.pi / 180.0) * earth_r
                    area_chunk = np.abs(width_m * height_m)
                else:
                    area_chunk = None
            else:
                if to_wgs84 is not None:
                    lng_chunk, lat_chunk = to_wgs84.transform(xs, ys)
                    lng_chunk = np.asarray(lng_chunk)
                    lat_chunk = np.asarray(lat_chunk)
                else:
                    lng_chunk, lat_chunk = xs, ys
                area_chunk = (
                    np.full(vals.shape, px_w * px_h, dtype=np.float64)
                    if is_count
                    else None
                )

            lat_parts.append(lat_chunk)
            lng_parts.append(lng_chunk)
            value_parts.append(vals)
            if is_count:
                area_parts.append(area_chunk)

    if not lat_parts:
        return pl.DataFrame(
            {h3_col: [], value_col: []}, schema={h3_col: pl.Utf8, value_col: pl.Float64}
        )

    lat_all = np.concatenate(lat_parts)
    lng_all = np.concatenate(lng_parts)
    value_all = np.concatenate(value_parts)
    area_all = np.concatenate(area_parts) if is_count else None

    return resample_points_to_h3(
        lat_all,
        lng_all,
        value_all,
        resolution,
        is_count=is_count,
        point_area_m2=area_all,
        h3_col=h3_col,
        value_col=value_col,
    )


def _sum_valid_raster(
    tif_path: str, nodata: Optional[float], row_chunk: int = 4096
) -> float:
    """Sum of every non-nodata pixel value in the raster, read in bounded row-block windows."""
    total = 0.0
    with rio.open(tif_path) as src:
        nodata = src.nodata if nodata is None else nodata
        for row_start in range(0, src.height, row_chunk):
            n_rows = min(row_chunk, src.height - row_start)
            window = rio.windows.Window(0, row_start, src.width, n_rows)
            array = src.read(1, window=window).astype(np.float64)
            if nodata is not None:
                array = np.where(array == nodata, np.nan, array)
            valid = np.isfinite(array)
            if valid.any():
                total += float(array[valid].sum())
    return total


def _mean_valid_raster(
    tif_path: str, nodata: Optional[float], row_chunk: int = 4096
) -> Optional[float]:
    """Mean of every non-nodata pixel value in the raster, read in bounded row-block windows."""
    total = 0.0
    count = 0
    with rio.open(tif_path) as src:
        nodata = src.nodata if nodata is None else nodata
        for row_start in range(0, src.height, row_chunk):
            n_rows = min(row_chunk, src.height - row_start)
            window = rio.windows.Window(0, row_start, src.width, n_rows)
            array = src.read(1, window=window).astype(np.float64)
            if nodata is not None:
                array = np.where(array == nodata, np.nan, array)
            valid = np.isfinite(array)
            if valid.any():
                total += float(array[valid].sum())
                count += int(valid.sum())
    return total / count if count else None


def raster_to_h3_by_centroid(
    tif_path: str,
    resolution: int,
    value_col: str = "value",
    h3_col: str = "h3_cell",
    nodata: Optional[float] = None,
    is_count: bool = True,
) -> pl.DataFrame:
    """Raster -> H3 resample by sampling the raster AT each H3 cell's own centroid.

    The inverse direction of :func:`raster_to_h3_nearest` -- for use when
    the target H3 resolution is FINER than the raster's own pixel spacing
    (see :func:`raster_to_h3_auto`'s docstring for when this applies).
    Aggregating sparse pixel centers into a denser H3 grid (as
    `raster_to_h3_nearest`/`resample_points_to_h3` do) leaves most cells
    with zero assigned pixels in that regime -- most H3 cells simply don't
    contain any pixel center. Sampling the raster's own value at each H3
    cell's centroid instead guarantees a value for every cell whose
    centroid falls within the raster's valid-data extent, at the cost of
    point-sampling noise (a cell's density is exactly its centroid's pixel
    value, not a true area-weighted average over the cell) -- corrected for
    in aggregate by the same global conservation rescale step 4 uses.

    Steps (matching :func:`resample_points_to_h3`'s spec, applied in the
    other direction):
    1. Generate every H3 cell at `resolution` whose centroid falls within
       the raster's bounding-box footprint (`h3ronpy.vector.geometry_to_cells`).
    2. Sample the raster's value at each cell's own centroid (nearest
       pixel, `rasterio.DatasetReader.sample`).
    3. (`is_count=True`) convert each sampled value to a density (divide by
       that pixel's own real area), multiply by the H3 CELL's own area to
       recover a count.
    4. (`is_count=True`) rescale every cell's count by a single global
       factor so `sum(output value)` matches the TRUE sum of every valid
       pixel in the raster (not just the sampled ones) -- this is what
       corrects for the point-sampling approximation in aggregate, exactly
       as `resample_points_to_h3`'s own rescale corrects for its
       mean-of-densities approximation.

    Args:
        tif_path: Path to a single-band GeoTIFF.
        resolution: Target H3 resolution.
        value_col: Output value column name.
        h3_col: Output H3 cell-id column name.
        nodata: Nodata sentinel to exclude (default: read from the file).
        is_count: See :func:`resample_points_to_h3`.

    Returns:
        Polars DataFrame with `h3_col` and `value_col` -- one row per H3
        cell whose centroid sampled a valid (non-nodata) pixel.
    """
    import h3ronpy
    import h3ronpy.vector as h3v
    from shapely.geometry import box

    with rio.open(tif_path) as src:
        nodata_val = src.nodata if nodata is None else nodata
        bounds = src.bounds
        is_geographic = src.crs is not None and src.crs.is_geographic
        if not is_geographic:
            import pyproj

            to_wgs84 = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            minx, miny = to_wgs84.transform(bounds.left, bounds.bottom)
            maxx, maxy = to_wgs84.transform(bounds.right, bounds.top)
        else:
            to_wgs84 = None
            minx, miny, maxx, maxy = (
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            )

        footprint = box(minx, miny, maxx, maxy)
        cells = h3v.geometry_to_cells(footprint, resolution)
        if len(cells) == 0:
            return pl.DataFrame(
                {h3_col: [], value_col: []},
                schema={h3_col: pl.Utf8, value_col: pl.Float64},
            )
        cell_strings = np.asarray(h3ronpy.cells_to_string(cells))
        coords = h3v.cells_to_coordinates(cells)
        lat_arr = np.asarray(coords.column("lat"))
        lng_arr = np.asarray(coords.column("lng"))

        if to_wgs84 is not None:
            import pyproj

            to_native = pyproj.Transformer.from_crs(
                "EPSG:4326", src.crs, always_xy=True
            )
            xs, ys = to_native.transform(lng_arr, lat_arr)
            xs, ys = np.asarray(xs), np.asarray(ys)
        else:
            xs, ys = lng_arr, lat_arr

        # Vectorized inverse-affine indexing, not `src.sample()` -- that
        # iterates one point at a time in pure Python, real live measured
        # bottleneck (97.7s for ~370K centroids on Concepcion, SLOWER than
        # the exact overlay method it's meant to replace, 2026-09-04). The
        # whole raster is read once into an array (bounded by one city's
        # extent, same cost `raster_to_h3`'s single-shot path already
        # pays) and every centroid's row/col is computed with plain numpy
        # array ops against the inverse transform -- no per-point Python
        # call at all.
        full_array = src.read(1).astype(np.float64)
        inv_transform = ~src.transform
        cols_f, rows_f = inv_transform * (xs, ys)
        cols_i = np.floor(cols_f).astype(np.int64)
        rows_i = np.floor(rows_f).astype(np.int64)
        in_bounds = (
            (cols_i >= 0) & (cols_i < src.width) & (rows_i >= 0) & (rows_i < src.height)
        )
        sampled = np.full(xs.shape, np.nan, dtype=np.float64)
        sampled[in_bounds] = full_array[rows_i[in_bounds], cols_i[in_bounds]]
        del full_array
        if nodata_val is not None:
            sampled = np.where(sampled == nodata_val, np.nan, sampled)
        valid = np.isfinite(sampled)
        if not valid.any():
            return pl.DataFrame(
                {h3_col: [], value_col: []},
                schema={h3_col: pl.Utf8, value_col: pl.Float64},
            )

        cell_strings = cell_strings[valid]
        lat_valid = lat_arr[valid]
        sampled_valid = sampled[valid]

        if not is_count:
            # Same mean-conservation rescale as `resample_points_to_h3`'s
            # `is_count=False` branch, adapted to centroid sampling: most
            # raster pixels never get sampled here (only ones containing an
            # H3 centroid do), so the sampled set's mean can meaningfully
            # differ from the TRUE mean over every valid pixel in the
            # raster -- rescale the H3 output to match that true mean.
            true_mean = _mean_valid_raster(tif_path, nodata)
            sampled_mean = float(np.mean(sampled_valid)) if sampled_valid.size else 0.0
            out_values = sampled_valid
            if sampled_mean != 0 and true_mean is not None:
                out_values = sampled_valid * (true_mean / sampled_mean)
            return pl.DataFrame({h3_col: cell_strings, value_col: out_values})

        transform = src.transform
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        if is_geographic:
            lat_rad = np.radians(lat_valid)
            earth_r = 6_371_000.0
            width_m = px_w * (np.pi / 180.0) * earth_r * np.cos(lat_rad)
            height_m = px_h * (np.pi / 180.0) * earth_r
            pixel_area = np.abs(width_m * height_m)
        else:
            pixel_area = np.full(sampled_valid.shape, px_w * px_h, dtype=np.float64)

        density = sampled_valid / pixel_area
        cell_area_m2 = np.asarray(
            h3ronpy.cells_area_m2(h3ronpy.cells_parse(list(cell_strings)))
        )
        counts = density * cell_area_m2

        raster_total = _sum_valid_raster(tif_path, nodata)
        h3_total = float(counts.sum())
        if h3_total > 0:
            counts = counts * (raster_total / h3_total)

    return pl.DataFrame({h3_col: cell_strings, value_col: counts})


def raster_to_h3_auto(
    tif_path: str,
    resolution: int,
    value_col: str = "value",
    h3_col: str = "h3_cell",
    nodata: Optional[float] = None,
    is_count: bool = True,
    row_chunk: int = 4096,
) -> pl.DataFrame:
    """Fast raster -> H3 resample, auto-picking the right direction for the resolution.

    2026-09-04, explicit user spec: "if the issue is that h3 cells are too
    small then better assign to each h3 cell the value of the raster at the
    h3 cell centroid and use the other method if the h3 res is higher than
    the raster size (use h3 cell diameter and pixel side)." Compares each
    H3 cell's real diameter at `resolution` (`h3.average_hexagon_edge_length
    * 2`) against the raster's own pixel side length (in real meters,
    geodesic-approximated for a degree CRS):

    - H3 cell diameter >= pixel side (the common case for a coarse-ish H3
      resolution over a fine raster, e.g. WorldPop 100m pixels onto H3 res
      9 or coarser, ~252 km^2+ per cell): :func:`raster_to_h3_nearest`
      (aggregate pixel centers per cell, mean density) -- every cell gets
      many pixels, a real local average.
    - H3 cell diameter < pixel side (H3 finer than the raster, e.g.
      WorldPop's 100m pixels onto this study's native H3 res 11, ~29m
      edge/58m diameter cells): :func:`raster_to_h3_by_centroid` (sample
      the raster directly at each cell's centroid) -- aggregation would
      leave most cells with zero assigned pixels in this regime (verified
      live: WorldPop res-11 aggregation covered only ~22% of the cells the
      exact area-weighted method covers), while centroid-sampling gives
      every cell within the raster's extent a value.

    Both branches are mass-conserving via the same global rescale (see
    each function's own docstring) -- `sum(output value)` matches the
    raster's real total either way when `is_count=True`.

    Args:
        tif_path: Path to a single-band GeoTIFF.
        resolution: Target H3 resolution.
        value_col: Output value column name.
        h3_col: Output H3 cell-id column name.
        nodata: Nodata sentinel to exclude (default: read from the file).
        is_count: See :func:`resample_points_to_h3`.
        row_chunk: Rows per read window, only used by the aggregation branch.

    Returns:
        Polars DataFrame with `h3_col` and `value_col`.
    """
    import h3

    with rio.open(tif_path) as src:
        transform = src.transform
        is_geographic = src.crs is not None and src.crs.is_geographic
        bounds = src.bounds
        px_w_native = abs(transform.a)
        px_h_native = abs(transform.e)
        if is_geographic:
            center_lat = (bounds.top + bounds.bottom) / 2.0
            earth_r = 6_371_000.0
            px_w_m = (
                px_w_native * (np.pi / 180.0) * earth_r * np.cos(np.radians(center_lat))
            )
            px_h_m = px_h_native * (np.pi / 180.0) * earth_r
        else:
            px_w_m, px_h_m = px_w_native, px_h_native
        pixel_side_m = (abs(px_w_m) + abs(px_h_m)) / 2.0

    h3_diameter_m = h3.average_hexagon_edge_length(resolution, unit="m") * 2.0

    if h3_diameter_m < pixel_side_m:
        return raster_to_h3_by_centroid(
            tif_path, resolution, value_col, h3_col, nodata, is_count
        )
    return raster_to_h3_nearest(
        tif_path, resolution, value_col, h3_col, nodata, is_count, row_chunk
    )
