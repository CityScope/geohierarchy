"""Aggregate line-geometry attribute data onto an arbitrary polygon layer.

This module is a thin convenience wrapper around :class:`~geohierarchy.core.GeoHierarchy`
for the common one-shot task of turning "edges" data -- any GeoDataFrame of
``LineString``/``MultiLineString`` geometries with attribute columns, such as
street network edges, rail lines, or utility lines -- into an aggregated
table on top of a polygon geometry layer (H3 cells from
:func:`~geohierarchy.utils.h3_cells`, administrative boundaries, or any other
polygon ``GeoDataFrame``). Internally it builds a two-level hierarchy (the
target polygons as the only level) and injects the edges via
:meth:`~geohierarchy.core.GeoHierarchy.add_vector_data` with
``geoweight_by="length"``, since overlap between a line and a polygon is
naturally weighted by the length of line inside each polygon, not by area.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import polars as pl

from .aggregation import AggregationStrategy, Max, Mean
from .core import GeoHierarchy


def edges_to_level(
    edges_gdf: gpd.GeoDataFrame,
    level_gdf: gpd.GeoDataFrame,
    id_col: str,
    columns: Union[str, List[str]],
    agg: Union[AggregationStrategy, Dict[str, AggregationStrategy], None] = None,
    geometry_col: str = "geometry",
    crs=None,
) -> gpd.GeoDataFrame:
    """Aggregate columns from a line-geometry GeoDataFrame onto a polygon level.

    Builds a single-level :class:`~geohierarchy.core.GeoHierarchy` from
    ``level_gdf``, injects ``edges_gdf``'s attribute columns into it with
    :meth:`~geohierarchy.core.GeoHierarchy.add_vector_data` (length-weighted,
    since ``edges_gdf`` holds line geometries), and returns the resulting
    polygon layer with just the requested columns.

    Args:
        edges_gdf: Source GeoDataFrame of line geometries (e.g. street
            network edges) carrying the attribute columns to aggregate.
        level_gdf: Target polygon GeoDataFrame to aggregate onto -- an H3
            grid from :func:`~geohierarchy.utils.h3_cells`, an
            administrative boundary layer, or any other polygon layer.
        id_col: Name of the unique id column in ``level_gdf``.
        columns: Column name or list of column names in ``edges_gdf`` to
            aggregate onto ``level_gdf``.
        agg: Aggregation strategy applied to every column in ``columns``,
            or a per-column mapping of ``{column: AggregationStrategy}``.
            Defaults to :class:`~geohierarchy.aggregation.Mean` for every
            column if not given.
        geometry_col: Name of the geometry column in ``edges_gdf``, in
            case it isn't already named ``"geometry"``.
        crs: Coordinate reference system the hierarchy (and the returned
            GeoDataFrame) should use. Defaults to ``level_gdf``'s own CRS.

    Returns:
        A copy of ``level_gdf`` (reprojected to ``crs``) with ``id_col``,
        geometry, and the aggregated ``columns`` -- one row per polygon.
    """
    if isinstance(columns, str):
        columns = [columns]

    if geometry_col != "geometry" and geometry_col in edges_gdf.columns:
        edges_gdf = edges_gdf.rename(columns={geometry_col: "geometry"}).set_geometry(
            "geometry"
        )

    resolved_agg: Union[AggregationStrategy, Dict[str, AggregationStrategy]] = (
        agg if agg is not None else Mean()
    )
    if isinstance(resolved_agg, dict):
        resolved_agg = {col: resolved_agg[col] for col in columns}

    hierarchy_crs = crs if crs is not None else level_gdf.crs

    gh = GeoHierarchy(crs=hierarchy_crs)
    gh.add_level("target", level_gdf, id_col=id_col, agg=resolved_agg)
    gh.add_vector_data(
        edges_gdf,
        level="target",
        columns=columns,
        agg=resolved_agg,
        geoweight_by="length",
    )
    gh.propagate()

    result = gh.get_level("target")
    target_id_col = gh.id_cols["target"]
    return result[[target_id_col, "geometry", *columns]].rename(
        columns={target_id_col: id_col}
    )


def _densify_lines(edges_proj: gpd.GeoDataFrame, columns: Sequence[str], step_m: float):
    """Sample every line in `edges_proj` into points roughly `step_m` apart, carrying `columns`.

    Pure coordinate math on the vertex array (`GeoSeries.get_coordinates()`
    plus linear interpolation between consecutive same-line vertices) --
    no per-line Shapely object, no Python loop over lines. The variable
    number of interpolated points per segment is built with one
    `np.repeat`/`np.concatenate` pass over every segment at once.

    Returns:
        `(xy, values)`: `xy` is an `(N, 2)` NumPy array of sample-point
        coordinates; `values` is a `{column: (N,) array}` dict.
    """
    coords = edges_proj.geometry.get_coordinates()
    idx = coords.index.to_numpy()
    x = coords["x"].to_numpy()
    y = coords["y"].to_numpy()
    # Positional row (0..n-1) within `edges_proj` for each vertex's parent line --
    # `coords.index` holds `edges_proj`'s (possibly non-default) index *labels*,
    # repeated once per vertex, mapped back to row order for indexing `columns`.
    row_of = edges_proj.index.get_indexer(coords.index)
    col_values = {c: edges_proj[c].to_numpy() for c in columns}

    same_line = idx[1:] == idx[:-1]
    x0, y0 = x[:-1][same_line], y[:-1][same_line]
    x1, y1 = x[1:][same_line], y[1:][same_line]
    seg_row = row_of[:-1][same_line]
    dx, dy = x1 - x0, y1 - y0
    seg_len = np.hypot(dx, dy)
    n_steps = np.maximum(1, np.ceil(seg_len / step_m).astype(np.int64))

    total = int(n_steps.sum())
    seg_id = np.repeat(np.arange(len(n_steps)), n_steps)
    starts = np.concatenate(([0], np.cumsum(n_steps)[:-1]))
    within = np.arange(total) - np.repeat(starts, n_steps)
    t = within / np.repeat(n_steps, n_steps)

    seg_x = x0[seg_id] + t * dx[seg_id]
    seg_y = y0[seg_id] + t * dy[seg_id]
    seg_rows = seg_row[seg_id]

    # Every original vertex too (covers each line's final point, at t=1, which the
    # per-segment sampling above never reaches, plus single-vertex/degenerate lines).
    all_x = np.concatenate([seg_x, x])
    all_y = np.concatenate([seg_y, y])
    all_rows = np.concatenate([seg_rows, row_of])
    values = {c: col_values[c][all_rows] for c in columns}
    return np.column_stack([all_x, all_y]), values


def edges_to_h3_by_distance(
    edges_gdf: gpd.GeoDataFrame,
    h3_cells: Sequence[str],
    columns: Union[str, List[str]],
    resolution: int,
    margin_m: float = 10.0,
    agg: Union[AggregationStrategy, Dict[str, AggregationStrategy], None] = None,
    fallback_radius_multiplier: Optional[float] = 5.0,
) -> pl.DataFrame:
    """Aggregate line-geometry columns onto H3 cells by centroid distance, not geometric overlap.

    An alternative to :func:`edges_to_level` for H3 targets specifically:
    instead of a GEOS polygon/line overlay (which can be expensive or even
    memory-unstable for very large edge counts -- e.g. `shapely.union_all`
    over 800K+ street edges failing outright with `GEOSException:
    std::bad_alloc`), "touching" is defined purely by distance -- an H3
    cell counts as touched by a line if any point on that line falls
    within ``margin_m`` of the cell's centroid, checked via a
    `scipy.spatial.cKDTree` radius query against densified line-sample
    points (see :func:`_densify_lines`). No Shapely geometry object is
    built anywhere in this path.

    Args:
        edges_gdf: Source GeoDataFrame of line geometries carrying the
            attribute columns to aggregate.
        h3_cells: Candidate H3 cells to test (e.g. every cell with
            population, to skip cells irrelevant to the caller).
        columns: Column name or list of column names in ``edges_gdf`` to
            aggregate onto touching cells.
        resolution: H3 resolution of ``h3_cells`` -- used to size the
            search radius (``margin_m`` beyond one cell's approximate
            diameter, `2 * h3.average_hexagon_edge_length(resolution)`).
        margin_m: Extra distance beyond one cell diameter still counted as
            "touching".
        agg: Aggregation strategy applied to every column, or a per-column
            mapping. Defaults to :class:`~geohierarchy.aggregation.Max`
            (matching this function's typical use: assigning each cell the
            best/highest value among nearby edges).
        fallback_radius_multiplier: Guarantees every cell in ``h3_cells``
            gets a value. Any cell that matched no edge within ``radius``
            (``margin_m + 2 * edge_len_m``) falls back to its single
            nearest edge-sample point, as long as that point is within
            ``fallback_radius_multiplier * edge_len_m`` (the H3 cell's
            approximate edge length at ``resolution``) of the cell
            centroid. Cells with no edge at all within that fallback
            distance are still left absent from the result -- callers
            that need an explicit "no coverage" sentinel (e.g. filling
            with 0.0) should do so themselves on the missing rows. Set to
            ``None`` to disable the fallback and keep the original
            behavior (unmatched cells simply absent).

    Returns:
        Polars DataFrame with ``h3_cell`` and ``columns``, one row per
        cell within ``margin_m`` of at least one edge. Cells with no
        nearby edge are simply absent.
    """
    import h3
    import h3ronpy
    import h3ronpy.vector as h3v
    import pyproj
    from scipy.spatial import cKDTree

    if isinstance(columns, str):
        columns = [columns]
    h3_cells = list(h3_cells)

    empty_schema = {"h3_cell": pl.Utf8, **{c: pl.Float64 for c in columns}}
    if not h3_cells or edges_gdf.empty:
        return pl.DataFrame({k: [] for k in empty_schema}, schema=empty_schema)

    resolved_agg: Union[AggregationStrategy, Dict[str, AggregationStrategy]] = (
        agg if agg is not None else Max()
    )
    per_column_agg: Dict[str, AggregationStrategy] = (
        resolved_agg
        if isinstance(resolved_agg, dict)
        else {c: resolved_agg for c in columns}
    )

    edges_proj = (
        edges_gdf
        if edges_gdf.crs is not None and edges_gdf.crs.is_projected
        else edges_gdf.to_crs(edges_gdf.estimate_utm_crs())
    )
    edge_len_m = h3.average_hexagon_edge_length(resolution, unit="m")
    radius = margin_m + 2 * edge_len_m
    step_m = max(edge_len_m / 2, 5.0)

    point_xy, point_values = _densify_lines(edges_proj, columns, step_m)
    tree = cKDTree(point_xy)

    cell_ids = h3ronpy.cells_parse(h3_cells)
    latlng = h3v.cells_to_coordinates(cell_ids)
    lat = np.asarray(latlng["lat"])
    lng = np.asarray(latlng["lng"])
    to_proj = pyproj.Transformer.from_crs("EPSG:4326", edges_proj.crs, always_xy=True)
    cell_x, cell_y = to_proj.transform(lng, lat)
    cell_xy = np.column_stack([cell_x, cell_y])

    neighbor_lists = tree.query_ball_point(cell_xy, r=radius)
    matched_idx = [i for i, neighbors in enumerate(neighbor_lists) if neighbors]
    if not matched_idx:
        # 2026-08-31: previously returned here unconditionally, which meant a
        # call whose candidate `h3_cells` happen to have ZERO direct matches
        # (e.g. a batch of candidate cells that are only reachable via the
        # nearest-street fallback, not within `radius` of any sample point)
        # skipped the fallback path entirely -- silently dropping cells that
        # `fallback_radius_multiplier` should have covered. Real bug this
        # surfaced: `h3_population.population_and_access_to_h3`'s per-chunk
        # candidate-cell batching (added to bound peak memory for
        # megaregion-scale AOIs) produces exactly such all-fallback-no-match
        # batches for cells near a batch boundary. Fall through to the same
        # fallback logic below instead of returning early -- `matched_result`
        # is simply the empty frame in that case.
        matched_result = pl.DataFrame(
            {k: [] for k in empty_schema}, schema=empty_schema
        )
    else:
        # Long/exploded (cell, neighbor-point) table -- one row per (candidate cell, nearby
        # sample point) pair -- so the configured `AggregationStrategy` can group-by/agg it
        # exactly like any other polars aggregation, instead of hardcoding e.g. a max().
        row_cell = np.repeat(
            [h3_cells[i] for i in matched_idx],
            [len(neighbor_lists[i]) for i in matched_idx],
        )
        row_neighbor = np.concatenate([neighbor_lists[i] for i in matched_idx])
        long_df = pl.DataFrame(
            {"h3_cell": row_cell, **{c: point_values[c][row_neighbor] for c in columns}}
        )

        agg_exprs = [
            expr for c in columns for expr in per_column_agg[c].upscale_aggs([c])
        ]
        matched_result = long_df.group_by("h3_cell").agg(agg_exprs)

    if not fallback_radius_multiplier:
        return matched_result

    matched_cells = set(matched_result["h3_cell"].to_list())
    missing_mask = np.array([c not in matched_cells for c in h3_cells])
    if not missing_mask.any():
        return matched_result

    missing_cells = [c for c, m in zip(h3_cells, missing_mask) if m]
    missing_xy = cell_xy[missing_mask]

    fallback_max_dist = fallback_radius_multiplier * edge_len_m
    nn_dist, nn_idx = tree.query(missing_xy, k=1)
    nn_dist = np.atleast_1d(nn_dist)
    nn_idx = np.atleast_1d(nn_idx)

    within = nn_dist <= fallback_max_dist
    if not within.any():
        return matched_result

    fallback_cells = [c for c, w in zip(missing_cells, within) if w]
    fallback_idx = nn_idx[within]
    fallback_df = pl.DataFrame(
        {
            "h3_cell": fallback_cells,
            **{c: point_values[c][fallback_idx] for c in columns},
        }
    )

    return pl.concat([matched_result, fallback_df], how="vertical")
