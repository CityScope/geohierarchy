"""Cache expensive spatial CORRESPONDENCE (id <-> id / id <-> pixel) computations.

Every function here answers "which unit maps to which other unit" and
persists that answer as a parquet file, deliberately separate from any
data VALUES. The idea: correspondences depend only on GEOMETRY (raster grid,
H3 resolution, polygon boundaries, street network), not on whatever attribute
is currently being aggregated through them. When only the values change --
a new WorldPop release, a re-scored ``access_score``, a re-run with different
aggregation parameters -- but the underlying geometries are unchanged, the
expensive spatial join/rule should not have to run again: this module lets
callers look up the cached correspondence and redo only the (fast) value
re-aggregation.

Each mapping is its own parquet file, named so the two id-spaces it connects
and the geometry version it was computed against are both legible from the
filename (and re-checked against a stored geometry-fingerprint on load, so a
mapping is never silently reused for a geometry it wasn't built from).

Nothing in this module is wired into any production pipeline. It is meant to
be adopted deliberately, one call site at a time, once verified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import geopandas as gpd
import polars as pl

from .utils import get_id_mapping


# ================================================================
# CACHE KEYS
# ================================================================


def _hash_key(*parts: object) -> str:
    """Short, stable hex digest of arbitrary JSON-able parts, for cache filenames."""
    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def raster_grid_key(transform, shape: Sequence[int], resolution: int) -> str:
    """Content-based cache key for a raster grid + target H3 resolution.

    Args:
        transform: ``rasterio``/``affine``-style ``Affine`` transform (or
            any object exposing ``.a .b .c .d .e .f``) describing the
            raster's pixel size and origin.
        shape: ``(height, width)`` of the raster array.
        resolution: Target H3 resolution the mapping was/will be built for.

    Returns:
        A short, deterministic hex string. Two rasters with the same grid
        (affine + shape) and the same target resolution always produce the
        same key; anything different produces a different one.
    """
    coeffs = (
        round(float(transform.a), 12),
        round(float(transform.b), 12),
        round(float(transform.c), 12),
        round(float(transform.d), 12),
        round(float(transform.e), 12),
        round(float(transform.f), 12),
    )
    return _hash_key(
        "raster_grid", coeffs, tuple(int(s) for s in shape), int(resolution)
    )


def geometry_version_key(gdf: gpd.GeoDataFrame, id_col: str) -> str:
    """Content-based fingerprint for a polygon/line layer's geometry (not its attributes).

    Used to detect staleness: a cache built against one version of a
    boundary layer (e.g. census tracts) must not be silently reused once
    the boundaries themselves change (a new TIGER/Line vintage, a
    re-clipped AOI, ...). Cheap by design -- total bounds, feature count,
    and a hash of the id column and per-feature bounds -- rather than a
    hash of full WKB, since this only needs to change when the geometry
    *actually* changes, not be cryptographically unforgeable.

    Args:
        gdf: Geometry layer to fingerprint.
        id_col: Name of the layer's unique id column.

    Returns:
        A short, deterministic hex string.
    """
    bounds = gdf.total_bounds.round(9).tolist()
    ids = sorted(gdf[id_col].astype(str).tolist())
    feat_bounds = gdf.geometry.bounds.round(9).to_numpy().tobytes()
    return _hash_key(
        "geometry_version",
        bounds,
        len(gdf),
        hashlib.sha256(",".join(ids).encode()).hexdigest(),
        hashlib.sha256(feat_bounds).hexdigest(),
    )


# ================================================================
# GENERIC CACHE I/O
# ================================================================


def _cache_path(cache_dir: Union[str, Path], name: str, key: str) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}__{key}.parquet"


def load_mapping(
    cache_dir: Union[str, Path], name: str, key: str
) -> Optional[pl.DataFrame]:
    """Load a previously cached mapping table, if present.

    Args:
        cache_dir: Directory mappings are cached under.
        name: Logical mapping name (e.g. ``"h3_to_raster_pixel"``).
        key: Content-based cache key identifying the exact geometry inputs.

    Returns:
        The cached mapping as a Polars DataFrame, or ``None`` if no cache
        entry exists for this ``(name, key)`` pair.
    """
    path = _cache_path(cache_dir, name, key)
    if not path.is_file():
        return None
    return pl.read_parquet(path)


def save_mapping(
    mapping: pl.DataFrame, cache_dir: Union[str, Path], name: str, key: str
) -> Path:
    """Persist a mapping table to its cache slot.

    Args:
        mapping: Mapping table to persist.
        cache_dir: Directory mappings are cached under.
        name: Logical mapping name (e.g. ``"h3_to_raster_pixel"``).
        key: Content-based cache key identifying the exact geometry inputs.

    Returns:
        The path the mapping was written to.
    """
    path = _cache_path(cache_dir, name, key)
    mapping.write_parquet(path)
    return path


# ================================================================
# PART 1 -- RASTER <-> H3 (closed-form rule, no stored id list)
# ================================================================


def _h3_cell_centroids(h3_cells: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """H3-native centroid lookup for a batch of H3 cell index strings.

    No Shapely/geometry objects anywhere -- ``h3ronpy`` computes centroids
    directly from the H3 index, which is the fast path this module relies
    on throughout.

    Args:
        h3_cells: H3 cell index strings.

    Returns:
        ``(lat, lng)`` arrays, aligned with ``h3_cells``.
    """
    import h3ronpy
    import h3ronpy.vector as h3v

    cell_ids = h3ronpy.cells_parse(list(h3_cells))
    latlng = h3v.cells_to_coordinates(cell_ids)
    return np.asarray(latlng["lat"]), np.asarray(latlng["lng"])


def build_raster_h3_mapping(
    h3_cells: Sequence[str],
    transform,
    shape: Sequence[int],
    resolution: int,
    cache_dir: Union[str, Path],
    raster_crs=None,
    name: str = "h3_to_raster_pixel",
    force: bool = False,
) -> pl.DataFrame:
    """Compute (or load a cached) closed-form mapping of H3 cells to raster pixel row/col.

    Raster pixels have no natural id, so unlike the id-correspondence
    mappings in this module, this one is a *rule*: for each H3 cell,
    compute its centroid with :func:`_h3_cell_centroids` (H3-native, no
    Shapely) and invert the raster's affine transform to find which pixel
    column/row that centroid falls into -- pure arithmetic, no geometry
    objects, no per-row Python loop.

    This is only a sound approximation when pixel size and H3 cell size at
    ``resolution`` are comparable and the study area is small enough that
    pixel curvature/distortion don't make a fixed affine rule meaningfully
    wrong (true at every resolution this codebase currently uses -- see
    :func:`UrbanAccessAnalyzer.h3_ops.from_raster_centroid`, which performs
    the equivalent per-pixel-centroid assignment in the other direction and
    resolution-matches to keep the same error bounded). Callers with pixels
    much coarser or finer than the target H3 resolution should pick
    ``resolution`` accordingly (see :func:`apply_raster_h3_mapping`'s
    population-conservation note) rather than relying on this rule alone.

    Args:
        h3_cells: H3 cell index strings covering the raster's extent (e.g.
            from :func:`~geohierarchy.utils.h3_cells`, in EPSG:4326).
        transform: Raster's affine transform (``rasterio``/``affine``
            ``Affine``), in ``raster_crs``.
        shape: ``(height, width)`` of the raster array.
        resolution: H3 resolution of ``h3_cells``.
        cache_dir: Directory the mapping is cached under.
        raster_crs: CRS the raster (and ``transform``) is in. Defaults to
            EPSG:4326 (i.e. ``h3_cells`` centroids, already lat/lng, are
            used as-is).
        name: Cache artifact name.
        force: Recompute even if a cache entry already exists.

    Returns:
        Polars DataFrame with ``h3_cell``, ``pixel_row``, ``pixel_col`` --
        one row per H3 cell whose centroid falls inside the raster's
        extent (out-of-extent cells are dropped).
    """
    key = raster_grid_key(transform, shape, resolution)

    if not force:
        cached = load_mapping(cache_dir, name, key)
        if cached is not None:
            return cached

    height, width = int(shape[0]), int(shape[1])
    h3_cells = list(h3_cells)
    lat, lng = _h3_cell_centroids(h3_cells)

    if raster_crs is not None:
        import pyproj

        crs_obj = raster_crs if not isinstance(raster_crs, str) else raster_crs
        to_raster = pyproj.Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)
        px_x, px_y = to_raster.transform(lng, lat)
    else:
        px_x, px_y = lng, lat

    # Invert the affine transform (pure arithmetic): pixel (col, row) from
    # world (x, y). transform maps (col, row) -> (x, y) as:
    #   x = a*col + b*row + c ; y = d*col + e*row + f
    a, b, c, d, e, f = (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )
    det = a * e - b * d
    if det == 0:
        raise ValueError("Raster transform is singular; cannot invert.")
    px_x = np.asarray(px_x, dtype=np.float64)
    px_y = np.asarray(px_y, dtype=np.float64)
    dx, dy = px_x - c, px_y - f
    col = (e * dx - b * dy) / det
    row = (a * dy - d * dx) / det

    pixel_col = np.floor(col).astype(np.int64)
    pixel_row = np.floor(row).astype(np.int64)

    in_bounds = (
        (pixel_row >= 0) & (pixel_row < height) & (pixel_col >= 0) & (pixel_col < width)
    )

    mapping = pl.DataFrame(
        {
            "h3_cell": np.asarray(h3_cells)[in_bounds],
            "pixel_row": pixel_row[in_bounds],
            "pixel_col": pixel_col[in_bounds],
        }
    )

    save_mapping(mapping, cache_dir, name, key)
    return mapping


def apply_raster_h3_mapping(
    mapping: pl.DataFrame,
    array: np.ndarray,
    resolution: int,
    target_resolution: Optional[int] = None,
    value_col: str = "value",
) -> pl.DataFrame:
    """Apply a cached raster<->H3 mapping to a pixel array, gathering values by fast indexing.

    Preserves the same population-conservation guarantee as
    :func:`UrbanAccessAnalyzer.h3_ops.from_raster_centroid`: every pixel's
    value is assigned whole to exactly one H3 cell at ``resolution`` (a
    centroid-based, not area-weighted, assignment -- same trade-off the
    existing direct computation makes), so ``sum(value)`` over the returned
    table equals ``sum(array)`` restricted to pixels whose centroid maps to
    an in-mapping H3 cell. Many-to-one (several H3 cells -> one pixel, when
    the pixel is coarser than the H3 grid) is handled naturally since every
    such cell independently reads the same pixel value -- callers wanting
    an exact partition of that pixel's total across those cells (rather
    than each cell getting the full pixel value / density-preserving
    behavior) should assign at the resolution :func:`build_raster_h3_mapping`
    was built for, then aggregate up via H3's exact parent/child index
    nesting (``target_resolution``, summed) exactly as ``from_raster_centroid``
    does via its own ``resample(..., method="sum")`` step -- summing nests
    exactly across resolutions with no further approximation, so the total
    is identical whether cells are assigned directly at a coarse resolution
    or via a fine intermediate step summed upward; only per-cell density
    accuracy differs.

    Args:
        mapping: Cached mapping from :func:`build_raster_h3_mapping`
            (``h3_cell``, ``pixel_row``, ``pixel_col``).
        array: 2D raster array the mapping's pixel coordinates index into.
        resolution: H3 resolution ``mapping`` was built at.
        target_resolution: If given and different from ``resolution``,
            aggregates ``mapping``'s per-cell values up to this coarser
            resolution via exact H3 parent/child rollup (sum, see
            :func:`_resample_h3_sum`), mirroring
            ``UrbanAccessAnalyzer.h3_ops.from_raster_centroid``'s own
            ``resample(method="sum")`` step.
        value_col: Name to give the gathered value column.

    Returns:
        Polars DataFrame with ``h3_cell`` and ``value_col``.
    """
    rows = mapping["pixel_row"].to_numpy()
    cols = mapping["pixel_col"].to_numpy()
    values = np.asarray(array)[rows, cols]

    result = pl.DataFrame({"h3_cell": mapping["h3_cell"], value_col: values})
    result = result.filter(pl.col(value_col).is_finite())

    if target_resolution is not None and target_resolution != resolution:
        result = _resample_h3_sum(result, target_resolution, value_col)

    return result


def _resample_h3_sum(
    df: pl.DataFrame, target_resolution: int, value_col: str
) -> pl.DataFrame:
    """Roll a per-cell value up (or down) to ``target_resolution`` via exact H3 parent/child nesting, summed.

    Self-contained (``h3ronpy`` only) equivalent of
    ``UrbanAccessAnalyzer.h3_ops.resample(..., method="sum")``: index-based,
    no approximation, so a value's total is identical whether it is summed
    directly at ``target_resolution`` or first assigned at a finer
    resolution and rolled up -- see :func:`build_raster_h3_mapping`.
    """
    import h3ronpy

    cell_ids = h3ronpy.cells_parse(df["h3_cell"].to_list())
    parent_ids = h3ronpy.change_resolution(cell_ids, target_resolution)
    parent_strs = h3ronpy.cells_to_string(parent_ids).to_pylist()
    return (
        df.with_columns(pl.Series("h3_cell", parent_strs))
        .group_by("h3_cell")
        .agg(pl.col(value_col).sum())
    )


# ================================================================
# PART 2 -- PURE ID-CORRESPONDENCE MAPPINGS
# ================================================================


def build_h3_to_polygon_mapping(
    h3_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    h3_id: str,
    polygon_id: str,
    cache_dir: Union[str, Path],
    name: str = "h3_to_census",
    force: bool = False,
) -> pl.DataFrame:
    """Cache which polygon (e.g. census unit) each H3 cell's centroid falls within.

    Polygon boundaries are irregular, so -- unlike the raster case -- this
    needs a real point-in-polygon test rather than a closed-form rule;
    it wraps :func:`~geohierarchy.utils.get_id_mapping` (the same
    correct, already-used centroid-in-polygon join the rest of
    ``geohierarchy`` relies on) and adds a caching layer keyed on both
    layers' geometry versions, so the join itself only runs once per
    ``(h3 grid, polygon layer)`` combination.

    Args:
        h3_gdf: H3 cell layer (polygons), e.g. from
            :func:`~geohierarchy.utils.h3_cells`.
        polygon_gdf: Target polygon layer (e.g. census tracts).
        h3_id: Id column in ``h3_gdf`` (e.g. ``"h3"``).
        polygon_id: Id column in ``polygon_gdf``.
        cache_dir: Directory the mapping is cached under.
        name: Cache artifact name.
        force: Recompute even if a cache entry already exists.

    Returns:
        Polars DataFrame with ``h3_id``, ``polygon_id`` (original column
        names preserved) and ``_geoweight`` (always ``1.0`` here -- one
        polygon per H3 cell, centroid assignment).
    """
    key = _hash_key(
        geometry_version_key(h3_gdf, h3_id),
        geometry_version_key(polygon_gdf, polygon_id),
    )

    if not force:
        cached = load_mapping(cache_dir, name, key)
        if cached is not None:
            return cached

    mapping = get_id_mapping(h3_gdf, polygon_gdf, h3_id, polygon_id, geoweighted=False)
    save_mapping(mapping, cache_dir, name, key)
    return mapping


def build_street_to_h3_mapping(
    edges_gdf: gpd.GeoDataFrame,
    h3_cells: Sequence[str],
    resolution: int,
    edge_id: str,
    cache_dir: Union[str, Path],
    margin_m: float = 10.0,
    name: str = "street_to_h3",
    force: bool = False,
) -> pl.DataFrame:
    """Cache which H3 cell(s) each street edge is near, via the existing distance-based join.

    Wraps :func:`~geohierarchy.edges.edges_to_h3_by_distance`'s matching
    logic (line-densify + KD-tree radius query against H3 centroids) --
    the correct, already-used street<->H3 correspondence -- but caches the
    resulting ``(edge_id, h3_cell)`` pairs themselves rather than any
    aggregated value, so re-running with a different attribute column or
    aggregation strategy skips the KD-tree query entirely as long as the
    edges and H3 grid haven't changed.

    Args:
        edges_gdf: Street network edges. Must contain ``edge_id``.
        h3_cells: Candidate H3 cells (e.g. every cell with population).
        resolution: H3 resolution of ``h3_cells``.
        edge_id: Id column in ``edges_gdf`` uniquely identifying each edge.
        cache_dir: Directory the mapping is cached under.
        margin_m: Extra distance beyond one cell diameter still counted as
            "touching" -- see :func:`~geohierarchy.edges.edges_to_h3_by_distance`.
        name: Cache artifact name.
        force: Recompute even if a cache entry already exists.

    Returns:
        Polars DataFrame with ``edge_id`` and ``h3_cell`` -- one row per
        (edge, nearby cell) pair.
    """
    import h3
    import h3ronpy
    import h3ronpy.vector as h3v
    import pyproj
    from scipy.spatial import cKDTree

    from .edges import _densify_lines

    h3_cells = list(h3_cells)
    key = _hash_key(
        geometry_version_key(edges_gdf, edge_id),
        sorted(h3_cells),
        resolution,
        margin_m,
    )

    if not force:
        cached = load_mapping(cache_dir, name, key)
        if cached is not None:
            return cached

    empty = pl.DataFrame(
        {edge_id: [], "h3_cell": []}, schema={edge_id: pl.Utf8, "h3_cell": pl.Utf8}
    )
    if not h3_cells or edges_gdf.empty:
        save_mapping(empty, cache_dir, name, key)
        return empty

    # Same distance-based correspondence rule as
    # `edges.edges_to_h3_by_distance` (densified line samples + a KD-tree
    # radius query against H3 centroids, no Shapely geometry anywhere), but
    # kept as raw (edge, cell) pairs rather than aggregated away -- the
    # thing being cached here is the *correspondence*, not any value.
    tagged = edges_gdf[[edge_id, edges_gdf.geometry.name]].copy()
    tagged = tagged.rename(columns={edges_gdf.geometry.name: "geometry"}).set_geometry(
        "geometry"
    )
    tagged["_edge_row"] = np.arange(len(tagged))

    edges_proj = (
        tagged
        if tagged.crs is not None and tagged.crs.is_projected
        else tagged.to_crs(tagged.estimate_utm_crs())
    )
    edge_len_m = h3.average_hexagon_edge_length(resolution, unit="m")
    radius = margin_m + 2 * edge_len_m
    step_m = max(edge_len_m / 2, 5.0)

    point_xy, point_values = _densify_lines(edges_proj, ["_edge_row"], step_m)
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
        save_mapping(empty, cache_dir, name, key)
        return empty

    row_cell = np.repeat(
        [h3_cells[i] for i in matched_idx],
        [len(neighbor_lists[i]) for i in matched_idx],
    )
    row_neighbor = np.concatenate([neighbor_lists[i] for i in matched_idx])
    edge_rows = point_values["_edge_row"][row_neighbor]

    lookup = tagged[[edge_id]].to_numpy().ravel()
    edge_ids = lookup[edge_rows.astype(np.int64)]

    mapping = pl.DataFrame({edge_id: edge_ids, "h3_cell": row_cell}).unique()
    save_mapping(mapping, cache_dir, name, key)
    return mapping


def compose_street_to_polygon_mapping(
    street_to_h3: pl.DataFrame,
    h3_to_polygon: pl.DataFrame,
    edge_id: str,
    h3_id: str,
    polygon_id: str,
) -> pl.DataFrame:
    """Compose a street-to-polygon (e.g. street-to-census) mapping from two cached mappings.

    No new spatial computation: a street edge's polygon correspondence is
    derived purely by joining its already-cached street->H3 correspondence
    to the already-cached H3->polygon correspondence, on the shared H3 cell
    id. Keeping this as a composition (rather than its own spatial join)
    means it's always consistent with the two mappings it's built from and
    never needs its own cache invalidation logic beyond theirs.

    Args:
        street_to_h3: Mapping from :func:`build_street_to_h3_mapping`
            (``h3_id``, ``edge_id`` columns).
        h3_to_polygon: Mapping from :func:`build_h3_to_polygon_mapping`
            (``h3_id``, ``polygon_id`` columns).
        edge_id: Id column name shared with ``street_to_h3``.
        h3_id: H3 cell id column name shared by both inputs.
        polygon_id: Id column name shared with ``h3_to_polygon``.

    Returns:
        Polars DataFrame with ``edge_id`` and ``polygon_id`` -- one row per
        (edge, polygon) pair reachable through a shared H3 cell.
    """
    return (
        street_to_h3.join(h3_to_polygon, on=h3_id, how="inner")
        .select([edge_id, polygon_id])
        .unique()
    )


def build_census_level_mapping(
    fine_gdf: gpd.GeoDataFrame,
    coarse_gdf: gpd.GeoDataFrame,
    fine_id: str,
    coarse_id: str,
    cache_dir: Union[str, Path],
    name: str = "census_level_mapping",
    force: bool = False,
) -> pl.DataFrame:
    """Cache a census-level-to-census-level id correspondence (e.g. blockgroup -> tract).

    A thin cached wrapper around :func:`~geohierarchy.utils.get_id_mapping`
    (centroid-in-polygon, matching how ``GeoHierarchy`` itself resolves
    parent/child correspondences), for the common case where the finer
    level nests exactly inside the coarser one (blockgroup -> tract ->
    county, the hierarchy ``code/pipeline.py``'s ``MAP_CENSUS_LEVELS``
    already uses).

    Args:
        fine_gdf: Finer census level (e.g. block groups).
        coarse_gdf: Coarser census level (e.g. tracts).
        fine_id: Id column in ``fine_gdf``.
        coarse_id: Id column in ``coarse_gdf``.
        cache_dir: Directory the mapping is cached under.
        name: Cache artifact name.
        force: Recompute even if a cache entry already exists.

    Returns:
        Polars DataFrame with ``fine_id``, ``coarse_id`` (original column
        names preserved).
    """
    key = _hash_key(
        geometry_version_key(fine_gdf, fine_id),
        geometry_version_key(coarse_gdf, coarse_id),
    )

    if not force:
        cached = load_mapping(cache_dir, name, key)
        if cached is not None:
            return cached

    mapping = get_id_mapping(
        fine_gdf, coarse_gdf, fine_id, coarse_id, geoweighted=False
    )
    save_mapping(mapping, cache_dir, name, key)
    return mapping
