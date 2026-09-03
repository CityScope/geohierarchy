"""Spatial utilities for geohierarchy mapping and raster processing."""

import polars as pl
import geopandas as gpd
import numpy as np
import rasterio as rio
from typing import Optional, Literal
from pyproj import Geod
import warnings


def geodesic_area(geom, geod=Geod(ellps="WGS84")) -> float:
    if geom is None or geom.is_empty:
        return 0.0

    # Handle polygons & multipolygons
    if geom.geom_type == "Polygon":
        area, _ = geod.geometry_area_perimeter(geom)
        return abs(area)

    if geom.geom_type == "MultiPolygon":
        return sum(abs(geod.geometry_area_perimeter(p)[0]) for p in geom.geoms)

    return 0.0


def geodesic_perimeter(geom, geod=Geod(ellps="WGS84")) -> float:
    if geom is None or geom.is_empty:
        return 0.0

    # Handle polygons & multipolygons
    if geom.geom_type == "Polygon":
        _, perimeter = geod.geometry_area_perimeter(geom)
        return abs(perimeter)

    if geom.geom_type == "MultiPolygon":
        return sum(abs(geod.geometry_area_perimeter(p)[1]) for p in geom.geoms)

    return 0.0


def geodesic_length(geom, geod=Geod(ellps="WGS84")) -> float:
    if geom is None or geom.is_empty:
        return 0.0

    # Handle LineString
    if geom.geom_type == "LineString":
        return geod.geometry_length(geom)

    # Handle MultiLineString
    if geom.geom_type == "MultiLineString":
        return sum(geod.geometry_length(line) for line in geom.geoms)

    return geodesic_perimeter(geom, geod)


def is_utm_reasonable(
    gdf: gpd.GeoDataFrame | gpd.GeoSeries,
    max_width_m: float = 750_000,
    max_height_m: float = 2_000_000,
    ellps=None,
) -> bool:
    """
    Check if a GeoDataFrame is reasonable for a single UTM projection.
    Uses the CRS ellipsoid directly.
    """
    if gdf.crs is None or not gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must have geographic CRS (degrees).")

    minx, miny, maxx, maxy = gdf.total_bounds

    if ellps is None:
        # Extract ellipsoid axes
        ellps = gdf.crs.ellipsoid.name.replace(" ", "")

    geod = Geod(ellps=ellps)

    # Compute width and height in meters along approximate edges
    midy = (miny + maxy) / 2
    # geod.inv returns az1, az2, distance
    _, _, width_m = geod.inv(minx, midy, maxx, midy)
    _, _, height_m = geod.inv(minx, miny, minx, maxy)

    return width_m <= max_width_m and height_m <= max_height_m


def area(
    gdf: gpd.GeoDataFrame | gpd.GeoSeries,
    max_width_m: float = 750_000,
    max_height_m: float = 2_000_000,
    ellps=None,
    geod=Geod(ellps="WGS84"),
    line_area=0,
    point_area=0,
):
    """Compute per-geometry area in square meters, CRS-aware.

    Uses the CRS's own units directly if already projected; otherwise
    reprojects to a local UTM zone when the data fits within one, and
    falls back to geodesic area computation on the ellipsoid otherwise.

    Args:
        gdf: Geometries to measure.
        max_width_m: Maximum east-west extent (meters) considered safe for
            a single UTM projection.
        max_height_m: Maximum north-south extent (meters) considered safe
            for a single UTM projection.
        ellps: Ellipsoid name passed to :class:`pyproj.Geod` for the
            geodesic fallback. Derived from ``gdf``'s CRS if ``None``.
        geod: :class:`pyproj.Geod` instance used for geodesic area
            computation.
        line_area: Area value assigned to line geometries, if not ``0``.
        point_area: Area value assigned to point geometries, if not ``0``.

    Returns:
        A pandas Series of areas aligned with ``gdf``.
    """
    if gdf.crs.is_projected:
        res = gdf.geometry.area
    else:
        gdf = gdf.to_crs(4326)

        if is_utm_reasonable(gdf, max_width_m, max_height_m, ellps):
            res = gdf.geometry.to_crs(gdf.estimate_utm_crs()).area
        else:
            res = gdf.to_crs(4326).geometry.map(
                lambda geom: geodesic_area(geom, geod=geod)
            )

    if (point_area == 0) and (line_area == 0):
        return res
    else:
        mask = gdf.geometry.geom_type.str.contains("Point")
        res.loc[mask] = point_area

        mask = gdf.geometry.geom_type.str.contains("Line")
        res.loc[mask] = line_area

    return res


def length(
    gdf: gpd.GeoDataFrame | gpd.GeoSeries,
    max_width_m: float = 750_000,
    max_height_m: float = 2_000_000,
    ellps=None,
    geod=Geod(ellps="WGS84"),
    point_length=0,
):
    """Compute per-geometry length in meters, CRS-aware.

    Same reprojection/fallback strategy as :func:`area`, applied to
    line/polygon-boundary length instead of area.

    Args:
        gdf: Geometries to measure.
        max_width_m: Maximum east-west extent (meters) considered safe for
            a single UTM projection.
        max_height_m: Maximum north-south extent (meters) considered safe
            for a single UTM projection.
        ellps: Ellipsoid name passed to :class:`pyproj.Geod` for the
            geodesic fallback. Derived from ``gdf``'s CRS if ``None``.
        geod: :class:`pyproj.Geod` instance used for geodesic length
            computation.
        point_length: Length value assigned to point geometries, if not
            ``0``.

    Returns:
        A pandas Series of lengths aligned with ``gdf``.
    """
    if gdf.crs.is_projected:
        res = gdf.geometry.area
    else:
        gdf = gdf.to_crs(4326)
        if is_utm_reasonable(gdf, max_width_m, max_height_m, ellps):
            res = gdf.geometry.to_crs(gdf.estimate_utm_crs()).length
        else:
            res = gdf.to_crs(4326).geometry.map(
                lambda geom: geodesic_length(geom, geod=geod)
            )

    if point_length == 0:
        return res
    else:
        mask = gdf.geometry.geom_type.str.contains("Point")
        res.loc[mask] = point_length

    return res


def get_geometry_types(gdf):
    if gdf is None or gdf.empty:
        raise ValueError("GeoDataFrame is empty or None")

    # vectorized geometry type extraction
    types = gdf.geometry.geom_type.unique()

    # normalize Multi* → base type
    base_types = {t.replace("Multi", "") for t in types if t is not None}

    if not base_types:
        raise ValueError("No valid geometries found")

    return list(base_types)


def vectorize_raster_polars(
    raster_array: np.ndarray, transform: rio.Affine
) -> pl.DataFrame:
    """
    Efficiently converts a raster array into a Polars DataFrame of pixel centroids.

    Args:
        raster_array (np.ndarray): 2D array of raster values.
        transform (rio.Affine): Affine transform of the raster.

    Returns:
        pl.DataFrame: DataFrame with columns [value, x, y, minx, miny, maxx, maxy].
    """
    rows, cols = raster_array.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    lon, lat = rio.transform.xy(transform, r, c)
    lon, lat = np.array(lon).flatten(), np.array(lat).flatten()
    vals = raster_array.flatten()
    dx, dy = abs(transform.a), abs(transform.e)

    return pl.DataFrame(
        {
            "value": vals,
            "x": lon,
            "y": lat,
            "minx": lon - dx / 2,
            "maxx": lon + dx / 2,
            "miny": lat - dy / 2,
            "maxy": lat + dy / 2,
        }
    ).filter(pl.col("value").is_not_null())


def get_id_mapping(
    src_gdf: gpd.GeoDataFrame,
    dst_gdf: gpd.GeoDataFrame,
    src_id: str,
    dst_id: str,
    geoweighted: bool = False,
    how: Optional[Literal["area", "length"]] = None,
    intersection_mode: Optional[
        Literal["centroid", "touches", "exact", "intersects"]
    ] = None,
) -> pl.DataFrame:
    """
    Creates a mapping table between two geometry levels.

    Args:
        src_gdf (gpd.GeoDataFrame): Source geometry (finer).
        dst_gdf (gpd.GeoDataFrame): Destination geometry (coarser).
        src_id (str): ID column in source layer.
        dst_id (str): ID column in destination layer.
        geoweighted (bool): Legacy flag, kept for backward compatibility.
            Ignored whenever ``intersection_mode`` is given explicitly;
            otherwise ``True`` behaves like ``intersection_mode="exact"``
            and ``False`` like ``intersection_mode="centroid"``.
        how: ``"area"`` or ``"length"``, forwarded to the ``"exact"`` path.
        intersection_mode: How ``src`` rows are paired with ``dst`` rows:

            - ``"centroid"``: a source polygon belongs to whichever
              destination polygon contains its centroid (this is the
              original, cheap ``geoweighted=False`` behavior). Every
              matched row gets ``_geoweight = 1.0``.
            - ``"touches"``: a source geometry belongs to *every*
              destination polygon that contains it (full containment, not
              just border-touching). If a source row is contained by ``N``
              destination rows, its ``_geoweight`` is ``1/N`` -- downstream
              additive aggregations (e.g. :class:`~geohierarchy.aggregation.Sum`)
              split its value across those ``N`` claims, while averaging
              aggregations (e.g. :class:`~geohierarchy.aggregation.Mean`)
              naturally average instead of divide, since the weights are
              equal.
            - ``"exact"``: like ``"touches"``, but ``_geoweight`` is the
              real intersection area (or length) fraction between each
              matched pair, not a plain ``1/N`` split (this is the
              original ``geoweighted=True`` behavior).
            - ``"intersects"``: a source geometry belongs to *every*
              destination polygon it geometrically touches at all --
              real ``predicate="intersects"`` on the original geometry, no
              centroid substitution -- with every match getting
              ``_geoweight = 1.0`` (no ``1/N`` split, no area-fraction
              overlay). 2026-09-01 fix: this is what ``"centroid"`` was
              always *meant* to approximate for a caller aggregating an
              attribute like transit-access level_of_service onto census
              polygons ("population weighted average of every hexagon cell
              touching this polygon") -- a hexagon whose true footprint
              overlaps a small/oddly-shaped polygon but whose *centroid*
              happens to fall just outside it (common for census blocks
              comparable in size to a single hexagon) was silently dropped
              from that polygon's aggregate, which could paint a real
              transit-served block with a wrong 0 despite every hexagon
              actually covering it showing real access. Costs the same
              index-based ``gpd.sjoin`` as ``"centroid"`` (no expensive
              ``gpd.overlay``), so it is not the slow path ``"exact"`` is.

            If ``None`` (default), falls back to the legacy ``geoweighted``
            bool for backward compatibility.

    Returns:
        pl.DataFrame: Mapping table with [src_id, dst_id, geoweight].
    """
    src = src_gdf[[src_id, "geometry"]].copy()
    dst = dst_gdf[[dst_id, "geometry"]].copy()

    if src.crs != dst.crs:
        dst = dst.to_crs(src.crs)

    if intersection_mode is None:
        intersection_mode = "exact" if geoweighted else "centroid"

    if intersection_mode == "touches":
        if "geometry_type" not in src.columns:
            src["geometry_type"] = src.geometry.geom_type.str.replace(
                "^Multi", "", regex=True
            )
        # Full containment (not just centroid-in-polygon): a destination
        # polygon "touches" (claims) a source row whenever it geometrically
        # contains it. `predicate="within"` on the *original* src geometry
        # (no centroid substitution) means a source row can legitimately
        # match more than one destination row (e.g. it straddles a
        # destination boundary and is reported as contained by both, or the
        # destination layer has overlapping polygons).
        joined = gpd.sjoin(src, dst, predicate="within", how="inner")
        mapping = joined[[src_id, dst_id]].drop_duplicates()
        counts = mapping.groupby(src_id)[dst_id].transform("count")
        mapping = mapping.assign(_geoweight=1.0 / counts)
        return pl.from_pandas(mapping)

    if intersection_mode == "intersects":
        # Real geometry-to-geometry intersects test -- no centroid
        # substitution (unlike "centroid" below) and no area/length overlay
        # (unlike "exact"). A source row can legitimately match more than
        # one destination row (it straddles a boundary); each match counts
        # fully (`_geoweight = 1.0`), matching `"touches"`'s per-row full
        # weight but without requiring full containment.
        joined = gpd.sjoin(src, dst, predicate="intersects", how="inner")
        mapping = joined[[src_id, dst_id]].drop_duplicates()
        mapping = mapping.assign(_geoweight=1.0)
        return pl.from_pandas(mapping)

    if intersection_mode == "exact":
        if how is None:
            if "geometry_type" in src.columns:
                geom_types = src["geometry_type"].drop_duplicates().to_list()
            else:
                geom_types = (
                    src.geometry.geom_type.str.replace("^Multi", "", regex=True)
                    .drop_duplicates()
                    .to_list()
                )

            if "Polygon" in geom_types:
                how = "area"
            else:
                how = "length"

        # Use cached area if available to avoid re-calculating
        if how == "area":
            if "area" in src_gdf.columns:
                src["area"] = src_gdf["area"]
            else:
                src["area"] = area(src.geometry, point_area=0, line_area=0)
        else:
            if "length" in src_gdf.columns:
                src["length"] = src_gdf["length"]
            else:
                src["length"] = length(src.geometry, point_length=0)

        inter = gpd.overlay(src, dst, how="intersection", keep_geom_type=False)
        if how == "area":
            inter["_geoweight"] = np.where(
                inter["area"] > 0,
                area(inter.geometry, point_area=0, line_area=0) / inter["area"],
                0.0,
            )
        else:
            inter["_geoweight"] = np.where(
                inter["length"] > 0,
                length(inter.geometry, point_length=0) / inter["length"],
                0.0,
            )

        mapping = inter[[src_id, dst_id, "_geoweight"]]
    else:
        if "geometry_type" not in src.columns:
            src["geometry_type"] = src.geometry.geom_type.str.replace(
                "^Multi", "", regex=True
            )

        mask = src["geometry_type"] == "Polygon"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            src.loc[mask, "geometry"] = src.loc[mask, "geometry"].centroid

        joined = gpd.sjoin(src, dst, predicate="intersects", how="inner")
        mapping = joined[[src_id, dst_id]].drop_duplicates(
            subset=[src_id], keep="first"
        )
        mapping["_geoweight"] = 1.0

    return pl.from_pandas(mapping)


def get_knn_mapping(
    src_gdf: gpd.GeoDataFrame,
    dst_gdf: gpd.GeoDataFrame,
    src_id: str,
    dst_id: str,
    k: int = 6,
    power: float = 2.0,
) -> pl.DataFrame:
    """Build an inverse-distance-weighted k-nearest-neighbor mapping, for smoothed downscaling.

    Unlike :func:`get_id_mapping` (which only pairs geometries that
    physically overlap, reproducing the source geometry's boundaries
    exactly), this pairs every ``src_gdf`` centroid with its ``k`` nearest
    ``dst_gdf`` centroids, weighted by inverse distance and normalized to
    sum to 1 within each ``src_id`` group. Used by
    :class:`~geohierarchy.aggregation.SmoothMean` so a downscaled value
    blends across several source cells instead of copying whichever one
    happens to contain (or nearly contain) a given destination cell,
    smoothing out the source geometry's hard boundaries.

    Also computes each row's polygon area (reusing an existing ``"area"``
    column if either GeoDataFrame already has one -- e.g. the ``"area"``
    column :class:`~geohierarchy.core.GeoHierarchy` stores on every level's
    geometry table -- rather than recomputing it) and carries it along as
    ``"_query_area"`` (area of the ``src_gdf`` row each mapping row is
    computed *for*) and ``"_neighbor_area"`` (area of the paired
    ``dst_gdf`` neighbor). :class:`~geohierarchy.aggregation.SmoothMean`
    uses these to convert an additive count to a density (dividing by
    whichever side the value's own row is on) before blending, and back to
    a count sized for the row being computed afterward. Meaningless (and
    unused) for non-polygon geometries.

    Args:
        src_gdf: Rows to compute neighbor weights for (the finer,
            "destination-of-propagation" geometries in a downscale).
        dst_gdf: Candidate neighbor geometries (the coarser,
            "source-of-propagation" geometries in a downscale).
        src_id: Id column in ``src_gdf``.
        dst_id: Id column in ``dst_gdf``.
        k: Number of nearest ``dst_gdf`` neighbors to blend per ``src_gdf`` row.
        power: Inverse-distance weighting exponent; higher values
            concentrate more weight on the nearest neighbor(s).

    Returns:
        Polars DataFrame with ``[src_id, dst_id, "_geoweight", "_query_area",
        "_neighbor_area"]``; ``"_geoweight"`` sums to 1 within each ``src_id`` group.
    """
    src = src_gdf[[src_id, "geometry"]].copy()
    dst = dst_gdf[[dst_id, "geometry"]].copy()
    if src.crs != dst.crs:
        dst = dst.to_crs(src.crs)

    query_areas = (
        src_gdf["area"] if "area" in src_gdf.columns else area(src_gdf)
    ).to_numpy()
    neighbor_areas = (
        dst_gdf["area"] if "area" in dst_gdf.columns else area(dst_gdf)
    ).to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        src_xy = np.column_stack([src.geometry.centroid.x, src.geometry.centroid.y])
        dst_xy = np.column_stack([dst.geometry.centroid.x, dst.geometry.centroid.y])

    k = max(1, min(k, len(dst_xy)))
    # Brute-force distance matrix: fine for the level sizes geohierarchy operates at.
    dist = np.sqrt(((src_xy[:, None, :] - dst_xy[None, :, :]) ** 2).sum(axis=2))
    nn_idx = np.argsort(dist, axis=1)[:, :k]
    nn_dist = np.take_along_axis(dist, nn_idx, axis=1)

    eps = (
        1e-9  # avoid division by zero when a src centroid coincides with a dst centroid
    )
    weights = 1.0 / np.power(nn_dist + eps, power)
    weights = weights / weights.sum(axis=1, keepdims=True)

    n_src, kk = nn_idx.shape
    src_ids = np.repeat(src[src_id].to_numpy(), kk)
    dst_ids = dst[dst_id].to_numpy()[nn_idx.ravel()]
    geoweights = weights.ravel()
    query_area_col = np.repeat(query_areas, kk)
    neighbor_area_col = neighbor_areas[nn_idx.ravel()]

    return pl.DataFrame(
        {
            src_id: src_ids,
            dst_id: dst_ids,
            "_geoweight": geoweights,
            "_query_area": query_area_col,
            "_neighbor_area": neighbor_area_col,
        }
    )


def h3_cells(
    geometry: "gpd.GeoDataFrame | gpd.GeoSeries", resolution: int
) -> gpd.GeoDataFrame:
    """Build an H3 cell grid covering a geometry, as polygon cells.

    Useful for building an H3 hierarchy level with :meth:`GeoHierarchy.add_level`
    -- H3 cells are plain polygons once vectorized, so they work as a core
    layer exactly like any other polygon geometry (they can also be used
    with :meth:`GeoHierarchy.add_vector_data` / :meth:`GeoHierarchy.add_raster_data`
    as non-layer source data).

    Args:
        geometry: Geometries whose combined extent the H3 grid should
            cover. Reprojected to EPSG:4326 first, since H3 operates on
            geographic coordinates.
        resolution: H3 resolution (0 = coarsest, ~15 = finest). See the
            H3 documentation for the approximate cell size at each level.

    Returns:
        A GeoDataFrame in EPSG:4326 with an ``"h3"`` column (the cell
        index as a hex string) and one polygon per covering cell.
    """
    import shapely.wkb
    from h3ronpy import cells_to_string
    from h3ronpy.vector import geometry_to_cells, cells_to_wkb_polygons

    geometry = geometry.to_crs(4326)
    union = (
        geometry.union_all() if hasattr(geometry, "union_all") else geometry.unary_union
    )

    cells = geometry_to_cells(union, resolution=resolution)
    wkb_arr = cells_to_wkb_polygons(cells)

    ids = [c.as_py() for c in cells_to_string(cells)]
    geoms = [shapely.wkb.loads(w.as_py()) for w in wkb_arr]

    return gpd.GeoDataFrame({"h3": ids}, geometry=geoms, crs="EPSG:4326")


def h3_cells_from_ids(cell_ids) -> gpd.GeoDataFrame:
    """Build polygons for an explicit, already-known list of H3 cell ids.

    Unlike :func:`h3_cells` (which *discovers* the cells covering a
    geometry via containment), this just vectorizes cells you already have
    ids for -- the tiled/chunked raster resamplers use it to materialize
    only one H3-resolution-5 tile's worth of child cells at a time (e.g.
    via ``h3.cell_to_children``), instead of building geometry for an
    entire city's fine-resolution grid in one shot.

    Args:
        cell_ids: Iterable of H3 cell id hex strings.

    Returns:
        A GeoDataFrame in EPSG:4326 with an ``"h3"`` column and one polygon
        per cell, same shape/columns as :func:`h3_cells`.
    """
    import shapely.wkb
    from h3ronpy import cells_parse
    from h3ronpy.vector import cells_to_wkb_polygons

    ids = list(cell_ids)
    cells = cells_parse(ids)
    wkb_arr = cells_to_wkb_polygons(cells)
    geoms = [shapely.wkb.loads(w.as_py()) for w in wkb_arr]

    return gpd.GeoDataFrame({"h3": ids}, geometry=geoms, crs="EPSG:4326")
