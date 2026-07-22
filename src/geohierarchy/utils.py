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
) -> pl.DataFrame:
    """
    Creates a mapping table between two geometry levels.

    Args:
        src_gdf (gpd.GeoDataFrame): Source geometry (finer).
        dst_gdf (gpd.GeoDataFrame): Destination geometry (coarser).
        src_id (str): ID column in source layer.
        dst_id (str): ID column in destination layer.
        geoweighted (bool): If True, computes area overlap fractions as 'geoweighted'.

    Returns:
        pl.DataFrame: Mapping table with [src_id, dst_id, geoweight].
    """
    src = src_gdf[[src_id, "geometry"]].copy()
    dst = dst_gdf[[dst_id, "geometry"]].copy()

    if src.crs != dst.crs:
        dst = dst.to_crs(src.crs)

    if geoweighted:
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
