"""Vector-tile (XYZ MVT `.pbf`) generation for a level's geometry.

For a given zoom range, enumerates the XYZ tiles covering a GeoDataFrame's
bounds, clips its geometries against the whole tile grid for each zoom in
one vectorized pass, encodes each tile's features with
``mapbox_vector_tile``, and writes it to ``{tiles_dir}/{level}/{z}/{x}/{y}.pbf``.

Tiles are written as **raw, uncompressed** protobuf bytes. It's tempting to
gzip them on disk, but Leaflet.VectorGrid fetches a tile with a plain
``fetch``/XHR and only gunzips it if the *HTTP response* carries a
``Content-Encoding: gzip`` header -- something a plain static host (GitHub
Pages, ``python -m http.server``, ...) will not add for an arbitrary
``.pbf`` extension. A gzip-on-disk tile is therefore fed straight to the
protobuf parser as-is and silently fails to decode, rendering nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd


def prep_level_gdf(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    property_cols: Iterable[str],
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """Reproject/trim a level's geometry once, up front, to the small
    ``(id_col + property_cols + geometry)`` frame every zoom's tiling pass
    actually needs.

    Split out of `write_level_tiles` so `HierarchyMap.build()` can do this
    once per level and then fan the (already-small) result out to one
    parallel job *per zoom* (see `write_level_zoom_tiles`) instead of one
    job per level -- levels differ hugely in tile count (a coarse polygon
    layer vs. a deep street-edge layer can differ by 2-3 orders of
    magnitude), so parallelizing only across levels leaves most cores idle
    while one big level's zooms grind through serially on a single core.
    Splitting by zoom too keeps every core busy regardless of how lopsided
    the levels are.

    Returns:
        ``(work, keep_cols)`` -- the reprojected/trimmed GeoDataFrame and
        the property columns kept in it (besides ``geometry``).
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


def write_level_zoom_tiles(
    work: gpd.GeoDataFrame,
    level_name: str,
    tiles_dir: str,
    z: int,
    keep_cols: List[str],
    buffer_frac: float = 0.02,
) -> List[Path]:
    """Tile one level's already-prepped geometry (`prep_level_gdf`) at a
    single zoom `z`. The per-(level, zoom) parallel unit `HierarchyMap.build`
    submits to its process pool."""
    if work.empty:
        return []

    minx, miny, maxx, maxy = work.total_bounds
    out_root = Path(tiles_dir) / level_name
    return _write_zoom_level(
        work, out_root, z, minx, miny, maxx, maxy, keep_cols, level_name, buffer_frac
    )


def write_level_tiles(
    gdf: gpd.GeoDataFrame,
    level_name: str,
    tiles_dir: str,
    min_zoom: int,
    max_zoom: int,
    id_col: str,
    property_cols: Iterable[str],
    buffer_frac: float = 0.02,
) -> List[Path]:
    """Tile one level's geometry across its assigned zoom range.

    Args:
        gdf: Level geometry (EPSG:4326), typically ``hierarchy.get_level(name)``.
        level_name: Name of the level; also used as the MVT layer name and
            the top-level directory under ``tiles_dir``.
        tiles_dir: Root output directory for all levels' tiles.
        min_zoom: Minimum zoom (inclusive) to generate tiles for.
        max_zoom: Maximum zoom (inclusive) to generate tiles for.
        id_col: Feature id column to keep in tile properties.
        property_cols: Additional attribute columns to keep in tile
            properties (styling/popups read these client-side).
        buffer_frac: Tile buffer as a fraction of tile width, used to avoid
            hairline seams between adjacent tiles for polygons/lines that
            cross a tile boundary.

    Returns:
        List of paths to every ``.pbf`` file written.
    """

    work, keep_cols = prep_level_gdf(gdf, id_col, property_cols)
    if work.empty:
        return []

    written: List[Path] = []
    for z in range(min_zoom, max_zoom + 1):
        written.extend(
            write_level_zoom_tiles(
                work, level_name, tiles_dir, z, keep_cols, buffer_frac
            )
        )

    return written


def _write_zoom_level(
    work: gpd.GeoDataFrame,
    out_root: Path,
    z: int,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    keep_cols: List[str],
    level_name: str,
    buffer_frac: float,
) -> List[Path]:
    """Clip ``work`` against every tile at zoom ``z`` in one vectorized pass and write each tile."""
    import mercantile
    import mapbox_vector_tile
    from shapely.geometry import box as shapely_box

    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zooms=[z]))
    if not tiles:
        return []

    tile_rows = []
    tile_bounds: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
    tile_geoms: Dict[Tuple[int, int], object] = {}
    for t in tiles:
        b = mercantile.bounds(t.x, t.y, t.z)
        width, height = b.east - b.west, b.north - b.south
        bw, bh = width * buffer_frac, height * buffer_frac
        buffered = shapely_box(b.west - bw, b.south - bh, b.east + bh, b.north + bh)
        tile_bounds[(t.x, t.y)] = (b.west, b.south, b.east, b.north)
        tile_geoms[(t.x, t.y)] = buffered
        tile_rows.append({"_tx": t.x, "_ty": t.y, "geometry": buffered})

    tile_grid = gpd.GeoDataFrame(tile_rows, geometry="geometry", crs=work.crs)

    # One spatial join for the whole zoom level, instead of a per-tile
    # sindex query -- this is what keeps tiling fast at deep zooms where a
    # level can span thousands of tiles. Only tiles with >=1 candidate
    # feature are ever clipped/encoded below.
    joined = gpd.sjoin(work, tile_grid, how="inner", predicate="intersects")
    if joined.empty:
        return []

    written: List[Path] = []
    for (tx, ty), idx in joined.groupby(["_tx", "_ty"]).indices.items():
        candidates = work.loc[joined.iloc[idx].index.unique()]
        west, south, east, north = tile_bounds[(tx, ty)]
        tile_geom = tile_geoms[(tx, ty)]

        clipped = gpd.clip(candidates, tile_geom)
        clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty]
        if clipped.empty:
            continue
        clipped = _flatten_geometry_collections(clipped)
        if clipped.empty:
            continue

        features = [
            {
                "geometry": row.geometry.__geo_interface__,
                "properties": {c: _json_safe(row[c]) for c in keep_cols},
            }
            for _, row in clipped.iterrows()
        ]

        layer = {"name": level_name, "features": features}
        encoded = mapbox_vector_tile.encode(
            [layer],
            default_options={"quantize_bounds": (west, south, east, north)},
        )

        out_path = out_root / str(z) / str(tx) / f"{ty}.pbf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(encoded)
        written.append(out_path)

    return written


def _flatten_geometry_collections(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Flatten any `GeometryCollection` rows into a single-type geometry.

    `gpd.clip` against a tile's (possibly buffered) bounding box can
    occasionally degenerate into a `GeometryCollection` -- e.g. a polygon
    whose boundary runs exactly along (or numerically touches) the clip
    edge can clip into a mix of a small polygon sliver plus a stray
    LineString/Point where the boundary grazed the edge. This is most
    likely for geometrically complex coastlines/islands (bays, harbors,
    archipelagos) but can in principle happen for any layer, so this is
    applied unconditionally rather than gated on a particular city or
    dataset. `mapbox_vector_tile.encode` (and vector tile encoders
    generally) only understand Point/LineString/Polygon (Multi* included),
    not GeometryCollection, so any collection has to be resolved before it
    reaches the encoder.

    For each collection, keep only the highest-dimension family present
    (polygons over lines over points) -- that's the geometry the clip was
    "supposed" to produce; the lower-dimension members are the degenerate
    slivers the edge-touching produced. Rows that end up empty (e.g. a
    collection with no salvageable parts) are dropped.
    """
    from shapely.geometry import (
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
    )

    def flatten(geom):
        if geom is None or geom.geom_type != "GeometryCollection":
            return geom
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if polys:
            return MultiPolygon(polys) if len(polys) > 1 else polys[0]
        lines = [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
        if lines:
            return MultiLineString(lines) if len(lines) > 1 else lines[0]
        pts = [g for g in geom.geoms if isinstance(g, Point) and not g.is_empty]
        if pts:
            return MultiPoint(pts) if len(pts) > 1 else pts[0]
        return None

    has_collection = gdf.geometry.geom_type == "GeometryCollection"
    if not has_collection.any():
        return gdf

    gdf = gdf.copy()
    gdf.loc[has_collection, "geometry"] = gdf.loc[has_collection, "geometry"].apply(
        flatten
    )
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]


def _json_safe(value):
    """Coerce a value to something mapbox_vector_tile can encode as an MVT property."""
    import math
    import numpy as np

    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
