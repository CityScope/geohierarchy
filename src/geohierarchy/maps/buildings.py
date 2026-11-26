"""Building footprints as a map layer: vector tiles (PMTiles) of the footprints with their height and the H3 cell of their centre.

`write_building_tiles(buildings, out_dir)` writes `<out_dir>/buildings.pmtiles` (layer `buildings`, ids `bid` = 1..n) with `height_m` (the
extrusion height) and `h3_<res>` (the cell the building's centre is in: a map that colours cells by a value colours the buildings by the value of
their cell, through the same feature-state as the hexagons). Footprints need no particular source: the cadaster loaders of pycensus (official
where there is one, GlobalBuildingAtlas elsewhere) return them in the same columns (`geometry`, `height_m`).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np

DEFAULT_HEIGHT_M = 6.0   # a footprint without a height (floors x 3 m would be better when known)


def building_table(buildings: gpd.GeoDataFrame, res: int = 11, min_height_m: float = 3.0, simplify_m: float = 0.8, min_area_m2: float = 12.0) -> gpd.GeoDataFrame:
    """`bid`, `height_m` (whole metres, filled, at least `min_height_m`), `h3_<res>` (the cell of the footprint's representative point) and
    `geometry` (4326), light: footprints are simplified to `simplify_m` (a building's outline needs no finer detail than that on a map), reduced
    to their outer ring (courtyards are not drawn), and those under `min_area_m2` (sheds, GBA noise) dropped."""
    import h3

    b = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].reset_index(drop=True)
    utm = b.estimate_utm_crs()
    g = b.geometry.to_crs(utm)
    g = g.simplify(simplify_m, preserve_topology=True)
    g = g.map(lambda x: x if x.geom_type != "MultiPolygon" else max(x.geoms, key=lambda p: p.area))   # a multi-part footprint: its main part
    g = g.map(lambda x: x if x.geom_type != "Polygon" or not len(x.interiors) else type(x)(x.exterior))
    keep = (g.area >= min_area_m2) & ~g.is_empty & g.is_valid
    b, g = b[keep.to_numpy()].reset_index(drop=True), g[keep.to_numpy()].reset_index(drop=True)
    b = b.assign(geometry=gpd.GeoSeries(g.to_numpy(), crs=utm).to_crs(4326).to_numpy()).set_geometry("geometry", crs=4326)
    pts = b.geometry.representative_point()
    h = b["height_m"].to_numpy(float) if "height_m" in b.columns else np.full(len(b), np.nan)
    h = np.where(np.isfinite(h) & (h > 0), np.maximum(h, min_height_m), DEFAULT_HEIGHT_M)
    cells = [h3.latlng_to_cell(y, x, res) for x, y in zip(pts.x, pts.y)]
    return gpd.GeoDataFrame({"bid": np.arange(1, len(b) + 1, dtype=np.int64), "height_m": np.round(h).astype(np.int16), f"h3_{res}": cells}, geometry=b.geometry.to_numpy(), crs=4326)


def write_building_tiles(buildings: gpd.GeoDataFrame, out_dir, res: int = 11, min_zoom: int = 14, max_zoom: int = 15) -> Path:
    """Write `<out_dir>/buildings.pmtiles` from footprints (`geometry`, `height_m`); returns its path."""
    from geohierarchy.maps.folium.fast_tiles import generate_pmtiles_from_geoparquet

    t = building_table(buildings, res)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "buildings.parquet"
        t.to_parquet(src)
        return generate_pmtiles_from_geoparquet(src, out / "buildings.pmtiles", "buildings", min_zoom=min_zoom, max_zoom=max_zoom)
