"""Tests for geohierarchy.spatial_cache."""

import numpy as np
import geopandas as gpd
import pytest
from affine import Affine
from shapely.geometry import box, LineString

from geohierarchy.utils import h3_cells
from geohierarchy.spatial_cache import (
    build_raster_h3_mapping,
    apply_raster_h3_mapping,
    build_h3_to_polygon_mapping,
    build_street_to_h3_mapping,
    compose_street_to_polygon_mapping,
    build_census_level_mapping,
    raster_grid_key,
    geometry_version_key,
)


RESOLUTION = 9


@pytest.fixture
def aoi_gdf():
    return gpd.GeoDataFrame(
        {"geometry": [box(-71.1, 42.3, -71.0, 42.4)]}, crs="EPSG:4326"
    )


@pytest.fixture
def h3_grid(aoi_gdf):
    return h3_cells(aoi_gdf, RESOLUTION)


@pytest.fixture
def raster_setup(aoi_gdf):
    """A small synthetic raster of population-like values covering the AOI."""
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    width, height = 40, 40
    px_w = (maxx - minx) / width
    px_h = (maxy - miny) / height
    transform = Affine(px_w, 0.0, minx, 0.0, -px_h, maxy)
    rng = np.random.default_rng(0)
    array = rng.uniform(0, 100, size=(height, width)).astype(np.float64)
    return array, transform, (height, width)


# ================================================================
# PART 1 -- raster <-> H3
# ================================================================


def test_raster_h3_mapping_direct_and_cache_agree(tmp_path, h3_grid, raster_setup):
    array, transform, shape = raster_setup
    cache_dir = tmp_path / "cache"

    mapping1 = build_raster_h3_mapping(
        h3_grid["h3"].tolist(), transform, shape, RESOLUTION, cache_dir
    )
    assert set(mapping1.columns) == {"h3_cell", "pixel_row", "pixel_col"}
    assert len(mapping1) > 0

    # Cache file exists and is reused (no recompute) on second call.
    cached_files = list(cache_dir.glob("h3_to_raster_pixel__*.parquet"))
    assert len(cached_files) == 1

    mapping2 = build_raster_h3_mapping(
        h3_grid["h3"].tolist(), transform, shape, RESOLUTION, cache_dir
    )
    assert mapping1.equals(mapping2)

    values1 = apply_raster_h3_mapping(mapping1, array, RESOLUTION)
    values2 = apply_raster_h3_mapping(mapping2, array, RESOLUTION)
    assert values1.sort("h3_cell").equals(values2.sort("h3_cell"))


def test_raster_h3_population_conservation(tmp_path, h3_grid, raster_setup):
    """Sum of gathered H3 values must never exceed / mismatch the source pixels actually assigned."""
    array, transform, shape = raster_setup
    cache_dir = tmp_path / "cache"

    mapping = build_raster_h3_mapping(
        h3_grid["h3"].tolist(), transform, shape, RESOLUTION, cache_dir
    )
    result = apply_raster_h3_mapping(mapping, array, RESOLUTION)

    # Every gathered value must be exactly the value of the pixel it was
    # assigned to read (centroid-assignment: no splitting, no distortion).
    rows = mapping["pixel_row"].to_numpy()
    cols = mapping["pixel_col"].to_numpy()
    expected = array[rows, cols]
    assert np.allclose(
        sorted(result["value"].to_list()),
        sorted(expected[np.isfinite(expected)].tolist()),
    )

    # Aggregate rollup to a coarser resolution must preserve the fine-level total exactly.
    coarse = apply_raster_h3_mapping(
        mapping, array, RESOLUTION, target_resolution=RESOLUTION - 1
    )
    assert coarse["value"].sum() == pytest.approx(result["value"].sum())


def test_raster_h3_mapping_cache_staleness(tmp_path, h3_grid, raster_setup):
    """A mapping cached for one raster grid must not be reused for a different one."""
    array, transform, shape = raster_setup
    cache_dir = tmp_path / "cache"

    key_a = raster_grid_key(transform, shape, RESOLUTION)

    # Different resolution -> different key.
    key_b = raster_grid_key(transform, shape, RESOLUTION - 1)
    assert key_a != key_b

    # Different transform (shifted origin) -> different key.
    transform2 = transform * Affine.translation(1, 1)
    key_c = raster_grid_key(transform2, shape, RESOLUTION)
    assert key_a != key_c

    # Different shape -> different key.
    key_d = raster_grid_key(transform, (shape[0] + 1, shape[1]), RESOLUTION)
    assert key_a != key_d

    mapping_a = build_raster_h3_mapping(
        h3_grid["h3"].tolist(), transform, shape, RESOLUTION, cache_dir
    )
    # Building with a different resolution must produce a distinct cache file
    # and not reuse mapping_a's contents.
    h3_grid_coarse = h3_cells(
        gpd.GeoDataFrame(
            {"geometry": [box(-71.1, 42.3, -71.0, 42.4)]}, crs="EPSG:4326"
        ),
        RESOLUTION - 1,
    )
    mapping_b = build_raster_h3_mapping(
        h3_grid_coarse["h3"].tolist(), transform, shape, RESOLUTION - 1, cache_dir
    )
    assert set(mapping_a["h3_cell"].to_list()).isdisjoint(
        set(mapping_b["h3_cell"].to_list())
    )

    cached_files = list(cache_dir.glob("h3_to_raster_pixel__*.parquet"))
    assert len(cached_files) == 2


def test_raster_h3_mapping_out_of_bounds_cells_dropped(tmp_path, raster_setup):
    array, transform, shape = raster_setup
    cache_dir = tmp_path / "cache"

    far_away = gpd.GeoDataFrame(
        {"geometry": [box(10, 10, 10.1, 10.1)]}, crs="EPSG:4326"
    )
    far_grid = h3_cells(far_away, RESOLUTION)

    mapping = build_raster_h3_mapping(
        far_grid["h3"].tolist(), transform, shape, RESOLUTION, cache_dir
    )
    assert len(mapping) == 0


# ================================================================
# PART 2 -- id correspondence mappings
# ================================================================


def test_h3_to_polygon_mapping(tmp_path, h3_grid):
    cache_dir = tmp_path / "cache"
    minx, miny, maxx, maxy = h3_grid.total_bounds
    midx = (minx + maxx) / 2
    polygons = gpd.GeoDataFrame(
        {
            "poly_id": ["west", "east"],
            "geometry": [
                box(minx, miny, midx, maxy),
                box(midx, miny, maxx, maxy),
            ],
        },
        crs="EPSG:4326",
    )

    mapping = build_h3_to_polygon_mapping(h3_grid, polygons, "h3", "poly_id", cache_dir)
    assert set(mapping.columns) >= {"h3", "poly_id"}
    assert len(mapping) > 0
    assert set(mapping["poly_id"].to_list()) <= {"west", "east"}

    cached_files = list(cache_dir.glob("h3_to_census__*.parquet"))
    assert len(cached_files) == 1

    # Reuse on identical inputs.
    mapping2 = build_h3_to_polygon_mapping(
        h3_grid, polygons, "h3", "poly_id", cache_dir
    )
    assert mapping.equals(mapping2)


def test_h3_to_polygon_mapping_staleness(tmp_path, h3_grid):
    cache_dir = tmp_path / "cache"
    minx, miny, maxx, maxy = h3_grid.total_bounds
    midx = (minx + maxx) / 2
    polygons = gpd.GeoDataFrame(
        {
            "poly_id": ["west", "east"],
            "geometry": [box(minx, miny, midx, maxy), box(midx, miny, maxx, maxy)],
        },
        crs="EPSG:4326",
    )
    polygons_moved = gpd.GeoDataFrame(
        {
            "poly_id": ["west", "east"],
            "geometry": [
                box(minx, miny, midx - 0.01, maxy),
                box(midx - 0.01, miny, maxx, maxy),
            ],
        },
        crs="EPSG:4326",
    )

    key_a = geometry_version_key(polygons, "poly_id")
    key_b = geometry_version_key(polygons_moved, "poly_id")
    assert key_a != key_b

    build_h3_to_polygon_mapping(h3_grid, polygons, "h3", "poly_id", cache_dir)
    build_h3_to_polygon_mapping(h3_grid, polygons_moved, "h3", "poly_id", cache_dir)
    cached_files = list(cache_dir.glob("h3_to_census__*.parquet"))
    assert len(cached_files) == 2


def test_street_to_h3_mapping_and_composition(tmp_path, h3_grid):
    cache_dir = tmp_path / "cache"
    minx, miny, maxx, maxy = h3_grid.total_bounds
    midy = (miny + maxy) / 2
    edges = gpd.GeoDataFrame(
        {
            "edge_id": [1, 2],
            "geometry": [
                LineString([(minx, midy), (maxx, midy)]),
                LineString([(minx, miny), (minx, maxy)]),
            ],
        },
        crs="EPSG:4326",
    )

    street_to_h3 = build_street_to_h3_mapping(
        edges, h3_grid["h3"].tolist(), RESOLUTION, "edge_id", cache_dir
    )
    assert set(street_to_h3.columns) == {"edge_id", "h3_cell"}
    assert len(street_to_h3) > 0
    # Each matched edge_id must be one of the real edge ids -- not collapsed away.
    assert set(street_to_h3["edge_id"].to_list()) <= {1, 2}

    midx = (minx + maxx) / 2
    polygons = gpd.GeoDataFrame(
        {
            "poly_id": ["west", "east"],
            "geometry": [box(minx, miny, midx, maxy), box(midx, miny, maxx, maxy)],
        },
        crs="EPSG:4326",
    )
    h3_to_poly = build_h3_to_polygon_mapping(
        h3_grid, polygons, "h3", "poly_id", cache_dir
    )
    h3_to_poly = h3_to_poly.rename({"h3": "h3_cell"})

    composed = compose_street_to_polygon_mapping(
        street_to_h3, h3_to_poly, "edge_id", "h3_cell", "poly_id"
    )
    assert set(composed.columns) == {"edge_id", "poly_id"}
    assert len(composed) > 0
    assert set(composed["edge_id"].to_list()) <= {1, 2}


def test_census_level_mapping(tmp_path):
    cache_dir = tmp_path / "cache"
    tracts = gpd.GeoDataFrame(
        {"tract_id": ["t1", "t2"], "geometry": [box(0, 0, 1, 2), box(1, 0, 2, 2)]},
        crs="EPSG:4326",
    )
    block_groups = gpd.GeoDataFrame(
        {
            "bg_id": ["b1", "b2", "b3", "b4"],
            "geometry": [
                box(0, 0, 1, 1),
                box(0, 1, 1, 2),
                box(1, 0, 2, 1),
                box(1, 1, 2, 2),
            ],
        },
        crs="EPSG:4326",
    )

    mapping = build_census_level_mapping(
        block_groups, tracts, "bg_id", "tract_id", cache_dir
    )
    lookup = dict(zip(mapping["bg_id"].to_list(), mapping["tract_id"].to_list()))
    assert lookup["b1"] == "t1"
    assert lookup["b2"] == "t1"
    assert lookup["b3"] == "t2"
    assert lookup["b4"] == "t2"
