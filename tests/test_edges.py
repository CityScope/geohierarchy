"""Tests for geohierarchy.edges.edges_to_level."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from geohierarchy.edges import edges_to_level
from geohierarchy.utils import h3_cells


@pytest.fixture
def edges_gdf():
    return gpd.GeoDataFrame(
        {
            "traffic": [10.0, 20.0, 30.0, 40.0],
            "geometry": [
                LineString([(0.0, 0.5), (1.0, 0.5)]),
                LineString([(0.5, 0.0), (0.5, 1.0)]),
                LineString([(0.0, 0.0), (0.5, 0.5)]),
                LineString([(0.5, 0.5), (1.0, 1.0)]),
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def grid_gdf():
    return gpd.GeoDataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "geometry": [
                box(0, 0, 0.5, 0.5),
                box(0, 0.5, 0.5, 1),
                box(0.5, 0, 1, 0.5),
                box(0.5, 0.5, 1, 1),
            ],
        },
        crs="EPSG:4326",
    )


def test_edges_to_level_on_manual_grid(edges_gdf, grid_gdf):
    result = edges_to_level(edges_gdf, grid_gdf, id_col="cell_id", columns="traffic")

    assert isinstance(result, gpd.GeoDataFrame)
    assert "cell_id" in result.columns
    assert "geometry" in result.columns
    assert "traffic" in result.columns
    assert len(result) == len(grid_gdf)
    assert result["traffic"].notna().all()
    assert (result["traffic"] >= 0).all()


def test_edges_to_level_on_h3_grid(edges_gdf):
    bbox_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    h3_grid = h3_cells(bbox_gdf, resolution=3)

    result = edges_to_level(edges_gdf, h3_grid, id_col="h3", columns="traffic")

    assert isinstance(result, gpd.GeoDataFrame)
    assert "h3" in result.columns
    assert "geometry" in result.columns
    assert "traffic" in result.columns
    assert len(result) == len(h3_grid)
    assert result["traffic"].notna().all()
    assert (result["traffic"] >= 0).all()
