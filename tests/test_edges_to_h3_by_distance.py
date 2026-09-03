"""Tests for geohierarchy.edges.edges_to_h3_by_distance, including the
nearest-edge fallback that guarantees every cell gets a value up to
`fallback_radius_multiplier * cell_edge_length` away.
"""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from geohierarchy.aggregation import Max
from geohierarchy.edges import edges_to_h3_by_distance
from geohierarchy.utils import h3_cells


@pytest.fixture
def far_corner_edge():
    # A single short line tucked into the corner of the bbox below, so most
    # cells covering the bbox are progressively farther from it.
    return gpd.GeoDataFrame(
        {
            "access_score": [42.0],
            "geometry": [LineString([(0.0010, 0.0010), (0.0020, 0.0020)])],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def bbox_h3(far_corner_edge):
    bbox_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 0.05, 0.05)]}, crs="EPSG:4326")
    grid = h3_cells(bbox_gdf, resolution=8)
    return grid["h3"].tolist()


def test_without_fallback_leaves_far_cells_absent(far_corner_edge, bbox_h3):
    result = edges_to_h3_by_distance(
        far_corner_edge,
        bbox_h3,
        columns="access_score",
        resolution=8,
        agg=Max(),
        fallback_radius_multiplier=None,
    )
    assert len(result) < len(bbox_h3)
    assert set(result["h3_cell"].to_list()).issubset(set(bbox_h3))
    assert (result["access_score"] == 42.0).all()


def test_fallback_covers_more_cells_than_no_fallback(far_corner_edge, bbox_h3):
    no_fallback = edges_to_h3_by_distance(
        far_corner_edge,
        bbox_h3,
        columns="access_score",
        resolution=8,
        agg=Max(),
        fallback_radius_multiplier=None,
    )
    with_fallback = edges_to_h3_by_distance(
        far_corner_edge,
        bbox_h3,
        columns="access_score",
        resolution=8,
        agg=Max(),
        fallback_radius_multiplier=5.0,
    )

    assert len(with_fallback) > len(no_fallback)
    assert set(no_fallback["h3_cell"].to_list()).issubset(
        set(with_fallback["h3_cell"].to_list())
    )
    # Still not necessarily every cell -- the bbox is much larger than
    # 5x a resolution-8 cell's edge length -- but strictly more coverage,
    # and every filled cell carries the one line's value.
    assert len(with_fallback) < len(bbox_h3)
    assert (with_fallback["access_score"] == 42.0).all()


def test_large_fallback_multiplier_covers_every_cell(far_corner_edge, bbox_h3):
    result = edges_to_h3_by_distance(
        far_corner_edge,
        bbox_h3,
        columns="access_score",
        resolution=8,
        agg=Max(),
        fallback_radius_multiplier=1000.0,
    )
    assert len(result) == len(bbox_h3)
    assert set(result["h3_cell"].to_list()) == set(bbox_h3)
    assert (result["access_score"] == 42.0).all()


def test_no_edges_returns_empty_frame(bbox_h3):
    empty_edges = gpd.GeoDataFrame(
        {"access_score": [], "geometry": []}, crs="EPSG:4326"
    )
    result = edges_to_h3_by_distance(
        empty_edges, bbox_h3, columns="access_score", resolution=8, agg=Max()
    )
    assert len(result) == 0
    assert "h3_cell" in result.columns
    assert "access_score" in result.columns
