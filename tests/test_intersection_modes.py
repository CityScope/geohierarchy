"""Tests for `intersection_mode` ("centroid" | "touches" | "exact") support.

Added alongside pyCensus's global schema/levels redesign, which replaces the
old `resample_intersections: bool` level flag with a three-way
`intersection_mode`. These tests cover `get_id_mapping` directly (each
mode's pairing/weighting semantics) and one end-to-end `GeoHierarchy` upscale
to check the `constant_total` invariant (population conservation) holds
under `intersection_mode="touches"`, mirroring the assert style used in
`CS_transitLOS/code/h3_population.py`.
"""

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import box

from geohierarchy.core import GeoHierarchy
from geohierarchy.aggregation import Sum
from geohierarchy.utils import get_id_mapping


@pytest.fixture
def overlapping_dst():
    # Two destination (coarse) polygons that overlap in [0.25, 0.75] x [0, 1].
    return gpd.GeoDataFrame(
        {"dst_id": ["D1", "D2"], "geometry": [box(0, 0, 0.75, 1), box(0.25, 0, 1, 1)]},
        crs="EPSG:4326",
    )


@pytest.fixture
def src_cells():
    return gpd.GeoDataFrame(
        {
            "src_id": ["S_only_in_overlap", "S_only_in_D1", "S_straddling"],
            "geometry": [
                box(0.4, 0.4, 0.5, 0.5),  # fully inside the D1/D2 overlap
                box(0.05, 0.05, 0.15, 0.15),  # fully inside D1 only
                box(
                    -0.1, 0.0, 1.1, 1.0
                ),  # wider than both D1 and D2, contained by neither
            ],
        },
        crs="EPSG:4326",
    )


def test_centroid_mode_assigns_single_owner(overlapping_dst, src_cells):
    mapping = get_id_mapping(
        src_cells, overlapping_dst, "src_id", "dst_id", intersection_mode="centroid"
    )
    # Every source row gets exactly one destination match, weight 1.0.
    assert mapping.group_by("src_id").len()["len"].max() == 1
    assert set(mapping["_geoweight"].to_list()) == {1.0}


def test_touches_mode_splits_across_all_containing_cells(overlapping_dst, src_cells):
    mapping = get_id_mapping(
        src_cells, overlapping_dst, "src_id", "dst_id", intersection_mode="touches"
    )
    by_src = {
        row["src_id"]: mapping.filter(pl.col("src_id") == row["src_id"])
        for row in mapping.select("src_id").unique().to_dicts()
    }

    # Fully inside the overlap zone -> contained by both D1 and D2 -> split 1/2 each.
    overlap_rows = by_src["S_only_in_overlap"]
    assert set(overlap_rows["dst_id"].to_list()) == {"D1", "D2"}
    assert overlap_rows["_geoweight"].to_list() == pytest.approx([0.5, 0.5])
    assert sum(overlap_rows["_geoweight"].to_list()) == pytest.approx(1.0)

    # Fully inside D1 only -> single claim, weight 1.0 (no split).
    d1_only_rows = by_src["S_only_in_D1"]
    assert d1_only_rows["dst_id"].to_list() == ["D1"]
    assert d1_only_rows["_geoweight"].to_list() == [1.0]

    # Not fully contained by either -> "within" predicate matches nothing.
    assert "S_straddling" not in by_src


def test_exact_mode_uses_real_intersection_area_not_full_cell_weight(overlapping_dst):
    # A source cell whose intersection with D1 is only a quarter of its own area.
    src = gpd.GeoDataFrame(
        {"src_id": ["S1"], "geometry": [box(0.5, 0.0, 1.0, 1.0)]}, crs="EPSG:4326"
    )
    dst = gpd.GeoDataFrame(
        {"dst_id": ["D1"], "geometry": [box(0.5, 0.0, 0.75, 1.0)]}, crs="EPSG:4326"
    )
    mapping = get_id_mapping(src, dst, "src_id", "dst_id", intersection_mode="exact")
    # Intersection area (0.25 x 1.0 = 0.25) / full src area (0.5 x 1.0 = 0.5) = 0.5,
    # not 1.0 as "touches"/"centroid" would report.
    assert mapping["_geoweight"].to_list() == pytest.approx([0.5], rel=1e-3)


def test_touches_mode_weights_sum_to_one_across_all_claiming_cells(
    overlapping_dst, src_cells
):
    # The core `constant_total` requirement for "touches": whatever a source
    # row's value gets split into across the N destination cells claiming
    # it, those N shares must sum back to the source's full weight (1.0) --
    # otherwise Sum's downstream weighted-sum machinery couldn't reconstruct
    # the original total.
    mapping = get_id_mapping(
        src_cells, overlapping_dst, "src_id", "dst_id", intersection_mode="touches"
    )
    totals = mapping.group_by("src_id").agg(pl.col("_geoweight").sum().alias("total"))
    assert totals["total"].to_list() == pytest.approx([1.0] * totals.height)


def test_sum_downscale_conserves_total(overlapping_dst):
    # Sum's actual `constant_total` mechanism lives in the downscale
    # direction: `divide_expr` splits a parent's value across its children
    # proportional to weight, and `consolidate_downscale` sums fragments
    # back -- so the children's total must exactly reproduce the parent's
    # value, mirroring `h3_population.py`'s population-conservation assert.
    dst = gpd.GeoDataFrame(
        {"dst_id": ["D1"], "population": [100], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:4326",
    )
    src = gpd.GeoDataFrame(
        {
            "src_id": ["S1", "S2", "S3", "S4"],
            "geometry": [
                box(0.0, 0.0, 0.5, 0.5),
                box(0.5, 0.0, 1.0, 0.5),
                box(0.0, 0.5, 0.5, 1.0),
                box(0.5, 0.5, 1.0, 1.0),
            ],
        },
        crs="EPSG:4326",
    )

    gh = GeoHierarchy(crs="EPSG:4326")
    gh.add_level("dst", dst, id_col="dst_id", agg=Sum(intersection_mode="exact"))
    gh.add_level(
        "src", src, id_col="src_id", agg=Sum(intersection_mode="exact"), parent="dst"
    )
    gh.propagate()

    result = gh.get_level("src")
    total_after = result["population"].sum()
    assert total_after == pytest.approx(
        100.0
    ), f"constant_total violated on downscale: 100 -> {total_after}"


def test_sum_downscale_splits_by_geoweight_not_just_weight_column():
    # Regression test for a bug where `divide_expr` (the downscale half of
    # `Sum`) computed its split ratio from `weight_column` alone, ignoring
    # `_geoweight` even when `geoweighted=True` -- so "exact" downscaling
    # into a child that only partially overlaps its parent still got the
    # same split weight as a child fully inside it, even though upscaling
    # (`sum_agg`) correctly discounted partial overlaps via `_geoweight`.
    #
    # D is the sole parent. C1 sticks half out of D (only half its own area
    # overlaps D -> _geoweight=0.5); C2 sits entirely inside D
    # (_geoweight=1.0). With no `weight_column`, a bug that ignores
    # `_geoweight` would split D's value 50/50 regardless; the correct,
    # area-weighted split is C1 : C2 = 0.5 : 1.0 -> 1/3 : 2/3.
    dst = gpd.GeoDataFrame(
        {"dst_id": ["D"], "population": [90], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:4326",
    )
    src = gpd.GeoDataFrame(
        {
            "src_id": ["C1", "C2"],
            "geometry": [
                box(-0.5, 0.0, 0.5, 1.0),  # area 1.0, half (0.5) overlaps D
                box(0.5, 0.0, 1.0, 1.0),  # area 0.5, fully inside D
            ],
        },
        crs="EPSG:4326",
    )

    gh = GeoHierarchy(crs="EPSG:4326")
    gh.add_level("dst", dst, id_col="dst_id", agg=Sum(intersection_mode="exact"))
    gh.add_level(
        "src", src, id_col="src_id", agg=Sum(intersection_mode="exact"), parent="dst"
    )
    gh.propagate()

    result = gh.get_level("src")
    values = dict(zip(result["src_id"].tolist(), result["population"].tolist()))
    # eff_w: C1=0.5, C2=1.0, sum=1.5 -> C1 = 90 * 0.5/1.5 = 30, C2 = 90 * 1.0/1.5 = 60.
    assert values["C1"] == pytest.approx(30.0, rel=1e-3)
    assert values["C2"] == pytest.approx(60.0, rel=1e-3)
    # And the split still conserves the parent's total.
    assert (values["C1"] + values["C2"]) == pytest.approx(90.0, rel=1e-3)
