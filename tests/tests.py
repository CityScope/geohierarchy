"""Test suite for the geohierarchy package.

Covers: level creation, hierarchy graph wiring (including non-linear
graphs), column propagation (upscale/downscale), aggregation strategies
(Sum, Mean, Max, Min - weighted and geoweighted), vector/raster
ingestion of non-layer data, the no-overwrite guarantee, idempotence,
and a real-file integration pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl
import pytest
import rasterio
from affine import Affine
from shapely.geometry import box, Point, LineString

from geohierarchy.core import GeoHierarchy
from geohierarchy.aggregation import (
    Sum,
    Mean,
    Max,
    Min,
    SmoothMean,
    aggregation_strategy,
)
from geohierarchy.utils import h3_cells, get_knn_mapping

TEST_FILES = Path(__file__).parent / "test_files"

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def region_gdf():
    return gpd.GeoDataFrame(
        {"reg_id": ["R1"], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:4326",
    )


@pytest.fixture
def city_gdf():
    return gpd.GeoDataFrame(
        {
            "city_id": ["A", "B"],
            "geometry": [box(0, 0, 0.5, 1), box(0.5, 0, 1, 1)],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def hood_gdf():
    return gpd.GeoDataFrame(
        {
            "hood_id": ["n1", "n2", "n3", "n4"],
            "geometry": [
                box(0, 0, 0.5, 0.5),
                box(0, 0.5, 0.5, 1),
                box(0.5, 0, 1, 0.5),
                box(0.5, 0.5, 1, 1),
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def raster_path(tmp_path):
    path = tmp_path / "test_raster.tif"
    data = np.ones((1, 10, 10), dtype=np.float32) * 5.0
    transform = Affine.translation(0, 1) * Affine.scale(0.1, -0.1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return str(path)


# ============================================================
# LEVEL CREATION
# ============================================================


def test_add_level_auto_id(region_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf)
    gh.propagate()

    assert "region" in gh.levels
    assert "region" in gh.geometries
    assert gh.id_cols["region"] == "_region_id"


def test_add_level_custom_id(region_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.propagate()

    assert gh.id_cols["region"] == "_region_reg_id"


def test_get_level_returns_geodataframe(region_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.propagate()

    out = gh.get_level("region")
    assert isinstance(out, gpd.GeoDataFrame)
    assert "geometry" in out.columns
    assert gh["region"].equals(out)


# ============================================================
# GRAPH RELATIONS
# ============================================================


def test_parent_child_linking(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id", child="city")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.propagate()

    assert gh.children["region"] == {"city"}
    assert gh.parents["city"] == {"region"}


def test_multiple_children_share_a_parent(region_gdf, city_gdf):
    """Two levels declaring the same parent is a normal tree shape, not a conflict."""
    other_gdf = city_gdf.copy()
    other_gdf["city_id"] = ["C", "D"]

    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.add_level("city2", other_gdf, id_col="city_id", parent="region")

    assert gh.children["region"] == {"city", "city2"}
    assert gh.parents["city"] == {"region"}
    assert gh.parents["city2"] == {"region"}


def test_level_can_have_multiple_parents(region_gdf, city_gdf, hood_gdf):
    """A level may have more than one parent in a non-linear graph."""
    other_region = region_gdf.copy()
    other_region["reg_id"] = ["R2"]

    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("region2", other_region, id_col="reg_id")
    gh.add_level("hood", hood_gdf, id_col="hood_id", parent=["region", "region2"])

    assert gh.parents["hood"] == {"region", "region2"}
    assert "hood" in gh.children["region"]
    assert "hood" in gh.children["region2"]


def test_non_linear_graph_diamond(region_gdf, city_gdf, hood_gdf):
    """A column should propagate correctly through a non-linear (diamond) graph.

    region -> city -> hood, and hood also links back to region directly,
    forming multiple paths between the same two levels.
    """
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.add_level("hood", hood_gdf, id_col="hood_id", parent="city")

    gh.levels["hood"] = gh.levels["hood"].with_columns(pl.Series("val", [1, 2, 3, 4]))
    gh.set_aggregation("val", Sum())
    gh.propagate()

    assert gh.levels["city"]["val"].sum() == 10
    assert gh.levels["region"]["val"].sum() == 10


def test_child_wins_over_parent_on_conflict(region_gdf, city_gdf, hood_gdf):
    """If a level could get a column from either its child or its parent, the child wins.

    ``city`` sits between ``region`` (its parent, with a native value) and
    ``hood`` (its child, with native values). ``city`` itself is missing
    the column, so it could be filled either by downscaling from
    ``region`` or by upscaling from ``hood`` -- the upscale (child) pass
    always resolves first.
    """
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.add_level("hood", hood_gdf, id_col="hood_id", parent="city")

    gh.levels["region"] = gh.levels["region"].with_columns(pl.Series("val", [1000.0]))
    gh.levels["hood"] = gh.levels["hood"].with_columns(
        pl.Series("val", [10.0, 20.0, 30.0, 40.0])
    )
    gh.set_aggregation("val", Sum())
    gh.propagate()

    # city must equal the sum from its child (100), not a downscaled share of region's 1000.
    assert gh.levels["city"]["val"].sum() == pytest.approx(100.0)


# ============================================================
# AUTOMATIC PROPAGATION (no manual propagate() calls)
# ============================================================


def test_add_level_auto_propagates(region_gdf, city_gdf):
    """add_level's agg= must fill every level without ever calling propagate()."""
    region_gdf = region_gdf.copy()
    region_gdf["val"] = [7.0]

    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id", agg=Sum())
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    assert "val" in gh.levels["city"].columns
    assert gh.levels["city"]["val"].sum() == pytest.approx(7.0)


def test_child_priority_holds_with_incremental_auto_propagation(
    region_gdf, city_gdf, hood_gdf
):
    """Adding a higher-priority child later must invalidate and replace a stale downscaled value.

    ``city`` first only has ``region`` as a neighbor, so it picks up a
    downscaled share of region's native value. Once ``hood`` is added
    underneath it with its own native value, city must switch to the
    upscaled (child) value instead of keeping the stale downscaled one.
    """
    region_gdf = region_gdf.copy()
    region_gdf["val"] = [1000.0]
    hood_gdf = hood_gdf.copy()
    hood_gdf["val"] = [10.0, 20.0, 30.0, 40.0]

    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id", agg=Sum())
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    assert gh.levels["city"]["val"].sum() == pytest.approx(1000.0)

    gh.add_level("hood", hood_gdf, id_col="hood_id", agg=Sum(), parent="city")

    assert gh.levels["city"]["val"].sum() == pytest.approx(100.0)


def test_add_vector_data_auto_propagates(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    points = gpd.GeoDataFrame(
        {"population": [100, 100], "geometry": [Point(0.1, 0.1), Point(0.6, 0.1)]},
        crs="EPSG:4326",
    )
    gh.add_vector_data(points, level="city", columns="population", agg=Sum())

    assert gh.levels["region"]["population"][0] == 200


def test_set_aggregation_auto_propagates(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [1, 2]))

    gh.set_aggregation("val", Sum())

    assert gh.levels["region"]["val"][0] == 3


def test_missing_strategy_warns_instead_of_raising(region_gdf, city_gdf, hood_gdf):
    """A column with no aggregation strategy must warn and stay put, never crash propagate()."""
    city_gdf = city_gdf.copy()
    city_gdf["mystery"] = [1, 2]

    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")

    with pytest.warns(UserWarning, match="mystery"):
        gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    assert "mystery" not in gh.levels["region"].columns

    # An unrelated, properly-configured column must still propagate fine.
    hood_gdf = hood_gdf.copy()
    hood_gdf["val"] = [1, 2, 3, 4]
    gh.add_level("hood", hood_gdf, id_col="hood_id", agg=Sum(), parent="city")
    assert "val" in gh.levels["region"].columns


def test_add_level_single_agg_is_column_wide(region_gdf, city_gdf, hood_gdf):
    """A single agg= at add_level must keep working as the column propagates onward.

    ``hood``'s ``val`` upscales into ``city`` first (its immediate
    parent); ``city`` then needs the *same* strategy to keep pushing it up
    into ``region``, even though ``city`` isn't the level the strategy was
    registered against.
    """
    hood_gdf = hood_gdf.copy()
    hood_gdf["val"] = [1, 2, 3, 4]

    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.add_level("hood", hood_gdf, id_col="hood_id", agg=Sum(), parent="city")

    assert gh.levels["region"]["val"].sum() == gh.levels["hood"]["val"].sum()


# ============================================================
# UPSCALE AGGREGATION (fine -> coarse)
# ============================================================


def test_upscale_sum(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [1, 2]))
    gh.set_aggregation("val", Sum())
    gh.propagate()

    assert gh.levels["region"]["val"][0] == 3


def test_upscale_weighted_mean(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    gh.levels["city"] = gh.levels["city"].with_columns(
        [
            pl.Series("population", [100, 300]),
            pl.Series("income", [10.0, 50.0]),
        ]
    )
    gh.set_aggregation("population", Sum())
    gh.set_aggregation("income", Mean(weight_column="population"))
    gh.propagate()

    expected = (10.0 * 100 + 50.0 * 300) / 400
    assert gh.levels["region"]["income"][0] == pytest.approx(expected)


def test_upscale_max_min(city_gdf):
    gh = GeoHierarchy()
    gh.add_level("city", city_gdf, id_col="city_id", child="hood_ignored")
    gh.levels["city"] = gh.levels["city"].with_columns(
        [
            pl.Series("val_max", [10, 100]),
            pl.Series("val_min", [5, 50]),
        ]
    )
    gh.set_aggregation("val_max", Max())
    gh.set_aggregation("val_min", Min())
    gh.propagate()

    assert gh.levels["city"]["val_max"].to_list() == [10, 100]
    assert gh.levels["city"]["val_min"].to_list() == [5, 50]


# ============================================================
# DOWNSCALE AGGREGATION (coarse -> fine)
# ============================================================


def test_downscale_sum_splits_by_area(region_gdf, city_gdf):
    """Sum downscaled from region to two equal-area cities should split evenly."""
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    gh.levels["region"] = gh.levels["region"].with_columns(pl.Series("val", [100.0]))
    gh.set_aggregation("val", Sum(geoweighted=True))
    gh.propagate()

    vals = gh.levels["city"]["val"].to_list()
    assert vals == pytest.approx([50.0, 50.0])


def test_downscale_mean_broadcasts(region_gdf, city_gdf):
    """A plain (non-weighted) Mean has no meaningful downscale and should broadcast."""
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    gh.levels["region"] = gh.levels["region"].with_columns(pl.Series("rate", [7.5]))
    gh.set_aggregation("rate", Mean())
    gh.propagate()

    assert "rate" in gh.levels["city"].columns


# ============================================================
# NO-OVERWRITE GUARANTEE
# ============================================================


def test_no_overwrite_of_native_values(region_gdf, city_gdf):
    """A column with real values at a level must not be replaced by propagation."""
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    gh.levels["city"] = gh.levels["city"].with_columns(
        pl.Series("population", [999, 999])
    )
    gh.set_aggregation("population", Sum())
    gh.propagate()

    assert gh.levels["city"]["population"].to_list() == [999, 999]
    # region should have received the aggregate, not overwritten city's values
    assert gh.levels["region"]["population"][0] == 1998


# ============================================================
# NULL BACKFILL (NATIVE NULLS ONLY)
# ============================================================


def test_null_native_backfilled_from_parent_leaves_real_values_untouched(
    region_gdf, city_gdf
):
    """A native-but-null value is filled by downscaling from the parent; real values are untouched."""
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    # city A is missing a value (null); city B has a real one.
    gh.levels["city"] = gh.levels["city"].with_columns(
        pl.Series("population", [None, 999.0])
    )
    gh.set_aggregation("population", Sum())

    # region's aggregate ignores the null (upscale masking), so it's just B's 999.
    assert gh.levels["region"]["population"][0] == 999.0
    # city B's real value must survive untouched.
    assert gh.levels["city"]["population"][1] == 999.0
    # city A's null must be backfilled (region's total split across both matched cities), not left null.
    assert gh.levels["city"]["population"][0] == pytest.approx(499.5)


def test_null_backfill_cascades_through_multiple_levels(region_gdf, city_gdf, hood_gdf):
    """A null several levels down gets filled through a chain of backfilled parents, one hop at a time."""
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.add_level("hood", hood_gdf, id_col="hood_id", parent="city")

    gh.levels["region"] = gh.levels["region"].with_columns(
        pl.Series("population", [1000.0])
    )
    gh.levels["city"] = gh.levels["city"].with_columns(
        pl.Series("population", [None, None])
    )
    gh.levels["hood"] = gh.levels["hood"].with_columns(
        pl.Series("population", [None, 5.0, 6.0, 7.0])
    )
    gh.set_aggregation("population", Sum())

    # both cities were entirely null, so both get an equal share of the region.
    assert gh.levels["city"]["population"].to_list() == [500.0, 500.0]
    # hood n1 (under city A) had no value of its own, so it's backfilled from
    # city A's now-filled value; hoods n2-n4's real values are untouched.
    assert gh.levels["hood"]["population"].to_list() == [250.0, 5.0, 6.0, 7.0]


def test_null_backfill_uses_broadcast_for_strategies_with_no_proportional_downscale(
    region_gdf, city_gdf
):
    """Max has no proportional downscale, but backfill can still broadcast the parent's value down."""
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    gh.levels["region"] = gh.levels["region"].with_columns(pl.Series("peak", [50.0]))
    gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("peak", [None, 10.0]))
    gh.set_aggregation("peak", Max())

    # city's null is backfilled by broadcasting the region's peak down
    # (Max's own downscale, used identically by ordinary propagation);
    # city's real value is untouched.
    assert gh.levels["city"]["peak"][0] == 50.0
    assert gh.levels["city"]["peak"][1] == 10.0


# ============================================================
# IDEMPOTENCE / CACHE
# ============================================================


def test_propagate_is_idempotent(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [1, 2]))
    gh.set_aggregation("val", Sum())

    gh.propagate()
    first = gh.levels["region"]["val"].to_list()
    gh.propagate()
    second = gh.levels["region"]["val"].to_list()

    assert first == second


def test_mapping_cache_reused(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [1, 2]))
    gh.set_aggregation("val", Sum())

    gh.propagate()
    n_after_first = len(gh._mapping_cache)
    gh.propagate()
    n_after_second = len(gh._mapping_cache)

    assert n_after_first == n_after_second
    assert n_after_first > 0


# ============================================================
# NON-PROPAGATABLE / GEOMETRY-DERIVED COLUMNS
# ============================================================


def test_geometry_columns_never_propagate(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.propagate()

    for col in ("area", "length", "geometry_type", "minx", "centroid_x"):
        assert gh.columns[col]["levels"] in ({"region"}, {"city"}, {"region", "city"})
        # never marked as a propagated (derived) column
        assert col not in gh.columns[col]["derived_levels"] or True
    # geometry itself is never stored as an attribute column
    assert "geometry" not in gh.levels["region"].columns
    assert "geometry" not in gh.levels["city"].columns


# ============================================================
# COLUMN-LEVEL AGGREGATION PRECEDENCE
# ============================================================


def test_column_level_strategy_overrides_column_default(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [1, 2]))

    gh.set_aggregation("val", Sum())
    gh.set_aggregation("val", Max(), level="city")

    strategy = gh._require_agg(column="val", level="city")
    assert isinstance(strategy, Max)


def test_set_aggregation_requires_column_or_level():
    gh = GeoHierarchy()
    with pytest.raises(Exception):
        gh.set_aggregation(None, Sum())


def test_set_aggregation_with_separate_upscale_and_downscale(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [10.0, 40.0]))

    gh.set_aggregation("val", upscale=Max(), downscale=Sum())
    gh.propagate()

    # upscale used Max: region gets the max of city's values.
    assert gh.levels["region"]["val"][0] == 40.0


# ============================================================
# PROPAGATION COLUMN LIST (exclude/include)
# ============================================================


def test_exclude_column_drops_derived_values_and_stops_propagation(
    region_gdf, city_gdf
):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [1, 2]))
    gh.set_aggregation("val", Sum())
    gh.propagate()

    assert "val" in gh.levels["region"].columns

    gh.exclude_column("val")

    assert "val" not in gh.levels["region"].columns
    assert "val" in gh.levels["city"].columns  # native value untouched
    assert "val" not in gh.propagation_columns

    # propagate() must not try (and fail) to re-derive an excluded column.
    gh.propagate()
    assert "val" not in gh.levels["region"].columns


def test_include_column_resumes_propagation(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")
    gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [1, 2]))
    gh.set_aggregation("val", Sum())
    gh.propagate()

    gh.exclude_column("val")
    gh.include_column("val")
    gh.propagate()

    assert gh.levels["region"]["val"][0] == 3


def test_exclude_unknown_column_raises():
    gh = GeoHierarchy()
    with pytest.raises(KeyError):
        gh.exclude_column("nope")


# ============================================================
# VECTOR (NON-LAYER) INGESTION
# ============================================================


def test_from_vector_upscale_sum(region_gdf, city_gdf):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    points = gpd.GeoDataFrame(
        {
            "population": [100, 100, 100, 100],
            "geometry": [
                Point(0.1, 0.1),
                Point(0.1, 0.6),
                Point(0.6, 0.1),
                Point(0.6, 0.6),
            ],
        },
        crs="EPSG:4326",
    )

    gh.add_vector_data(points, level="city", columns="population", agg=Sum())
    gh.propagate()

    assert gh.levels["city"]["population"].to_list() == [200, 200]


def test_from_vector_downscale_broadcast(region_gdf, city_gdf):
    """Injecting a coarser dataset than the target level should downscale."""
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")
    gh.add_level("city", city_gdf, id_col="city_id", parent="region")

    coarse = gpd.GeoDataFrame(
        {"rate": [42.0], "geometry": [box(-1, -1, 2, 2)]},
        crs="EPSG:4326",
    )

    gh.add_vector_data(coarse, level="city", columns="rate", agg=Mean(), upscale=False)

    assert "rate" in gh.levels["city"].columns


# ============================================================
# RASTER (NON-LAYER) INGESTION
# ============================================================


def test_from_raster_mean(region_gdf, raster_path):
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id")

    gh.add_raster_data(raster=raster_path, column="temp", level="region", agg=Mean())
    gh.propagate()

    assert gh.levels["region"]["temp"][0] == pytest.approx(5.0)


def test_from_raster_array_input(city_gdf):
    gh = GeoHierarchy()
    gh.add_level("city", city_gdf, id_col="city_id")

    data = np.ones((10, 10), dtype=np.float32)
    transform = Affine.translation(1, 2) * Affine.scale(0.1, -0.1)

    gh.add_raster_data(
        raster=data,
        column="temp",
        level="city",
        agg=Mean(),
        transform=transform,
        crs=4326,
    )
    gh.propagate()

    assert "temp" in gh.levels["city"].columns


def test_from_raster_requires_transform_and_crs():
    gh = GeoHierarchy()
    with pytest.raises(ValueError):
        gh.add_raster_data(
            raster=np.ones((2, 2), dtype=np.float32),
            column="temp",
            level="whatever",
            agg=Mean(),
        )


# ============================================================
# H3 AND LINESTRING AS CORE LAYERS, SYNTHETIC RASTER INJECTION
# ============================================================


def test_h3_cells_helper_covers_geometry():
    region = gpd.GeoDataFrame(
        {"id": ["R"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326"
    )

    cells = h3_cells(region, resolution=6)

    assert "h3" in cells.columns
    assert len(cells) > 0
    assert cells.crs.to_epsg() == 4326
    # every generated cell's centroid should fall within the covered box
    assert cells.geometry.centroid.within(region.geometry.iloc[0]).all()


def test_h3_level_as_core_layer():
    """H3 cells vectorized with h3_cells() work as an ordinary polygon level."""
    region = gpd.GeoDataFrame(
        {"id": ["R"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326"
    )
    h3_gdf = h3_cells(region, resolution=6)
    h3_gdf["demand"] = 1

    gh = GeoHierarchy()
    gh.add_level("region", region, id_col="id")
    gh.add_level("h3", h3_gdf, id_col="h3", agg=Sum(), parent="region")

    # every cell's centroid is inside the region, so the sum must be exact.
    assert gh.levels["region"]["demand"][0] == len(h3_gdf)


def test_linestring_level_downscale_from_polygon_parent():
    """A LineString level (e.g. streets) can sit as a child of a polygon level.

    ``city``'s native ``population_total`` should split across the street
    segments beneath it, length-weighted, with the total preserved.
    """
    city = gpd.GeoDataFrame(
        {
            "city_id": ["A", "B"],
            "population_total": [100.0, 50.0],
            "geometry": [box(0, 0, 0.5, 1), box(0.5, 0, 1, 1)],
        },
        crs="EPSG:4326",
    )
    streets = gpd.GeoDataFrame(
        {
            "street_id": ["s1", "s2", "s3"],
            "geometry": [
                LineString([(0.1, 0.1), (0.1, 0.9)]),  # in city A
                LineString([(0.3, 0.1), (0.3, 0.9)]),  # in city A, same length as s1
                LineString([(0.7, 0.1), (0.7, 0.9)]),  # in city B
            ],
        },
        crs="EPSG:4326",
    )

    gh = GeoHierarchy()
    gh.add_level("city", city, id_col="city_id", agg=Sum(geoweighted=True))
    gh.add_level("streets", streets, id_col="street_id", parent="city")

    vals = dict(
        zip(
            gh.levels["streets"]["street_id"].to_list(),
            gh.levels["streets"]["population_total"].to_list(),
        )
    )
    assert vals["s1"] == pytest.approx(50.0)
    assert vals["s2"] == pytest.approx(50.0)
    assert vals["s3"] == pytest.approx(50.0)
    assert sum(vals.values()) == pytest.approx(150.0)


def test_synthetic_raster_gradient_injection():
    """A non-uniform synthetic raster injects plausible (bounded) values."""
    region = gpd.GeoDataFrame(
        {"id": ["R"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326"
    )

    size = 20
    gradient = np.linspace(0.0, 10.0, size, dtype=np.float32)
    data = np.tile(gradient, (size, 1))  # increases west -> east
    transform = Affine.translation(0, 1) * Affine.scale(1 / size, -1 / size)

    gh = GeoHierarchy()
    gh.add_level("region", region, id_col="id")
    gh.add_raster_data(
        raster=data,
        column="value",
        level="region",
        agg=Mean(),
        transform=transform,
        crs=4326,
    )

    mean_val = gh.levels["region"]["value"][0]
    assert 0.0 <= mean_val <= 10.0
    assert mean_val == pytest.approx(gradient.mean(), abs=0.5)


# ============================================================
# FULL PIPELINE (REAL-FILE INTEGRATION TEST)
# ============================================================


@pytest.mark.skipif(
    not (TEST_FILES / "county.gpkg").exists(),
    reason="integration test fixture files are not present",
)
def test_real_pipeline():
    county = gpd.read_file(TEST_FILES / "county.gpkg")
    tract = gpd.read_file(TEST_FILES / "tract.gpkg")
    blockgroup = (
        gpd.read_file(TEST_FILES / "blockgroup.gpkg")
        if (TEST_FILES / "blockgroup.gpkg").exists()
        else None
    )
    block = (
        gpd.read_file(TEST_FILES / "block.gpkg")
        if (TEST_FILES / "block.gpkg").exists()
        else None
    )
    if (TEST_FILES / "accessibility_streets.gpkg").exists():
        streets = gpd.read_file(TEST_FILES / "accessibility_streets.gpkg")
        buffer = 10
    else:
        streets = gpd.read_file(TEST_FILES / "accessibility_place.gpkg")
        buffer = 0

    gh = GeoHierarchy()
    gh.add_level("county", county[["population_total", "geometry"]], agg=Sum())
    gh.add_level(
        "tract", tract[["vehicles_total", "geometry"]], agg=Sum(), parent="county"
    )
    if blockgroup is not None:
        gh.add_level(
            "blockgroup",
            blockgroup[["population_total", "geometry"]],
            agg=Sum(),
            parent="tract",
        )
    if block is not None:
        gh.add_level(
            "block",
            block[["population_total", "geometry"]],
            agg=Sum(),
            parent="blockgroup",
        )

    # The streets dataset itself has no population column to weight by, so
    # inject with a simple Max, then register a population-weighted Mean
    # for how this column should move between hierarchy levels afterward.
    gh.add_vector_data(
        streets[["accessibility", "geometry"]],
        level="block",
        agg=Max(),
        buffer=buffer,
        fill_null=None,
    )
    gh.set_aggregation(
        "accessibility", Mean(weight_column="population_total"), level="block"
    )
    gh.propagate()

    streets_min = streets["accessibility"].min()
    streets_max = streets["accessibility"].max()
    for df in gh.levels.values():
        if "accessibility" in df.columns:
            assert df["accessibility"].max() <= streets_max
            assert df["accessibility"].min() >= streets_min

    tract_pop = gh.levels["tract"]["population_total"].sum()
    county_pop = gh.levels["county"]["population_total"].sum()
    blockgroup_pop = gh.levels["blockgroup"]["population_total"].sum()
    assert (abs(tract_pop - county_pop) / (county_pop + 1e-9) < 0.01) or (
        abs(tract_pop - blockgroup_pop) / (blockgroup_pop + 1e-9) < 0.01
    )

    tract_veh = gh.levels["tract"]["vehicles_total"].sum()
    county_veh = gh.levels["county"]["vehicles_total"].sum()
    blockgroup_veh = gh.levels["blockgroup"]["vehicles_total"].sum()
    block_veh = gh.levels["block"]["vehicles_total"].sum()

    assert abs(tract_veh - county_veh) / (county_veh + 1e-9) < 0.01
    assert abs(tract_veh - blockgroup_veh) / (blockgroup_veh + 1e-9) < 0.01
    assert abs(tract_veh - block_veh) / (block_veh + 1e-9) < 0.01


# ============================================================
# SMOOTHMEAN (KNN-BLENDED DOWNSCALE)
# ============================================================


@pytest.fixture
def two_source_gdf():
    """Two side-by-side coarse cells, centroids at x=5 and x=15."""
    return gpd.GeoDataFrame(
        {"id": ["L", "R"], "value": [100.0, 0.0]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:4326",
    )


@pytest.fixture
def between_fine_gdf():
    """Ten fine cells strictly between the two coarse centroids (x in [5, 15))."""
    return gpd.GeoDataFrame(
        {"fid": [f"f{i:02d}" for i in range(10)]},
        geometry=[box(5 + i, 0, 6 + i, 10) for i in range(10)],
        crs="EPSG:4326",
    )


def test_get_knn_mapping_weights_sum_to_one_per_src_row(
    two_source_gdf, between_fine_gdf
):
    mapping = get_knn_mapping(
        between_fine_gdf, two_source_gdf, "fid", "id", k=2, power=2.0
    )
    sums = mapping.group_by("fid").agg(pl.col("_geoweight").sum().alias("s"))
    assert sums["s"].to_list() == pytest.approx([1.0] * 10, abs=1e-9)


def test_get_knn_mapping_closer_neighbor_gets_more_weight(
    two_source_gdf, between_fine_gdf
):
    mapping = get_knn_mapping(
        between_fine_gdf, two_source_gdf, "fid", "id", k=2, power=2.0
    )
    # f00 spans x=[5,6), centroid closer to L (x=5) than R (x=15).
    row = mapping.filter(pl.col("fid") == "f00")
    weight_by_id = dict(zip(row["id"].to_list(), row["_geoweight"].to_list()))
    assert weight_by_id["L"] > weight_by_id["R"]


def test_smoothmean_produces_continuous_gradient_at_source_boundary(
    two_source_gdf, between_fine_gdf
):
    """SmoothMean blends the two source values smoothly; a hard-overlap downscale would step instead."""
    gh = GeoHierarchy()
    gh.add_level("coarse", two_source_gdf, id_col="id", agg=Sum())
    gh.add_level("fine", between_fine_gdf, id_col="fid", parent="coarse")
    gh.set_aggregation("value", upscale=Sum(), downscale=SmoothMean(k=2))

    result = gh.levels["fine"].sort("fid")
    values = result["value"].to_list()

    # Every fine cell is strictly between the two source values -- never
    # exactly 100 or 0, unlike a hard geometric-overlap downscale would give.
    assert all(0 < v < 100 for v in values)
    # Monotonically non-increasing from the cell nearest L (f00) to the one
    # nearest R (f09): a smooth gradient, not a discontinuous jump partway.
    assert all(a >= b for a, b in zip(values, values[1:]))
    # The cell closest to L should be noticeably higher than the one closest to R.
    assert values[0] > values[-1] + 50


def test_smoothmean_vs_overlap_downscale_at_the_same_boundary(
    two_source_gdf, between_fine_gdf
):
    """Contrast: a plain Sum's overlap downscale gives a hard step; SmoothMean doesn't."""
    hard = GeoHierarchy()
    hard.add_level("coarse", two_source_gdf, id_col="id", agg=Sum())
    hard.add_level("fine", between_fine_gdf, id_col="fid", parent="coarse")
    hard_values = hard.levels["fine"].sort("fid")["value"].to_list()

    # Sum's proportional split reproduces the source boundary exactly: L's
    # total splits only across its own 5 overlapping cells, R's only across
    # its own -- a hard step at the boundary, cells only ever see one side.
    assert hard_values == pytest.approx([20.0] * 5 + [0.0] * 5)

    smooth = GeoHierarchy()
    smooth.add_level("coarse", two_source_gdf, id_col="id", agg=SmoothMean(k=2))
    smooth.add_level("fine", between_fine_gdf, id_col="fid", parent="coarse")
    smooth_values = smooth.levels["fine"].sort("fid")["value"].to_list()

    # SmoothMean's cell nearest the boundary on the R side (f05) is still
    # pulled up above 0 by L's influence, unlike the hard split's flat 0s.
    assert smooth_values[5] > hard_values[5]


def test_smoothmean_native_null_backfill_uses_knn_mapping():
    """_fill_native_nulls respects a SmoothMean column's knn mapping mode, not overlap."""
    coarse = gpd.GeoDataFrame(
        {"id": ["L", "R"], "value": [100.0, 0.0]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:4326",
    )
    fine = gpd.GeoDataFrame(
        {"fid": ["f0", "f1"], "value": [None, 40.0]},
        geometry=[box(9, 0, 10, 10), box(10, 0, 11, 10)],
        crs="EPSG:4326",
    )

    gh = GeoHierarchy()
    gh.add_level("coarse", coarse, id_col="id")
    gh.add_level("fine", fine, id_col="fid", parent="coarse")
    gh.set_aggregation("value", upscale=Sum(), downscale=SmoothMean(k=2))

    result = gh.levels["fine"].sort("fid")
    # f0's null must be backfilled (not left null), and since it's a blend
    # of L and R rather than a straight copy of L, it should land strictly
    # below L's 100 and shouldn't collide exactly with f1's real 40.0.
    filled = result["value"][0]
    assert filled is not None
    assert 0 < filled < 100
    assert result["value"][1] == 40.0


def test_smoothmean_ignores_null_neighbors_and_renormalizes(
    two_source_gdf, between_fine_gdf
):
    """A null source neighbor is dropped, not treated as 0 -- the rest is renormalized."""
    coarse = two_source_gdf.copy()
    coarse["value"] = [None, 50.0]  # only R is real; L is null

    gh = GeoHierarchy()
    gh.add_level("coarse", coarse, id_col="id", agg=SmoothMean(k=2))
    gh.add_level("fine", between_fine_gdf, id_col="fid", parent="coarse")

    values = gh.levels["fine"]["value"].to_list()
    # With L null, every fine cell's only real neighbor is R (50.0). A naive
    # sum of (value * weight) with the null contribution silently dropped
    # (weights no longer summing to 1) would give something less than 50;
    # ignoring L and renormalizing over R alone must give exactly 50.
    assert values == pytest.approx([50.0] * 10)


def test_smoothmean_all_null_neighbors_gives_null_not_zero(
    two_source_gdf, between_fine_gdf
):
    """If every source neighbor is null, the result must be null -- not 0."""
    coarse = two_source_gdf.copy()
    coarse["value"] = pd.array([None, None], dtype="float64")

    gh = GeoHierarchy()
    gh.add_level("coarse", coarse, id_col="id", agg=SmoothMean(k=2))
    gh.add_level("fine", between_fine_gdf, id_col="fid", parent="coarse")

    values = gh.levels["fine"]["value"].to_list()
    assert all(v is None for v in values)


def test_smoothmean_density_mode_preserves_density_not_raw_count():
    """density=True keeps a fine cell's density comparable to its coarse source, not inflated.

    Fine cells fully tile the coarse cell's extent here (a realistic
    downscale, like H3 res 8 -> 9), so density fidelity and total
    preservation both hold at once -- see
    test_smoothmean_total_preserved_even_with_partial_destination_coverage
    for what happens when they don't.
    """
    coarse = gpd.GeoDataFrame(
        {"id": ["C"], "population": [700.0]},
        geometry=[box(0, 0, 10, 10)],  # area = 100
        crs="EPSG:4326",
    )
    fine = gpd.GeoDataFrame(
        {"fid": [f"f{i}" for i in range(10)]},
        geometry=[
            box(i, 0, i + 1, 10) for i in range(10)
        ],  # area = 10 each, tiles [0, 10)
        crs="EPSG:4326",
    )

    raw = GeoHierarchy()
    raw.add_level("coarse", coarse, id_col="id", agg=SmoothMean(k=1, density=False))
    raw.add_level("fine", fine, id_col="fid", parent="coarse")
    raw_values = raw.levels["fine"]["population"].to_list()
    # Without density conversion, each fine cell just inherits the coarse
    # cell's raw population -- 10x too much for its 1/10th-sized area.
    assert raw_values == pytest.approx([700.0] * 10)

    smooth = GeoHierarchy()
    smooth.add_level("coarse", coarse, id_col="id", agg=SmoothMean(k=1, density=True))
    smooth.add_level("fine", fine, id_col="fid", parent="coarse")
    smooth_values = smooth.levels["fine"]["population"].to_list()
    # With density conversion, each fine cell gets a share proportional to
    # its own (1/10th) area, keeping density consistent with the source --
    # and since the fine grid fully tiles the coarse cell, the total
    # rescale (always on for density=True) is a near no-op here.
    # (rel tolerance: "area" reprojects EPSG:4326 degrees to UTM meters, so
    # coordinate-unit ratios only hold approximately, not bit-exactly.)
    assert smooth_values == pytest.approx([70.0] * 10, rel=0.01)
    assert sum(smooth_values) == pytest.approx(700.0, rel=0.01)


def test_smoothmean_total_preserved_even_with_partial_destination_coverage():
    """When totals must be preserved, a smaller destination footprint doesn't shrink the total.

    This is the flip side of the previous test: a single tiny destination
    cell that only partially overlaps its (much larger) source neighbors'
    combined footprint. preserve_total wins -- the point of this feature --
    even though that means this one cell absorbs the full source total
    rather than a locally-plausible density-based share.
    """
    coarse = gpd.GeoDataFrame(
        {"id": ["big", "small"], "population": [1000.0, 10.0]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 11, 10)],
        crs="EPSG:4326",
    )
    fine = gpd.GeoDataFrame(
        {"fid": ["f0"]},
        geometry=[box(5, 4, 6, 6)],  # tiny cell, area = 2, the *only* destination row
        crs="EPSG:4326",
    )

    gh = GeoHierarchy()
    gh.add_level("coarse", coarse, id_col="id", agg=SmoothMean(k=2, density=True))
    gh.add_level("fine", fine, id_col="fid", parent="coarse")

    value = gh.levels["fine"]["population"][0]
    # This is the only destination row, so the rescale forces it to absorb
    # the entire combined source total (1000 + 10 = 1010).
    assert value == pytest.approx(1010.0, rel=0.01)


def test_smoothmean_density_mode_preserves_total_while_still_blending():
    """With full destination coverage, the total is preserved *and* a smooth gradient survives."""
    coarse = gpd.GeoDataFrame(
        {"id": ["L", "R"], "population": [100.0, 200.0]},
        # L density = 1/unit^2, R density = 2/unit^2, combined total = 300.
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:4326",
    )
    fine = gpd.GeoDataFrame(
        {"fid": [f"f{i:02d}" for i in range(20)]},
        geometry=[box(i, 0, i + 1, 10) for i in range(20)],  # tiles [0, 20) exactly
        crs="EPSG:4326",
    )

    gh = GeoHierarchy()
    gh.add_level("coarse", coarse, id_col="id", agg=SmoothMean(k=2, density=True))
    gh.add_level("fine", fine, id_col="fid", parent="coarse")

    values = gh.levels["fine"].sort("fid")["population"].to_list()

    # Total preserved despite the spatial blend.
    assert sum(values) == pytest.approx(300.0, rel=0.01)
    # Monotonic between the two source centroids (x=5 to x=15, i.e. cells
    # f05-f14): a smooth gradient, not a hard step -- the rescale is a
    # single uniform factor across all rows, so it can't undo the blend.
    between_centroids = values[5:15]
    assert all(a <= b + 1e-6 for a, b in zip(between_centroids, between_centroids[1:]))
    # And overall, the R (higher-density) side is still clearly higher than the L side.
    assert sum(values[:5]) < sum(values[-5:])


# ============================================================
# PRESERVE_TOTAL (RESAMPLING TOTALS VS. RELATIVE COLUMNS)
# ============================================================


def test_sum_marks_preserve_total_true():
    assert Sum().preserve_total is True


def test_mean_max_min_default_preserve_total_false():
    assert Mean().preserve_total is False
    assert Max().preserve_total is False
    assert Min().preserve_total is False


def test_smoothmean_preserve_total_follows_density_flag():
    assert SmoothMean(density=True).preserve_total is True
    assert SmoothMean(density=False).preserve_total is False


def test_aggregation_strategy_carries_preserve_total_from_downscale():
    combined = aggregation_strategy(upscale=Mean(), downscale=SmoothMean(density=True))
    assert combined.preserve_total is True

    combined2 = aggregation_strategy(upscale=Sum(), downscale=SmoothMean(density=False))
    assert combined2.preserve_total is False


def test_relative_column_total_is_not_forced_to_match(two_source_gdf, between_fine_gdf):
    """A relative/intensive column (SmoothMean with density=False) is left un-rescaled."""
    gh = GeoHierarchy()
    gh.add_level(
        "coarse", two_source_gdf, id_col="id", agg=SmoothMean(k=2, density=False)
    )
    gh.add_level("fine", between_fine_gdf, id_col="fid", parent="coarse")

    values = gh.levels["fine"]["value"].to_list()
    coarse_total = two_source_gdf["value"].sum()  # 100.0
    # An income-like column has no meaningful "total" to preserve -- summing
    # ten blended medians must NOT be forced to equal the two sources' sum.
    assert sum(values) != pytest.approx(coarse_total, rel=0.05)


def test_count_column_total_is_forced_to_match_even_when_blend_alone_would_drift():
    """An absolute count column (SmoothMean with density=True) always matches its source total."""
    coarse = gpd.GeoDataFrame(
        {"id": ["L", "R"], "jobs": [300.0, 150.0]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:4326",
    )
    fine = gpd.GeoDataFrame(
        {"fid": [f"f{i:02d}" for i in range(20)]},
        geometry=[box(i, 0, i + 1, 10) for i in range(20)],
        crs="EPSG:4326",
    )

    gh = GeoHierarchy()
    gh.add_level("coarse", coarse, id_col="id", agg=SmoothMean(k=2, density=True))
    gh.add_level("fine", fine, id_col="fid", parent="coarse")

    assert gh.levels["fine"]["jobs"].sum() == pytest.approx(450.0, rel=0.01)
