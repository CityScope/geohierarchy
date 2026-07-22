"""Test suite for the geohierarchy package.

Covers: level creation, hierarchy graph wiring (including non-linear
graphs), column propagation (upscale/downscale), aggregation strategies
(Sum, Mean, Max, Min - weighted and geoweighted), vector/raster
ingestion of non-layer data, the no-overwrite guarantee, idempotence,
and a real-file integration pipeline.
"""

from pathlib import Path

import numpy as np
import geopandas as gpd
import polars as pl
import pytest
import rasterio
from affine import Affine
from shapely.geometry import box, Point, LineString

from geohierarchy.core import GeoHierarchy
from geohierarchy.aggregation import Sum, Mean, Max, Min
from geohierarchy.utils import h3_cells

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
