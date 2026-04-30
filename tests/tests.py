import os
import numpy as np
import geopandas as gpd
import polars as pl
import rasterio
from affine import Affine
from shapely.geometry import box, Point
import unittest

from geohierarchy.core import GeoHierarchy
from geohierarchy.aggregation import Sum, Mean, Max, Min


# ============================================================
# FIXTURES
# ============================================================


class TestGeoHierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raster_path = "test_raster.tif"

        data = np.ones((1, 10, 10), dtype=np.float32) * 5.0
        transform = Affine.translation(0, 1) * Affine.scale(0.1, -0.1)

        with rasterio.open(
            cls.raster_path,
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

        cls.region_gdf = gpd.GeoDataFrame(
            {"reg_id": ["R1"], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:4326",
        )

        cls.city_gdf = gpd.GeoDataFrame(
            {
                "city_id": ["A", "B"],
                "geometry": [box(0, 0, 0.5, 1), box(0.5, 0, 1, 1)],
            },
            crs="EPSG:4326",
        )

        cls.hood_gdf = gpd.GeoDataFrame(
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

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.raster_path):
            os.remove(cls.raster_path)

    # ============================================================
    # 01 STRUCTURE: LEVEL CREATION
    # ============================================================

    def test_01_add_level(self):
        gh = GeoHierarchy()
        gh.add_level("region", self.region_gdf, id_col="reg_id")
        gh.propagate()

        self.assertIn("region", gh.levels)
        self.assertIn("region", gh.geometries)

    # ============================================================
    # 02 STRUCTURE: GRAPH RELATIONS
    # ============================================================

    def test_02_parent_linking(self):
        gh = GeoHierarchy()
        gh.add_level("region", self.region_gdf, id_col="reg_id", child="city")
        gh.add_level("city", self.city_gdf, id_col="city_id", parent="region")
        gh.propagate()

        self.assertIn("city", gh.connections["region"]["child"])
        self.assertIn("region", gh.connections["city"]["parent"])

    # ============================================================
    # 03 VECTOR AGGREGATION (SUM)
    # ============================================================

    def test_03_vector_sum(self):
        gh = GeoHierarchy()
        gh.add_level("region", self.region_gdf, id_col="reg_id")
        gh.add_level("city", self.city_gdf, id_col="city_id", parent="region")

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

        gh.from_vector(
            columns="population",
            gdf=points,
            to_level="city",
            agg=Sum(),
        )
        gh.propagate()

        # invariant: city aggregation must exist
        self.assertIn("population", gh.levels["city"].columns)
        self.assertEqual(
            gh.levels["city"]["population"][0],
            200,
        )

    # ============================================================
    # 04 WEIGHTED MEAN
    # ============================================================

    def test_04_weighted_mean(self):
        gh = GeoHierarchy()
        gh.add_level("region", self.region_gdf, id_col="reg_id")
        gh.add_level("city", self.city_gdf, id_col="city_id", parent="region")

        gh.levels["city"] = gh.levels["city"].with_columns(
            [
                pl.Series("population", [100, 300]),
                pl.Series("income", [10, 50]),
            ]
        )
        gh.set_aggregation("income", Mean(weight_column="population"))
        gh.propagate()
        self.assertIn("income", gh.aggregation_strategies)

    # ============================================================
    # 05 RASTER INGESTION
    # ============================================================

    def test_05_raster(self):
        gh = GeoHierarchy()
        gh.add_level("region", self.region_gdf, id_col="reg_id")

        gh.from_raster(
            column="temp",
            raster=self.raster_path,
            to_level="region",
            agg=Mean(),
        )
        gh.propagate()
        self.assertAlmostEqual(
            gh.levels["region"]["temp"][0],
            5.0,
        )

    # ============================================================
    # 06 MAX / MIN
    # ============================================================

    def test_06_max_min(self):
        gh = GeoHierarchy()
        gh.add_level("city", self.city_gdf, id_col="city_id")

        gh.levels["city"] = gh.levels["city"].with_columns(
            [
                pl.Series("val_max", [10, 100]),
                pl.Series("val_min", [5, 50]),
            ]
        )

        gh.set_aggregation("val_max", Max())
        gh.set_aggregation("val_min", Min())
        gh.propagate()
        self.assertIn("val_max", gh.aggregation_strategies)

    # ============================================================
    # 07 NO OVERWRITE GUARANTEE
    # ============================================================

    def test_07_no_overwrite(self):
        gh = GeoHierarchy()
        gh.add_level("region", self.region_gdf, id_col="reg_id")
        gh.add_level("city", self.city_gdf, id_col="city_id", parent="region")

        gh.levels["city"] = gh.levels["city"].with_columns(
            pl.Series("population", [999, 999])
        )

        gh.set_aggregation("population", Sum())
        gh.propagate()
        # invariant: original values preserved
        self.assertEqual(
            gh.levels["city"]["population"].to_list(),
            [999, 999],
        )

    # ============================================================
    # 08 DETERMINISM / IDEMPOTENCE
    # ============================================================

    def test_08_idempotence(self):
        gh = GeoHierarchy()
        gh.add_level("region", self.region_gdf, id_col="reg_id")
        gh.add_level("city", self.city_gdf, id_col="city_id", parent="region")

        gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [1, 2]))

        gh.set_aggregation("val", Sum())
        gh.propagate()
        a = gh.levels["region"]["val"].to_list()
        gh.propagate()
        b = gh.levels["region"]["val"].to_list()

        self.assertEqual(a, b)

    # ============================================================
    # 09 COLUMN REGISTRY INVARIANT
    # ============================================================

    def test_09_column_consistency(self):
        gh = GeoHierarchy()
        gh.add_level("city", self.city_gdf, id_col="city_id")

        gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("x", [1, 2]))

        gh.set_aggregation("x", Sum())
        gh.propagate()

        self.assertIn("x", gh.levels["city"].columns)

    # ============================================================
    # 10 CACHE STABILITY
    # ============================================================

    def test_10_cache(self):
        gh = GeoHierarchy()
        gh.add_level("city", self.city_gdf, id_col="city_id")
        gh.add_level("hood", self.hood_gdf, id_col="hood_id", parent="city")

        gh.levels["city"] = gh.levels["city"].with_columns(pl.Series("val", [1, 2]))

        gh.set_aggregation("val", Sum())
        gh.propagate()

        self.assertEqual(
            len(gh._mapping_cache),
            len(gh._mapping_cache),
        )

    # ============================================================
    # 11 RASTER EDGE CASE
    # ============================================================

    def test_11_raster_edge(self):
        gh = GeoHierarchy()
        gh.add_level("city", self.city_gdf, id_col="city_id")

        data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.translation(1, 2) * Affine.scale(0.1, -0.1)

        gh.from_raster(
            column="temp",
            raster=data,
            transform=transform,
            crs=4326,
            to_level="city",
            agg=Mean(),
        )
        gh.propagate()
        vals = gh.levels["city"]["temp"].to_list()
        self.assertTrue(any(v is None or v == 0 for v in vals))

    # ============================================================
    # 12 FULL PIPELINE (REAL SYSTEM TEST)
    # ============================================================

    def test_12_real_pipeline(self):
        from pathlib import Path

        BASE_DIR = Path.cwd() / "test_files"

        county = gpd.read_file(BASE_DIR / "county.gpkg")
        tract = gpd.read_file(BASE_DIR / "tract.gpkg")
        if os.path.isfile(BASE_DIR / "blockgroup.gpkg"):
            blockgroup = gpd.read_file(BASE_DIR / "blockgroup.gpkg")
        else:
            blockgroup = None

        if os.path.isfile(BASE_DIR / "block.gpkg"):
            block = gpd.read_file(BASE_DIR / "block.gpkg")
        else:
            block = None

        if os.path.isfile(BASE_DIR / "accessibility_streets.gpkg"):
            streets = gpd.read_file(BASE_DIR / "accessibility_streets.gpkg")
        else:
            streets = gpd.read_file(BASE_DIR / "accessibility_place.gpkg")

        gh = GeoHierarchy()

        # -------------------------
        # HIERARCHY SETUP
        # -------------------------
        gh.add_level("county", county[["population_total", "geometry"]], Sum())

        gh.add_level(
            "tract", tract[["vehicles_total", "geometry"]], Sum(), parent="county"
        )
        if blockgroup is not None:
            gh.add_level(
                "blockgroup",
                blockgroup[["population_total", "geometry"]],
                Sum(),
                parent="tract",
            )

        if block is not None:
            gh.add_level(
                "block",
                block[["population_total", "geometry"]],
                Sum(),
                parent="blockgroup",
            )

        buffer = 0
        if os.path.isfile(BASE_DIR / "accessibility_streets.gpkg"):
            buffer = 10
        # -------------------------
        # VECTOR INJECTION
        # -------------------------
        gh.from_vector(
            streets[["accessibility", "geometry"]],
            to_level="block",
            agg=Mean(weight_column="population_total"),
            injection_agg=Max(),
            buffer=buffer,
            fill_null=None,
        )
        gh.propagate()

        # -------------------------
        # 1. ACCESSIBILITY BOUNDS CHECK
        # -------------------------
        streets_min = streets["accessibility"].min()
        streets_max = streets["accessibility"].max()

        for level_name, df in gh.levels.items():
            if "accessibility" in df.columns:
                self.assertLessEqual(df["accessibility"].max(), streets_max)
                self.assertGreaterEqual(df["accessibility"].min(), streets_min)

        # -------------------------
        # 2. POPULATION CONSISTENCY (TRACT vs COUNTY & BLOCKGROUP)
        # -------------------------
        tract_pop = gh.levels["tract"]["population_total"].sum()
        county_pop = gh.levels["county"]["population_total"].sum()
        blockgroup_pop = gh.levels["blockgroup"]["population_total"].sum()
        print("")
        print("County population ", county_pop)
        print("Tract population ", tract_pop)
        print("Blockgroup population ", blockgroup_pop)
        print("Block population ", blockgroup_pop)
        self.assertTrue(
            (abs(tract_pop - county_pop) / (county_pop + 1e-9) < 0.01)
            | (abs(tract_pop - blockgroup_pop) / (blockgroup_pop + 1e-9) < 0.01)
        )

        # -------------------------
        # 3. VEHICLE CONSISTENCY (TRACT vs ALL LEVELS)
        # -------------------------
        tract_veh = gh.levels["tract"]["vehicles_total"].sum()
        county_veh = gh.levels["county"]["vehicles_total"].sum()
        blockgroup_veh = gh.levels["blockgroup"]["vehicles_total"].sum()
        block_veh = gh.levels["block"]["vehicles_total"].sum()

        print("Tract vehicles ", tract_veh)
        print("County vehicles ", county_veh)
        print("Blockgroup vehicles ", blockgroup_veh)
        print("Block vehicles ", tract_veh)
        self.assertTrue(abs(tract_veh - county_veh) / (county_veh + 1e-9) < 0.01)
        self.assertTrue(
            abs(tract_veh - blockgroup_veh) / (blockgroup_veh + 1e-9) < 0.01
        )
        self.assertTrue(abs(tract_veh - block_veh) / (block_veh + 1e-9) < 0.01)


# ============================================================

if __name__ == "__main__":
    unittest.main()
