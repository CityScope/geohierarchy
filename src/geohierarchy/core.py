"""
GeoHierarchy: hierarchical spatial aggregation system with controlled propagation.

Key rules:
- Geometry/meta columns are NEVER propagated
- Only user-defined attributes propagate
- Each propagating column MUST have an aggregation strategy
- No overwrites (unless forced by registry invalidation)
- No duplicate propagation
- Strict single-source column rule
"""

from __future__ import annotations

from typing import Dict, Optional, Union, Any, Literal, Set, List
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl
from pyproj import CRS

from .utils import vectorize_raster_polars, get_id_mapping, area, length
from .aggregation import AggregationStrategy


# ================================================================
# CONSTANTS: NEVER PROPAGATE THESE
# ================================================================
NON_PROPAGATABLE_COLS: Set[str] = {
    "area",
    "length",
    "minx",
    "miny",
    "maxx",
    "maxy",
    "centroid_x",
    "centroid_y",
    "geometry_type",
    "geometry",
}


class GeoHierarchy:
    """
    Hierarchical spatial system with strict propagation control.

    Core idea:
    Every column that can propagate MUST have:
    - exactly one aggregation strategy
    - exactly one origin level (no ambiguity)
    """

    # ================================================================
    # INIT
    # ================================================================

    def __init__(self, crs: Any = "EPSG:4326"):
        self.crs = CRS(crs)

        self.levels: Dict[str, pl.DataFrame] = {}
        self.geometries: Dict[str, gpd.GeoDataFrame] = {}
        self.id_cols: Dict[str, str] = {}
        self.geoweight_by: Dict[str, str] = {}

        self.connections: Dict[str, Dict[str, str]] = {}

        self.columns: Dict[str, Dict[str, Set[str]]] = {}

        self.aggregation_strategies: Dict[str, AggregationStrategy] = {}

        self._mapping_cache: Dict[str, pl.DataFrame] = {}

        self.non_propagable_cols: Set[str] = set(NON_PROPAGATABLE_COLS)

    # ================================================================
    # INTERNAL HELPERS
    # ================================================================

    def _init_level(self, name: str) -> None:
        if name not in self.connections:
            self.connections[name] = {"parent": None, "child": None}

    def _init_column(self, col: str) -> None:
        if col not in self.columns:
            self.columns[col] = {
                "native_levels": set(),
                "derived_levels": set(),
                "levels": set(),
            }

    def _is_propagatable(self, col: str) -> bool:
        return col not in self.non_propagable_cols

    def _require_agg(self, col: str) -> AggregationStrategy:
        if col not in self.aggregation_strategies:
            raise ValueError(f"Missing aggregation strategy for column '{col}'")
        return self.aggregation_strategies[col]

    # ================================================================
    # AGGREGATION REGISTRY (NEW)
    # ================================================================

    def set_aggregation(
        self,
        column: str,
        agg: AggregationStrategy,
    ) -> None:
        """
        Set or overwrite aggregation strategy for a column.

        If strategy changes:
        - invalidate derived propagation state
        - keep only native levels
        - force recomputation on propagate()
        """

        if column not in self.columns:
            self._init_column(column)
            for level, df in self.levels.items():
                level_cols = df.collect_schema().names()
                if column in level_cols:
                    self.columns[column]["native_levels"].add(level)
                    self.columns[column]["levels"].add(level)

        old = self.aggregation_strategies.get(column)

        if (old is not None) and (old is not agg):
            # RESET derived state if strategy changes
            self.columns[column]["derived_levels"].clear()
            self.columns[column]["levels"].clear()
            self.columns[column]["levels"].update(self.columns[column]["native_levels"])

        self.aggregation_strategies[column] = agg

    # ================================================================
    # LEVEL CREATION
    # ================================================================

    def add_level(
        self,
        name: str,
        gdf: gpd.GeoDataFrame,
        agg: Optional[
            Union[AggregationStrategy, Dict[str, AggregationStrategy]]
        ] = None,
        id_col: Optional[str] = None,
        parent: Optional[str] = None,
        child: Optional[str] = None,
        geoweight_by: Optional[Literal["area", "length"]] = None,
    ) -> None:
        """
        Add a spatial level.

        agg rules:
        - None → ERROR if any column missing aggregation
        - AggregationStrategy → applied to ALL columns
        - Dict[column, AggregationStrategy] → per-column override
        """

        self._init_level(name)

        # ---------------- ID ----------------
        if id_col is None:
            gdf[f"_{name}_id"] = np.arange(len(gdf))
            id_col = f"_{name}_id"
            self.non_propagable_cols.add(id_col)
        else:
            self.non_propagable_cols.add(id_col)
            gdf[f"_{name}_{id_col}"] = gdf[id_col]
            id_col = f"_{name}_{id_col}"
            self.non_propagable_cols.add(id_col)

        gdf = gdf.to_crs(self.crs)

        # ---------------- geometry features ----------------
        gdf["area"] = area(gdf)
        gdf["length"] = length(gdf)
        gdf[["minx", "miny", "maxx", "maxy"]] = gdf.geometry.bounds

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf["centroid_x"] = gdf.geometry.centroid.x
            gdf["centroid_y"] = gdf.geometry.centroid.y

        gdf["geometry_type"] = gdf.geometry.geom_type.str.replace(
            "^Multi", "", regex=True
        )

        # ---------------- geoweight ----------------
        if geoweight_by is None:
            geom_types = gdf["geometry_type"].drop_duplicates().to_list()
            if "Polygon" in geom_types:
                geoweight_by = "area"
            else:
                geoweight_by = "length"

        self.geoweight_by[name] = geoweight_by

        # ---------------- store geometry ----------------
        self.id_cols[name] = id_col
        self.geometries[name] = gdf[[id_col, gdf.geometry.name, geoweight_by]]

        # ---------------- store attributes ----------------
        df = pl.from_pandas(pd.DataFrame(gdf.drop(columns=[gdf.geometry.name])))

        self.levels[name] = df

        # ---------------- column registry ----------------
        for col in df.columns:
            self._init_column(col)
            self.columns[col]["native_levels"].add(name)
            self.columns[col]["levels"].add(name)

            # enforce aggregation rule
            if col in self.non_propagable_cols:
                continue

            if agg is None:
                if col in self.aggregation_strategies:
                    continue
                else:
                    warnings.warn(
                        f"Missing aggregation for column '{col}'. Call .set_aggregation to solve it.",
                        category=UserWarning,
                    )

            if isinstance(agg, dict):
                if col in agg:
                    self.set_aggregation(col, agg[col])
            else:
                if col not in self.aggregation_strategies:
                    self.set_aggregation(col, agg)

        # ---------------- graph ----------------
        if parent:
            self._init_level(parent)
            self.connections[name]["parent"] = parent
            for level in self.connections.keys():
                if (level != name) and (self.connections[level]["parent"] == parent):
                    raise Exception(
                        f"Level connection conflict. {parent} is parent from both {name} and {level}."
                    )

        if child:
            self._init_level(child)
            self.connections[name]["child"] = child
            for level in self.connections.keys():
                if (level != name) and (self.connections[level]["child"] == child):
                    raise Exception(
                        f"Level connection conflict. {child} is child from both {name} and {level}."
                    )

    # ================================================================
    # PROPAGATION
    # ================================================================

    def propagate(self) -> None:
        """
        4-phase propagation with strict direction + key control.
        """

        columns = [c for c in self.columns if self._is_propagatable(c)]

        all_levels = set(self.levels.keys())

        # If there are any columns that should be overwritten delete them here

        for lev in self.levels.keys():
            delete_cols = []
            lev_cols = [c for c in self.levels[lev].columns if c in columns]
            for c in lev_cols:
                if c in self.columns.keys() and lev not in self.columns[c]["levels"]:
                    delete_cols.append(c)

            if len(delete_cols) > 0:
                self.levels[lev] = self.levels[lev].drop(delete_cols)

        self._propagate_columns(columns, upscale=True, connection_key="child")
        self._propagate_columns(columns, upscale=True, connection_key="parent")
        self._propagate_columns(columns, upscale=False, connection_key="parent")
        self._propagate_columns(columns, upscale=False, connection_key="child")

        # ============================================================
        # FINAL VALIDATION
        # ============================================================

        for c in columns:
            if self.columns[c]["levels"] != all_levels:
                raise Exception(
                    f"Column '{c}' did not reach all levels: {self.columns[c]['levels']}"
                )

    # ================================================================
    # EDGE PROPAGATION
    # ================================================================

    def _propagate_columns(self, columns, upscale=True, connection_key="parent"):
        def total_coverage():
            return sum(
                sum(c in self.levels[lev].columns for lev in self.levels)
                for c in columns
            )

        total_coverage_value = len(self.levels) * len(self.columns)
        prev = -1
        for _ in range(len(self.levels)):
            curr = total_coverage()
            if (curr == total_coverage_value) or (curr == prev):
                break
            prev = curr

            for level_1 in self.levels:
                level_2 = self.connections[level_1][connection_key]
                if level_2 not in self.levels.keys():
                    continue

                if upscale:
                    if connection_key == "parent":
                        src_level = level_1
                        dst_level = level_2
                    else:
                        src_level = level_2
                        dst_level = level_1
                else:
                    if connection_key == "parent":
                        src_level = level_2
                        dst_level = level_1
                    else:
                        src_level = level_1
                        dst_level = level_2

                available = [c for c in columns if c in self.levels[src_level].columns]
                missing = [
                    c for c in available if c not in self.levels[dst_level].columns
                ]

                if not missing:
                    continue

                self._propagate_edge(missing, src_level, dst_level, upscale=upscale)

                for c in missing:
                    self.columns[c]["levels"].add(dst_level)
                    self.columns[c]["derived_levels"].add(dst_level)

    def _propagate_edge(
        self, columns: List[str], src: str, dst: str, upscale=True
    ) -> None:
        """
        Propagate columns from src → dst.
        """
        aggs = [self._require_agg(col) for col in columns]

        geoweighted = False
        for agg in aggs:
            if agg.geoweighted:
                geoweighted = True
                break

        if upscale:
            cache_key = (
                f"_id_mapping_{src}_{dst}_{geoweighted}_{self.geoweight_by[src]}"
            )

            if cache_key not in self._mapping_cache:
                self._mapping_cache[cache_key] = get_id_mapping(
                    self.geometries[src],
                    self.geometries[dst],
                    self.id_cols[src],
                    self.id_cols[dst],
                    geoweighted=geoweighted,
                    how=self.geoweight_by[src],
                )
        else:
            cache_key = (
                f"_id_mapping_{dst}_{src}_{geoweighted}_{self.geoweight_by[dst]}"
            )

            if cache_key not in self._mapping_cache:
                self._mapping_cache[cache_key] = get_id_mapping(
                    self.geometries[dst],
                    self.geometries[src],
                    self.id_cols[dst],
                    self.id_cols[src],
                    geoweighted=geoweighted,
                    how=self.geoweight_by[dst],
                )

        mapping = self._mapping_cache[cache_key]

        selected_cols = set(columns)
        selected_cols.add(self.id_cols[src])
        # include weight column if needed
        for agg in aggs:
            if hasattr(agg, "weight_column"):
                selected_cols.add(agg.weight_column)

        data = self.levels[src].select(list(selected_cols))

        joined = mapping.join(data, on=self.id_cols[src])
        if upscale:
            exprs = []
            for i in range(len(columns)):
                col = columns[i]
                agg = aggs[i]
                exprs += agg.upscale_aggs([col])

            result = joined.group_by(self.id_cols[dst]).agg(exprs)
        else:
            exprs = []
            for i in range(len(columns)):
                col = columns[i]
                agg = aggs[i]
                exprs += agg.downscale_exprs([col], self.id_cols[src])

            result = joined.with_columns(exprs) if exprs else joined

        self.levels[dst] = self.levels[dst].join(
            result.select([self.id_cols[dst], *columns]),
            on=self.id_cols[dst],
            how="left",
        )

    # ================================================================
    # VECTOR INPUT
    # ================================================================

    def from_vector(
        self,
        gdf: gpd.GeoDataFrame,
        to_level: str,
        columns: Optional[Union[str, List[str]]] = None,
        agg: Optional[
            Union[AggregationStrategy, Dict[str, AggregationStrategy]]
        ] = None,
        injection_agg: Optional[
            Union[AggregationStrategy, Dict[str, AggregationStrategy]]
        ] = None,
        geoweight_by: Optional[Literal["area", "length"]] = None,
        upscale: Optional[bool] = None,
        buffer=0,
        fill_null: Union[int, float, dict[str, int], dict[str, float], None] = 0,
    ) -> None:
        """
        Inject vector data into a level.
        """
        if isinstance(columns, str):
            columns = [columns]

        if injection_agg is None:
            injection_agg = agg

        if injection_agg is None:
            raise Exception("injection_agg keyword is needed")

        gdf = gdf.copy()

        if gdf.geometry.name != "geometry":
            gdf = gdf.rename(columns={gdf.geometry.name: "geometry"})
            gdf = gdf.set_geometry("geometry")

        if columns is None:
            columns = [
                col for col in gdf.columns if col not in self.non_propagable_cols
            ]

        gdf = gdf.to_crs(self.crs)

        is_geoweghted = False
        if agg is None:
            is_geoweghted = False
        elif isinstance(agg, dict):
            for a in agg.values():
                if hasattr(a, "geoweighted") and a.geoweighted:
                    is_geoweghted = True
                    break
        else:
            if hasattr(agg, "geoweighted") and agg.geoweighted:
                is_geoweghted = True

        if is_geoweghted or upscale is None:
            if geoweight_by is None:
                geom_types = (
                    gdf.geometry.geom_type.str.replace("^Multi", "", regex=True)
                    .drop_duplicates()
                    .to_list()
                )
                if "Polygon" in geom_types:
                    geoweight_by = "area"
                else:
                    geoweight_by = "length"

            if geoweight_by == "area":
                gdf["area"] = area(gdf)
            elif geoweight_by == "length":
                gdf["length"] = length(gdf)

        tid = "_tid"
        gdf[tid] = np.arange(len(gdf))

        if upscale is None:
            if (
                gdf[geoweight_by].median()
                < self.levels[to_level].select(pl.col(geoweight_by).median()).item()
            ):
                upscale = True
            else:
                upscale = False

        if buffer > 0:
            if upscale:
                dst_gdf = self.geometries[to_level].copy()
                if "geometry_type" not in dst_gdf.columns:
                    dst_gdf["geometry_type"] = dst_gdf.geometry.geom_type.str.replace(
                        "^Multi", "", regex=True
                    )

                mask = dst_gdf["geometry_type"] == "Polygon"
                dst_gdf = dst_gdf.to_crs(dst_gdf.estimate_utm_crs())
                dst_gdf.geometry = dst_gdf.geometry.buffer(buffer, resolution=2)
                dst_gdf.loc[mask, "geometry"] = dst_gdf.loc[mask, "geometry"].boundary
                dst_gdf = dst_gdf.to_crs(self.crs)
                dst_gdf = dst_gdf.drop(columns=["geometry_type"])
            else:
                if "geometry_type" not in gdf.columns:
                    gdf["geometry_type"] = gdf.geometry.geom_type.str.replace(
                        "^Multi", "", regex=True
                    )

                mask = gdf["geometry_type"] == "Polygon"
                gdf = gdf.to_crs(gdf.estimate_utm_crs())
                gdf.geometry = gdf.geometry.buffer(buffer, resolution=2)
                gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].boundary
                gdf = gdf.to_crs(self.crs)
                gdf = gdf.drop(columns=["geometry_type"])
        else:
            dst_gdf = self.geometries[to_level]

        if upscale:
            mapping = get_id_mapping(
                gdf,
                dst_gdf,
                tid,
                self.id_cols[to_level],
                geoweighted=injection_agg.geoweighted,
                how=geoweight_by,
            )
        else:
            mapping = get_id_mapping(
                dst_gdf,
                gdf,
                self.id_cols[to_level],
                tid,
                geoweighted=injection_agg.geoweighted,
                how=geoweight_by,
            )

        for col in columns:
            if col in self.columns:
                self.columns[col]["derived_levels"].clear()
                self.columns[col]["levels"].clear()
                self.columns[col]["levels"].update(self.columns[col]["native_levels"])

        col_selection = [c for c in gdf.columns if c not in self.non_propagable_cols]
        df_pl = pl.from_pandas(gdf[col_selection])

        joined = mapping.join(df_pl, on=tid)

        if upscale:
            if isinstance(injection_agg, dict):
                new_columns = []
                exprs = []
                for col, agg_col in injection_agg.items():
                    if col in columns:
                        new_columns.append(col)
                        exprs += agg_col.upscale_aggs([col])

                result = joined.group_by(self.id_cols[to_level]).agg(exprs)
                columns = new_columns
            else:
                result = injection_agg.upscale(
                    joined,
                    self.id_cols[to_level],
                    columns,
                )
        else:
            if isinstance(injection_agg, dict):
                new_columns = []
                exprs = []
                for col, agg_col in injection_agg.items():
                    if col in columns:
                        new_columns.append(col)
                        exprs += agg_col.downscale_exprs([col], self.id_cols[tid])

                result = joined.with_columns(exprs) if exprs else joined
                columns = new_columns
            else:
                result = injection_agg.downscale(
                    joined,
                    self.id_cols[tid],
                    columns,
                )

        self.levels[to_level] = (
            self.levels[to_level]
            .drop(columns, strict=False)
            .join(
                result[self.id_cols[to_level], *columns],
                on=self.id_cols[to_level],
                how="left",
            )
        )

        self.levels[to_level] = self.levels[to_level].with_columns(
            [pl.col(col).fill_nan(None) for col in columns]
        )

        if isinstance(fill_null, (float, int)):
            self.levels[to_level] = self.levels[to_level].with_columns(
                [pl.col(col).fill_null(fill_null) for col in columns]
            )
        elif isinstance(fill_null, dict):
            self.levels[to_level] = self.levels[to_level].with_columns(
                [pl.col(col).fill_null(fill) for col, fill in fill_null.items()]
            )

        # ---------------- column registry ----------------
        for col in columns:
            self._init_column(col)
            self.columns[col]["native_levels"].add(to_level)
            self.columns[col]["levels"].add(to_level)

            # enforce aggregation rule
            if col in self.non_propagable_cols:
                continue

            if agg is None:
                if col in self.aggregation_strategies:
                    continue
                else:
                    warnings.warn(
                        f"Missing aggregation for column '{col}'. Call .set_aggregation to solve it.",
                        category=UserWarning,
                    )

            if isinstance(agg, dict):
                if col in agg:
                    self.set_aggregation(col, agg[col])
            else:
                if col not in self.aggregation_strategies:
                    self.set_aggregation(col, agg)

    # ================================================================
    # RASTER INPUT
    # ================================================================

    def from_raster(
        self,
        raster: Union[np.ndarray, str],
        column: str,
        to_level: str,
        agg: Optional[
            Union[AggregationStrategy, Dict[str, AggregationStrategy]]
        ] = None,
        injection_agg: Optional[
            Union[AggregationStrategy, Dict[str, AggregationStrategy]]
        ] = None,
        upscale: bool = True,
        transform=None,
        crs=None,
    ) -> None:
        """
        Inject raster data into hierarchy.
        """

        if isinstance(raster, str):
            import rasterio as rio

            with rio.open(raster) as src:
                raster = src.read(1)
                transform = src.transform
                crs = src.crs

        if transform is None or crs is None:
            raise ValueError(
                "Raster metadata incomplete: 'transform' and 'crs' are required."
            )

        df = vectorize_raster_polars(raster, transform)

        gdf = gpd.GeoDataFrame(
            df.select(pl.col("value").alias(column)).to_pandas(),
            geometry=gpd.points_from_xy(df["x"], df["y"]),
            crs=crs,
        ).to_crs(self.crs)

        self.from_vector(
            gdf,
            to_level,
            [column],
            agg,
            injection_agg,
            geoweight_by="area",
            upscale=upscale,
        )

    # ================================================================
    # ACCESS
    # ================================================================

    def get_level(self, name: str) -> gpd.GeoDataFrame:
        return self.geometries[name][[self.id_cols[name], "geometry"]].merge(
            self.levels[name].to_pandas(),
            on=self.id_cols[name],
        )

    def __getitem__(self, key: str) -> gpd.GeoDataFrame:
        return self.get_level(key)
