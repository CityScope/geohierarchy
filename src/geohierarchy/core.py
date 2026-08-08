"""GeoHierarchy: hierarchical spatial data management with controlled column propagation.

Key rules:
    - Geometry/meta columns are never propagated across levels.
    - Only user-defined attribute columns propagate.
    - Every propagating column must resolve to exactly one aggregation
      strategy for the level it propagates from.
    - A level's native (user-provided) non-null values are never
      overwritten by propagation. Null native values are the one
      exception: they are backfilled from a parent level (downscaled),
      since a null means "no value collected here", not "a real value
      of nothing" -- see :meth:`GeoHierarchy._fill_native_nulls`.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Union, Any, Literal, Set, List
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl
from pyproj import CRS

from .utils import (
    vectorize_raster_polars,
    get_id_mapping,
    get_knn_mapping,
    area,
    length,
)
from .aggregation import AggregationStrategy, aggregation_strategy


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

_LevelNames = Union[str, List[str]]


def _as_list(value: Optional[_LevelNames]) -> List[str]:
    """Normalize a ``None`` / ``str`` / ``list[str]`` argument to a list of names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


class GeoHierarchy:
    """A graph of spatial levels with controlled, direction-aware column propagation.

    A :class:`GeoHierarchy` holds a set of named "levels", each backed by a
    geometry (polygons, lines, points, H3 cells, ...) and a table of
    attribute columns. Levels are wired into a graph by passing ``parent``
    and/or ``child`` level names to :meth:`add_level`. The graph does not
    need to be a tree: a level can have several parents and several
    children (e.g. two levels can both be a parent of the same child, or
    one level can feed into two unrelated hierarchy branches).

    Every attribute column that should be shared across levels needs
    exactly one :class:`~geohierarchy.aggregation.AggregationStrategy`,
    registered with :meth:`set_aggregation` (or passed directly to
    :meth:`add_level`). :meth:`propagate` then walks the graph and fills
    in that column everywhere it is missing: values move from a level to
    its parents by aggregation ("upscaling") and from a level to its
    children by disaggregation ("downscaling"). A level's own native
    values -- whether provided at creation or injected later with
    :meth:`add_vector_data` / :meth:`add_raster_data` -- are never
    overwritten by propagated values.

    Example:
        >>> gh = GeoHierarchy()
        >>> gh.add_level("county", county_gdf, id_col="geoid", agg=Sum())
        >>> gh.add_level("tract", tract_gdf, id_col="geoid", agg=Sum(), parent="county")
        >>> gh.add_vector_data(streets_gdf, level="tract", agg=Mean())
        >>> gh.propagate()
        >>> gh["county"]  # GeoDataFrame with columns filled in at every level

    Attributes:
        crs: Common coordinate reference system all level geometries are
            reprojected to.
        levels: Mapping of level name to its Polars attribute table
            (geometry excluded).
        geometries: Mapping of level name to a GeoDataFrame with just the
            id column, geometry, and geoweighting column.
        id_cols: Mapping of level name to the name of its unique id column.
        geoweight_by: Mapping of level name to whether area or length is
            used to weight overlaps involving that level.
        parents: Mapping of level name to the set of its parent level
            names in the level graph.
        children: Mapping of level name to the set of its child level
            names in the level graph.
        columns: Mapping of column name to bookkeeping about which levels
            hold native values, which hold values derived through
            propagation, and which levels currently have the column at all.
        aggregations: Registry of aggregation strategies, keyed by
            ``"column"``, ``"level"``, and ``"column_level"``.
        non_propagable_cols: Column names (geometry-derived metadata, id
            columns, ...) that are always excluded from propagation and
            can never be re-enabled.
        excluded_columns: Column names that would otherwise be eligible
            for propagation but have been opted out with
            :meth:`exclude_column`. Unlike ``non_propagable_cols``, this
            is meant to be toggled at runtime with :meth:`exclude_column`
            / :meth:`include_column`.
    """

    # ================================================================
    # INIT
    # ================================================================

    def __init__(self, crs: Any = "EPSG:4326") -> None:
        """Initialize an empty hierarchy.

        Args:
            crs: Coordinate reference system (anything accepted by
                :class:`pyproj.CRS`) that every level's geometry is
                reprojected to when added.
        """
        self.crs = CRS(crs)

        self.levels: Dict[str, pl.DataFrame] = {}
        self.geometries: Dict[str, gpd.GeoDataFrame] = {}
        self.id_cols: Dict[str, str] = {}
        self.geoweight_by: Dict[str, str] = {}

        self.parents: Dict[str, Set[str]] = {}
        self.children: Dict[str, Set[str]] = {}

        self.columns: Dict[str, Dict[str, Set[str]]] = {}

        self.aggregations: Dict[str, Dict[str, AggregationStrategy]] = {
            "column": {},
            "level": {},
            "column_level": {},
        }

        self._mapping_cache: Dict[str, pl.DataFrame] = {}

        self.non_propagable_cols: Set[str] = set(NON_PROPAGATABLE_COLS)
        self.excluded_columns: Set[str] = set()

    # ================================================================
    # INTERNAL HELPERS
    # ================================================================

    def _init_level(self, name: str) -> None:
        """Ensure a level has an entry in the level graph.

        Args:
            name: Level name to register if not already present.
        """
        self.parents.setdefault(name, set())
        self.children.setdefault(name, set())

    def _link(
        self,
        name: str,
        parent: Optional[_LevelNames] = None,
        child: Optional[_LevelNames] = None,
    ) -> None:
        """Wire a level to its parent(s) and/or child(ren) in the level graph.

        Links are always recorded on both endpoints, so a level's parents
        and children can be read from either side of the graph.

        Args:
            name: Level being linked.
            parent: Name or list of names of ``name``'s parent level(s).
            child: Name or list of names of ``name``'s child level(s).
        """
        for p in _as_list(parent):
            self._init_level(p)
            self.parents[name].add(p)
            self.children[p].add(name)

        for c in _as_list(child):
            self._init_level(c)
            self.children[name].add(c)
            self.parents[c].add(name)

    def _init_column(self, col: str) -> None:
        """Ensure a column has an entry in the column registry.

        Args:
            col: Column name to register if not already present.
        """
        if col not in self.columns:
            self.columns[col] = {
                "native_levels": set(),
                "derived_levels": set(),
                "levels": set(),
            }

    def _is_propagatable(self, col: str) -> bool:
        """Check whether a column is eligible for cross-level propagation.

        Args:
            col: Column name to check.

        Returns:
            ``True`` unless the column is structurally non-propagable
            (geometry metadata, id columns, etc.) or has been opted out
            at runtime with :meth:`exclude_column`.
        """
        return (col not in self.non_propagable_cols) and (
            col not in self.excluded_columns
        )

    @property
    def propagation_columns(self) -> Set[str]:
        """The current set of columns eligible for cross-level propagation.

        This is the live "column list" :meth:`propagate` acts on: every
        registered column except geometry metadata/id columns and any
        column excluded with :meth:`exclude_column`.
        """
        return {c for c in self.columns if self._is_propagatable(c)}

    def _require_agg(
        self,
        column: Optional[str] = None,
        level: Optional[str] = None,
        _raise: bool = True,
    ) -> Optional[AggregationStrategy]:
        """Resolve the aggregation strategy registered for a column/level.

        Resolution order: an exact ``(column, level)`` match takes
        precedence, then a column-wide strategy, then a level-wide default.

        Args:
            column: Column name to resolve a strategy for.
            level: Level name to resolve a strategy for.
            _raise: If ``True``, raise when no strategy is found; if
                ``False``, emit a warning and return ``None`` instead.

        Returns:
            The resolved :class:`AggregationStrategy`, or ``None`` if none
            was found and ``_raise`` is ``False``.

        Raises:
            Exception: If neither ``column`` nor ``level`` is given.
            ValueError: If ``_raise`` is ``True`` and no strategy is found.
        """
        if (column is None) and (level is None):
            raise Exception("Keywords 'column', 'level' or both are required")

        if (column is not None) and (level is not None):
            value = f"column_{column}_-_level_{level}"
            if value in self.aggregations["column_level"]:
                return self.aggregations["column_level"][value]

        if (column is not None) and (column in self.aggregations["column"]):
            return self.aggregations["column"][column]

        if (level is not None) and (level in self.aggregations["level"]):
            return self.aggregations["level"][level]

        label = f"column '{column}' " if column is not None else ""
        label += f"level '{level}'" if level is not None else ""

        if _raise:
            raise ValueError(f"Missing aggregation strategy for {label}")

        warnings.warn(
            f"Missing aggregation strategy for {label}. "
            "Call .set_aggregation() to solve it.",
            category=UserWarning,
        )
        return None

    # ================================================================
    # AGGREGATION REGISTRY
    # ================================================================

    def set_aggregation(
        self,
        column: Optional[str],
        agg: Optional[AggregationStrategy] = None,
        level: Optional[str] = None,
        *,
        upscale: Optional[AggregationStrategy] = None,
        downscale: Optional[AggregationStrategy] = None,
        _auto_propagate: bool = True,
    ) -> None:
        """Register the aggregation strategy for a column.

        Call this with just ``column`` to set a column-wide default
        strategy, with just ``level`` (``column=None``) to set the default
        for every propagatable column native to that level, or with both
        to override the strategy for that specific column/level
        combination -- which takes precedence over the other two forms
        when resolving what to use during propagation.

        Changing an existing strategy invalidates any values already
        derived from it, and this call triggers :meth:`propagate` itself
        (unless the column doesn't exist at any level yet), so there's no
        need to call it manually afterward.

        Args:
            column: Column name the strategy applies to. May be ``None``
                only if ``level`` is given.
            agg: Aggregation strategy instance to register, used for both
                upscaling and downscaling. Omit if you pass ``upscale``
                and/or ``downscale`` instead.
            level: Level name the strategy applies to.
            upscale: Strategy whose upscale (fine -> coarse) behavior is
                used. Combine with ``downscale`` to give a column
                different upscale and downscale rules -- for example
                ``set_aggregation("income", upscale=Mean(weight_column="population"), downscale=Max())``.
                Ignored if ``agg`` is given.
            downscale: Strategy whose downscale (coarse -> fine) behavior
                is used, paired with ``upscale``. If ``upscale`` is given
                without ``downscale``, ``upscale``'s own downscale
                behavior is kept.

        Raises:
            Exception: If neither ``column`` nor ``level`` is provided, or
                if none of ``agg``/``upscale``/``downscale`` is given.
        """
        if agg is None:
            if upscale is None:
                raise Exception(
                    "One of 'agg' or 'upscale' (optionally with 'downscale') is required"
                )
            agg = aggregation_strategy(upscale, downscale)

        if (column is not None) and (level is not None):
            mode, key = "column_level", f"column_{column}_-_level_{level}"
        elif column is not None:
            mode, key = "column", column
        elif level is not None:
            mode, key = "level", level
        else:
            raise Exception("Keywords 'column', 'level' or both are required")

        if (column is not None) and (column not in self.columns):
            self._init_column(column)
            for lvl, df in self.levels.items():
                if column in df.columns:
                    self.columns[column]["native_levels"].add(lvl)
                    self.columns[column]["levels"].add(lvl)

        old = self.aggregations[mode].get(key)
        if (old is not None) and (old is not agg):
            if mode == "column_level":
                self.columns[column]["derived_levels"].discard(level)
                self.columns[column]["levels"].discard(level)
            elif mode == "column":
                self.columns[column]["derived_levels"].clear()
                self.columns[column]["levels"] = set(
                    self.columns[column]["native_levels"]
                )
            elif mode == "level":
                for col in self.columns:
                    self.columns[col]["derived_levels"].discard(level)
                    self.columns[col]["levels"].discard(level)

        self.aggregations[mode][key] = agg

        # Only auto-propagate if there's actually something to propagate:
        # a bare column-wide/level-wide strategy registered ahead of any
        # matching data would otherwise fail the "reached every level"
        # check in propagate() for a column that (correctly) exists
        # nowhere yet.
        if _auto_propagate and self.levels:
            if (column is None) or self.columns[column]["native_levels"]:
                self.propagate()

    # ================================================================
    # PROPAGATION COLUMN LIST
    # ================================================================

    def exclude_column(self, column: str) -> None:
        """Opt a column out of cross-level propagation.

        The column keeps its native values at whatever levels it already
        has them (from :meth:`add_level`, :meth:`add_vector_data`, or
        :meth:`add_raster_data`), but is dropped from every other level it
        had reached through a previous :meth:`propagate` call, and future
        calls to :meth:`propagate` skip it entirely. No aggregation
        strategy is required for an excluded column.

        Args:
            column: Column name to exclude.

        Raises:
            KeyError: If ``column`` is not a known column.
        """
        if column not in self.columns:
            raise KeyError(f"Unknown column '{column}'")

        self.excluded_columns.add(column)

        native = self.columns[column]["native_levels"]
        for lev in self.levels:
            if (lev not in native) and (column in self.levels[lev].columns):
                self.levels[lev] = self.levels[lev].drop(column)

        self.columns[column]["derived_levels"].clear()
        self.columns[column]["levels"] = set(native)

        if self.levels:
            self.propagate()

    def include_column(self, column: str) -> None:
        """Re-enable propagation for a column previously excluded.

        Has no effect if ``column`` was never excluded. Triggers
        :meth:`propagate` itself, which fills the column in at every
        reachable level using whichever aggregation strategy is
        registered for it (which must be set separately, e.g. with
        :meth:`set_aggregation`).

        Args:
            column: Column name to re-enable.
        """
        self.excluded_columns.discard(column)

        if self.levels:
            self.propagate()

    # ================================================================
    # LEVEL CREATION
    # ================================================================

    def add_level(
        self,
        name: str,
        gdf: gpd.GeoDataFrame,
        id_col: Optional[str] = None,
        agg: Optional[
            Union[AggregationStrategy, Dict[str, AggregationStrategy]]
        ] = None,
        parent: Optional[_LevelNames] = None,
        child: Optional[_LevelNames] = None,
        geoweight_by: Optional[Literal["area", "length"]] = None,
    ) -> None:
        """Add a spatial level (a hierarchy layer) built from a GeoDataFrame.

        The geometry is reprojected to the hierarchy's CRS, and derived
        geometry metadata (area, length, bounds, centroid, geometry type)
        is computed and stored as non-propagable columns. The remaining
        columns become candidate propagation columns and must each resolve
        to an aggregation strategy, either supplied here via ``agg`` or
        registered separately with :meth:`set_aggregation`.

        This triggers :meth:`propagate` itself, so there's no need to call
        it manually afterward. If ``parent``/``child`` connects this level
        to levels that already hold propagated (non-native) values, those
        values are dropped first and recomputed, since the new edge might
        now offer a higher-priority source (a child always outranks a
        parent -- see :meth:`propagate`).

        Args:
            name: Unique name for the level.
            gdf: Source geometries and attributes for the level.
            id_col: Name of an existing column in ``gdf`` to use as the
                unique id. If ``None``, a positional id is generated.
            agg: Aggregation strategy applied to every propagatable column
                in ``gdf`` if a single :class:`AggregationStrategy` is
                given, or a per-column mapping if a dict is given. Either
                way it's registered column-wide (via
                :meth:`set_aggregation`), so it keeps applying to that
                column at every level it later propagates from, not just
                this one. If ``None``, a strategy must already be
                registered for each column (a warning is emitted
                otherwise).
            parent: Name or list of names of this level's parent(s) in the
                level graph. A level may have any number of parents.
            child: Name or list of names of this level's child(ren) in the
                level graph. A level may have any number of children.
            geoweight_by: Whether to weight spatial overlaps by ``"area"``
                or ``"length"`` when this level acts as the source of a
                geoweighted aggregation. Defaults to ``"area"`` for
                polygonal geometries and ``"length"`` otherwise.
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
            geoweight_by = "area" if "Polygon" in geom_types else "length"

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

            if col in self.non_propagable_cols:
                continue

            if agg is None:
                self._require_agg(column=col, level=name, _raise=False)
            elif isinstance(agg, dict):
                # Registered column-wide (not tied to this level as
                # source), so it keeps working as the column moves on to
                # further levels. Use set_aggregation(col, agg, level=...)
                # afterward for a level-specific override instead.
                self.set_aggregation(col, agg[col], _auto_propagate=False)
            else:
                self.set_aggregation(col, agg, _auto_propagate=False)

        # ---------------- graph ----------------
        had_links = bool(_as_list(parent)) or bool(_as_list(child))
        self._link(name, parent=parent, child=child)

        if had_links:
            # A new edge might offer a higher-priority (child) source for
            # a column some other level already derived from a parent.
            self._invalidate_derived()

        self.propagate()

    # ================================================================
    # PROPAGATION
    # ================================================================

    def _invalidate_derived(self, columns: Optional[List[str]] = None) -> None:
        """Drop propagated (non-native) values so the next propagate() redoes them.

        Used whenever the graph topology changes (a new parent/child edge
        added by :meth:`add_level`), since a newly linked level might now
        offer a higher-priority source for a column another level already
        derived a value from before that edge existed.

        Args:
            columns: Column names to invalidate. Defaults to every
                registered column.
        """
        for col in columns if columns is not None else list(self.columns):
            info = self.columns.get(col)
            if not info or not info["derived_levels"]:
                continue
            for lev in list(info["derived_levels"]):
                if (lev in self.levels) and (col in self.levels[lev].columns):
                    self.levels[lev] = self.levels[lev].drop(col)
            info["derived_levels"].clear()
            info["levels"] = set(info["native_levels"])

    def propagate(self) -> None:
        """Fill in missing columns across every level of the hierarchy graph.

        :meth:`add_level`, :meth:`set_aggregation`, :meth:`add_vector_data`,
        :meth:`add_raster_data`, :meth:`exclude_column`, and
        :meth:`include_column` all call this automatically after they
        change anything that could affect propagation, so it normally
        never needs to be called directly. The one exception is mutating
        ``hierarchy.levels[name]`` in place (e.g. for quick experiments):
        that bypasses the column registry entirely, so call
        :meth:`propagate` yourself afterward -- and prefer registering any
        new native data through :meth:`set_aggregation`, :meth:`add_level`,
        or :meth:`add_vector_data` instead, since a column that has
        already been registered once only has its native levels rescanned
        the first time, not on every :meth:`set_aggregation` call.

        Every propagatable column is pushed from each level to its parents
        (aggregation / "upscaling") and pushed from each level to its
        children (disaggregation / "downscaling"), repeating until no more
        levels can be filled in. Columns that already hold values at a
        level (native or previously derived) are left untouched, so
        calling this repeatedly is idempotent.

        The upscale pass always runs to completion before the downscale
        pass starts, so **when a level could receive a column both from a
        child (via aggregation) and from a parent (via disaggregation),
        the child's value wins**: the level is filled from its child
        before the downscale pass ever considers filling it from its
        parent.

        Any propagatable column stripped of its level membership (e.g.
        because its aggregation strategy changed) is dropped from that
        level's table before propagation resumes.

        A column that needs to cross an edge but has no aggregation
        strategy registered for the source level is never fatal: it's
        skipped (left missing at levels it couldn't reach) and a
        ``UserWarning`` is emitted pointing at :meth:`set_aggregation`.
        This matters because propagation now runs automatically after
        almost every call, so one column missing a strategy should never
        block unrelated changes elsewhere in the hierarchy.
        """

        columns = [c for c in self.columns if self._is_propagatable(c)]
        all_levels = set(self.levels.keys())

        for lev in self.levels:
            lev_cols = [c for c in self.levels[lev].columns if c in columns]
            delete_cols = [c for c in lev_cols if lev not in self.columns[c]["levels"]]
            if delete_cols:
                self.levels[lev] = self.levels[lev].drop(delete_cols)

        self._propagate_columns(columns, upscale=True)
        self._propagate_columns(columns, upscale=False)
        self._fill_native_nulls(columns)

        for c in columns:
            missing_levels = all_levels - self.columns[c]["levels"]
            if missing_levels:
                warnings.warn(
                    f"Column '{c}' did not reach level(s) {sorted(missing_levels)}. "
                    "This usually means an aggregation strategy is missing for "
                    "some level along the way -- call .set_aggregation() to solve it.",
                    category=UserWarning,
                )

    # ================================================================
    # EDGE PROPAGATION
    # ================================================================

    def _propagate_columns(self, columns: List[str], upscale: bool = True) -> None:
        """Propagate columns along every parent/child edge with a worklist.

        Rather than re-scanning the whole graph on every pass, this keeps
        a queue of levels whose available columns just changed and only
        re-examines the edges leaving those levels, so each edge is
        touched exactly as many times as it actually receives new
        columns. Column availability per level is tracked as an in-memory
        set instead of re-reading each Polars DataFrame's column list on
        every check.

        Walking the ``parents`` side of the graph alone is enough to reach
        every edge exactly once (each edge is recorded symmetrically in
        both ``parents`` and ``children`` by :meth:`_link`).

        Args:
            columns: Propagatable column names to consider.
            upscale: If ``True``, propagate from each level to its
                parent(s) (fine -> coarse); if ``False``, propagate from
                each level to its child(ren) (coarse -> fine).
        """
        columns_set = set(columns)
        available: Dict[str, Set[str]] = {
            lev: columns_set.intersection(self.levels[lev].columns)
            for lev in self.levels
        }

        queue = deque(self.levels.keys())
        queued = set(queue)

        while queue:
            level = queue.popleft()
            queued.discard(level)

            neighbors = self.parents[level] if upscale else self.children[level]
            for other in neighbors:
                if other not in self.levels:
                    continue

                # upscale: level's data flows up into its parent (other).
                # downscale: level's data flows down into its child (other).
                src, dst = level, other

                missing = available[src] - available[dst]
                if not missing:
                    continue

                propagated = self._propagate_edge(
                    list(missing), src, dst, upscale=upscale
                )
                if not propagated:
                    continue

                available[dst].update(propagated)
                for c in propagated:
                    self.columns[c]["levels"].add(dst)
                    self.columns[c]["derived_levels"].add(dst)

                if dst not in queued:
                    queue.append(dst)
                    queued.add(dst)

    def _propagate_edge(
        self, columns: List[str], src: str, dst: str, upscale: bool = True
    ) -> List[str]:
        """Propagate a batch of columns across a single graph edge.

        Builds (and caches) a geometric id mapping between ``src`` and
        ``dst``, joins the source columns through it, applies each
        column's aggregation strategy, and left-joins the result into the
        destination level's table. Columns with no aggregation strategy
        registered for ``src`` are skipped (with a warning emitted by
        :meth:`_require_agg`) rather than aborting the whole edge.

        Args:
            columns: Propagatable column names to move from ``src`` to
                ``dst``. All must share the same source level.
            src: Name of the level the columns currently exist on.
            dst: Name of the level receiving the columns.
            upscale: If ``True``, aggregate many ``src`` rows into each
                ``dst`` row (fine -> coarse); if ``False``, disaggregate
                each ``src`` row across matching ``dst`` rows (coarse ->
                fine).

        Returns:
            The subset of ``columns`` that actually had a strategy and
            were propagated.
        """
        resolved_columns = []
        aggs = []
        for col in columns:
            agg = self._require_agg(col, src, _raise=False)
            if agg is not None:
                resolved_columns.append(col)
                aggs.append(agg)

        if not resolved_columns:
            return []

        # Columns can mix mapping modes (e.g. a plain Sum column and a
        # SmoothMean column reaching the same edge in the same pass), so
        # each mode gets its own mapping and its own join/aggregate step.
        by_mapping: Dict[str, tuple] = {}
        for col, agg in zip(resolved_columns, aggs):
            by_mapping.setdefault(agg.mapping, ([], []))
            by_mapping[agg.mapping][0].append(col)
            by_mapping[agg.mapping][1].append(agg)

        results = []
        for mode, (mode_columns, mode_aggs) in by_mapping.items():
            mapping = self._get_or_build_mapping(mode, mode_aggs, src, dst, upscale)
            results.append(
                self._aggregate_edge_batch(
                    mode_columns, mode_aggs, mapping, src, dst, upscale
                )
            )

        result = results[0]
        for extra in results[1:]:
            result = result.join(extra, on=self.id_cols[dst], how="full", coalesce=True)

        self.levels[dst] = self.levels[dst].join(
            result.select([self.id_cols[dst], *resolved_columns]),
            on=self.id_cols[dst],
            how="left",
        )

        return resolved_columns

    def _get_or_build_mapping(
        self,
        mode: str,
        aggs: List[AggregationStrategy],
        src: str,
        dst: str,
        upscale: bool,
    ) -> pl.DataFrame:
        """Build (and cache) the id mapping for one mapping mode ("overlap" or "knn").

        Args:
            mode: ``"overlap"`` or ``"knn"``.
            aggs: Strategies sharing this mode for the current batch --
                used to read ``geoweighted``/``knn_k``/``knn_power``.
            src: Source level name.
            dst: Destination level name.
            upscale: Propagation direction.

        Returns:
            The mapping DataFrame for this mode.
        """
        if upscale:
            geo_a, geo_b, id_a, id_b = (
                self.geometries[src],
                self.geometries[dst],
                self.id_cols[src],
                self.id_cols[dst],
            )
        else:
            geo_a, geo_b, id_a, id_b = (
                self.geometries[dst],
                self.geometries[src],
                self.id_cols[dst],
                self.id_cols[src],
            )

        if mode == "knn":
            agg = aggs[0]
            cache_key = f"_knn_mapping_{id_a}_{id_b}_{agg.knn_k}_{agg.knn_power}"
            if cache_key not in self._mapping_cache:
                self._mapping_cache[cache_key] = get_knn_mapping(
                    geo_a, geo_b, id_a, id_b, k=agg.knn_k, power=agg.knn_power
                )
        else:
            geoweighted = any(agg.geoweighted for agg in aggs)
            how = self.geoweight_by[src] if upscale else self.geoweight_by[dst]
            cache_key = f"_id_mapping_{id_a}_{id_b}_{geoweighted}_{how}"
            if cache_key not in self._mapping_cache:
                self._mapping_cache[cache_key] = get_id_mapping(
                    geo_a, geo_b, id_a, id_b, geoweighted=geoweighted, how=how
                )

        return self._mapping_cache[cache_key]

    def _aggregate_edge_batch(
        self,
        columns: List[str],
        aggs: List[AggregationStrategy],
        mapping: pl.DataFrame,
        src: str,
        dst: str,
        upscale: bool,
    ) -> pl.DataFrame:
        """Join ``src`` data through ``mapping`` and aggregate one batch of same-mode columns.

        Args:
            columns: Column names to aggregate, all sharing one mapping mode.
            aggs: Their aggregation strategies, same order as ``columns``.
            mapping: Id mapping for this batch, from :meth:`_get_or_build_mapping`.
            src: Source level name.
            dst: Destination level name.
            upscale: Propagation direction.

        Returns:
            One row per destination id, with ``columns`` aggregated.
        """
        selected_cols = set(columns)
        selected_cols.add(self.id_cols[src])
        for agg in aggs:
            if getattr(agg, "weight_column", None):
                selected_cols.add(agg.weight_column)

        data = self.levels[src].select(list(selected_cols))
        joined = mapping.join(data, on=self.id_cols[src])

        if upscale:
            exprs = []
            for col, agg in zip(columns, aggs):
                exprs += agg.upscale_aggs([col])
            return joined.group_by(self.id_cols[dst]).agg(exprs)

        exprs = []
        for col, agg in zip(columns, aggs):
            exprs += agg.downscale_exprs([col], self.id_cols[src])
        result = joined.with_columns(exprs) if exprs else joined

        # A destination row can overlap more than one source row (e.g. a
        # street crossing two tracts, or several knn neighbors), which
        # leaves one fragment row per source it overlaps here. Collapse
        # those fragments back to a single row per destination id, letting
        # each strategy decide how (Sum/SmoothMean add them, most others
        # just keep one).
        consolidate_exprs = []
        for col, agg in zip(columns, aggs):
            consolidate_exprs += agg.consolidate_downscale([col])
        result = result.group_by(self.id_cols[dst]).agg(consolidate_exprs)

        return self._rescale_to_preserve_totals(result, columns, aggs, src)

    def _rescale_to_preserve_totals(
        self,
        result: pl.DataFrame,
        columns: List[str],
        aggs: List[AggregationStrategy],
        src: str,
    ) -> pl.DataFrame:
        """Correct a downscaled column back to matching its source total, where required.

        Most strategies (:class:`~geohierarchy.aggregation.Sum`'s proportional
        split) already reproduce the source total exactly by construction.
        :class:`~geohierarchy.aggregation.SmoothMean`'s spatial blending
        doesn't -- it optimizes for a smooth surface, not an exact
        partition -- so any column whose strategy has
        ``preserve_total=True`` (an absolute/additive quantity, as opposed
        to a relative one with no real total to preserve) gets uniformly
        rescaled here so its grand total across ``result`` matches its
        grand total in ``self.levels[src]``.

        Args:
            result: One row per destination id, columns already aggregated.
            columns: Column names in ``result`` to consider.
            aggs: Their aggregation strategies, same order as ``columns``.
            src: Source level name, to read the target total from.

        Returns:
            ``result`` with any ``preserve_total`` column rescaled in place.
        """
        rescale_exprs = []
        for col, agg in zip(columns, aggs):
            if not agg.preserve_total:
                continue

            src_total = self.levels[src][col].sum()
            dst_total = result[col].sum()
            if not src_total or not dst_total:
                continue

            rescale_exprs.append((pl.col(col) * (src_total / dst_total)).alias(col))

        return result.with_columns(rescale_exprs) if rescale_exprs else result

    # ================================================================
    # NULL BACKFILL (NATIVE NULLS ONLY)
    # ================================================================

    def _fill_native_nulls(self, columns: List[str]) -> None:
        """Backfill null cells in a present column from a parent, leaving real values untouched.

        A column can be present at a level (native, or already derived by
        :meth:`_propagate_columns`) while still holding nulls -- e.g. a
        source that legitimately has no value for some rows. Ordinary
        propagation never revisits that column for this level, since
        "present" is all it checks. This fills exactly the null rows, by
        downscaling from a parent that has the column, without touching any
        row that already holds a value. Runs coarsest-to-finest with a
        worklist so a grandparent's values can cascade down through a
        parent that itself just got backfilled.

        Args:
            columns: Propagatable column names to consider.
        """
        queue = deque(self.levels.keys())
        queued = set(queue)

        while queue:
            lev = queue.popleft()
            queued.discard(lev)
            changed = False

            for col in columns:
                if col not in self.levels[lev].columns:
                    continue
                remaining = self.levels[lev][col].null_count()
                if remaining == 0:
                    continue

                for parent in self.parents.get(lev, ()):
                    if (
                        parent not in self.levels
                        or col not in self.levels[parent].columns
                    ):
                        continue
                    agg = self._require_agg(col, parent, _raise=False)
                    if agg is None:
                        continue

                    filled = self._downscale_fill_column(col, agg, src=parent, dst=lev)
                    new_remaining = filled[col].null_count()
                    if new_remaining < remaining:
                        self.levels[lev] = filled
                        changed = True
                        remaining = new_remaining
                    if remaining == 0:
                        break

            if changed:
                for child in self.children.get(lev, ()):
                    if child in self.levels and child not in queued:
                        queue.append(child)
                        queued.add(child)

    def _downscale_fill_column(
        self, col: str, agg: AggregationStrategy, src: str, dst: str
    ) -> pl.DataFrame:
        """Compute ``col``'s downscaled values from ``src`` and coalesce them into ``dst``'s nulls.

        Args:
            col: Column to fill.
            agg: Aggregation strategy registered for ``col`` at ``src``.
                Its ``mapping`` mode (``"overlap"`` or ``"knn"``) picks
                which id-mapping function pairs the rows.
            src: Parent level to downscale from.
            dst: Level whose null ``col`` rows should be filled.

        Returns:
            ``dst``'s table with ``col``'s null rows filled wherever
            ``src`` had a value to offer; non-null rows are unchanged.
        """
        if agg.mapping == "knn":
            cache_key = f"_knn_mapping_{dst}_{src}_{agg.knn_k}_{agg.knn_power}"
            if cache_key not in self._mapping_cache:
                self._mapping_cache[cache_key] = get_knn_mapping(
                    self.geometries[dst],
                    self.geometries[src],
                    self.id_cols[dst],
                    self.id_cols[src],
                    k=agg.knn_k,
                    power=agg.knn_power,
                )
        else:
            cache_key = (
                f"_id_mapping_{dst}_{src}_{agg.geoweighted}_{self.geoweight_by[dst]}"
            )
            if cache_key not in self._mapping_cache:
                self._mapping_cache[cache_key] = get_id_mapping(
                    self.geometries[dst],
                    self.geometries[src],
                    self.id_cols[dst],
                    self.id_cols[src],
                    geoweighted=agg.geoweighted,
                    how=self.geoweight_by[dst],
                )
        mapping = self._mapping_cache[cache_key]

        selected_cols = {col, self.id_cols[src]}
        if getattr(agg, "weight_column", None):
            selected_cols.add(agg.weight_column)
        if not selected_cols.issubset(self.levels[src].columns):
            return self.levels[dst]

        data = self.levels[src].select(list(selected_cols))
        joined = mapping.join(data, on=self.id_cols[src])

        exprs = agg.downscale_exprs([col], self.id_cols[src])
        candidate = joined.with_columns(exprs) if exprs else joined
        consolidate_exprs = agg.consolidate_downscale([col])
        candidate = candidate.group_by(self.id_cols[dst]).agg(consolidate_exprs)
        candidate = candidate.select(
            [self.id_cols[dst], pl.col(col).alias("_fill_candidate")]
        )

        return (
            self.levels[dst]
            .join(candidate, on=self.id_cols[dst], how="left")
            .with_columns(
                pl.coalesce([pl.col(col), pl.col("_fill_candidate")]).alias(col)
            )
            .drop("_fill_candidate")
        )

    # ================================================================
    # VECTOR INPUT (NON-LAYER DATA)
    # ================================================================

    def add_vector_data(
        self,
        gdf: gpd.GeoDataFrame,
        level: str,
        columns: Optional[Union[str, List[str]]] = None,
        agg: Union[AggregationStrategy, Dict[str, AggregationStrategy], None] = None,
        geoweight_by: Optional[Literal["area", "length"]] = None,
        upscale: Optional[bool] = None,
        buffer: float = 0,
        fill_null: Union[int, float, Dict[str, Union[int, float]], None] = 0,
    ) -> None:
        """Inject columns from an external (non-layer) vector dataset into a level.

        Unlike :meth:`add_level`, ``gdf`` does not become part of the
        hierarchy graph -- it is only used as a one-off source of column
        values for ``level``. The direction of resampling is auto-detected
        (a finer source upscales into ``level``, a coarser source
        downscales) unless ``upscale`` is given explicitly. The strategy
        passed in ``agg`` is also registered for the injected columns
        going forward, so it is what :meth:`propagate` later uses to move
        them to every other level -- which happens automatically as part
        of this call, so there's no need to call :meth:`propagate`
        manually afterward.

        Args:
            gdf: External geometries and attributes to inject from.
            level: Name of the level to inject the columns into.
            columns: Column name or list of column names in ``gdf`` to
                inject. If ``None``, every propagatable column is used.
            agg: Aggregation strategy (or per-column mapping) used both
                for this injection and registered for future propagation
                of the column.
            geoweight_by: Whether to weight overlaps by ``"area"`` or
                ``"length"``. Auto-detected from ``gdf``'s geometry type
                if not given.
            upscale: Force the resampling direction. If ``None``, it is
                inferred by comparing the median geoweight of ``gdf`` to
                that of ``level``.
            buffer: Optional buffer distance (in the hierarchy CRS'
                units) applied to whichever side is polygonal, useful for
                capturing near-miss overlaps (e.g. streets near block
                boundaries).
            fill_null: Value(s) used to fill nulls left after injection --
                a scalar applied to every injected column, a per-column
                mapping, or ``None`` to leave nulls as-is.

        Raises:
            Exception: If ``agg`` is not given.
        """
        if isinstance(columns, str):
            columns = [columns]

        if agg is None:
            raise Exception("The 'agg' keyword is required")

        gdf = gdf.copy()
        if gdf.geometry.name != "geometry":
            gdf = gdf.rename(columns={gdf.geometry.name: "geometry"})
            gdf = gdf.set_geometry("geometry")

        if columns is None:
            columns = [
                col for col in gdf.columns if col not in self.non_propagable_cols
            ]

        gdf = gdf.to_crs(self.crs)

        if isinstance(agg, dict):
            is_geoweighted = any(getattr(a, "geoweighted", False) for a in agg.values())
        else:
            is_geoweighted = getattr(agg, "geoweighted", False)

        if is_geoweighted or upscale is None:
            if geoweight_by is None:
                geom_types = (
                    gdf.geometry.geom_type.str.replace("^Multi", "", regex=True)
                    .drop_duplicates()
                    .to_list()
                )
                geoweight_by = "area" if "Polygon" in geom_types else "length"

            if geoweight_by == "area":
                gdf["area"] = area(gdf)
            else:
                gdf["length"] = length(gdf)

        tid = "_tid"
        gdf[tid] = np.arange(len(gdf))

        if upscale is None:
            source_median = gdf[geoweight_by].median()
            level_median = (
                self.levels[level].select(pl.col(geoweight_by).median()).item()
            )
            upscale = source_median < level_median

        if buffer > 0:
            if upscale:
                dst_gdf = self.geometries[level].copy()
                if "geometry_type" not in dst_gdf.columns:
                    dst_gdf["geometry_type"] = dst_gdf.geometry.geom_type.str.replace(
                        "^Multi", "", regex=True
                    )
                mask = dst_gdf["geometry_type"] == "Polygon"
                dst_gdf = dst_gdf.to_crs(dst_gdf.estimate_utm_crs())
                dst_gdf.geometry = dst_gdf.geometry.buffer(buffer, resolution=2)
                dst_gdf.loc[mask, "geometry"] = dst_gdf.loc[mask, "geometry"].boundary
                dst_gdf = dst_gdf.to_crs(self.crs).drop(columns=["geometry_type"])
            else:
                if "geometry_type" not in gdf.columns:
                    gdf["geometry_type"] = gdf.geometry.geom_type.str.replace(
                        "^Multi", "", regex=True
                    )
                mask = gdf["geometry_type"] == "Polygon"
                gdf = gdf.to_crs(gdf.estimate_utm_crs())
                gdf.geometry = gdf.geometry.buffer(buffer, resolution=2)
                gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].boundary
                gdf = gdf.to_crs(self.crs).drop(columns=["geometry_type"])
        else:
            dst_gdf = self.geometries[level]

        if upscale:
            mapping = get_id_mapping(
                gdf,
                dst_gdf,
                tid,
                self.id_cols[level],
                geoweighted=is_geoweighted,
                how=geoweight_by,
            )
        else:
            mapping = get_id_mapping(
                dst_gdf,
                gdf,
                self.id_cols[level],
                tid,
                geoweighted=is_geoweighted,
                how=geoweight_by,
            )

        # Injecting a column resets its propagation state to just this level,
        # so it gets recomputed everywhere else on the next propagate().
        for col in columns:
            if col in self.columns:
                self.columns[col]["derived_levels"].clear()
                self.columns[col]["levels"].clear()
                self.columns[col]["levels"].update(self.columns[col]["native_levels"])

        col_selection = [c for c in gdf.columns if c not in self.non_propagable_cols]
        df_pl = pl.from_pandas(gdf[col_selection])
        joined = mapping.join(df_pl, on=tid)

        if upscale:
            if isinstance(agg, dict):
                columns = [c for c in columns if c in agg]
                exprs = [e for c in columns for e in agg[c].upscale_aggs([c])]
                result = joined.group_by(self.id_cols[level]).agg(exprs)
            else:
                result = agg.upscale(joined, self.id_cols[level], columns)
        else:
            if isinstance(agg, dict):
                columns = [c for c in columns if c in agg]
                exprs = [e for c in columns for e in agg[c].downscale_exprs([c], tid)]
                result = joined.with_columns(exprs) if exprs else joined
            else:
                result = agg.downscale(joined, tid, columns)

        self.levels[level] = (
            self.levels[level]
            .drop(columns, strict=False)
            .join(
                result.select([self.id_cols[level], *columns]),
                on=self.id_cols[level],
                how="left",
            )
        )

        self.levels[level] = self.levels[level].with_columns(
            [pl.col(col).fill_nan(None) for col in columns]
        )

        if isinstance(fill_null, (float, int)):
            self.levels[level] = self.levels[level].with_columns(
                [pl.col(col).fill_null(fill_null) for col in columns]
            )
        elif isinstance(fill_null, dict):
            self.levels[level] = self.levels[level].with_columns(
                [pl.col(col).fill_null(fill) for col, fill in fill_null.items()]
            )

        # ---------------- column registry ----------------
        for col in columns:
            self._init_column(col)
            self.columns[col]["native_levels"].add(level)
            self.columns[col]["levels"].add(level)

            if col in self.non_propagable_cols:
                continue

            if isinstance(agg, dict):
                if col in agg:
                    self.set_aggregation(col, agg[col])
            else:
                self.set_aggregation(col, agg)

    # ================================================================
    # RASTER INPUT (NON-LAYER DATA)
    # ================================================================

    def add_raster_data(
        self,
        raster: Union[np.ndarray, str],
        column: str,
        level: str,
        agg: Union[AggregationStrategy, Dict[str, AggregationStrategy], None] = None,
        upscale: bool = True,
        transform=None,
        crs=None,
    ) -> None:
        """Inject a raster band's values into a level as pixel-centroid points.

        The raster is vectorized into point geometries at each pixel
        centroid (non-null cells only), then delegated to
        :meth:`add_vector_data` with ``geoweight_by="area"``.

        Args:
            raster: Either a path to a raster file (opened with
                ``rasterio``) or an in-memory 2D array of values. If an
                array, ``transform`` and ``crs`` must be supplied.
            column: Name to give the injected column.
            level: Name of the level to inject the column into.
            agg: Aggregation strategy (or per-column mapping) used for
                this injection and registered for future propagation. See
                :meth:`add_vector_data`.
            upscale: Whether to aggregate pixels into level polygons
                (``True``) or disaggregate level values onto pixels
                (``False``).
            transform: Affine transform for ``raster`` when given as an
                array. Required if ``raster`` is not a file path.
            crs: Coordinate reference system for ``raster`` when given as
                an array. Required if ``raster`` is not a file path.

        Raises:
            ValueError: If ``raster`` is an array and either ``transform``
                or ``crs`` is missing.
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

        self.add_vector_data(
            gdf,
            level=level,
            columns=[column],
            agg=agg,
            geoweight_by="area",
            upscale=upscale,
        )

    # ================================================================
    # ACCESS
    # ================================================================

    def get_level(self, name: str) -> gpd.GeoDataFrame:
        """Return a level's geometry joined with its attribute table.

        Args:
            name: Name of the level to retrieve.

        Returns:
            A GeoDataFrame combining the level's geometry with every
            attribute column currently stored for it.
        """
        return self.geometries[name][[self.id_cols[name], "geometry"]].merge(
            self.levels[name].to_pandas(),
            on=self.id_cols[name],
        )

    def __getitem__(self, key: str) -> gpd.GeoDataFrame:
        """Alias for :meth:`get_level`, e.g. ``hierarchy["city"]``.

        Args:
            key: Name of the level to retrieve.

        Returns:
            A GeoDataFrame combining the level's geometry with every
            attribute column currently stored for it.
        """
        return self.get_level(key)
