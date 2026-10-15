"""Cells with no data to sample (e.g. an H3 hexagon with no street): three ways to handle them.

Typical case: street scores are averaged into hexagons, but some hexagons contain people and no street. Their
score is ``NaN`` ("no data"), and what to do with them is a modelling choice. Three conventions are used in
the CityScope studies, all available here through :func:`handle_empty_cells`:

``"nearest_value"``
    Copy the values of the **nearest** hexagon that has data (center to center). The people stay where they
    live and get the score of the closest served place. Optional ``max_distance``.
``"relocate_weight"``
    Move the **population** (any weight column) of an empty hexagon to the nearest hexagon with data, within
    ``max_distance`` (default 300 m); farther away it is excluded (and counted in ``attrs["excluded_weight"]``).
    The empty hexagon ends with weight 0. Totals are conserved except for the excluded remainder. Used by the
    city science network study.
``"ignore"``
    Leave them empty. Statistics must then skip them: use :func:`weighted_mean`, which removes the weight of
    empty cells from the denominator. Used by the USA study.

Example::

    from geohierarchy import handle_empty_cells, weighted_mean

    hexes = handle_empty_cells(hexes, ["access_score"], method="nearest_value")
    weighted_mean(hexes["access_score"], hexes["population"])
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EMPTY_CELL_METHODS = ("nearest_value", "relocate_weight", "ignore")
DEFAULT_RELOCATE_DISTANCE_M = (
    300.0  # the city science network study's cap (see module docstring)
)


def empty_mask(gdf: gpd.GeoDataFrame, columns: Sequence[str]) -> pd.Series:
    """``True`` for the cells with no data: every one of ``columns`` is ``NaN``."""
    return gdf[list(columns)].isna().all(axis=1)


def _centers_xy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Cell centers in meters (a projected CRS is used as is, otherwise the local UTM zone)."""
    projected = (
        gdf
        if (gdf.crs is not None and gdf.crs.is_projected)
        else gdf.to_crs(gdf.estimate_utm_crs())
    )
    c = projected.geometry.centroid
    return np.column_stack([c.x.to_numpy(), c.y.to_numpy()])


def _nearest(gdf: gpd.GeoDataFrame, targets: pd.Series, donors: pd.Series):
    """For every target cell: the nearest donor cell and the distance in meters."""
    from scipy.spatial import cKDTree

    xy = _centers_xy(gdf)
    dist, pos = cKDTree(xy[donors.to_numpy()]).query(xy[targets.to_numpy()])
    return gdf.index[donors][pos], dist


def fill_from_nearest(
    gdf: gpd.GeoDataFrame,
    columns: Sequence[str],
    weight_col: Optional[str] = "population",
    max_distance: Optional[float] = None,
    empty: Optional[pd.Series] = None,
    filled_col: str = "filled",
) -> gpd.GeoDataFrame:
    """Method ``"nearest_value"``: empty cells with weight copy ``columns`` from the nearest cell with data.

    Args:
        gdf: Cells (polygons) with ``columns``.
        columns: Value columns to copy.
        weight_col: Only cells with ``weight_col > 0`` (e.g. people) are filled; ``None`` fills every empty cell.
        max_distance: Largest center-to-center distance (m) to copy from; farther cells stay empty.
        empty: Mask of the empty cells (default: all of ``columns`` are ``NaN``).
        filled_col: Boolean column marking the cells that were filled.

    Returns:
        A copy of ``gdf`` with the cells filled and ``filled_col`` set.
    """
    columns = list(columns)
    out = gdf.copy()
    empty = empty_mask(out, columns) if empty is None else empty
    out[filled_col] = False
    targets = empty & (out[weight_col] > 0 if weight_col else True)
    donors = ~empty
    if not targets.any() or not donors.any():
        return out
    donor_idx, dist = _nearest(out, targets, donors)
    target_idx = out.index[targets]
    if max_distance is not None:
        ok = dist <= max_distance
        donor_idx, target_idx = donor_idx[ok], target_idx[ok]
    out.loc[target_idx, columns] = out.loc[donor_idx, columns].to_numpy()
    out.loc[target_idx, filled_col] = True
    return out


def relocate_weight(
    gdf: gpd.GeoDataFrame,
    columns: Sequence[str],
    weight_col: str = "population",
    max_distance: float = DEFAULT_RELOCATE_DISTANCE_M,
    empty: Optional[pd.Series] = None,
) -> gpd.GeoDataFrame:
    """Method ``"relocate_weight"``: move the weight of empty cells to the nearest cell with data.

    Within ``max_distance`` (m, center to center) the weight of every empty cell is added to its nearest cell
    with data and the empty cell is left with weight 0; farther away it is excluded. The excluded total is
    logged and stored in ``out.attrs["excluded_weight"]``.
    """
    columns = list(columns)
    out = gdf.copy()
    empty = empty_mask(out, columns) if empty is None else empty
    out.attrs["excluded_weight"] = 0.0
    movers = empty & (out[weight_col] > 0)
    donors = ~empty
    if not movers.any():
        return out
    weights = out.loc[movers, weight_col].astype(float)
    if not donors.any():
        out.attrs["excluded_weight"] = float(weights.sum())
        out.loc[movers, weight_col] = 0
        return out
    donor_idx, dist = _nearest(out, movers, donors)
    within = dist <= max_distance
    gained = (
        pd.Series(weights.to_numpy()[within])
        .groupby(np.asarray(donor_idx)[within])
        .sum()
    )
    out[weight_col] = out[weight_col].astype(float)
    out.loc[gained.index, weight_col] += gained
    out.loc[movers, weight_col] = 0.0
    excluded = float(weights.to_numpy()[~within].sum())
    out.attrs["excluded_weight"] = excluded
    if excluded:
        logger.info(
            "relocate_weight: %.0f of %s excluded (no cell with data within %.0f m)",
            excluded,
            weight_col,
            max_distance,
        )
    return out


def weighted_mean(values, weights, groups=None):
    """Weighted mean that **ignores empty values**: their weight leaves the denominator (method ``"ignore"``).

    Args:
        values: Values, ``NaN`` where a cell has no data.
        weights: Weights (e.g. population), same length.
        groups: Optional group of each cell (e.g. the parent cell): returns one mean per group.

    Returns:
        A float (``NaN`` if there is no weight), or a Series per group (``NaN`` where a group has no data).
    """
    v = pd.Series(np.asarray(values, dtype=float))
    w = pd.Series(np.asarray(weights, dtype=float))
    ok = v.notna()
    num = (v.where(ok, 0.0) * w).where(ok, 0.0)
    den = w.where(ok, 0.0)
    if groups is None:
        return float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    g = pd.Series(np.asarray(groups))
    n, d = num.groupby(g).sum(), den.groupby(g).sum()
    return n / d.where(d > 0)


def handle_empty_cells(
    gdf: gpd.GeoDataFrame,
    columns: Sequence[str],
    method: str = "nearest_value",
    weight_col: str = "population",
    max_distance: Optional[float] = None,
    empty: Optional[pd.Series] = None,
) -> gpd.GeoDataFrame:
    """Handle the cells with no data with one of :data:`EMPTY_CELL_METHODS` (see the module docstring).

    Args:
        gdf: Cells (polygons), with ``columns`` (``NaN`` = no data) and ``weight_col`` (e.g. population).
        columns: Value columns that define "no data" (all ``NaN``) and are copied by ``"nearest_value"``.
        method: ``"nearest_value"``, ``"relocate_weight"`` or ``"ignore"``.
        weight_col: Weight of each cell (people).
        max_distance: Meters. ``"nearest_value"``: default no limit. ``"relocate_weight"``: default 300.
        empty: Mask of the empty cells (default: all of ``columns`` are ``NaN``).

    Returns:
        A new GeoDataFrame. ``"nearest_value"`` adds a ``filled`` column; ``"relocate_weight"`` sets
        ``attrs["excluded_weight"]``; ``"ignore"`` returns the cells unchanged.
    """
    if method not in EMPTY_CELL_METHODS:
        raise ValueError(f"method must be one of {EMPTY_CELL_METHODS}, got {method!r}")
    if method == "ignore":
        return gdf.copy()
    if method == "nearest_value":
        return fill_from_nearest(gdf, columns, weight_col, max_distance, empty)
    return relocate_weight(
        gdf,
        columns,
        weight_col,
        DEFAULT_RELOCATE_DISTANCE_M if max_distance is None else max_distance,
        empty,
    )
