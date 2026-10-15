import geopandas as gpd
import numpy as np
import pytest
import shapely

from geohierarchy import (
    EMPTY_CELL_METHODS,
    empty_mask,
    fill_from_nearest,
    handle_empty_cells,
    relocate_weight,
    weighted_mean,
)


def _grid():
    """5x5 grid of 100 m cells (EPSG:32630). Only the left column (x=0) has data; 10 people per cell."""
    cells, value = [], []
    for i in range(5):
        for j in range(5):
            cells.append(shapely.box(i * 100, j * 100, (i + 1) * 100, (j + 1) * 100))
            value.append(float(j + 1) if i == 0 else np.nan)
    return gpd.GeoDataFrame(
        {"score": value, "population": 10}, geometry=cells, crs=32630
    )


def test_empty_mask():
    g = _grid()
    assert empty_mask(g, ["score"]).sum() == 20


def test_nearest_value_copies_row_wise_from_closest_cell():
    out = fill_from_nearest(_grid(), ["score"])
    assert out["score"].notna().all() and out["filled"].sum() == 20
    # cell (i=3, j=2) has its nearest data cell in the left column at the same height: score j+1 = 3
    row = out[(out.geometry.centroid.x == 350) & (out.geometry.centroid.y == 250)].iloc[
        0
    ]
    assert row["score"] == 3.0


def test_nearest_value_only_fills_cells_with_weight_and_respects_max_distance():
    g = _grid()
    g.loc[g.geometry.centroid.x == 450, "population"] = (
        0  # nobody lives in the last column
    )
    out = fill_from_nearest(g, ["score"])
    assert (
        out.loc[out.geometry.centroid.x == 450, "score"].isna().all()
        and out["score"].notna().sum() == 20
    )
    capped = fill_from_nearest(
        _grid(), ["score"], max_distance=150
    )  # reaches only the next column
    assert capped["filled"].sum() == 5 and capped["score"].notna().sum() == 10


def test_relocate_weight_moves_people_to_nearest_cell_with_data_and_conserves_total():
    out = relocate_weight(_grid(), ["score"], max_distance=1000)
    assert out["population"].sum() == 250 and out.attrs["excluded_weight"] == 0
    assert (out.loc[out["score"].isna(), "population"] == 0).all()
    assert out.loc[out["score"].notna(), "population"].sum() == 250
    # cells of row j go to the left-column cell of the same row: 5 cells + itself = 50 people
    assert (out.loc[out["score"].notna(), "population"] == 50).all()


def test_relocate_weight_excludes_what_is_too_far():
    out = relocate_weight(
        _grid(), ["score"], max_distance=150
    )  # only the next column can move
    assert out.attrs["excluded_weight"] == 150 and out["population"].sum() == 100
    assert (out.loc[out["score"].notna(), "population"] == 20).all()


def test_weighted_mean_ignores_empty_cells():
    assert weighted_mean([1.0, np.nan, 3.0], [10, 1000, 30]) == pytest.approx(
        (10 + 90) / 40
    )
    assert np.isnan(weighted_mean([np.nan], [5]))
    by = weighted_mean(
        [1.0, np.nan, 3.0, np.nan], [10, 10, 10, 10], groups=["a", "a", "b", "b"]
    )
    assert by["a"] == 1.0 and by["b"] == 3.0


def test_handle_empty_cells_dispatch():
    g = _grid()
    assert handle_empty_cells(g, ["score"], "ignore")["score"].isna().sum() == 20
    assert handle_empty_cells(g, ["score"], "nearest_value")["score"].isna().sum() == 0
    out = handle_empty_cells(
        g, ["score"], "relocate_weight"
    )  # default cap 300 m: the last column (400 m) is excluded
    assert out["population"].sum() == 200 and out.attrs["excluded_weight"] == 50
    assert set(EMPTY_CELL_METHODS) == {"nearest_value", "relocate_weight", "ignore"}
    with pytest.raises(ValueError):
        handle_empty_cells(g, ["score"], "teleport")
