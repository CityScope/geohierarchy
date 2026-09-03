"""Tests for geohierarchy.raster_resample: area-weighted, mass-conserving raster -> polygon resampling."""

import numpy as np
import geopandas as gpd
import pytest
import rasterio as rio
from shapely.geometry import box

from geohierarchy.raster_resample import (
    raster_to_polygons,
    raster_to_h3,
    raster_to_h3_tiled,
)


def _square_polys(crs="EPSG:32633"):
    """Two 1x1 squares side by side: [0,1]x[0,1] id='left', [1,2]x[0,1] id='right'."""
    return gpd.GeoDataFrame(
        {"id": ["left", "right"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=crs,
    )


def test_pixel_entirely_inside_one_polygon():
    """A pixel fully inside one polygon contributes 100% of its value to it."""
    array = np.array([[10.0]])
    transform = rio.Affine(
        1, 0, 0.0, 0, -1, 1.0
    )  # single 1x1 pixel covering [0,1]x[0,1]
    polys = _square_polys()
    result = raster_to_polygons(array, transform, polys.crs, polys, id_col="id")
    d = dict(zip(result["id"].to_list(), result["value"].to_list()))
    # 'right' may appear with a ~0 value since the pixel's bbox touches its shared
    # edge (zero-area intersection) -- only 'left' should carry real mass.
    assert d.get("left") == pytest.approx(10.0)
    assert d.get("right", 0.0) == pytest.approx(0.0, abs=1e-9)


def test_pixel_split_across_two_polygons_area_weighted():
    """A pixel straddling two equal-area regions of two polygons splits 50/50."""
    # Pixel spans x in [0.5, 1.5], y in [0,1]: half in 'left' ([0,1]x[0,1]),
    # half in 'right' ([1,2]x[0,1]).
    array = np.array([[20.0]])
    transform = rio.Affine(1, 0, 0.5, 0, -1, 1.0)
    polys = _square_polys()
    result = raster_to_polygons(array, transform, polys.crs, polys, id_col="id")
    d = dict(zip(result["id"].to_list(), result["value"].to_list()))
    assert d["left"] == pytest.approx(10.0)
    assert d["right"] == pytest.approx(10.0)
    assert sum(d.values()) == pytest.approx(20.0)


def test_full_coverage_no_touched_polygon_missing():
    """Every polygon touched by a valid pixel appears in the output."""
    # 4 pixels each fully inside a distinct 1x1 cell of a 2x2 grid of polygons.
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    transform = rio.Affine(1, 0, 0.0, 0, -1, 2.0)
    polys = gpd.GeoDataFrame(
        {"id": ["a", "b", "c", "d"]},
        geometry=[box(0, 1, 1, 2), box(1, 1, 2, 2), box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:32633",
    )
    result = raster_to_polygons(array, transform, polys.crs, polys, id_col="id")
    assert set(result["id"].to_list()) == {"a", "b", "c", "d"}
    d = dict(zip(result["id"].to_list(), result["value"].to_list()))
    assert d == {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}


def test_exact_mass_conservation_random_grid():
    """sum(pixel values) == sum(output polygon values), exactly, on an irregular random-offset case."""
    rng = np.random.default_rng(0)
    n = 20
    array = rng.random((n, n)) * 100
    # Pixel grid offset by an irregular fraction relative to a coarser polygon grid,
    # forcing most pixels to straddle boundaries.
    transform = rio.Affine(1.0, 0, 0.37, 0, -1.0, n - 0.21)
    cell = 3.0
    polys_geo = []
    ids = []
    for i in range(-1, n // int(cell) + 2):
        for j in range(-1, n // int(cell) + 2):
            polys_geo.append(box(i * cell, j * cell, (i + 1) * cell, (j + 1) * cell))
            ids.append(f"{i}_{j}")
    polys = gpd.GeoDataFrame({"id": ids}, geometry=polys_geo, crs="EPSG:32633")

    result = raster_to_polygons(array, transform, polys.crs, polys, id_col="id")
    assert result["value"].sum() == pytest.approx(array.sum(), rel=1e-9)
    assert result["id"].n_unique() == result.height  # no duplicate polygon rows


def test_density_bound_property():
    """Every output polygon's density lies within the min/max density of the pixels it touches."""
    rng = np.random.default_rng(1)
    n = 15
    array = rng.random((n, n)) * 50 + 1  # keep positive
    transform = rio.Affine(1.0, 0, 0.6, 0, -1.0, n - 0.4)
    pixel_area = 1.0
    cell = 4.0
    polys_geo, ids = [], []
    for i in range(-1, n // int(cell) + 2):
        for j in range(-1, n // int(cell) + 2):
            polys_geo.append(box(i * cell, j * cell, (i + 1) * cell, (j + 1) * cell))
            ids.append(f"{i}_{j}")
    polys = gpd.GeoDataFrame({"id": ids}, geometry=polys_geo, crs="EPSG:32633")
    poly_area = cell * cell

    result = raster_to_polygons(array, transform, polys.crs, polys, id_col="id")

    # For each output polygon, find pixels whose bbox intersects it and check density bound.
    height, width = array.shape
    raster_box = box(
        transform.c,
        transform.f + height * transform.e,
        transform.c + width * transform.a,
        transform.f,
    )
    for pid, val in zip(result["id"].to_list(), result["value"].to_list()):
        i, j = map(int, pid.split("_"))
        poly_geom = box(i * cell, j * cell, (i + 1) * cell, (j + 1) * cell)
        if not raster_box.contains(poly_geom):
            # Skip polygons only partially covered by the raster: the
            # uncovered remainder contributes 0 to the area-weighted average,
            # which can legitimately pull the cell's density outside the
            # range of its *touching pixels'* densities -- that's a raster
            # coverage-extent edge effect, not a violation of the density
            # bound property (which is about pixels a polygon actually
            # touches, not padding area with no pixel at all).
            continue
        touching_vals = []
        for r in range(height):
            for c in range(width):
                x0 = transform.c + c * transform.a
                x1 = x0 + transform.a
                y0 = transform.f + r * transform.e
                y1 = y0 + transform.e
                pb = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                if pb.intersects(poly_geom) and pb.intersection(poly_geom).area > 1e-12:
                    touching_vals.append(array[r, c])
        if not touching_vals:
            continue
        densities = [v / pixel_area for v in touching_vals]
        cell_density = val / poly_area
        assert min(densities) - 1e-9 <= cell_density <= max(densities) + 1e-9


def test_raster_to_h3_wrapper_conserves_mass():
    """The H3 convenience wrapper also conserves total mass over the covered extent."""
    n = 30
    rng = np.random.default_rng(2)
    array = rng.random((n, n)) * 10
    # A small raster in a projected CRS near the equator/origin so degrees-scale
    # coordinates aren't absurd; h3_cells reprojects to 4326 internally.
    transform = rio.Affine(0.0005, 0, 10.0, 0, -0.0005, 45.0)
    result = raster_to_h3(array, transform, "EPSG:4326", resolution=9)
    assert result.height > 0
    assert result["value"].sum() == pytest.approx(array.sum(), rel=1e-6)


def _write_test_tif(path, array, transform, crs="EPSG:4326", nodata=None):
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def test_raster_to_h3_tiled_matches_single_shot(tmp_path):
    """The tiled/chunked resampler (memory-bounded, per-tile subprocess or serial)
    gives the same result as the single-shot `raster_to_h3`, mass-conservation
    included -- see `raster_to_h3_tiled`'s docstring for why splitting work by H3
    parent/child hierarchy is exact (no double count, no gap) across tile
    boundaries."""
    n = 30
    rng = np.random.default_rng(3)
    array = (rng.random((n, n)) * 10).astype("float64")
    transform = rio.Affine(0.0005, 0, 10.0, 0, -0.0005, 45.0)
    tif_path = str(tmp_path / "test.tif")
    _write_test_tif(tif_path, array, transform)

    single = raster_to_h3(array, transform, "EPSG:4326", resolution=9)
    tiled_serial = raster_to_h3_tiled(
        tif_path, resolution=9, tile_resolution=8, max_workers=1
    )

    assert tiled_serial.height > 0
    assert tiled_serial["value"].sum() == pytest.approx(array.sum(), rel=1e-6)

    single_by_id = dict(zip(single["h3_cell"].to_list(), single["value"].to_list()))
    tiled_by_id = dict(
        zip(tiled_serial["h3_cell"].to_list(), tiled_serial["value"].to_list())
    )
    assert single_by_id.keys() == tiled_by_id.keys()
    for k in single_by_id:
        assert single_by_id[k] == pytest.approx(tiled_by_id[k], abs=1e-6)


def test_raster_to_h3_tiled_parallel_matches_serial(tmp_path):
    """Real multi-process execution (max_workers=2) gives the same result as serial."""
    n = 30
    rng = np.random.default_rng(4)
    array = (rng.random((n, n)) * 10).astype("float64")
    transform = rio.Affine(0.0005, 0, 10.0, 0, -0.0005, 45.0)
    tif_path = str(tmp_path / "test.tif")
    _write_test_tif(tif_path, array, transform)

    serial = raster_to_h3_tiled(
        tif_path, resolution=9, tile_resolution=8, max_workers=1
    )
    parallel = raster_to_h3_tiled(
        tif_path, resolution=9, tile_resolution=8, max_workers=2
    )

    assert parallel["value"].sum() == pytest.approx(serial["value"].sum(), rel=1e-9)
    assert set(parallel["h3_cell"].to_list()) == set(serial["h3_cell"].to_list())
