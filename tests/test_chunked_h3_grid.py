"""Tests for geohierarchy.chunked_h3_grid: per-tile GeoParquet output, never one big in-memory grid."""

import h3
import polars as pl
import pytest

from geohierarchy.chunked_h3_grid import (
    build_h3_grid_chunked,
    partition_h3_table,
    read_h3_grid_chunked_columns,
)


def _synthetic_grid(
    n_tiles: int = 3,
    cells_per_tile: int = 20,
    resolution: int = 9,
    tile_resolution: int = 5,
):
    """A small multi-tile fixture: real H3 cells spanning `n_tiles` distinct tile-resolution ancestors.

    Starts from a handful of real world points far enough apart that their
    resolution-`tile_resolution` ancestors are guaranteed distinct, then
    walks each one's local neighborhood at `resolution` via `grid_disk` to
    get real, valid child cells (not synthetic/fake ids) -- matching how the
    real pipeline's `h3_cell` column is always populated.
    """
    seed_latlngs = [
        (40.0, -75.0),
        (51.5, -0.1),
        (35.7, 139.7),
        (-33.9, 18.4),
        (1.3, 103.8),
    ][:n_tiles]
    tiles: dict[str, list[str]] = {}
    for lat, lng in seed_latlngs:
        center = h3.latlng_to_cell(lat, lng, resolution)
        tile_id = h3.cell_to_parent(center, tile_resolution)
        ring = h3.grid_disk(center, 2)
        # Keep only cells that truly belong to this seed's own tile (a couple
        # of ring cells near the edge could in principle fall under a
        # different tile-resolution ancestor -- excluding them keeps this
        # fixture's expected tile count exact).
        own = [c for c in ring if h3.cell_to_parent(c, tile_resolution) == tile_id][
            :cells_per_tile
        ]
        tiles[tile_id] = own
    return tiles


def _grid_table(tiles: dict) -> pl.DataFrame:
    rows = []
    for tile_id, cells in tiles.items():
        for i, c in enumerate(cells):
            rows.append(
                {"h3_cell": c, "population": float(i + 1), "level_of_service": 0.5}
            )
    return pl.DataFrame(rows)


def test_partition_is_exact_no_duplicates_no_drops():
    """Every input row appears in exactly one tile's partition, under the correct tile key."""
    tiles = _synthetic_grid()
    df = _grid_table(tiles)
    parts = partition_h3_table(df, tile_resolution=5)

    assert set(parts.keys()) == set(tiles.keys())
    total_in = df.height
    total_out = sum(p.height for p in parts.values())
    assert total_out == total_in  # no drops, no duplicates

    for tile_id, cells in tiles.items():
        got_cells = set(parts[tile_id]["h3_cell"].to_list())
        assert got_cells == set(cells)
        # Every cell in this partition really is a child of this tile id.
        for c in got_cells:
            assert h3.cell_to_parent(c, 5) == tile_id


def test_build_h3_grid_chunked_writes_one_file_per_tile(tmp_path):
    tiles = _synthetic_grid()
    df = _grid_table(tiles)
    out_dir = tmp_path / "chunks"

    # max_workers=1: serial, in-process -- avoids spawn-context ProcessPoolExecutor
    # overhead/sandboxing issues in a test environment, exercises the same
    # `_tile_worker` function a real pooled run would call.
    paths = build_h3_grid_chunked(df, str(out_dir), tile_resolution=5, max_workers=1)

    assert len(paths) == len(tiles)
    for p in paths:
        assert p.startswith(str(out_dir))

    import geopandas as gpd

    all_cells = set()
    for p in paths:
        gdf = gpd.read_parquet(p)
        assert "geometry" in gdf.columns
        assert gdf.geometry.notna().all()
        assert (gdf.geometry.geom_type == "Polygon").all()
        all_cells.update(gdf["h3_cell"].tolist())

    assert all_cells == set(df["h3_cell"].to_list())


def test_build_h3_grid_chunked_matches_unchunked_h3_ops_to_gdf(tmp_path):
    """Per-tile geometry/attributes are identical to building the whole grid in one shot."""
    from UrbanAccessAnalyzer import h3_ops

    tiles = _synthetic_grid()
    df = _grid_table(tiles)

    reference = h3_ops.to_gdf(df, h3_column="h3_cell").set_index("h3_cell").sort_index()

    paths = build_h3_grid_chunked(
        df, str(tmp_path / "chunks"), tile_resolution=5, max_workers=1
    )
    import geopandas as gpd
    import pandas as pd

    chunked = (
        pd.concat([gpd.read_parquet(p) for p in paths])
        .set_index("h3_cell")
        .sort_index()
    )

    assert list(chunked.index) == list(reference.index)
    assert (
        chunked["population"].to_numpy() == reference["population"].to_numpy()
    ).all()
    for cell in reference.index:
        assert chunked.loc[cell, "geometry"].equals_exact(
            reference.loc[cell, "geometry"], tolerance=1e-9
        )


def test_empty_input_returns_no_files(tmp_path):
    df = pl.DataFrame(
        {"h3_cell": [], "population": []},
        schema={"h3_cell": pl.Utf8, "population": pl.Float64},
    )
    paths = build_h3_grid_chunked(
        df, str(tmp_path / "chunks"), tile_resolution=5, max_workers=1
    )
    assert paths == []


def test_read_h3_grid_chunked_columns_rejects_geometry():
    with pytest.raises(ValueError):
        read_h3_grid_chunked_columns(["a.parquet"], ["h3_cell", "geometry"])


def test_read_h3_grid_chunked_columns_matches_eager_full_merge(tmp_path):
    """The polars-lazy multi-file column scan produces identical results to an eager full concat+select."""
    tiles = _synthetic_grid(n_tiles=3, cells_per_tile=15)
    df = _grid_table(tiles)
    paths = build_h3_grid_chunked(
        df, str(tmp_path / "chunks"), tile_resolution=5, max_workers=1
    )

    lazy_result = (
        read_h3_grid_chunked_columns(
            paths, ["h3_cell", "population", "level_of_service"]
        )
        .collect()
        .sort("h3_cell")
    )

    import geopandas as gpd
    import pandas as pd

    eager_merge = pd.concat(
        [
            gpd.read_parquet(p)[["h3_cell", "population", "level_of_service"]]
            for p in paths
        ]
    )
    eager_merge = pl.from_pandas(eager_merge).sort("h3_cell")

    assert lazy_result.equals(eager_merge)

    # A citywide aggregate (population-weighted mean access, the same shape
    # of computation the Distribution/Regression stats panel needs) computed
    # from the lazy multi-file scan must match the same aggregate computed
    # by eagerly merging every tile's full frame first.
    lazy_mean = (
        read_h3_grid_chunked_columns(paths, ["population", "level_of_service"])
        .select(
            (pl.col("population") * pl.col("level_of_service")).sum()
            / pl.col("population").sum()
        )
        .collect()
        .item()
    )
    eager_mean = (
        eager_merge["population"] * eager_merge["level_of_service"]
    ).sum() / eager_merge["population"].sum()
    assert lazy_mean == pytest.approx(eager_mean)
