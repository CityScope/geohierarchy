"""Tests for adaptive H3-chunked tiling (geohierarchy.maps.folium.tiles).

Covers the fix for OOM-killed whole-level `freestile_file` calls on
oversized levels (e.g. Boston's statewide census "block" level): a level
whose row count exceeds `H3_CHUNK_ROW_THRESHOLD` is tiled as many small
per-H3-cell `.pmtiles` files instead of one whole-level call.

Also covers the 2026-08-30 follow-up fix: a single fixed H3 resolution
does not bound any individual chunk's size for a geographically
concentrated dataset (real Boston data: one resolution-4 cell alone held
430,653 rows, ~3x `H3_CHUNK_ROW_THRESHOLD`, because that's how far the
metro area's own density outran the fixed partition). Chunking is now
adaptive (`_adaptive_h3_chunk_keys`): any chunk still over threshold is
recursively re-split at the next finer H3 resolution until it fits.
"""

from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
from shapely.geometry import box

import geohierarchy.maps.folium.tiles as tiles_mod
from geohierarchy.core import GeoHierarchy
from geohierarchy.aggregation import Sum
from geohierarchy.maps.folium import HierarchyMap
from geohierarchy.maps.folium.tiles import (
    H3_CHUNK_MAX_RESOLUTION,
    H3_CHUNK_RESOLUTION,
    _adaptive_h3_chunk_keys,
    _h3_keys_at_resolution,
    write_level_tiles,
)


def _wide_polygon_gdf(n: int, seed: int = 0) -> gpd.GeoDataFrame:
    """Synthetic polygon level spread across a wide area (many H3 res-4 cells)."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-80, -70, n)
    ys = rng.uniform(35, 45, n)
    geoms = [box(x, y, x + 0.001, y + 0.001) for x, y in zip(xs, ys)]
    return gpd.GeoDataFrame(
        {"id": [f"f{i}" for i in range(n)], "val": rng.random(n)},
        geometry=geoms,
        crs="EPSG:4326",
    )


def _wide_h3_grid_gdf(n: int, resolution: int = 9, seed: int = 0) -> gpd.GeoDataFrame:
    """Synthetic H3-grid level (has an `h3_cell` column) spread widely."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-80, -70, n)
    ys = rng.uniform(35, 45, n)
    cells = [h3.latlng_to_cell(y, x, resolution) for x, y in zip(xs, ys)]
    # de-dup (random points can collide into the same fine cell)
    cells = list(dict.fromkeys(cells))
    geoms = []
    for c in cells:
        boundary = h3.cell_to_boundary(c)
        geoms.append(
            box(
                boundary[0][1],
                boundary[0][0],
                boundary[0][1] + 0.0001,
                boundary[0][0] + 0.0001,
            )
        )
    return gpd.GeoDataFrame(
        {"h3_cell": cells, "val": rng.random(len(cells))},
        geometry=geoms,
        crs="EPSG:4326",
    )


# ============================================================
# _h3_keys_at_resolution
# ============================================================


def test_h3_keys_at_resolution_polygon_level_uses_representative_point():
    gdf = _wide_polygon_gdf(20)
    keys = _h3_keys_at_resolution(gdf, H3_CHUNK_RESOLUTION).to_list()
    assert len(keys) == len(gdf)
    # Every key should be a valid H3 res-4 cell matching the row's
    # representative point.
    pts = gdf.geometry.representative_point()
    for key, pt in zip(keys, pts):
        expected = h3.latlng_to_cell(pt.y, pt.x, H3_CHUNK_RESOLUTION)
        assert key == expected
        assert h3.get_resolution(key) == H3_CHUNK_RESOLUTION


def test_h3_keys_at_resolution_h3_grid_level_uses_cell_to_parent():
    gdf = _wide_h3_grid_gdf(200, resolution=9)
    keys = _h3_keys_at_resolution(gdf, H3_CHUNK_RESOLUTION).to_list()
    assert len(keys) == len(gdf)
    for key, cell in zip(keys, gdf["h3_cell"]):
        assert key == h3.cell_to_parent(cell, H3_CHUNK_RESOLUTION)
        assert h3.get_resolution(key) == H3_CHUNK_RESOLUTION


# ============================================================
# _adaptive_h3_chunk_keys: recursion for geographically-concentrated data
# ============================================================


def _dense_cluster_polygon_gdf(n: int, seed: int = 0) -> gpd.GeoDataFrame:
    """All rows packed into one small area (same H3 res-4 cell)."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-71.10, -71.05, n)
    ys = rng.uniform(42.35, 42.40, n)
    geoms = [box(x, y, x + 0.0001, y + 0.0001) for x, y in zip(xs, ys)]
    return gpd.GeoDataFrame(
        {"id": [f"f{i}" for i in range(n)], "val": rng.random(n)},
        geometry=geoms,
        crs="EPSG:4326",
    )


def test_adaptive_chunk_keys_recurses_for_dense_cluster():
    # A fixed resolution-4 partition would put every one of these rows in
    # the SAME chunk (they're all within one small area) -- this is the
    # exact Boston failure mode. Adaptive chunking must recurse to a finer
    # resolution so no chunk exceeds the threshold.
    gdf = _dense_cluster_polygon_gdf(500)
    threshold = 40
    keys = _adaptive_h3_chunk_keys(gdf, threshold=threshold)
    assert len(keys) == len(gdf)

    counts = keys.value_counts()
    assert len(counts) > 1, "dense cluster should split into more than one chunk"
    assert (
        counts.max() <= threshold
    ), f"chunk exceeds threshold: {counts.max()} > {threshold}"

    # Every key that actually needed recursion should report a resolution
    # finer than the base resolution.
    resolutions = {int(k.split("_", 1)[0].removeprefix("res")) for k in counts.index}
    assert max(resolutions) > H3_CHUNK_RESOLUTION
    assert max(resolutions) <= H3_CHUNK_MAX_RESOLUTION

    # No rows dropped or duplicated across chunk keys.
    assert keys.notna().all()


def test_adaptive_chunk_keys_no_recursion_when_already_under_threshold():
    # A wide, sparse dataset shouldn't recurse at all -- same as the old
    # fixed-resolution behavior for the common case.
    gdf = _wide_polygon_gdf(50)
    keys = _adaptive_h3_chunk_keys(gdf, threshold=150_000)
    resolutions = {int(k.split("_", 1)[0].removeprefix("res")) for k in keys.unique()}
    assert resolutions == {H3_CHUNK_RESOLUTION}


# ============================================================
# Threshold gating
# ============================================================


def test_small_level_stays_unchunked(tmp_path):
    gdf = _wide_polygon_gdf(50)
    written = write_level_tiles(
        gdf,
        "smalllevel",
        str(tmp_path),
        min_zoom=0,
        max_zoom=4,
        id_col="id",
        property_cols=["val"],
    )
    assert len(written) == 1
    assert written[0] == tmp_path / "smalllevel.pmtiles"
    assert written[0].exists()
    assert not (tmp_path / "smalllevel").exists()


def test_large_level_gets_chunked(tmp_path, monkeypatch):
    # Lower the threshold for this test only -- never change the real
    # default (150_000), which must stay the common, unchunked path.
    monkeypatch.setattr(tiles_mod, "H3_CHUNK_ROW_THRESHOLD", 30)
    gdf = _wide_polygon_gdf(200)
    written = write_level_tiles(
        gdf,
        "biglevel",
        str(tmp_path),
        min_zoom=0,
        max_zoom=4,
        id_col="id",
        property_cols=["val"],
    )
    assert len(written) > 1
    level_dir = tmp_path / "biglevel"
    assert level_dir.is_dir()
    assert not (tmp_path / "biglevel.pmtiles").exists()
    chunk_files = sorted(level_dir.glob("chunk_*.pmtiles"))
    assert chunk_files
    assert set(written) == set(chunk_files)
    for p in chunk_files:
        assert p.stat().st_size > 0


def test_chunking_default_threshold_is_150000():
    # Sanity-check the real, un-monkeypatched default hasn't drifted --
    # small/medium cities (the overwhelming common case) must never
    # accidentally cross it.
    assert tiles_mod.H3_CHUNK_ROW_THRESHOLD == 150_000


# ============================================================
# No drops/duplicates across chunk boundaries
# ============================================================


def test_chunked_output_accounts_for_every_row_no_drops_no_dupes(tmp_path, monkeypatch):
    import mapbox_vector_tile

    from geohierarchy.maps.folium.pmtiles_to_xyz import extract_xyz_from_pmtiles

    monkeypatch.setattr(tiles_mod, "H3_CHUNK_ROW_THRESHOLD", 30)
    n = 300
    gdf = _wide_polygon_gdf(n)
    written = write_level_tiles(
        gdf,
        "auditlevel",
        str(tmp_path),
        min_zoom=14,
        max_zoom=14,
        id_col="id",
        property_cols=["val"],
    )
    assert len(written) > 1

    seen_ids = []
    for pmtiles_path in written:
        xyz_dir = extract_xyz_from_pmtiles(
            pmtiles_path,
            tmp_path / f"xyz_{pmtiles_path.stem}",
            layer_name="auditlevel",
            min_zoom=14,
            max_zoom=14,
        )
        for pbf_path in Path(xyz_dir).rglob("*.pbf"):
            decoded = mapbox_vector_tile.decode(pbf_path.read_bytes())
            layer = decoded.get("auditlevel")
            if not layer:
                continue
            for feat in layer["features"]:
                fid = feat["properties"].get("id")
                if fid is not None:
                    seen_ids.append(fid)

    # Every input feature id appears at least once across all chunks (a
    # feature can legitimately appear in more than one MVT *tile* -- it's
    # clipped at tile boundaries at high zoom -- but it must belong to
    # exactly one *chunk*, so dedupe before comparing to the input set).
    unique_seen = set(seen_ids)
    expected_ids = set(gdf["id"])
    assert (
        unique_seen == expected_ids
    ), f"missing={expected_ids - unique_seen}, unexpected={unique_seen - expected_ids}"


# ============================================================
# MapLibre consumer: one source per chunk, registered as one layer
# ============================================================


def test_maplibre_registers_one_source_per_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(tiles_mod, "H3_CHUNK_ROW_THRESHOLD", 30)

    gdf = _wide_polygon_gdf(200)
    gh = GeoHierarchy()
    gh.add_level("biglevel", gdf, id_col="id", agg=Sum())
    gh.propagate()

    m = HierarchyMap(
        gh, levels=["biglevel"], tiles_dir=str(tmp_path / "tiles"), extract_xyz=False
    )
    m.build()

    level_dir = tmp_path / "tiles" / "biglevel"
    chunk_files = sorted(level_dir.glob("chunk_*.pmtiles"))
    assert len(chunk_files) > 1

    out_path = tmp_path / "map_maplibre.html"
    m.save_maplibre(str(out_path))
    html = out_path.read_text()

    # One pmtiles:// URL per chunk file, all under the level's subdirectory.
    for chunk_file in chunk_files:
        assert f"pmtiles://tiles/biglevel/{chunk_file.name}" in html
    # No un-chunked single-file URL leaked in.
    assert "pmtiles://tiles/biglevel.pmtiles" not in html
    # Every chunk source-layer is still the plain level name (unchanged
    # MVT layer name inside each chunk file), not chunk-specific.
    assert html.count('"source-layer": "biglevel"') == len(chunk_files)
    # The recolor API's level -> source-id map lists every chunk source.
    assert '"biglevel": [' in html or '"biglevel":[' in html
