"""Tests for geohierarchy.maps.folium: resolution assignment, tiling, and HTML rendering."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from geohierarchy.core import GeoHierarchy
from geohierarchy.aggregation import Sum
from geohierarchy.maps.folium import HierarchyMap, MultiHierarchyMap, ColorSpec
from geohierarchy.maps.folium.resolution import (
    assign_resolutions,
    validate_resolutions,
    MIN_ZOOM,
    MAX_ZOOM,
)
from geohierarchy.maps.folium.tiles import write_level_tiles

# ============================================================
# FIXTURES (mirrors tests.py conventions)
# ============================================================


# Small (~0.002 degree, roughly city-block scale) synthetic AOI, so that
# even a level's tiles get generated all the way to MAX_NATIVE_TILE_ZOOM
# (18) without producing an intractable number of files -- a level
# spanning a whole state/country would only ever be tiled across its own
# (typically low) auto-assigned zoom sub-range in real use.
S = 0.002


@pytest.fixture
def region_gdf():
    return gpd.GeoDataFrame(
        {"reg_id": ["R1"], "geometry": [box(0, 0, S, S)], "pop": [1000]},
        crs="EPSG:4326",
    )


@pytest.fixture
def city_gdf():
    return gpd.GeoDataFrame(
        {
            "city_id": ["A", "B"],
            "geometry": [box(0, 0, S / 2, S), box(S / 2, 0, S, S)],
            "pop": [400, 600],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def hood_gdf():
    return gpd.GeoDataFrame(
        {
            "hood_id": ["n1", "n2", "n3", "n4"],
            "geometry": [
                box(0, 0, S / 2, S / 2),
                box(0, S / 2, S / 2, S),
                box(S / 2, 0, S, S / 2),
                box(S / 2, S / 2, S, S),
            ],
            "pop": [100, 150, 200, 250],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def hierarchy2(region_gdf, city_gdf):
    """A 2-level hierarchy: region (coarse) -> city (fine)."""
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id", agg=Sum())
    gh.add_level("city", city_gdf, id_col="city_id", agg=Sum(), parent="region")
    gh.propagate()
    return gh


@pytest.fixture
def hierarchy3(region_gdf, city_gdf, hood_gdf):
    """A 3-level hierarchy: region -> city -> hood."""
    gh = GeoHierarchy()
    gh.add_level("region", region_gdf, id_col="reg_id", agg=Sum())
    gh.add_level("city", city_gdf, id_col="city_id", agg=Sum(), parent="region")
    gh.add_level("hood", hood_gdf, id_col="hood_id", agg=Sum(), parent="city")
    gh.propagate()
    return gh


# ============================================================
# RESOLUTION ASSIGNMENT
# ============================================================


def test_assign_resolutions_covers_full_range_2_levels(hierarchy2):
    ranges = assign_resolutions(hierarchy2)
    assert set(ranges) == {"region", "city"}
    validate_resolutions(ranges)  # no gaps/overlaps, full [0,25]


def test_assign_resolutions_covers_full_range_3_levels(hierarchy3):
    ranges = assign_resolutions(hierarchy3)
    assert set(ranges) == {"region", "city", "hood"}
    validate_resolutions(ranges)


def test_assign_resolutions_orders_coarsest_first(hierarchy3):
    ranges = assign_resolutions(hierarchy3)
    # region is the biggest polygon, so it should own the lowest zooms;
    # hood is the smallest, so it should own the highest zooms.
    assert ranges["region"][0] == MIN_ZOOM
    assert ranges["hood"][1] == MAX_ZOOM
    assert ranges["region"][1] < ranges["hood"][0]


def test_assign_resolutions_subset_of_levels(hierarchy3):
    ranges = assign_resolutions(hierarchy3, levels=["region", "city"])
    assert set(ranges) == {"region", "city"}
    validate_resolutions(ranges)


def test_assign_resolutions_single_level(hierarchy2):
    ranges = assign_resolutions(hierarchy2, levels=["region"])
    assert ranges == {"region": (MIN_ZOOM, MAX_ZOOM)}


def test_validate_resolutions_raises_on_gap():
    with pytest.raises(ValueError, match="gap"):
        validate_resolutions({"a": (0, 10), "b": (12, 25)})


def test_validate_resolutions_raises_on_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_resolutions({"a": (0, 12), "b": (10, 25)})


def test_validate_resolutions_raises_on_missing_start():
    with pytest.raises(ValueError, match="gap"):
        validate_resolutions({"a": (1, 25)})


def test_validate_resolutions_raises_on_missing_end():
    with pytest.raises(ValueError, match="gap"):
        validate_resolutions({"a": (0, 24)})


def test_validate_resolutions_accepts_contiguous_partition():
    validate_resolutions({"a": (0, 10), "b": (11, 25)})  # should not raise


# ============================================================
# HierarchyMap.set_resolution (manual override validation)
# ============================================================


def test_set_resolution_manual_override_used(hierarchy3):
    m = HierarchyMap(hierarchy3, levels=["region", "city", "hood"])
    m.set_resolution("region", 0, 5)
    ranges = m.resolve_zoom_ranges()
    assert ranges["region"] == (0, 5)
    validate_resolutions(ranges)


def test_set_resolution_all_manual_gap_raises(hierarchy2):
    m = HierarchyMap(hierarchy2, levels=["region", "city"])
    m.set_resolution("region", 0, 10)
    m.set_resolution("city", 12, 25)
    with pytest.raises(ValueError, match="gap"):
        m.resolve_zoom_ranges()


def test_set_resolution_all_manual_overlap_raises(hierarchy2):
    m = HierarchyMap(hierarchy2, levels=["region", "city"])
    m.set_resolution("region", 0, 15)
    m.set_resolution("city", 10, 25)
    with pytest.raises(ValueError, match="overlap"):
        m.resolve_zoom_ranges()


# ============================================================
# TILING
# ============================================================


def test_write_level_tiles_produces_pmtiles_file(hierarchy2, tmp_path):
    # PMTiles (via freestiler) is now the primary output format -- a single
    # .pmtiles archive per level, replacing the old per-z/x/y .pbf tree.
    # See fast_tiles.py / tiles.py write_level_tiles(use_xyz=False, default).
    gdf = hierarchy2.get_level("region")
    written = write_level_tiles(
        gdf,
        "region",
        str(tmp_path),
        min_zoom=0,
        max_zoom=2,
        id_col=hierarchy2.id_cols["region"],
        property_cols=["pop"],
    )
    assert len(written) > 0
    for p in written:
        assert p.exists()
        assert p.suffix == ".pmtiles"
        assert p.stat().st_size > 0


def test_write_level_tiles_readable_mvt(hierarchy2, tmp_path):
    import mapbox_vector_tile
    from geohierarchy.maps.folium.pmtiles_to_xyz import extract_xyz_from_pmtiles

    gdf = hierarchy2.get_level("region")
    written = write_level_tiles(
        gdf,
        "region",
        str(tmp_path),
        min_zoom=15,
        max_zoom=16,
        id_col=hierarchy2.id_cols["region"],
        property_cols=["pop"],
    )
    assert written
    pmtiles_path = written[0]
    assert pmtiles_path.suffix == ".pmtiles"

    # Extract one XYZ tile back out of the PMTiles archive and confirm the
    # bytes are genuinely decodable MVT -- this is what a static host serving
    # extracted XYZ (or a pmtiles-protocol client unpacking a tile) needs.
    xyz_dir = extract_xyz_from_pmtiles(
        pmtiles_path, tmp_path / "xyz", min_zoom=15, max_zoom=16
    )
    pbf_files = list(Path(xyz_dir).rglob("*.pbf"))
    assert pbf_files
    raw = pbf_files[0].read_bytes()
    decoded = mapbox_vector_tile.decode(raw)
    assert len(decoded) > 0
    first_layer = next(iter(decoded.values()))
    assert len(first_layer["features"]) > 0


# ============================================================
# HierarchyMap.save()
# ============================================================


def test_hierarchy_map_save_produces_html(hierarchy2, tmp_path):
    m = HierarchyMap(
        hierarchy2, levels=["region", "city"], tiles_dir=str(tmp_path / "tiles")
    )
    m.configure_level(
        "region", style=ColorSpec(column="pop", cmap="viridis"), popup_fields=["pop"]
    )
    m.build()
    out_path = tmp_path / "map.html"
    m.save(str(out_path))

    assert out_path.exists()
    html = out_path.read_text()

    # Leaflet.VectorGrid plugin pulled from CDN.
    assert "leaflet.vectorgrid" in html.lower()
    # One L.vectorGrid.protobuf(...) call per rendered level.
    assert html.count("L.vectorGrid.protobuf(") == 2
    # Basemap tile URL present (default cartodb_positron).
    assert "basemaps.cartocdn.com" in html
    # Tile URL template for at least one level.
    assert "tiles/region/" in html or "tiles\\/region\\/" in html

    # Regression: opening the saved HTML as a bare file:// path silently
    # fails to load any geometry (basemap still works, since that's an
    # ordinary cross-origin https image request; local vector-tile
    # fetch() calls are blocked under file://) -- the page must detect
    # this and warn instead of just showing an empty map.
    assert "file-protocol-warning" in html
    assert 'window.location.protocol === "file:"' in html

    # Regression: the vectorGrid-layer init code must be deferred to
    # `window.load` and must NOT be nested inside branca's own
    # <script>...</script> wrapper (a literal <script> tag nested inside
    # that block breaks HTML parsing and silently kills the plugin load;
    # calling this code before the map variable/plugin exist throws a
    # ReferenceError). Both bugs previously produced a blank map with no
    # tile requests ever made.
    assert "window.addEventListener('load'" in html
    # The number of opening/closing <script> tags must match -- a nested
    # <script> tag (as opposed to a sibling one) breaks HTML parsing
    # without necessarily changing the count, so also check every
    # <script ...> up to the VectorGrid tag has already been closed.
    assert html.count("<script") == html.count("</script>")
    vg_idx = html.index('<script src="https://unpkg.com/leaflet.vectorgrid')
    preceding = html[:vg_idx]
    assert preceding.count("<script") == preceding.count("</script>")


def test_hierarchy_map_save_custom_basemap(hierarchy2, tmp_path):
    m = HierarchyMap(
        hierarchy2,
        levels=["region", "city"],
        tiles_dir=str(tmp_path / "tiles"),
        basemap="osm",
    )
    m.build()
    out_path = tmp_path / "map.html"
    m.save(str(out_path))
    html = out_path.read_text()
    assert "tile.openstreetmap.org" in html


def test_basemap_not_registered_in_layer_control(hierarchy2, tmp_path):
    """Regression: the basemap TileLayer must not join the same non-overlay
    (radio) layer-control group as MultiHierarchyMap's named groups -- if it
    does, selecting "H3"/"Polygons" silently swaps out the basemap too,
    since Leaflet treats all non-overlay layers as mutually exclusive.
    """
    from geohierarchy.maps.folium.basemap import build_tile_layer

    layer = build_tile_layer("cartodb_positron")
    assert layer.control is False


def test_hierarchy_map_build_is_idempotent(hierarchy2, tmp_path):
    m = HierarchyMap(
        hierarchy2, levels=["region", "city"], tiles_dir=str(tmp_path / "tiles")
    )
    m.build()
    first_count = len(list((tmp_path / "tiles").rglob("*.pbf")))
    m.build()
    second_count = len(list((tmp_path / "tiles").rglob("*.pbf")))
    assert first_count == second_count
    assert first_count > 0


# ============================================================
# MultiHierarchyMap
# ============================================================


def test_multi_hierarchy_map_emits_non_overlay_groups(hierarchy2, hierarchy3, tmp_path):
    m_a = HierarchyMap(hierarchy2, levels=["region", "city"])
    m_b = HierarchyMap(hierarchy3, levels=["hood"])

    mm = MultiHierarchyMap(
        {"Coarse": m_a, "Fine": m_b},
        default="Coarse",
        tiles_dir=str(tmp_path / "tiles"),
    )
    mm.build()
    out_path = tmp_path / "multi.html"
    mm.save(str(out_path))

    assert out_path.exists()
    html = out_path.read_text()

    # Both group names appear (as FeatureGroup layer-control entries).
    assert "Coarse" in html
    assert "Fine" in html
    # Non-overlay (radio-button) semantics: folium/Leaflet render overlay=False
    # FeatureGroups as base_layers in L.control.layers, which the browser
    # renders as mutually-exclusive radio buttons (vs. overlays' checkboxes).
    base_layers_block = html[html.find("base_layers") : html.find("overlays")]
    assert '"Coarse"' in base_layers_block
    assert '"Fine"' in base_layers_block
    # Both groups' vector layers are present.
    assert html.count("L.vectorGrid.protobuf(") == 3  # region, city, hood


# ============================================================
# MapLibre (PMTiles-native, no XYZ/pbf extraction)
# ============================================================


def test_hierarchy_map_save_maplibre_produces_html(hierarchy2, tmp_path):
    m = HierarchyMap(
        hierarchy2,
        levels=["region", "city"],
        tiles_dir=str(tmp_path / "tiles"),
        extract_xyz=False,
    )
    m.configure_level(
        "region", style=ColorSpec(column="pop", cmap="viridis"), popup_fields=["pop"]
    )
    m.build()
    out_path = tmp_path / "map_maplibre.html"
    m.save_maplibre(str(out_path))

    assert out_path.exists()
    html = out_path.read_text()

    # PMTiles protocol registered, no L.vectorGrid.protobuf/.pbf XYZ usage.
    assert "maplibregl" in html
    assert "pmtiles.Protocol" in html
    assert "L.vectorGrid.protobuf(" not in html
    assert "pmtiles://tiles/region.pmtiles" in html
    assert "pmtiles://tiles/city.pmtiles" in html
    # Only .pmtiles files were built (no per-tile XYZ .pbf tree).
    tiles_dir = tmp_path / "tiles"
    assert any(tiles_dir.glob("*.pmtiles"))
    assert not list(tiles_dir.rglob("*.pbf"))


def test_hierarchy_map_default_still_extracts_xyz_for_folium(hierarchy2, tmp_path):
    """Regression: `extract_xyz` defaults True so `save()` (Leaflet.VectorGrid,
    which needs the XYZ .pbf tree) keeps working unmodified for existing
    callers -- only opt-in `extract_xyz=False` (used by MapLibre-only
    callers) skips the extraction."""
    m = HierarchyMap(hierarchy2, levels=["region"], tiles_dir=str(tmp_path / "tiles"))
    m.build()
    assert list((tmp_path / "tiles").rglob("*.pbf"))


def test_multi_hierarchy_map_save_maplibre_produces_html(
    hierarchy2, hierarchy3, tmp_path
):
    m_a = HierarchyMap(hierarchy2, levels=["region", "city"])
    m_b = HierarchyMap(hierarchy3, levels=["hood"])

    mm = MultiHierarchyMap(
        {"Coarse": m_a, "Fine": m_b},
        default="Coarse",
        tiles_dir=str(tmp_path / "tiles"),
    )
    mm.build()
    out_path = tmp_path / "multi_maplibre.html"
    mm.save_maplibre(str(out_path))

    assert out_path.exists()
    html = out_path.read_text()

    assert "maplibregl" in html
    assert "pmtiles.Protocol" in html
    assert "L.vectorGrid.protobuf(" not in html
    # Both groups' namespaced pmtiles sources are present.
    assert "Coarse:region" in html
    assert "Coarse:city" in html
    assert "Fine:hood" in html
    # Radio-button switcher for the two named (mutually exclusive) groups.
    assert 'name="__base_group"' in html
    assert 'value="Coarse"' in html
    assert 'value="Fine"' in html
