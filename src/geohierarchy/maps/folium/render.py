"""HierarchyMap / MultiHierarchyMap: orchestrate tiling, styling, legend, and .save()."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .basemap import build_tile_layer, MAP_MAX_ZOOM
from .legend import build_legend
from .resolution import assign_resolutions, validate_resolutions, fill_around_manual
from .style import ColorSpec, build_vector_tile_layer_styles_js
from .tiles import write_level_tiles

VECTORGRID_CDN = (
    "https://unpkg.com/leaflet.vectorgrid@1.3.0/dist/Leaflet.VectorGrid.bundled.js"
)

# PMTiles Leaflet plugin CDN
PMTILES_LEAFLET_CDN = (
    "https://unpkg.com/@protomaps/leaflet-pmtiles@latest/dist/leaflet-pmtiles.min.js"
)

# Opening the saved HTML directly (double-click / file://) silently fails:
# the basemap still renders (ordinary cross-origin https image requests,
# unaffected), but every vector-tile fetch is a same-origin-only local
# `fetch()` that browsers block under file://, with no visible error --
# just an empty map once you toggle past the basemap. Detecting and
# surfacing that beats a silent no-op.
_FILE_PROTOCOL_WARNING_HTML = """
<div id="file-protocol-warning" style="
    display:none; position:fixed; top:10px; left:50%; transform:translateX(-50%);
    z-index:10000; background:#fff3cd; color:#664d03; border:1px solid #ffda6a;
    padding:8px 14px; border-radius:4px; font-family:Arial, sans-serif;
    font-size:13px; box-shadow:0 1px 4px rgba(0,0,0,0.3); max-width:90%;">
  This map was opened as a local file (<code>file://</code>), so the
  geometry layers can't load -- browsers block local tile requests from a
  <code>file://</code> page. Serve this folder with a static server instead,
  e.g. <code>python -m http.server</code>, then open it via
  <code>http://localhost:...</code>.
</div>
<script>
if (window.location.protocol === "file:") {
  document.addEventListener("DOMContentLoaded", function() {
    var el = document.getElementById("file-protocol-warning");
    if (el) { el.style.display = "block"; }
  });
}
</script>
"""

# Actually generating .pbf tiles all the way out to zoom 25 is intractable
# (tile counts grow as 4^z): like a raster basemap, each level's vector
# tiles are only ever generated up to a practical native zoom, and Leaflet
# reuses/upscales the deepest generated tile for anything beyond that,
# exactly like TileLayer's max_native_zoom/max_zoom split in basemap.py.
#
# **Raised from 14 to 18 (2026-08-12), at the user's explicit request**: 14
# (~10 m/px) was originally chosen as "finer than any of this package's
# geometries need," but that assumption doesn't hold once you can zoom a
# street-edges layer in past building scale -- at z14, a single tile pixel
# spans ~10 real meters, so upscaled overzoom rendering of narrow street
# geometry looks visibly blocky/pixelated exactly as reported. 18 (~0.6
# m/px) is sub-meter precision, well past visible pixelation for street-
# level data, at the cost of slower tile generation (tile count grows as
# 4^z, so z18 generates roughly 4^4 = 256x the tiles of z14 for any level
# whose zoom range actually reaches that deep -- multiple extra minutes of
# `build()` time is the explicit tradeoff being made here for visual
# quality). Levels whose own zoom range tops out below 18 are unaffected
# (this is a ceiling, not a floor -- see `min(max_z, MAX_NATIVE_TILE_ZOOM)`
# below).
MAX_NATIVE_TILE_ZOOM = 18


@dataclass
class MapLayer:
    """Per-level rendering configuration for one hierarchy level.

    Attributes:
        style: A :class:`ColorSpec` describing color/fill declaratively.
        style_js: Raw JS function body ``function(properties){...}``,
            bypassing ``style`` entirely. Takes precedence over ``style``.
        popup_fields: Attribute columns to show in a default popup table.
        popup_js: Raw JS function ``function(properties){ return '<html>'; }``
            building the popup content, bypassing ``popup_fields``.
        tooltip_js: Raw JS function building tooltip content, analogous to
            ``popup_js``.
        legend_html: Raw HTML overriding the auto-generated legend box for
            this level.
        resolution: Explicit ``(min_zoom, max_zoom)`` override for this
            level, bypassing auto-assignment.
        buffer_frac: Tile buffer as a fraction of tile width, forwarded to
            :func:`~geohierarchy.maps.folium.tiles.write_level_tiles`
            (default there is 2%). Point layers rendered as client-side
            circles (a fixed pixel radius, not a geometry that scales with
            zoom) need a bigger buffer than polygon layers do -- a point
            just inside a tile's edge still gets its circle clipped by that
            tile's own canvas bounds unless the *neighboring* tile also
            carries a copy of it to draw the overflow, and 2% of a tile's
            geographic width can be far fewer pixels than an 18px circle
            radius at high zoom.
        layer_type: Registry key from
            :mod:`geohierarchy.maps.layers` (e.g. ``"polygon"``,
            ``"circle"``, ``"street_overlay"``), classifying how this
            level's tiles render (filled area / radius-scaled points /
            stroked line). Defaults to ``"polygon"`` when unset. Read by
            :class:`geohierarchy.maps.maplibre.render.MapLibreHierarchyMap`
            to pick the right MapLibre layer type/paint shape; the Folium
            renderer currently infers the same thing from ``style_js``, so
            this field is additive (safe to leave unset for existing
            Folium-only code).
        native_zoom_range: Optional ``(native_min_zoom, native_max_zoom)``
            override, decoupling the zoom range tiles are actually BUILT at
            from the (generally much wider) range they're DISPLAYED at
            (``resolution``/the auto-assigned display band). Leaflet.
            VectorGrid reuses the nearest native zoom's tiles (scaled) for
            any display zoom outside ``[native_min_zoom, native_max_zoom]``
            -- so a level whose content doesn't meaningfully change in
            complexity across zoom (e.g. a single, always-visible
            border-only overlay spanning the full [0, 25] display band) can
            build tiles at, say, only zoom 8-14 and still display correctly
            everywhere, instead of independently tiling all 19 zooms up to
            ``MAX_NATIVE_TILE_ZOOM``. Only safe for that kind of stable-
            content, wide-display-band level -- narrowing this for a level
            whose geometry/detail genuinely differs a lot by zoom would make
            far-zoomed-in views reuse an overly-coarse tile. ``None`` (the
            default) keeps the existing behavior: native range = the
            display range, capped at ``MAX_NATIVE_TILE_ZOOM``.
    """

    style: Optional[ColorSpec] = None
    style_js: Optional[str] = None
    popup_fields: Optional[List[str]] = None
    popup_js: Optional[str] = None
    tooltip_js: Optional[str] = None
    legend_html: Optional[str] = None
    resolution: Optional[Any] = None
    buffer_frac: Optional[float] = None
    native_zoom_range: Optional[Any] = None
    layer_type: Optional[str] = None
    maplibre_paint: Optional[Dict[str, Any]] = None
    """Raw MapLibre GL `paint` dict, used verbatim by
    :class:`geohierarchy.maps.maplibre.render.MapLibreHierarchyMap` in place
    of translating ``style``/``style_js`` (which is Leaflet/Folium-specific
    JS and has no MapLibre equivalent). Takes precedence over ``style``.
    Callers whose per-level coloring is expressed as ``style_js`` (a raw JS
    function -- e.g. `transitlos.map.build`'s score-based red/yellow/green
    ramp) must additionally set this to get the equivalent MapLibre
    ``["interpolate", ...]``/``["step", ...]`` paint expression; MapLibre
    output otherwise falls back to a flat default fill color for that
    level."""


def _default_style_js(level_name: str) -> str:
    return (
        "function(properties) {\n"
        "  return {color: '#3388ff', weight: 1, fill: true, "
        "fillColor: '#3388ff', fillOpacity: 0.4};\n"
        "}"
    )


def _popup_js_from_fields(fields: List[str]) -> str:
    fields_json = json.dumps(fields)
    return (
        "function(properties) {\n"
        f"  var fields = {fields_json};\n"
        "  var html = '<table>';\n"
        "  fields.forEach(function(f) {\n"
        "    html += '<tr><td style=\"font-weight:600;padding-right:6px;\">' + f + "
        "'</td><td>' + properties[f] + '</td></tr>';\n"
        "  });\n"
        "  html += '</table>';\n"
        "  return html;\n"
        "}"
    )


class HierarchyMap:
    """Renders a whole :class:`~geohierarchy.core.GeoHierarchy` as one interactive Leaflet map.

    Each level owns a zoom-resolution range; exactly one level's vector
    tiles are visible at any given zoom.
    """

    def __init__(
        self,
        hierarchy,
        levels: Optional[Sequence[str]] = None,
        basemap: Union[str, dict, Any] = "cartodb_positron",
        tiles_dir: Optional[str] = None,
        resolutions: Optional[Dict[str, Any]] = None,
        use_pmtiles: bool = True,
        extract_xyz: bool = True,
    ):
        """
        Args:
            hierarchy: A :class:`~geohierarchy.core.GeoHierarchy`.
            levels: Level names to render. Defaults to all of
                ``hierarchy.geometries``.
            basemap: Preset name / dict / ``folium.TileLayer``. See
                :func:`geohierarchy.maps.folium.basemap.build_tile_layer`.
            tiles_dir: Directory vector tiles are written to by
                :meth:`build`. Required before calling :meth:`build`.
            resolutions: Optional manual ``{level: (min_zoom, max_zoom)}``
                overrides, equivalent to calling :meth:`set_resolution` for
                each entry.
            use_pmtiles: If True (default), generate and use PMTiles format.
                If False, use legacy XYZ PBF directories for backward compatibility.
            extract_xyz: If True (default), also extract a `{z}/{x}/{y}.pbf`
                XYZ directory tree from each level's `.pmtiles` during
                :meth:`build` -- required by :meth:`save` (Folium's
                Leaflet.VectorGrid-based renderer has no native PMTiles
                support). Set False when only :meth:`save_maplibre` will be
                called (MapLibre reads `.pmtiles` directly) to skip this
                extraction entirely -- for large levels this is the
                difference between a single `.pmtiles` archive and an
                extracted tree of tens/hundreds of thousands of loose files.
        """
        self.hierarchy = hierarchy
        self.levels: List[str] = (
            list(levels) if levels is not None else list(hierarchy.geometries.keys())
        )
        self.basemap = basemap
        self.tiles_dir = tiles_dir
        self.use_pmtiles = use_pmtiles
        self.extract_xyz = extract_xyz
        self._layers: Dict[str, MapLayer] = {name: MapLayer() for name in self.levels}
        self._manual_resolutions: Dict[str, Any] = {}
        if resolutions:
            for name, zr in resolutions.items():
                self.set_resolution(name, zr[0], zr[1])
        self._built_tiles: List[Path] = []
        # Opt-in: when set (see `set_fallback_level`), the MapLibre renderer
        # (`geohierarchy.maps.maplibre.render`) widens this level's rendered
        # `maxzoom` to the full range regardless of its own
        # `resolve_zoom_ranges()`-assigned band, so it keeps rendering
        # (MapLibre overzooms its last-generated tile automatically -- no
        # extra tiling needed) underneath every finer level instead of
        # disappearing once a finer level's band starts. Tile *generation*
        # (`build()`) is unaffected -- this level is still only physically
        # tiled across its own normal band; deeper zooms just reuse/scale
        # that same tile. `None` (default): no change, strict partition as
        # before. Folium's renderer (`save()`) ignores this -- Leaflet's
        # VectorGrid layer switching has no equivalent overzoom mechanism.
        self.fallback_level: Optional[str] = None

    def set_fallback_level(self, level: str) -> "HierarchyMap":
        """Mark `level` (normally the coarsest) as an always-rendered fallback -- see `fallback_level`."""
        if level not in self.levels:
            raise KeyError(
                f"Level '{level}' is not part of this HierarchyMap ({self.levels})"
            )
        self.fallback_level = level
        return self

    # ------------------------------------------------------------------
    def configure_level(self, name: str, **kwargs) -> "HierarchyMap":
        """Set per-level rendering options (see :class:`MapLayer` fields).

        Args:
            name: Level name (must be one of ``self.levels``).
            **kwargs: Any :class:`MapLayer` field.

        Returns:
            ``self``, for chaining.
        """
        if name not in self._layers:
            raise KeyError(
                f"Level '{name}' is not part of this HierarchyMap ({self.levels})"
            )
        layer = self._layers[name]
        for k, v in kwargs.items():
            if not hasattr(layer, k):
                raise TypeError(f"MapLayer has no field '{k}'")
            setattr(layer, k, v)
        return self

    def set_resolution(
        self, level: str, min_zoom: int, max_zoom: int
    ) -> "HierarchyMap":
        """Manually pin a level's zoom range, overriding auto-assignment.

        Args:
            level: Level name.
            min_zoom: Inclusive minimum zoom.
            max_zoom: Inclusive maximum zoom.

        Returns:
            ``self``, for chaining.
        """
        self._manual_resolutions[level] = (min_zoom, max_zoom)
        return self

    # ------------------------------------------------------------------
    def resolve_zoom_ranges(self) -> Dict[str, Any]:
        """Compute the final zoom range for every level (manual overrides + auto-fill).

        Levels with a manual range (via :meth:`set_resolution` or the
        constructor's ``resolutions=``) keep it; the rest are auto-assigned
        with :func:`assign_resolutions` restricted to those remaining
        levels and validated together against the manual ones so the
        overall set still partitions ``[0, 25]``.

        Returns:
            Mapping of level name to ``(min_zoom, max_zoom)``.
        """
        manual = dict(self._manual_resolutions)

        if not manual:
            result: Dict[str, Any] = assign_resolutions(
                self.hierarchy, levels=self.levels
            )
        else:
            result = fill_around_manual(self.hierarchy, self.levels, manual)

        validate_resolutions(result)
        return result

    # ------------------------------------------------------------------
    def build(self, free_level_data: bool = False) -> "HierarchyMap":
        """Generate vector tiles for every level under ``tiles_dir``.

        Args:
            free_level_data: If True, once every level's tiles are written,
                shrink `self.hierarchy.levels[name]` down to just its
                `minx`/`miny`/`maxx`/`maxy` bounds columns (dropping every
                attribute/geometry column). Real 2026-08-26 OOM fix: a
                `MultiHierarchyMap` with several named groups (e.g.
                Boston's hexagons/circles/census, each itself several H3
                resolutions or census levels) keeps EVERY group's full
                `GeoHierarchy` -- the complete attribute table `add_level`
                copied in, for every level -- resident for the entire
                `MultiHierarchyMap.build()` loop, even though each group's
                own `.build()` call only ever touches one group's data at a
                time; nothing previously freed a finished group's data
                before the next group's `.build()` ran, so peak memory was
                the SUM of every group's full attribute tables, not the max
                of any one. Only safe to set when nothing downstream needs
                this hierarchy's per-feature attribute data again --
                `level_bounds` (map centering) still works after this (it
                only reads the four bounds columns kept here), but
                `get_level` (`_vectorgrid_js_and_legend`'s `layer.style is
                not None` branch, and the MapLibre renderer's equivalent)
                would come back empty. `transitlos.map.build.build_city_map`
                only ever configures levels with `style_js`/`maplibre_paint`
                (never the `style=` `ColorSpec` param), so `layer.style` is
                always `None` there and this is safe for that caller;
                defaults to False so every other caller's behavior is
                unchanged.

        Idempotent/re-runnable: re-running overwrites existing tile files.

        Parallelized per **(level, zoom)**, not per level: levels can differ
        by orders of magnitude in tile count -- not just from zoom-band
        width, but also from sheer feature density/count (a real rebuild
        showed census at ~1.17M tiles vs. ~80k for the next largest level,
        despite comparable band widths, simply because census polygons are
        far more numerous/complex per unit area than street edges) -- so
        splitting work only across levels leaves most cores idle while the
        single biggest level's zooms grind through serially on one core.
        Each level is reprojected/trimmed once (`prep_level_gdf`) and then
        every zoom in its range becomes its own job in one shared pool sized
        to the machine's core count, so a lopsided level's zooms run
        alongside every other level's instead of after them.

        (A more aggressive version of this -- tiling each level at a single
        native zoom and letting Leaflet.VectorGrid's `minNativeZoom`/
        `maxNativeZoom` reuse it across the whole display band, the way
        raster tiles already do -- was considered and is NOT done here.
        Some levels in this codebase are configured with very wide display
        bands (e.g. a single-level overlay spanning the full [0, 25] zoom
        range), and pinning `minNativeZoom` to a deep native zoom for a
        level like that would make Leaflet fetch `4^(native_z - display_z)`
        tiles to cover one screen at a zoomed-out display level -- a
        regression, not an improvement, for exactly the levels with the
        widest bands. That needs each level's natural resolution vs. its
        display band width worked out deliberately, with real
        browser-verified zoomed-out behavior, not a blanket rule; flagged
        as a real follow-up lever, not applied blindly.)

        Returns:
            ``self``, for chaining.
        """
        if not self.tiles_dir:
            raise ValueError("tiles_dir must be set before calling build()")

        zoom_ranges = self.resolve_zoom_ranges()

        # Use fast freestiler-based tiling when available
        # Generate all zooms for each level at once (much faster)
        self._built_tiles = []

        for name in self.levels:
            id_col = self.hierarchy.id_cols[name]
            layer = self._layers[name]
            popup_fields = layer.popup_fields or []
            style_cols = [layer.style.column] if layer.style is not None else []
            property_cols = list(dict.fromkeys([*popup_fields, *style_cols]))

            # Pull only the attribute columns tiling actually needs (id +
            # popup_fields + the active style column) rather than every
            # column ever added to the level's attribute table. On a
            # metro-scale H3 level with 100+ census/derived columns, the old
            # unconditional `get_level(name)` did a full Polars->pandas
            # conversion + merge of every column before this loop's own
            # `write_level_tiles` trimmed it right back down to
            # `property_cols` a few lines later -- peak memory was paying
            # for the full wide table even though only a handful of columns
            # ever reached the tiler. See core.GeoHierarchy.get_level's
            # `columns` parameter.
            gdf = self.hierarchy.get_level(name, columns=property_cols)

            min_z, max_z = zoom_ranges[name]
            if layer.native_zoom_range is not None:
                native_min_z, native_max_z = layer.native_zoom_range
                native_max_z = min(native_max_z, MAX_NATIVE_TILE_ZOOM)
            else:
                native_max_z = min(max_z, MAX_NATIVE_TILE_ZOOM)
                native_min_z = min(min_z, native_max_z)

            buffer_frac = 0.02 if layer.buffer_frac is None else layer.buffer_frac

            # Use fast method with freestiler. PMTiles (primary format) is
            # always generated; the XYZ PBF directory tree is only extracted
            # on top of it when `self.extract_xyz` is True (default -- needed
            # by `save()`'s Leaflet.VectorGrid renderer). Callers that only
            # need `save_maplibre()` (reads .pmtiles directly) should
            # construct with `extract_xyz=False` to skip this extra,
            # potentially very large, per-tile-file step entirely.
            result = write_level_tiles(
                gdf,
                name,
                self.tiles_dir,
                min_zoom=native_min_z,
                max_zoom=native_max_z,
                id_col=id_col,
                property_cols=property_cols,
                buffer_frac=buffer_frac,
                use_xyz=self.extract_xyz,
                extract_xyz_from_pmtiles=self.extract_xyz,
            )
            self._built_tiles.extend(result)

            # Drop this level's (trimmed, but still potentially large on a
            # metro-scale grid) GeoDataFrame and force an immediate
            # collection before moving to the next level -- cheap and safe,
            # same pattern already applied to `_join_worldpop_global_schema`
            # in CS_transitLOS's pipeline.py for the same class of issue.
            # Refcounting alone would normally free `gdf` as soon as it's
            # reassigned next iteration, but an explicit collect() here
            # guards against any lingering reference (e.g. held inside
            # `write_level_tiles`'s temp-file writing path) outliving the
            # loop iteration, and matters most for `MultiHierarchyMap.build()`,
            # which calls this method once per named/overlay group in a
            # single process -- without this, a wide level's memory could
            # still be pending collection when the next group's `build()`
            # call materializes its own level.
            del gdf
            gc.collect()

        if free_level_data:
            import polars as pl

            for name in self.levels:
                df = self.hierarchy.levels.get(name)
                if df is None:
                    continue
                keep_cols = [
                    c for c in ("minx", "miny", "maxx", "maxy") if c in df.columns
                ]
                self.hierarchy.levels[name] = (
                    df.select(keep_cols) if keep_cols else pl.DataFrame()
                )
            gc.collect()

        return self

    # ------------------------------------------------------------------
    def _vectorgrid_js_and_legend(
        self,
        map_var: str,
        zoom_ranges: Dict[str, Any],
        target_var: Optional[str] = None,
        key_prefix: str = "",
    ):
        """Build the per-level vectorGrid JS blocks and the combined legend element.

        Args:
            map_var: JS variable name of the ``folium.Map``/Leaflet map,
                used for popup placement and the legend's zoomend listener.
            zoom_ranges: Per-level zoom ranges.
            target_var: JS variable name each level's vectorGrid layer is
                added to (``.addTo(target_var)``). Defaults to ``map_var``
                (add directly to the map); :class:`MultiHierarchyMap` passes
                a ``FeatureGroup`` variable instead so a whole group's
                layers can be shown/hidden together via the layer control.
            key_prefix: Namespaces the JS var name and the
                ``window.__vgLayers`` registry key for every level in this
                call (e.g. ``"hexagons:"``). Without this, two
                :class:`HierarchyMap`s sharing level names (as
                :class:`MultiHierarchyMap` groups typically do) would
                overwrite each other's ``window.__vgLayers`` entries and JS
                var names, silently leaving only the last-built group
                redrawable/reachable from page-level controls.
        """
        target_var = target_var or map_var

        gdfs = {}
        vectorgrid_js_blocks = []
        safe_prefix = "".join(c if c.isalnum() else "_" for c in key_prefix)

        for name in self.levels:
            layer = self._layers[name]
            # Materialized only for the levels that actually need the data:
            # `get_level` re-joins the level's geometry with its whole
            # attribute table (a pandas merge over every row), which on a
            # metro-scale H3 level is multiple GB of pure waste when the
            # caller supplied `style_js`/`legend_html` and nothing here ever
            # reads a value. A level's data is needed only to derive a
            # `ColorSpec`'s style function or to infer its legend
            # domain/categories -- i.e. only when `layer.style` is set.
            if layer.style is not None:
                gdfs[name] = gdf = self.hierarchy.get_level(name)
            else:
                gdf = None

            if layer.style_js:
                style_body = layer.style_js
            elif layer.style is not None:
                style_body = layer.style.style_js(gdf, var_name=f"style_{name}")
            else:
                style_body = _default_style_js(name)

            if layer.popup_js:
                popup_body = layer.popup_js
            elif layer.popup_fields:
                popup_body = _popup_js_from_fields(layer.popup_fields)
            else:
                popup_body = None

            min_z, max_z = zoom_ranges[name]
            layer_for_native = self._layers[name]
            if layer_for_native.native_zoom_range is not None:
                native_min_z, native_max_z = layer_for_native.native_zoom_range
                native_max_z = min(native_max_z, MAX_NATIVE_TILE_ZOOM)
            else:
                native_max_z = min(max_z, MAX_NATIVE_TILE_ZOOM)
                native_min_z = min_z
            # Always use XYZ PBF URLs for Leaflet.VectorGrid compatibility
            # PMTiles files are generated as the primary format (fast, compressed),
            # and XYZ PBF directories are extracted from them for browser use.
            url = f"{self.tiles_dir}/{name}/{{z}}/{{x}}/{{y}}.pbf"
            style_entry = build_vector_tile_layer_styles_js(name, style_body)

            var_name = f"vg_{safe_prefix}{name}"
            registry_key = f"{key_prefix}{name}"
            block = [
                f"var {var_name} = L.vectorGrid.protobuf({json.dumps(url)}, {{",
                f"  minZoom: {min_z},",
                f"  maxZoom: {max_z},",
                f"  minNativeZoom: {native_min_z},",
                f"  maxNativeZoom: {native_max_z},",
                "  rendererFactory: L.canvas.tile,",
                f"  vectorTileLayerStyles: {{ {style_entry} }},",
                f"  interactive: {str(bool(popup_body or layer.tooltip_js)).lower()},",
                "});",
                f"{var_name}.addTo({target_var});",
                # Exposes every created vector-tile layer in a global registry, keyed by
                # (group-prefixed) level name, so page-level custom controls (e.g. an
                # opacity-by-field dropdown) can call `.redraw()` on the right layer(s)
                # after changing a style-affecting JS global -- VectorGrid doesn't restyle
                # already-rendered tiles on its own when a style function's external
                # inputs change. The prefix keeps two HierarchyMaps sharing level names
                # (as MultiHierarchyMap groups typically do) from clobbering each other.
                "window.__vgLayers = window.__vgLayers || {};",
                f"window.__vgLayers[{json.dumps(registry_key)}] = {var_name};",
            ]
            if popup_body:
                block.append(
                    f"{var_name}.on('click', function(e) {{\n"
                    f"  var popupFn = {popup_body};\n"
                    f"  L.popup().setLatLng(e.latlng).setContent(popupFn(e.layer.properties)).openOn({map_var});\n"
                    f"}});"
                )
            if layer.tooltip_js:
                block.append(
                    f"{var_name}.bindTooltip(function(e) {{\n"
                    f"  var tooltipFn = {layer.tooltip_js};\n"
                    f"  return tooltipFn(e.properties);\n"
                    f"}});"
                )
            vectorgrid_js_blocks.append("\n".join(block))

        legend = build_legend(self._layers, zoom_ranges, gdfs)
        return "\n\n".join(vectorgrid_js_blocks), legend

    # ------------------------------------------------------------------
    def _build_folium_map(self, existing_map=None, target=None, key_prefix: str = ""):
        """Build (or augment) a folium.Map with this HierarchyMap's layers.

        Args:
            existing_map: If given, layers are added to this map instead of
                a freshly created one (used by :class:`MultiHierarchyMap`).
            target: If given, a folium element (e.g. a ``FeatureGroup``)
                whose JS variable each level's vectorGrid layer is added to
                instead of the map directly (used by
                :class:`MultiHierarchyMap` so a whole group's layers toggle
                together via the layer control).
            key_prefix: Forwarded to :meth:`_vectorgrid_js_and_legend` to
                namespace ``window.__vgLayers`` keys/JS var names (see its
                docstring) -- required whenever multiple `HierarchyMap`s
                sharing level names are combined onto one page.
        """
        import folium

        zoom_ranges = self.resolve_zoom_ranges()

        if existing_map is None:
            # Extent only -- taken from the level's stored bounds columns
            # rather than by rebuilding the whole level (see `level_bounds`).
            bounds = self.hierarchy.level_bounds(
                self.levels[0]
            )  # minx, miny, maxx, maxy
            center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
            m = folium.Map(
                location=center, zoom_start=12, max_zoom=MAP_MAX_ZOOM, tiles=None
            )
            base_layer = build_tile_layer(self.basemap)
            base_layer.add_to(m)
            m.get_root().html.add_child(folium.Element(_FILE_PROTOCOL_WARNING_HTML))
        else:
            m = existing_map

        map_var = m.get_name()
        target_var = target.get_name() if target is not None else map_var
        js_blocks, legend = self._vectorgrid_js_and_legend(
            map_var, zoom_ranges, target_var=target_var, key_prefix=key_prefix
        )

        # Where this element ends up in folium's generated document is not
        # something we control precisely (folium/branca assemble the page
        # from several independently-ordered blocks), and it has bitten
        # this code twice already: once placed before the `map_XXXX`
        # variable's own declaration (ReferenceError), once nested inside
        # branca's already-<script>-wrapped `script` block (a literal
        # nested <script> tag breaks HTML parsing outright). The robust
        # fix is to stop caring about placement: always defer to the
        # window `load` event before touching `L.vectorGrid` or the map
        # variable, since by the time `load` fires every synchronous
        # <script> tag on the page -- Leaflet core, the VectorGrid plugin,
        # and folium's own map-init script -- has already run, regardless
        # of where in the document this element landed.
        script_el = folium.Element(
            f'<script src="{VECTORGRID_CDN}"></script>\n'
            f"<script>\n"
            f"window.addEventListener('load', function() {{\n"
            f"{js_blocks}\n"
            f"}});\n"
            f"</script>\n"
        )
        m.get_root().html.add_child(script_el)

        if legend is not None:
            m.add_child(legend)

        return m

    # ------------------------------------------------------------------
    def to_folium(self):
        """Build and return the underlying ``folium.Map``, without saving to disk."""
        return self._build_folium_map()

    def save(self, path: str) -> str:
        """Render the map (basemap + vector-tile layers + legend) to an HTML file.

        Args:
            path: Output HTML file path.

        Returns:
            The path saved to.
        """
        m = self._build_folium_map()
        import folium

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(path)
        return path

    # ------------------------------------------------------------------
    def save_maplibre(
        self, path: str, basemap: Union[str, dict] = None, title: str = "Map"
    ) -> str:
        """Render this map's base layers via MapLibre GL JS + PMTiles instead of Folium/Leaflet.

        Base-layer parity only (fill/circle/line coloring, click popups,
        hover) -- the scenario editor and stats panel remain Folium-only.
        See :class:`geohierarchy.maps.maplibre.render.MapLibreHierarchyMap`.
        Requires :meth:`build` to have already been run (PMTiles must
        exist under ``self.tiles_dir``), and ``path`` must be saved next to
        ``self.tiles_dir`` (same relative-URL convention as :meth:`save`).
        """
        from ..maplibre.render import save_maplibre as _save_maplibre, DEFAULT_BASEMAP

        return _save_maplibre(
            self, path, basemap=basemap or DEFAULT_BASEMAP, title=title
        )


class MultiHierarchyMap:
    """Combines several :class:`HierarchyMap` instances as mutually-exclusive Leaflet layers.

    Each named group is rendered as a ``folium.FeatureGroup(overlay=False)``,
    which ``folium.LayerControl`` renders as a radio-button group (Leaflet's
    ``baseLayer`` semantics: only one non-overlay layer is ever active at a
    time) -- giving the "select H3 or Polygons" toggle with no custom JS.

    ``overlay_hierarchy_maps`` additionally supports independently
    togglable (checkbox) layers on the same map -- e.g. a streets layer and
    a separate "development opportunity" layer that should each turn on/off
    on their own, not as part of the exclusive shape choice.
    """

    def __init__(
        self,
        named_hierarchy_maps: Dict[str, HierarchyMap],
        default: Optional[str] = None,
        basemap: Union[str, dict, Any] = "cartodb_positron",
        tiles_dir: Optional[str] = None,
        overlay_hierarchy_maps: Optional[Dict[str, HierarchyMap]] = None,
        overlay_show: Optional[Dict[str, bool]] = None,
    ):
        """
        Args:
            named_hierarchy_maps: Mapping of display name -> :class:`HierarchyMap`,
                rendered as mutually-exclusive (radio) base layers.
            default: Name of the base-layer group shown by default. Defaults
                to the first key.
            basemap: Shared basemap for the combined map.
            tiles_dir: If given, applied to every child :class:`HierarchyMap`
                (in both `named_hierarchy_maps` and `overlay_hierarchy_maps`)
                that doesn't already have its own ``tiles_dir`` set
                (namespaced under a per-group subdirectory to avoid
                collisions between groups sharing level names).
            overlay_hierarchy_maps: Mapping of display name -> :class:`HierarchyMap`,
                rendered as independently togglable (checkbox) overlay
                layers, each shown/hidden without affecting the others or
                the base-layer choice.
            overlay_show: Optional per-overlay initial visibility (defaults
                to visible/`True`).
        """
        if not named_hierarchy_maps:
            raise ValueError(
                "MultiHierarchyMap requires at least one named HierarchyMap"
            )
        self.named_hierarchy_maps = named_hierarchy_maps
        self.overlay_hierarchy_maps = overlay_hierarchy_maps or {}
        self.overlay_show = overlay_show or {}
        self.default = default or next(iter(named_hierarchy_maps))
        self.basemap = basemap
        self.tiles_dir = tiles_dir

        all_maps = {**named_hierarchy_maps, **self.overlay_hierarchy_maps}
        for group_name, hmap in all_maps.items():
            if hmap.tiles_dir is None and tiles_dir is not None:
                hmap.tiles_dir = str(Path(tiles_dir) / group_name)

    # ------------------------------------------------------------------
    def build(self, free_level_data: bool = False) -> "MultiHierarchyMap":
        """Build tiles for every child :class:`HierarchyMap` (base layers and overlays).

        Args:
            free_level_data: Forwarded to each child :meth:`HierarchyMap.build`
                -- see its docstring. Frees a finished group's full attribute
                data before the next group's `.build()` materializes its own,
                so peak memory is the max of any one group's data instead of
                the sum of all of them.
        """
        for hmap in list(self.named_hierarchy_maps.values()) + list(
            self.overlay_hierarchy_maps.values()
        ):
            hmap.build(free_level_data=free_level_data)
        return self

    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """Render all groups onto one map, each as a non-overlay (radio-button) layer.

        Args:
            path: Output HTML file path.

        Returns:
            The path saved to.
        """
        import folium

        first = next(iter(self.named_hierarchy_maps.values()))
        bounds = first.hierarchy.level_bounds(first.levels[0])
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

        m = folium.Map(
            location=center, zoom_start=12, max_zoom=MAP_MAX_ZOOM, tiles=None
        )
        base_layer = build_tile_layer(self.basemap)
        base_layer.add_to(m)
        m.get_root().html.add_child(folium.Element(_FILE_PROTOCOL_WARNING_HTML))

        for group_name, hmap in self.named_hierarchy_maps.items():
            fg = folium.FeatureGroup(
                name=group_name, overlay=False, show=(group_name == self.default)
            )
            fg.add_to(m)
            hmap._build_folium_map(
                existing_map=m, target=fg, key_prefix=f"{group_name}:"
            )

        for group_name, hmap in self.overlay_hierarchy_maps.items():
            fg = folium.FeatureGroup(
                name=group_name,
                overlay=True,
                show=self.overlay_show.get(group_name, True),
            )
            fg.add_to(m)
            hmap._build_folium_map(
                existing_map=m, target=fg, key_prefix=f"{group_name}:"
            )

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(path)
        return path

    # ------------------------------------------------------------------
    def save_maplibre(
        self,
        path: str,
        basemap: Union[str, dict] = None,
        title: str = "Map",
        radius_field_domains: Optional[Dict[str, tuple]] = None,
        opacity_field_domains: Optional[Dict[str, tuple]] = None,
        circle_fields: Optional[List[str]] = None,
        opacity_fields: Optional[List[str]] = None,
        default_circle_field: Optional[str] = None,
        field_labels: Optional[Dict[str, str]] = None,
        radius_field_domains_by_res: Optional[Dict[int, Dict[str, tuple]]] = None,
        circle_zoom_bands: Optional[Dict[int, tuple]] = None,
        special_group_levels: Optional[List[str]] = None,
        special_group_level_labels: Optional[Dict[str, str]] = None,
        special_group_name: str = "special",
    ) -> str:
        """Render every group via MapLibre GL JS + PMTiles, toggled by a radio-button switcher.

        Base-layer parity only, same caveat as :meth:`HierarchyMap.save_maplibre`:
        no scenario editor, no stats panel -- those remain Folium-only
        (see :meth:`save`). Each named group (e.g. "hexagons"/"census") is
        rendered as a mutually-exclusive base layer (matching :meth:`save`'s
        Leaflet radio-button semantics); overlay groups (e.g. "streets",
        "development") are independently togglable checkboxes.
        Requires every child :class:`HierarchyMap` to have already been
        ``.build()``-run (PMTiles must exist under each ``tiles_dir``).

        `radius_field_domains`/`opacity_field_domains`/`circle_fields`/
        `opacity_fields`/`default_circle_field`: optional, mirror Folium's
        `_control_panel_html`'s "Circle size by"/"Opacity by"/"Contrast"
        controls (see `transitlos.map.build._control_panel_html`) -- when
        given, `save_multi_maplibre` adds the matching live dropdowns/
        sliders to the layer-switcher panel. Omitted entirely when not
        passed (caller has no such fields), matching prior behavior.
        """
        from ..maplibre.render import (
            save_multi_maplibre as _save_multi_maplibre,
            DEFAULT_BASEMAP,
        )

        return _save_multi_maplibre(
            self,
            path,
            basemap=basemap or DEFAULT_BASEMAP,
            title=title,
            radius_field_domains=radius_field_domains,
            opacity_field_domains=opacity_field_domains,
            circle_fields=circle_fields,
            opacity_fields=opacity_fields,
            default_circle_field=default_circle_field,
            field_labels=field_labels,
            radius_field_domains_by_res=radius_field_domains_by_res,
            circle_zoom_bands=circle_zoom_bands,
            special_group_levels=special_group_levels,
            special_group_level_labels=special_group_level_labels,
            special_group_name=special_group_name,
        )
