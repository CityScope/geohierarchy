"""HierarchyMap / MultiHierarchyMap: orchestrate tiling, styling, legend, and .save()."""

from __future__ import annotations

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
# 18 (~0.6 m/px) is massive overkill for polygon/H3-cell data -- it makes
# tile generation multiple minutes slow for a single city because tile
# count grows as 4^z. 14 (~10 m/px) is already finer than any of this
# package's geometries need and keeps `build()` in the range of seconds.
MAX_NATIVE_TILE_ZOOM = 14


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
    """

    style: Optional[ColorSpec] = None
    style_js: Optional[str] = None
    popup_fields: Optional[List[str]] = None
    popup_js: Optional[str] = None
    tooltip_js: Optional[str] = None
    legend_html: Optional[str] = None
    resolution: Optional[Any] = None


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
        """
        self.hierarchy = hierarchy
        self.levels: List[str] = (
            list(levels) if levels is not None else list(hierarchy.geometries.keys())
        )
        self.basemap = basemap
        self.tiles_dir = tiles_dir
        self._layers: Dict[str, MapLayer] = {name: MapLayer() for name in self.levels}
        self._manual_resolutions: Dict[str, Any] = {}
        if resolutions:
            for name, zr in resolutions.items():
                self.set_resolution(name, zr[0], zr[1])
        self._built_tiles: List[Path] = []

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
    def build(self) -> "HierarchyMap":
        """Generate vector tiles for every level under ``tiles_dir``.

        Idempotent/re-runnable: re-running overwrites existing tile files.

        Returns:
            ``self``, for chaining.
        """
        if not self.tiles_dir:
            raise ValueError("tiles_dir must be set before calling build()")

        zoom_ranges = self.resolve_zoom_ranges()
        self._built_tiles = []
        for name in self.levels:
            gdf = self.hierarchy.get_level(name)
            id_col = self.hierarchy.id_cols[name]
            layer = self._layers[name]
            popup_fields = layer.popup_fields or []
            style_cols = [layer.style.column] if layer.style is not None else []
            property_cols = list(dict.fromkeys([*popup_fields, *style_cols]))
            min_z, max_z = zoom_ranges[name]
            native_max_z = min(max_z, MAX_NATIVE_TILE_ZOOM)
            native_min_z = min(min_z, native_max_z)
            written = write_level_tiles(
                gdf,
                name,
                self.tiles_dir,
                native_min_z,
                native_max_z,
                id_col,
                property_cols,
            )
            self._built_tiles.extend(written)
        return self

    # ------------------------------------------------------------------
    def _vectorgrid_js_and_legend(
        self,
        map_var: str,
        zoom_ranges: Dict[str, Any],
        target_var: Optional[str] = None,
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
        """
        target_var = target_var or map_var

        gdfs = {}
        # style_entries = []
        # popup_entries = []
        vectorgrid_js_blocks = []

        for name in self.levels:
            layer = self._layers[name]
            gdf = self.hierarchy.get_level(name)
            gdfs[name] = gdf

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
            native_max_z = min(max_z, MAX_NATIVE_TILE_ZOOM)
            url = f"{self.tiles_dir}/{name}/{{z}}/{{x}}/{{y}}.pbf"
            style_entry = build_vector_tile_layer_styles_js(name, style_body)

            var_name = f"vg_{name}"
            block = [
                f"var {var_name} = L.vectorGrid.protobuf({json.dumps(url)}, {{",
                f"  minZoom: {min_z},",
                f"  maxZoom: {max_z},",
                f"  maxNativeZoom: {native_max_z},",
                "  rendererFactory: L.canvas.tile,",
                f"  vectorTileLayerStyles: {{ {style_entry} }},",
                f"  interactive: {str(bool(popup_body or layer.tooltip_js)).lower()}",
                "});",
                f"{var_name}.addTo({target_var});",
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
    def _build_folium_map(self, existing_map=None, target=None):
        """Build (or augment) a folium.Map with this HierarchyMap's layers.

        Args:
            existing_map: If given, layers are added to this map instead of
                a freshly created one (used by :class:`MultiHierarchyMap`).
            target: If given, a folium element (e.g. a ``FeatureGroup``)
                whose JS variable each level's vectorGrid layer is added to
                instead of the map directly (used by
                :class:`MultiHierarchyMap` so a whole group's layers toggle
                together via the layer control).
        """
        import folium

        zoom_ranges = self.resolve_zoom_ranges()

        if existing_map is None:
            gdf0 = self.hierarchy.get_level(self.levels[0])
            bounds = gdf0.total_bounds  # minx, miny, maxx, maxy
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
            map_var, zoom_ranges, target_var=target_var
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


class MultiHierarchyMap:
    """Combines several :class:`HierarchyMap` instances as mutually-exclusive Leaflet layers.

    Each named group is rendered as a ``folium.FeatureGroup(overlay=False)``,
    which ``folium.LayerControl`` renders as a radio-button group (Leaflet's
    ``baseLayer`` semantics: only one non-overlay layer is ever active at a
    time) -- giving the "select H3 or Polygons" toggle with no custom JS.
    """

    def __init__(
        self,
        named_hierarchy_maps: Dict[str, HierarchyMap],
        default: Optional[str] = None,
        basemap: Union[str, dict, Any] = "cartodb_positron",
        tiles_dir: Optional[str] = None,
    ):
        """
        Args:
            named_hierarchy_maps: Mapping of display name -> :class:`HierarchyMap`.
            default: Name of the group shown by default. Defaults to the
                first key.
            basemap: Shared basemap for the combined map.
            tiles_dir: If given, applied to every child :class:`HierarchyMap`
                that doesn't already have its own ``tiles_dir`` set
                (namespaced under a per-group subdirectory to avoid
                collisions between groups sharing level names).
        """
        if not named_hierarchy_maps:
            raise ValueError(
                "MultiHierarchyMap requires at least one named HierarchyMap"
            )
        self.named_hierarchy_maps = named_hierarchy_maps
        self.default = default or next(iter(named_hierarchy_maps))
        self.basemap = basemap
        self.tiles_dir = tiles_dir

        for group_name, hmap in named_hierarchy_maps.items():
            if hmap.tiles_dir is None and tiles_dir is not None:
                hmap.tiles_dir = str(Path(tiles_dir) / group_name)

    # ------------------------------------------------------------------
    def build(self) -> "MultiHierarchyMap":
        """Build tiles for every child :class:`HierarchyMap`."""
        for hmap in self.named_hierarchy_maps.values():
            hmap.build()
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
        gdf0 = first.hierarchy.get_level(first.levels[0])
        bounds = gdf0.total_bounds
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
            hmap._build_folium_map(existing_map=m, target=fg)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(path)
        return path
