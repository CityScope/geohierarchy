"""PMTiles rendering support for HierarchyMap with Leaflet + @protomaps/leaflet-pmtiles.

This module provides PMTiles-compatible rendering that preserves all interactive
features from the original XYZ tile implementation while using single .pmtiles files.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# PMTiles Leaflet plugin CDN
PMTiles_Leaflet_CDN = (
    "https://unpkg.com/@protomaps/leaflet-pmtiles@latest/dist/leaflet-pmtiles.min.js"
)


def build_pmtiles_layer_js(
    pmtiles_url: str,
    layer_name: str,
    min_zoom: int,
    max_zoom: int,
    min_native_zoom: int,
    max_native_zoom: int,
    style_js: str,
    popup_js: str,
    tooltip_js: Optional[str] = None,
) -> str:
    """Generate JavaScript for a PMTiles vector layer.

    This creates a Leaflet layer using @protomaps/leaflet-pmtiles that provides
    the same vector styling and interactivity as the original Leaflet.VectorGrid.

    Args:
        pmtiles_url: URL to the .pmtiles file (relative or absolute)
        layer_name: Internal layer name for this source
        min_zoom: Minimum display zoom
        max_zoom: Maximum display zoom
        min_native_zoom: Minimum zoom where tiles are natively rendered
        max_native_zoom: Maximum zoom where tiles are natively rendered
        style_js: JavaScript function for styling features
        popup_js: JavaScript function for popup content
        tooltip_js: Optional JavaScript function for tooltip content

    Returns:
        JavaScript code to create and configure the PMTiles layer
    """
    layer_var = f"vg_{layer_name.replace('-', '_').replace('.', '_')}"

    js = f"""
// Load PMTiles layer for {layer_name}
var {layer_var} = new L.PMTilesVector("{pmtiles_url}", {{
    minZoom: {min_zoom},
    maxZoom: {max_zoom},
    minNativeZoom: {min_native_zoom},
    maxNativeZoom: {max_native_zoom},
    layerName: "{layer_name}",
    rendererFactory: L.canvas.tile,
    vectorTileLayerStyles: {style_js},
    interactive: true,
    getColor: undefined,
    getPopupContent: {popup_js},
"""

    if tooltip_js:
        js += f"    getTooltipContent: {tooltip_js},"

    js += "\n});\n"

    return js


def build_pmtiles_hierarchy_map_js(
    pmtiles_layers: Dict[str, Dict[str, Any]],
    tile_urls: Dict[str, str],
    default_shape: str,
    basemap_url: str,
    basemap_attribution: str,
    initial_view: Dict[str, Any],
    extra_js: str = "",
) -> str:
    """Build complete JavaScript for a PMTiles-based HierarchyMap.

    Args:
        pmtiles_layers: Dictionary of layer configs (same format as HierarchyMap._layers)
        tile_urls: Mapping of level_name -> PMTiles file URL
        default_shape: Name of the default shape layer to show
        basemap_url: Base map tile URL template
        basemap_attribution: Base map attribution
        initial_view: Dict with location, zoom_start, etc.
        extra_js: Additional JavaScript to include

    Returns:
        Complete JavaScript code for the map
    """
    # Build layer initialization JS
    layer_js_parts = []
    layer_vars = []

    for level_name, config in pmtiles_layers.items():
        url = tile_urls.get(level_name, "")
        if not url:
            continue

        style_js = config.get("style_js", "function(properties) { return {}; }")
        popup_js = config.get("popup_js", "function(properties) { return ''; }")
        tooltip_js = config.get("tooltip_js", None)
        min_zoom = config.get("min_zoom", 0)
        max_zoom = config.get("max_zoom", 25)
        min_native_zoom = config.get("min_native_zoom", min_zoom)
        max_native_zoom = config.get("max_native_zoom", max_zoom)

        layer_js = build_pmtiles_layer_js(
            url,
            level_name,
            min_zoom,
            max_zoom,
            min_native_zoom,
            max_native_zoom,
            style_js,
            popup_js,
            tooltip_js,
        )
        layer_js_parts.append(layer_js)
        layer_vars.append(f"{level_name.replace('-', '_').replace('.', '_')}")

    # Build the map initialization
    map_init = f"""
var map = L.map('map', {{
    center: [{initial_view.get('location', [0, 0])[1]}, {initial_view.get('location', [0, 0])[0]}],
    zoom: {initial_view.get('zoom_start', 12)},
    minZoom: {initial_view.get('min_zoom', 0)},
    maxZoom: {initial_view.get('max_zoom', 25)},
}});

// Add basemap
L.tileLayer('{basemap_url}', {{
    attribution: '{basemap_attribution}'
}}).addTo(map);

"""

    # Layer control setup - group shape layers together
    shape_layers_js = ""
    if len(layer_vars) > 1:
        shape_layers_js = f"""
// Create layer groups for mutually exclusive shape layers
var shapeLayers = {{}};
{''.join(f"shapeLayers['{ln}'] = {lv};\n" for ln, lv in zip(pmtiles_layers.keys(), layer_vars))}

// Add only the default shape layer initially
shapeLayers['{default_shape}'].addTo(map);

// Add layer control
L.control.layers(null, shapeLayers, {{
    collapsed: false
}}).addTo(map);
"""
    else:
        # Single layer, just add it
        shape_layers_js = f"{layer_vars[0]}.addTo(map);\n"

    all_js = map_init + extra_js + "\n".join(layer_js_parts) + "\n" + shape_layers_js

    return all_js


def build_pmtiles_html(
    pmtiles_layers: Dict[str, Dict[str, Any]],
    tile_urls: Dict[str, str],
    default_shape: str,
    basemap_url: str,
    basemap_attribution: str,
    initial_view: Dict[str, Any],
    html_id: str = "map",
    width: str = "100%",
    height: str = "100%",
    extra_js: str = "",
    extra_css: str = "",
    extra_html: str = "",
) -> str:
    """Build complete HTML for a PMTiles-based map.

    Args:
        pmtiles_layers: Dictionary of layer configs
        tile_urls: Mapping of level_name -> PMTiles file URL
        default_shape: Default shape layer name
        basemap_url: Base map URL template
        basemap_attribution: Base map attribution
        initial_view: Map view settings
        html_id: HTML element ID for the map
        width: CSS width for map container
        height: CSS height for map container
        extra_js: Additional JavaScript
        extra_css: Additional CSS
        extra_html: Additional HTML

    Returns:
        Complete HTML document
    """
    map_js = build_pmtiles_hierarchy_map_js(
        pmtiles_layers,
        tile_urls,
        default_shape,
        basemap_url,
        basemap_attribution,
        initial_view,
        extra_js,
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>PMTiles Map</title>

    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css" />

    <!-- Leaflet JS -->
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>

    <!-- PMTiles Leaflet plugin -->
    <script src="{PMTiles_Leaflet_CDN}"></script>

    <!-- jQuery for UI -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>

    <!-- Bootstrap for styling -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css" />
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js"></script>

    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css" />

    <style>
        html, body {{ width: 100%; height: 100%; margin: 0; padding: 0; }}
        #map {{ width: {width}; height: {height}; }}
        {extra_css}
    </style>

    <script>
        // Fix for Leaflet.VectorGrid click events
        if (window.L && L.DomEvent && !L.DomEvent.fakeStop) {{
            L.DomEvent.fakeStop = L.DomEvent._fakeStop || function(e) {{ e._stopped = true; }};
        }}

        {map_js}

        {extra_js}
    </script>
</head>
<body>
    <div id="{html_id}"></div>
    {extra_html}
</body>
</html>
"""
    return html
