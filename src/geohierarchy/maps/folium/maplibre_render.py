"""MapLibre GL JS rendering for PMTiles and H3J/H3T formats."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

MAPLIBRE_CDN = "https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.js"
PMTILES_BUNDLE_CDN = "https://unpkg.com/pmtiles@4.5.0/dist/pmtiles.js"
H3J_H3T_CDN = (
    "https://unpkg.com/@maplibre/maplibre-gl-geo-h3j-h3t@latest/dist/h3j_h3t.js"
)

BASEMAP_STYLES = {
    "positron": {
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
    },
    "dark_matter": {
        "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
    },
    "osm": {
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "&copy; OpenStreetMap contributors",
    },
}


def _get_basemap(basemap: str) -> Dict[str, str]:
    if isinstance(basemap, str) and basemap in BASEMAP_STYLES:
        return BASEMAP_STYLES[basemap]
    return basemap


def _js_value(value: Any) -> str:
    """Convert a Python value to a JavaScript-compatible JSON string."""
    # Use json.dumps which produces valid JavaScript object notation
    # with proper escaping of quotes and special characters
    return json.dumps(value, ensure_ascii=False)


def generate_maplibre_html(
    output_path: Union[str, Path],
    sources: Dict[str, Dict[str, Any]],
    layers: List[Dict[str, Any]],
    basemap: str = "positron",
    center: Optional[Sequence[float]] = None,
    zoom: int = 12,
    title: str = "Map",
    include_h3j_h3t: bool = False,
    include_pmtiles: bool = False,
    legend_html: Optional[str] = None,
    extra_css: Optional[str] = None,
    extra_js: Optional[str] = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    basemap = _get_basemap(basemap)
    all_sources = {
        "basemap": {
            "type": "raster",
            "tiles": [basemap["url"]],
            "attribution": basemap["attribution"],
        },
        **sources,
    }

    # Build style as JavaScript object, not JSON string
    style_obj = {"version": 8, "sources": all_sources, "layers": layers}

    if center is None:
        center = [-98.58, 39.83]

    html_parts = [
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n',
        '    <meta charset="UTF-8">\n',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n',
        f"    <title>{title}</title>\n",
        # Load MapLibre as UMD (sets global maplibregl)
        f'    <script src="{MAPLIBRE_CDN}"></script>\n',
    ]

    # Load PMTiles bundled version (sets global pmtiles namespace)
    if include_pmtiles:
        html_parts.append(f'    <script src="{PMTILES_BUNDLE_CDN}"></script>\n')

    if include_h3j_h3t:
        html_parts.append(f'    <script src="{H3J_H3T_CDN}"></script>\n')

    html_parts.append("    <style>\n")
    html_parts.append("        * { margin: 0; padding: 0; box-sizing: border-box; }\n")
    html_parts.append(
        "        body, html { width: 100%; height: 100%; overflow: hidden; }\n"
    )
    html_parts.append("        #map { width: 100%; height: 100%; }\n")
    if extra_css:
        html_parts.append(f"        {extra_css}\n")
    html_parts.append("    </style>\n")
    html_parts.append("</head>\n<body>\n")
    html_parts.append('    <div id="map"></div>\n')
    if legend_html:
        html_parts.append(f'    <div id="legend">{legend_html}</div>\n')

    # Initialization script - all regular scripts now
    html_parts.append("    <script>\n")

    if include_pmtiles:
        html_parts.append("        // Register PMTiles protocol\n")
        html_parts.append("        let protocol = new pmtiles.Protocol();\n")
        html_parts.append('        maplibregl.addProtocol("pmtiles", protocol.tile);\n')

    if include_h3j_h3t:
        html_parts.append("        // Register H3J/H3T protocol\n")
        html_parts.append("        if (window.h3jH3t) {\n")
        html_parts.append(
            '            maplibregl.addProtocol("h3tiles", window.h3jH3t.protocol);\n'
        )
        html_parts.append("        }\n")

    # Build style as JS object inline
    style_js = _js_value(style_obj)
    center_js = _js_value(center)

    html_parts.append("        const map = new maplibregl.Map({\n")
    html_parts.append("            container: 'map',\n")
    html_parts.append(f"            style: {style_js},\n")
    html_parts.append(f"            center: {center_js},\n")
    html_parts.append(f"            zoom: {zoom},\n")
    html_parts.append("            maxZoom: 25\n")
    html_parts.append("        });\n")
    html_parts.append("        // Add navigation controls\n")
    html_parts.append(
        "        map.addControl(new maplibregl.NavigationControl(), 'top-right');\n"
    )

    if extra_js:
        html_parts.append(f"        {extra_js}\n")

    html_parts.append("    </script>\n")
    html_parts.append("</body>\n</html>\n")

    with open(output_path, "w") as f:
        f.write("".join(html_parts))

    return output_path


def generate_mixed_html(
    pmtiles_paths: Optional[Dict[str, Path]] = None,
    h3j_paths: Optional[Dict[str, Path]] = None,
    h3t_dirs: Optional[Dict[str, Path]] = None,
    output_path: Union[str, Path] = "map.html",
    layer_styles: Optional[Dict[str, Dict[str, Any]]] = None,
    zoom_ranges: Optional[Dict[str, tuple]] = None,
    basemap: str = "positron",
    center: Optional[Sequence[float]] = None,
    zoom: int = 12,
    title: str = "Map",
    **kwargs: Any,
) -> Path:
    all_sources = {}
    all_layers = []
    extra_js_parts = []

    include_h3j_h3t = bool(h3j_paths or h3t_dirs)
    include_pmtiles = bool(pmtiles_paths)

    # Process PMTiles
    if pmtiles_paths:
        for layer_name, pmtiles_path in pmtiles_paths.items():
            rel_path = (
                str(pmtiles_path.relative_to(output_path.parent))
                if pmtiles_path.is_absolute()
                else str(pmtiles_path)
            )
            all_sources[layer_name] = {"type": "vector", "url": f"pmtiles://{rel_path}"}

            layer_style = layer_styles.get(layer_name, {}) if layer_styles else {}
            # Get layer type from style or default to fill
            layer_type = layer_style.get("type", "fill")
            layer_style.setdefault("type", layer_type)
            layer_style.setdefault("source", layer_name)
            # For PMTiles vector sources, source-layer must match the layer name in the PMTiles
            layer_style.setdefault("source-layer", layer_name)
            # Each layer needs an id
            layer_style.setdefault("id", f"{layer_name}_{layer_type}")
            # Remove unsupported properties for MapLibre GL JS
            if "paint" in layer_style and isinstance(layer_style["paint"], dict):
                layer_style["paint"].pop("fill-outline-width", None)
            layer_style.setdefault(
                "paint", {"fill-color": "#3388ff", "fill-opacity": 0.4}
            )

            if zoom_ranges and layer_name in zoom_ranges:
                layer_style["minzoom"] = zoom_ranges[layer_name][0]
                layer_style["maxzoom"] = zoom_ranges[layer_name][1]

            all_layers.append(layer_style)

    # Process H3J
    if h3j_paths:
        for layer_name, h3j_path in h3j_paths.items():
            rel_path = (
                str(h3j_path.relative_to(output_path.parent))
                if h3j_path.is_absolute()
                else str(h3j_path)
            )
            all_sources[layer_name] = {"type": "geojson", "data": rel_path}

            layer_style = layer_styles.get(layer_name, {}) if layer_styles else {}
            layer_style.setdefault("type", "fill")
            layer_style.setdefault("source", layer_name)
            layer_style.setdefault(
                "paint", {"fill-color": "#3388ff", "fill-opacity": 0.4}
            )

            if zoom_ranges and layer_name in zoom_ranges:
                layer_style["minzoom"] = zoom_ranges[layer_name][0]
                layer_style["maxzoom"] = zoom_ranges[layer_name][1]

            all_layers.append(layer_style)
            extra_js_parts.append(
                f"            map.addH3JSource('{layer_name}', {{data: '{rel_path}', geometry_type: 'Polygon', promoteId: true}});\n"
            )

    # Process H3T
    if h3t_dirs:
        for layer_name, h3t_dir in h3t_dirs.items():
            all_sources[layer_name] = {
                "type": "vector",
                "tiles": [f"h3tiles://{h3t_dir}/{{z}}/{{x}}/{{y}}.h3t"],
                "minzoom": 0,
                "maxzoom": 15,
            }

            layer_style = layer_styles.get(layer_name, {}) if layer_styles else {}
            layer_style.setdefault("type", "fill")
            layer_style.setdefault("source", layer_name)
            layer_style.setdefault(
                "paint", {"fill-color": "#3388ff", "fill-opacity": 0.4}
            )

            if zoom_ranges and layer_name in zoom_ranges:
                layer_style["minzoom"] = zoom_ranges[layer_name][0]
                layer_style["maxzoom"] = zoom_ranges[layer_name][1]

            all_layers.append(layer_style)
            extra_js_parts.append(
                f"            map.addH3TSource('{layer_name}', {{tiles: ['h3tiles://{h3t_dir}/{{z}}/{{x}}/{{y}}.h3t'], geometry_type: 'Polygon'}});\n"
            )

    extra_js = "\n".join(extra_js_parts) if extra_js_parts else None

    return generate_maplibre_html(
        output_path,
        sources=all_sources,
        layers=all_layers,
        basemap=basemap,
        center=center,
        zoom=zoom,
        title=title,
        include_h3j_h3t=include_h3j_h3t,
        include_pmtiles=include_pmtiles,
        extra_js=extra_js,
        **kwargs,
    )
