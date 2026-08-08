"""Basemap presets and helpers for building the underlying raster tile layer."""

from __future__ import annotations

from typing import Any, Dict, Union

MAP_MAX_ZOOM = 25

BASEMAPS: Dict[str, Dict[str, Any]] = {
    "cartodb_positron": {
        "tiles": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attr": (
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        "name": "CartoDB Positron",
        "max_native_zoom": 20,
        "subdomains": "abcd",
    },
    "cartodb_dark_matter": {
        "tiles": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attr": (
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        "name": "CartoDB Dark Matter",
        "max_native_zoom": 20,
        "subdomains": "abcd",
    },
    "osm": {
        "tiles": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        "name": "OpenStreetMap",
        "max_native_zoom": 19,
    },
    "google_roadmap": {
        "tiles": "https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "attr": "&copy; Google",
        "name": "Google Roadmap",
        "max_native_zoom": 20,
        "subdomains": "0123",
    },
    "google_satellite": {
        "tiles": "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attr": "&copy; Google",
        "name": "Google Satellite",
        "max_native_zoom": 20,
        "subdomains": "0123",
    },
    "google_hybrid": {
        "tiles": "https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "attr": "&copy; Google",
        "name": "Google Hybrid",
        "max_native_zoom": 20,
        "subdomains": "0123",
    },
    "google_terrain": {
        "tiles": "https://mt{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        "attr": "&copy; Google",
        "name": "Google Terrain",
        "max_native_zoom": 20,
        "subdomains": "0123",
    },
    "esri_satellite": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": (
            "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, "
            "and the GIS User Community"
        ),
        "name": "Esri World Imagery",
        "max_native_zoom": 19,
    },
}


def build_tile_layer(basemap: Union[str, dict, Any]):
    """Build a ``folium.TileLayer`` from a preset name, dict, or passthrough instance.

    Args:
        basemap: One of: a preset name (key of :data:`BASEMAPS`), a
            ``folium.TileLayer`` instance (returned unchanged), or a dict
            with ``tiles``/``attr``/``name``/``max_native_zoom`` (and
            optionally ``subdomains``) describing a custom XYZ provider.

    Returns:
        A ``folium.TileLayer`` configured with ``max_zoom=25`` on the map
        side and the provider's real ``max_native_zoom``, so Leaflet
        upscales tiles beyond what the provider natively serves.

    Raises:
        KeyError: If ``basemap`` is a string not present in ``BASEMAPS``.
        TypeError: If ``basemap`` is not a recognized type.
    """
    import folium

    if isinstance(basemap, folium.TileLayer):
        return basemap

    if isinstance(basemap, str):
        if basemap not in BASEMAPS:
            raise KeyError(
                f"Unknown basemap preset '{basemap}'. Available: {sorted(BASEMAPS)}"
            )
        spec = BASEMAPS[basemap]
    elif isinstance(basemap, dict):
        spec = basemap
    else:
        raise TypeError(
            f"basemap must be a preset name, dict, or folium.TileLayer; got {type(basemap)}"
        )

    kwargs = dict(
        tiles=spec["tiles"],
        attr=spec["attr"],
        name=spec.get("name", "basemap"),
        max_zoom=MAP_MAX_ZOOM,
        max_native_zoom=spec.get("max_native_zoom", 19),
        # control=False: Leaflet's layer control puts every non-overlay
        # ("base") layer into one shared radio group. If the basemap
        # registered here too, picking "H3"/"Polygons" in
        # MultiHierarchyMap's toggle -- itself built from non-overlay
        # FeatureGroups so it renders as radio buttons -- would silently
        # swap out the basemap along with it, since Leaflet treats them as
        # mutually exclusive alternatives. The basemap should always stay
        # on; it's just not one of the choices in that control.
        control=False,
        overlay=False,
    )
    if "subdomains" in spec:
        kwargs["subdomains"] = spec["subdomains"]

    return folium.TileLayer(**kwargs)
