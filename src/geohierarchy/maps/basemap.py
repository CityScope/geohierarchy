"""MapLibre vector-tile background maps for Folium maps.

Folium/Leaflet only draws raster tiles on its own. This module adds a
MapLibre GL vector-tile style (served by OpenFreeMap, no API key) as the
background of a Folium map, through the ``maplibre-gl-leaflet`` bridge:

    m = folium.Map(location=[42.37, -71.12], zoom_start=13, tiles=None)
    add_basemap(m, "positron")

Styles: https://openfreemap.org
"""

from __future__ import annotations

import folium

from .folium.maplibre_basemap import MAPLIBRE_CSS, MAPLIBRE_JS, MAPLIBRE_LEAFLET_JS, MapLibreBasemap  # noqa: F401  (one implementation)

BASEMAP_STYLES = {
    "positron": "https://tiles.openfreemap.org/styles/positron",  # light, minimal
    "bright": "https://tiles.openfreemap.org/styles/bright",
    "liberty": "https://tiles.openfreemap.org/styles/liberty",
    "dark": "https://tiles.openfreemap.org/styles/dark",
}
DEFAULT_BASEMAP = "positron"

ATTRIBUTION = (
    '&copy; <a href="https://openfreemap.org">OpenFreeMap</a> '
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)


def add_basemap(m: folium.Map, style: str = DEFAULT_BASEMAP) -> folium.Map:
    """Use a MapLibre vector-tile style as the background of a Folium map.

    Create the map with ``tiles=None`` so no raster background is added.

    Args:
        m: The Folium map.
        style: A key of :data:`BASEMAP_STYLES` (``"positron"``, ``"bright"``,
            ``"liberty"``, ``"dark"``) or the URL of any MapLibre style JSON.

    Returns:
        ``m``, for chaining.
    """
    MapLibreBasemap(BASEMAP_STYLES.get(style, style), attribution=ATTRIBUTION).add_to(m)
    return m
