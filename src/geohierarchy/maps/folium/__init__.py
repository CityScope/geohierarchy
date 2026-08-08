"""geohierarchy.maps.folium: interactive multi-resolution Leaflet maps for a GeoHierarchy.

Example:
    >>> from geohierarchy.maps.folium import HierarchyMap, ColorSpec
    >>> m = HierarchyMap(hierarchy, basemap="cartodb_positron", tiles_dir="tiles")
    >>> m.configure_level("tract", style=ColorSpec(column="population", cmap="viridis"))
    >>> m.build()
    >>> m.save("map.html")

Nothing here is imported by plain ``import geohierarchy`` -- folium,
branca, mercantile, and mapbox_vector_tile are only required once this
subpackage itself is imported.
"""

from .render import HierarchyMap, MultiHierarchyMap, MapLayer
from .style import ColorSpec
from .resolution import assign_resolutions
from .basemap import BASEMAPS

__all__ = [
    "HierarchyMap",
    "MultiHierarchyMap",
    "MapLayer",
    "ColorSpec",
    "assign_resolutions",
    "BASEMAPS",
]
