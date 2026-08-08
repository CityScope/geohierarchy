"""geohierarchy.maps: interactive map rendering built on top of GeoHierarchy.

Only lazily imports heavy optional dependencies (folium, branca, mercantile,
mapbox-vector-tile) via the ``folium`` subpackage -- plain ``import
geohierarchy`` never requires them.
"""

from . import folium  # noqa: F401

__all__ = ["folium"]
