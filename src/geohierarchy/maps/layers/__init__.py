"""Modular layer-type registry, shared by the Folium and MapLibre renderers.

Every level a :class:`~geohierarchy.maps.folium.render.HierarchyMap` renders
(hexagon-per-resolution, circle, census polygon, street/development overlay,
...) reduces to one of a small number of geometry-rendering *primitives*
("kinds") -- filled polygons, radius-scaled circle points, or stroked lines
-- styled by a :class:`~geohierarchy.maps.folium.style.ColorSpec`. This
module names that primitive-per-level-type mapping as an explicit registry
(:data:`LAYER_TYPES`) instead of leaving it implicit/duplicated in each
renderer, so:

* Folium (Leaflet.VectorGrid) and MapLibre (PMTiles) renderers can both
  consume the same registry entry for a given layer type and stay visually
  consistent without hand-syncing style logic in two places.
* Adding a genuinely new layer type (e.g. building footprints) is a
  one-time :func:`register_layer_type` call, not a rewrite of either
  renderer.

See :mod:`geohierarchy.maps.maplibre.render` for the consumer that turns a
registry entry + a :class:`~geohierarchy.maps.folium.render.MapLayer` into
an actual MapLibre GL source/layer pair.
"""

from __future__ import annotations

from .registry import (
    LayerKind,
    LayerTypeSpec,
    register_layer_type,
    get_layer_type,
    list_layer_types,
    LAYER_TYPES,
)

__all__ = [
    "LayerKind",
    "LayerTypeSpec",
    "register_layer_type",
    "get_layer_type",
    "list_layer_types",
    "LAYER_TYPES",
]
