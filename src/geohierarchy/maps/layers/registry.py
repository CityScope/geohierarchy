"""The layer-type registry itself: :class:`LayerTypeSpec` + register/get/list.

A "layer type" here is a *rendering* classification (how a level's tiles
turn into pixels: filled polygons, circle-markers, or stroked lines --
optionally with hover/click interactivity), not a data classification.
Two very different hierarchy levels -- H3 hexagons at some resolution and
census tracts -- are typically the *same* layer type (``"polygon"``)
because they render identically (filled area, popup on click); what
differs between them (color column, popup fields, zoom band) already lives
in :class:`~geohierarchy.maps.folium.render.MapLayer`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

LayerKind = Literal["polygon", "circle", "line"]


@dataclass(frozen=True)
class LayerTypeSpec:
    """A registered layer type.

    Attributes:
        name: Registry key, e.g. ``"hexagon"``, ``"circle"``,
            ``"census_polygon"``, ``"street_overlay"``,
            ``"development_overlay"``.
        kind: Geometry-rendering primitive -- see :data:`LayerKind`. Drives
            which MapLibre layer ``"type"`` is emitted
            (fill/circle/line) and which branch of
            :meth:`~geohierarchy.maps.folium.style.ColorSpec.maplibre_paint`
            / Leaflet vectorTileLayerStyles shape (``fillColor`` vs.
            ``radius`` vs. stroke-only) applies.
        interactive: Whether click/hover (popup + hover-highlight) should be
            wired up by default for this layer type. Overlay/annotation
            layers (e.g. a border-only development-priority overlay) often
            set this ``False``.
        default_paint: Optional MapLibre paint overrides merged UNDER
            whatever :meth:`ColorSpec.maplibre_paint` produces (e.g. a
            circle layer type's base radius). Renderer-supplied paint keys
            win.
        description: Human-readable summary, surfaced by
            :func:`list_layer_types` for discoverability.
    """

    name: str
    kind: LayerKind
    interactive: bool = True
    default_paint: Dict[str, object] = field(default_factory=dict)
    description: str = ""


LAYER_TYPES: Dict[str, LayerTypeSpec] = {}


def register_layer_type(
    spec: LayerTypeSpec, *, overwrite: bool = False
) -> LayerTypeSpec:
    """Register a new layer type.

    Args:
        spec: The :class:`LayerTypeSpec` to register.
        overwrite: If False (default), raises if ``spec.name`` is already
            registered -- catches accidental duplicate registration rather
            than silently shadowing an existing type.

    Returns:
        ``spec``, for convenient use as ``FOO = register_layer_type(...)``.
    """
    if not overwrite and spec.name in LAYER_TYPES:
        raise ValueError(
            f"Layer type '{spec.name}' is already registered "
            f"(pass overwrite=True to replace it)"
        )
    LAYER_TYPES[spec.name] = spec
    return spec


def get_layer_type(name: Optional[str]) -> LayerTypeSpec:
    """Look up a registered layer type by name.

    Args:
        name: Registry key, or ``None`` to get the default (``"polygon"``)
            -- the common case for most hierarchy levels (hexagons, census
            polygons, any other filled-area level).

    Returns:
        The matching :class:`LayerTypeSpec`.

    Raises:
        KeyError: If ``name`` isn't registered. Message lists what IS
            registered, since this is a config/typo error a caller should
            fix, not something to fall back silently from.
    """
    key = name or "polygon"
    try:
        return LAYER_TYPES[key]
    except KeyError:
        raise KeyError(
            f"Unknown layer type '{key}'. Registered types: "
            f"{sorted(LAYER_TYPES)}. Register new ones with "
            f"geohierarchy.maps.layers.register_layer_type()."
        ) from None


def list_layer_types() -> Dict[str, str]:
    """Return ``{name: description}`` for every registered layer type."""
    return {name: spec.description for name, spec in sorted(LAYER_TYPES.items())}


# ---------------------------------------------------------------------
# Built-in layer types, covering every rendering shape currently used by
# geohierarchy/transitLOS's Folium maps (see transitlos/map/build.py's
# `_polygon_style_js`, `_circle_style_js`, `_edges_style_js`,
# `_development_style_js`). New layer types (e.g. building footprints) are
# just another `register_layer_type(...)` call -- no renderer changes.
# ---------------------------------------------------------------------

register_layer_type(
    LayerTypeSpec(
        name="polygon",
        kind="polygon",
        interactive=True,
        description=(
            "Filled polygon area, popup on click. Default for any hierarchy "
            "level with area geometry: H3 hexagons (any resolution), census "
            "tracts/block groups/counties, or a generic geohierarchy level."
        ),
    )
)

register_layer_type(
    LayerTypeSpec(
        name="hexagon",
        kind="polygon",
        interactive=True,
        description="H3 hexagon-per-resolution level. Same rendering as 'polygon'; named separately for registry discoverability.",
    )
)

register_layer_type(
    LayerTypeSpec(
        name="census_polygon",
        kind="polygon",
        interactive=True,
        description="Census geography level (tract/block group/county/...). Same rendering as 'polygon'.",
    )
)

register_layer_type(
    LayerTypeSpec(
        name="circle",
        kind="circle",
        interactive=True,
        default_paint={"circle-radius": 6},
        description=(
            "Centroid points rendered as radius-scaled circles (radius driven "
            "client-side by a selectable numeric field)."
        ),
    )
)

register_layer_type(
    LayerTypeSpec(
        name="street_overlay",
        kind="line",
        interactive=False,
        description="Stroke-only line overlay (e.g. street/edge network), no fill, typically zoom-gated.",
    )
)

register_layer_type(
    LayerTypeSpec(
        name="development_overlay",
        kind="line",
        interactive=False,
        description="Stroke-only border overlay flagging a categorical condition (e.g. development priority), transparent otherwise.",
    )
)
