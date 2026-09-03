"""Declarative styling for vector-tile layers.

Leaflet.VectorGrid styles tiles client-side via a ``vectorTileLayerStyles``
callback that receives each feature's ``properties``, so a Python-side
``style`` can't be a plain callable -- it has to be codegen'd into a small
JS snippet. :class:`ColorSpec` is the declarative way to build that
snippet (and the matching branca colormap for the legend); a raw
``style_js`` string bypasses it entirely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class ColorSpec:
    """Declarative color/style rule for one hierarchy level's vector tiles.

    Attributes:
        column: Feature property to color by.
        cmap: Name of a branca/matplotlib colormap (continuous) or a list
            of hex colors (categorical, cycled/keyed by sorted category).
        domain: ``(min, max)`` for continuous data. Computed from the data
            by the caller if not given.
        categories: Explicit category -> color mapping for
            ``kind="categorical"``. If omitted, categories are assigned
            colors from ``cmap`` in sorted order.
        kind: ``"continuous"`` (branca linear colormap) or
            ``"categorical"`` (discrete swatches).
        opacity: Fill opacity.
        weight: Stroke weight (px).
        color: Stroke color. Defaults to the fill color if not given.
        fill: Whether polygons/points are filled.
        nodata_color: Color used when ``column`` is missing/null on a
            feature.
    """

    column: str
    cmap: Any = "viridis"
    domain: Optional[Tuple[float, float]] = None
    categories: Optional[Dict[str, str]] = None
    kind: Literal["continuous", "categorical"] = "continuous"
    opacity: float = 0.7
    weight: float = 1.0
    color: Optional[str] = None
    fill: bool = True
    nodata_color: str = "#888888"

    # ------------------------------------------------------------------
    def branca_colormap(self, gdf=None):
        """Build the matching :mod:`branca.colormap` object.

        Args:
            gdf: Optional GeoDataFrame/DataFrame used to infer ``domain``
                from ``self.column`` when it wasn't given explicitly.

        Returns:
            A ``branca.colormap.ColorMap`` instance (linear for
            continuous, step for categorical).
        """
        import branca.colormap as bcm

        if self.kind == "categorical":
            cats = self._resolved_categories(gdf)
            colors = list(cats.values())
            index = list(range(len(colors) + 1))
            colormap = bcm.StepColormap(
                colors, index=index, vmin=0, vmax=len(colors), caption=self.column
            )
            colormap.tick_labels = list(cats.keys())
            return colormap

        vmin, vmax = self._resolved_domain(gdf)
        cmap_name = self.cmap if isinstance(self.cmap, str) else "viridis"
        # Try the exact name first (branca's colorbrewer-derived names are mixed-case,
        # e.g. "RdYlGn_09" -- `.capitalize()` would mangle that to "Rdylgn_09" and never
        # match, silently falling back to viridis for every non-single-word cmap name).
        base = getattr(bcm.linear, cmap_name, None)
        if base is None:
            base = getattr(bcm.linear, cmap_name.capitalize(), None)
        if base is None:
            base = bcm.linear.viridis
        colormap = base.scale(vmin, vmax)
        colormap.caption = self.column
        return colormap

    def _resolved_domain(self, gdf=None) -> Tuple[float, float]:
        if self.domain is not None:
            return self.domain
        if gdf is not None and self.column in getattr(gdf, "columns", []):
            series = gdf[self.column].dropna()
            if len(series) > 0:
                return float(series.min()), float(series.max())
        return (0.0, 1.0)

    def _resolved_categories(self, gdf=None) -> Dict[str, str]:
        if self.categories is not None:
            return self.categories
        palette = (
            self.cmap
            if isinstance(self.cmap, (list, tuple))
            else _DEFAULT_CATEGORICAL_PALETTE
        )
        cats = []
        if gdf is not None and self.column in getattr(gdf, "columns", []):
            cats = sorted({str(v) for v in gdf[self.column].dropna().unique()})
        if not cats:
            cats = ["_default"]
        return {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}

    # ------------------------------------------------------------------
    def style_js(self, gdf=None, var_name: str = "style") -> str:
        """Generate the JS style function body for this spec.

        Returns:
            JS source defining ``function {var_name}(properties) {{ ... }}``
            returning a Leaflet path-style object, suitable for use inside
            a ``vectorTileLayerStyles`` map.
        """
        stroke_color = self.color or "'#333333'"
        if self.color:
            stroke_color = json.dumps(self.color)

        if self.kind == "categorical":
            cats = self._resolved_categories(gdf)
            lookup = json.dumps(cats)
            return (
                f"function {var_name}(properties) {{\n"
                f"  var lookup = {lookup};\n"
                f"  var key = String(properties[{json.dumps(self.column)}]);\n"
                f"  var fill = lookup.hasOwnProperty(key) ? lookup[key] : {json.dumps(self.nodata_color)};\n"
                f"  return {{color: {stroke_color}, weight: {self.weight}, "
                f"fill: {str(self.fill).lower()}, fillColor: fill, fillOpacity: {self.opacity}}};\n"
                f"}}"
            )

        vmin, vmax = self._resolved_domain(gdf)
        colormap = self.branca_colormap(gdf)
        # Sample the colormap into a lookup table the browser can
        # interpolate over without needing branca/matplotlib client-side.
        n_samples = 32
        samples = []
        for i in range(n_samples + 1):
            v = vmin + (vmax - vmin) * i / n_samples
            samples.append(colormap(v))
        samples_json = json.dumps(samples)
        return (
            f"function {var_name}(properties) {{\n"
            f"  var v = properties[{json.dumps(self.column)}];\n"
            f"  var samples = {samples_json};\n"
            f"  var vmin = {vmin}, vmax = {vmax};\n"
            f"  var fill = {json.dumps(self.nodata_color)};\n"
            f"  if (v !== null && v !== undefined && !isNaN(v)) {{\n"
            f"    var t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0;\n"
            f"    t = Math.max(0, Math.min(1, t));\n"
            f"    var idx = Math.round(t * (samples.length - 1));\n"
            f"    fill = samples[idx];\n"
            f"  }}\n"
            f"  return {{color: {stroke_color}, weight: {self.weight}, "
            f"fill: {str(self.fill).lower()}, fillColor: fill, fillOpacity: {self.opacity}}};\n"
            f"}}"
        )

    # ------------------------------------------------------------------
    def maplibre_fill_color_expr(self, gdf=None) -> Any:
        """Build a MapLibre GL style expression for this spec's fill color.

        Mirrors :meth:`style_js` (same sampling/lookup logic) but targets
        MapLibre's declarative expression language instead of a Leaflet.
        VectorGrid callback, so a Folium-vector-tile level and a MapLibre
        PMTiles layer sourced from the *same* tiles reach the same visual
        result -- this is the piece that lets
        :mod:`geohierarchy.maps.maplibre` reach base-layer parity with
        :mod:`geohierarchy.maps.folium` without re-deriving color logic.

        Returns:
            A MapLibre expression (JSON-serializable list), usable directly
            as a ``"fill-color"``/``"circle-color"``/``"line-color"`` paint
            value.
        """
        if self.kind == "categorical":
            cats = self._resolved_categories(gdf)
            expr: List[Any] = ["match", ["to-string", ["get", self.column]]]
            for cat, color in cats.items():
                expr.extend([cat, color])
            expr.append(self.nodata_color)
            return expr

        vmin, vmax = self._resolved_domain(gdf)
        colormap = self.branca_colormap(gdf)
        n_samples = 16
        stops: List[Any] = []
        for i in range(n_samples + 1):
            v = vmin + (vmax - vmin) * i / n_samples
            stops.extend([v, colormap(v)])
        return [
            "case",
            ["==", ["typeof", ["get", self.column]], "number"],
            ["interpolate", ["linear"], ["get", self.column], *stops],
            self.nodata_color,
        ]

    def maplibre_paint(self, kind: str = "polygon", gdf=None) -> Dict[str, Any]:
        """Build a full MapLibre ``paint`` dict for this spec.

        Args:
            kind: One of ``"polygon"``, ``"circle"``, ``"line"`` -- the
                geometry-rendering primitive, matching
                :class:`geohierarchy.maps.layers.registry.LayerKind`.
            gdf: Optional data used to infer domain/categories (see
                :meth:`maplibre_fill_color_expr`).
        """
        color_expr = self.maplibre_fill_color_expr(gdf)
        stroke = self.color or "#333333"
        if kind == "circle":
            return {
                "circle-color": color_expr,
                "circle-opacity": self.opacity,
                "circle-stroke-color": stroke,
                "circle-stroke-width": self.weight,
                "circle-radius": 6,
            }
        if kind == "line":
            return {
                "line-color": color_expr,
                "line-width": self.weight,
                "line-opacity": self.opacity,
            }
        # polygon (default)
        paint: Dict[str, Any] = {}
        if self.fill:
            paint["fill-color"] = color_expr
            paint["fill-opacity"] = self.opacity
        paint["fill-outline-color"] = stroke
        return paint


_DEFAULT_CATEGORICAL_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def build_vector_tile_layer_styles_js(layer_name: str, style_js_body: str) -> str:
    """Wrap a per-layer style function into a ``vectorTileLayerStyles`` entry.

    Args:
        layer_name: Vector tile layer name (matches the layer name used
            when encoding tiles in :mod:`tiles.py`, typically the level
            name).
        style_js_body: JS source produced by :meth:`ColorSpec.style_js` or
            a user-supplied raw function body.

    Returns:
        A JS object-literal fragment: ``"<layer_name>": <function>``.
    """
    return f"{json.dumps(layer_name)}: {style_js_body}"
