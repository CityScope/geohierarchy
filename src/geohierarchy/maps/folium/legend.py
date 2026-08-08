"""Zoom-aware legend rendering: one small fixed-position box per configured layer.

Only the legend belonging to whichever level's zoom range contains the
current map zoom is shown, mirroring the vector-tile visibility rule (each
level's tiles only render within their own ``[minZoom, maxZoom]``).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from branca.element import MacroElement, Template


def _continuous_legend_html(div_id: str, title: str, colormap) -> str:
    vmin, vmax = colormap.vmin, colormap.vmax
    stops = []
    n = 10
    for i in range(n + 1):
        v = vmin + (vmax - vmin) * i / n
        stops.append(f"{colormap(v)} {100 * i / n:.1f}%")
    gradient = ", ".join(stops)
    return f"""
    <div id="{div_id}" class="hierarchy-legend" style="display:none;">
      <div style="font-weight:600;margin-bottom:4px;">{title}</div>
      <div style="width:140px;height:12px;background:linear-gradient(to right, {gradient});
                  border:1px solid #999;"></div>
      <div style="display:flex;justify-content:space-between;font-size:11px;margin-top:2px;">
        <span>{vmin:.3g}</span><span>{vmax:.3g}</span>
      </div>
    </div>
    """


def _categorical_legend_html(
    div_id: str, title: str, categories: Dict[str, str]
) -> str:
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="width:12px;height:12px;background:{color};display:inline-block;'
        f'margin-right:6px;border:1px solid #999;"></span>'
        f'<span style="font-size:11px;">{cat}</span></div>'
        for cat, color in categories.items()
    )
    return f"""
    <div id="{div_id}" class="hierarchy-legend" style="display:none;">
      <div style="font-weight:600;margin-bottom:4px;">{title}</div>
      {rows}
    </div>
    """


class HierarchyLegend(MacroElement):
    """A branca MacroElement rendering one fixed-position legend box per level.

    Each box is toggled via a ``map.on('zoomend', ...)`` listener so only
    the box whose level's ``[min_zoom, max_zoom]`` contains the current
    zoom is visible.
    """

    def __init__(
        self,
        entries: List[Tuple[str, str, Tuple[int, int]]],
        position: str = "bottom-right",
    ):
        """
        Args:
            entries: List of ``(div_id, html, (min_zoom, max_zoom))`` tuples,
                one per level that has a legend.
            position: CSS corner: "bottom-right", "bottom-left",
                "top-right", or "top-left".
        """
        super().__init__()
        self._name = "HierarchyLegend"
        self.entries = entries
        self.position = position

        pos_css = {
            "bottom-right": "bottom:20px;right:10px;",
            "bottom-left": "bottom:20px;left:10px;",
            "top-right": "top:80px;right:10px;",
            "top-left": "top:80px;left:10px;",
        }.get(position, "bottom:20px;right:10px;")

        boxes_html = "".join(html for _, html, _ in entries)
        ranges_js = json.dumps({div_id: list(zr) for div_id, _, zr in entries})
        container_id = f"hierarchy-legend-container-{id(self)}"

        self._template = Template(f"""
        {{% macro html(this, kwargs) %}}
        <div id="{container_id}" style="
            position:fixed; {pos_css} z-index:9999; background:white;
            padding:8px 10px; border-radius:4px; box-shadow:0 1px 4px rgba(0,0,0,0.4);
            font-family:Arial, sans-serif;">
          {boxes_html}
        </div>
        {{% endmacro %}}

        {{% macro script(this, kwargs) %}}
        (function() {{
          var map = {{{{this._parent.get_name()}}}};
          var ranges = {ranges_js};
          function updateLegend() {{
            var z = map.getZoom();
            Object.keys(ranges).forEach(function(id) {{
              var el = document.getElementById(id);
              if (!el) return;
              var r = ranges[id];
              el.style.display = (z >= r[0] && z <= r[1]) ? 'block' : 'none';
            }});
          }}
          map.on('zoomend', updateLegend);
          map.whenReady(updateLegend);
        }})();
        {{% endmacro %}}
        """)


def build_legend(
    layers: Dict[str, "object"],
    resolutions: Dict[str, Tuple[int, int]],
    gdfs: Optional[Dict] = None,
) -> Optional[HierarchyLegend]:
    """Build a :class:`HierarchyLegend` from a mapping of level name -> ``MapLayer``.

    Args:
        layers: Mapping of level name to :class:`~geohierarchy.maps.folium.render.MapLayer`.
        resolutions: Zoom-range assignment per level, from :func:`assign_resolutions`.
        gdfs: Optional mapping of level name to its GeoDataFrame, used to
            infer a ``ColorSpec``'s domain/categories when not given
            explicitly.

    Returns:
        A :class:`HierarchyLegend`, or ``None`` if no layer has legend
        content (no ``ColorSpec``/``legend_html``).
    """
    entries = []
    for i, (level_name, layer) in enumerate(layers.items()):
        if level_name not in resolutions:
            continue
        zr = resolutions[level_name]
        div_id = f"legend_{level_name}"

        html = None
        if getattr(layer, "legend_html", None):
            html = f'<div id="{div_id}" class="hierarchy-legend" style="display:none;">{layer.legend_html}</div>'
        elif getattr(layer, "style", None) is not None:
            spec = layer.style
            gdf = (gdfs or {}).get(level_name)
            if spec.kind == "categorical":
                cats = spec._resolved_categories(gdf)
                html = _categorical_legend_html(div_id, spec.column, cats)
            else:
                colormap = spec.branca_colormap(gdf)
                html = _continuous_legend_html(div_id, spec.column, colormap)

        if html is not None:
            entries.append((div_id, html, zr))

    if not entries:
        return None
    return HierarchyLegend(entries)
