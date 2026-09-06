"""MapLibreHierarchyMap: render a HierarchyMap's PMTiles via MapLibre GL JS.

This is the MapLibre counterpart to
:class:`geohierarchy.maps.folium.render.HierarchyMap`, targeting *base-layer
parity* with it: same zoom-banded levels, same
:class:`~geohierarchy.maps.folium.style.ColorSpec` coloring, same
click-popup/hover interactivity -- sourced from the exact same per-level
``.pmtiles`` files ``HierarchyMap.build()`` already writes (no separate
tile-generation step). It deliberately does NOT attempt to port the full
Folium scenario editor or stats panel (route drawing, grade-separation
paint mode, Distribution/Regression/ANOVA tabs, live
``window.__h3AccessOverrides`` recolor) -- Folium keeps sole ownership of
that until a later pass; see ``transitlos/map/build.py``.

Which MapLibre layer *type* (fill/circle/line) a given hierarchy level
gets, and whether it's interactive, comes from the modular registry in
:mod:`geohierarchy.maps.layers` via each level's
``MapLayer.layer_type`` -- adding a new layer type there (e.g. building
footprints) does not require touching this renderer.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..layers import get_layer_type
from ..folium.render import HierarchyMap, MAX_NATIVE_TILE_ZOOM

MAPLIBRE_CDN = "https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.js"
MAPLIBRE_CSS_CDN = "https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.css"
PMTILES_CDN = "https://unpkg.com/pmtiles@4.5.0/dist/pmtiles.js"

# CartoDB Positron/Dark Matter were replaced by the OpenFreeMap equivalents
# below (`openfreemap_positron`/`openfreemap_dark` -- same light/dark niche,
# no API key, no signup, OSM-based, MapLibre-native vector styles instead of
# a third-party raster tile CDN). `openstreetmap`/`esri_imagery` stay as
# raster options; `openfreemap_bright`/`openfreemap_liberty` and the three
# Google raster sources are new additions.
#
# Two distinct shapes live in this dict, both tagged with an explicit
# "type":
#   - "raster": the original shape -- a raster XYZ `tiles` URL template
#     list + `attribution`, swapped in as a single `type: 'raster'`
#     source/layer pair (see `_client_basemap_js` below).
#   - "vector-style": an OpenFreeMap-style full MapLibre style document,
#     identified only by its `url` (fetched client-side at basemap-switch
#     time -- see `_client_basemap_js`'s docstring for why this isn't
#     fetched at build time) + a manually-supplied `attribution` (OpenFreeMap
#     style JSON does not embed a `sources[...].attribution` field the way
#     CARTO/OSM raster tiles do, so this is hardcoded to OpenFreeMap's
#     requested attribution text instead of being read out of the style).
BASEMAP_STYLES = {
    "openfreemap_positron": {
        "type": "vector-style",
        "url": "https://tiles.openfreemap.org/styles/positron",
        "attribution": (
            '&copy; <a href="https://openfreemap.org">OpenFreeMap</a> '
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        ),
    },
    "openfreemap_bright": {
        "type": "vector-style",
        "url": "https://tiles.openfreemap.org/styles/bright",
        "attribution": (
            '&copy; <a href="https://openfreemap.org">OpenFreeMap</a> '
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        ),
    },
    "openfreemap_liberty": {
        "type": "vector-style",
        "url": "https://tiles.openfreemap.org/styles/liberty",
        "attribution": (
            '&copy; <a href="https://openfreemap.org">OpenFreeMap</a> '
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        ),
    },
    "openfreemap_dark": {
        "type": "vector-style",
        "url": "https://tiles.openfreemap.org/styles/dark",
        "attribution": (
            '&copy; <a href="https://openfreemap.org">OpenFreeMap</a> '
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        ),
    },
    "openstreetmap": {
        "type": "raster",
        "tiles": [
            "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
        ],
        "attribution": "&copy; OpenStreetMap contributors",
    },
    "esri_imagery": {
        "type": "raster",
        "tiles": [
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ],
        "attribution": "Esri, Maxar, Earthstar Geographics",
    },
    "google_roadmap": {
        "type": "raster",
        "tiles": [
            f"https://mt{i}.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}"
            for i in range(4)
        ],
        "attribution": "&copy; Google",
    },
    "google_satellite": {
        "type": "raster",
        "tiles": [
            f"https://mt{i}.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}"
            for i in range(4)
        ],
        "attribution": "&copy; Google",
    },
    "google_hybrid": {
        "type": "raster",
        "tiles": [
            f"https://mt{i}.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}"
            for i in range(4)
        ],
        "attribution": "&copy; Google",
    },
}

# Default basemap: OpenFreeMap Positron -- the direct replacement for the
# old CartoDB-Positron default, same light/minimal look.
DEFAULT_BASEMAP = "openfreemap_positron"

# Labels shown in the basemap-switcher dropdown.
BASEMAP_LABELS = {
    "openfreemap_positron": "OpenFreeMap Positron (light)",
    "openfreemap_bright": "OpenFreeMap Bright",
    "openfreemap_liberty": "OpenFreeMap Liberty",
    "openfreemap_dark": "OpenFreeMap Dark",
    "openstreetmap": "OpenStreetMap",
    "esri_imagery": "Esri World Imagery (satellite)",
    "google_roadmap": "Google Maps",
    "google_satellite": "Google Satellite",
    "google_hybrid": "Google Hybrid",
}


def _extract_score_interpolator(
    color_expr: Any,
) -> Optional[tuple[str, List[List[Any]]]]:
    """Pull ``(column, [[v0,c0], [v1,c1], ...])`` out of a fill/circle/line color expression.

    Matches two shapes, both produced by this codebase's own paint builders:

    1. ``ColorSpec.maplibre_fill_color_expr``'s wrapped form:
       ``["case", ["==", ["typeof", ["get", col]], "number"],
         ["interpolate", ["linear"], ["get", col], v0, c0, v1, c1, ...], nodata]``.
    2. ``transitlos.map.build._score_maplibre_paint``'s bare form (used for
       every hexagon/circle/census/street level `build_city_map` actually
       configures -- round 18 found the "case"-only match above never fired
       for ANY of them, silently leaving `window.__scoreInterpolators` empty
       and every `computeAccess()` color lookup a no-op):
       ``["interpolate", ["linear"], ["get", col], v0, c0, v1, c1, ...]``, OR
       (as of this round's item-8 null-guard fix, which wraps the bare
       ``["get", col]`` in a ``coalesce`` so a feature missing ``col``
       doesn't throw "Expected value to be of type number, but found null
       instead.") ``["interpolate", ["linear"], ["coalesce", ["get", col],
       fallback], v0, c0, v1, c1, ...]``.

    Used by :func:`save_multi_maplibre` to give the scenario-editor's
    ``computeAccess()`` port a way to turn a recomputed numeric score back
    into the *exact same* color a feature would have gotten baked into the
    tile, without duplicating the color ramp itself in JS -- see that
    function's ``window.__colorForValue`` for the consumer.
    """
    interp = color_expr
    if (
        isinstance(color_expr, list)
        and len(color_expr) == 4
        and color_expr[0] == "case"
    ):
        interp = color_expr[2]
    if not (
        isinstance(interp, list) and len(interp) >= 4 and interp[0] == "interpolate"
    ):
        return None
    get_expr = interp[2]
    if isinstance(get_expr, list) and len(get_expr) == 3 and get_expr[0] == "coalesce":
        get_expr = get_expr[1]
    if not (isinstance(get_expr, list) and len(get_expr) == 2 and get_expr[0] == "get"):
        return None
    column = get_expr[1]
    pairs = interp[3:]
    stops = [[pairs[i], pairs[i + 1]] for i in range(0, len(pairs) - 1, 2)]
    if not stops:
        return None
    return column, stops


def _client_basemap_js(
    basemap_styles: Dict[str, Any],
    initial_key: str,
    before_id_json: str,
    with_controls: bool,
) -> str:
    """Client-side basemap engine shared by both single-group and
    multi-group MapLibre HTML output.

    Handles both basemap shapes in ``BASEMAP_STYLES``:

    - ``"raster"``: unchanged from the original mechanism -- a single
      ``type: 'raster'`` source+layer pair named ``'basemap'``.
    - ``"vector-style"``: a full OpenFreeMap-shaped MapLibre style document
      (``sources``/``layers``/``sprite``/``glyphs``), which cannot be
      represented as a single source/layer. Its style JSON is fetched
      *client-side* (``fetch()``, cached in ``__bmVectorCache`` so
      re-selecting a style already seen this session doesn't re-fetch) --
      deliberately NOT at Python build time, so building a city's map.html
      never needs network access to tiles.openfreemap.org and never fails
      or slows down a pipeline run if that host is briefly unreachable.
      Every one of the style's sources/layers is added under an
      ``__bm_``-namespaced id (so they can never collide with this
      renderer's own PMTiles sources/layers, or with a second vector
      basemap's ids on a later switch) and inserted, in the style's own
      order, right before ``__basemapBeforeId`` -- the exact same
      insertion point the single raster layer already used, so overlay
      z-order is unaffected either way. The style's own ``sprite``/
      ``glyphs`` are wired via ``map.setSprite``/``map.setGlyphs`` (present
      on MapLibre GL JS 3.1+; this project pins 5.13.0) so icons/labels
      resolve; skipped with a console warning if the pinned MapLibre build
      doesn't have them rather than throwing.

    Whichever shape is active, ``__bmCurrentSourceIds``/
    ``__bmCurrentLayerIds`` track exactly what belongs to "the current
    basemap" so the next switch (to another vector style, or back to a
    plain raster one) can cleanly remove it first -- generalizing the old
    single well-known-id-``'basemap'`` removal invariant to N ids.

    Opacity/B&W controls (only wired when ``with_controls`` -- the
    multi-group switcher build has the slider/checkbox in its DOM, the
    single-group ``build_html`` does not): ``raster-opacity``/
    ``raster-saturation`` are real paint properties only on ``type:
    'raster'`` layers, so for a vector-style basemap this applies an
    approximate per-layer-type opacity (``fill-opacity``/``line-opacity``/
    ``circle-opacity``/``background-opacity``/``icon-opacity``+
    ``text-opacity`` for symbol layers) instead, and the B&W checkbox is
    simply disabled while a vector style is active -- there is no vector
    paint-property equivalent to raster desaturation short of rewriting
    every layer's color expression, which was judged too fragile to ship;
    the checkbox re-enables automatically on switching back to a raster
    basemap. This tradeoff is deliberate, not an oversight.
    """
    js = f"""
    const __basemapStyles = {json.dumps(basemap_styles)};
    const __basemapBeforeId = {before_id_json};
    const __bmVectorCache = {{}};
    let __bmCurrentSourceIds = [];
    let __bmCurrentLayerIds = [];
    let __bmCurrentType = null;

    function __bmOpacityPropsForType(type) {{
      if (type === 'fill') return ['fill-opacity'];
      if (type === 'line') return ['line-opacity'];
      if (type === 'circle') return ['circle-opacity'];
      if (type === 'background') return ['background-opacity'];
      if (type === 'fill-extrusion') return ['fill-extrusion-opacity'];
      if (type === 'symbol') return ['icon-opacity', 'text-opacity'];
      if (type === 'raster') return ['raster-opacity'];
      return [];
    }}

    function __bmRemoveCurrent() {{
      __bmCurrentLayerIds.forEach(function(id) {{ if (map.getLayer(id)) map.removeLayer(id); }});
      __bmCurrentSourceIds.forEach(function(id) {{ if (map.getSource(id)) map.removeSource(id); }});
      __bmCurrentSourceIds = [];
      __bmCurrentLayerIds = [];
    }}

    function __bmCurrentOpacityBwState() {{
      var opEl = document.getElementById('basemapOpacitySlider');
      var bwEl = document.getElementById('basemapBwCheckbox');
      return {{
        opacity: opEl ? opEl.value / 100 : 1,
        bw: bwEl ? bwEl.checked : false,
      }};
    }}

    function __bmApplyOpacityBw() {{
      var state = __bmCurrentOpacityBwState();
      var bwEl = document.getElementById('basemapBwCheckbox');
      if (__bmCurrentType === 'raster') {{
        if (bwEl) bwEl.disabled = false;
        if (map.getLayer('basemap')) {{
          map.setPaintProperty('basemap', 'raster-opacity', state.opacity);
          map.setPaintProperty('basemap', 'raster-saturation', state.bw ? -1 : 0);
        }}
      }} else {{
        // Vector style: no desaturation equivalent (see docstring) -- grey
        // out the B&W control rather than silently ignoring it.
        if (bwEl) bwEl.disabled = true;
        __bmCurrentLayerIds.forEach(function(id) {{
          if (!map.getLayer(id)) return;
          var layer = map.getLayer(id);
          __bmOpacityPropsForType(layer.type).forEach(function(prop) {{
            try {{ map.setPaintProperty(id, prop, state.opacity); }} catch (err) {{ /* not all layers accept every prop */ }}
          }});
        }});
      }}
    }}

    function __bmApplyRaster(bm) {{
      map.addSource('basemap', {{
        type: 'raster', tiles: bm.tiles, tileSize: 256, attribution: bm.attribution || '',
      }});
      map.addLayer({{id: 'basemap', type: 'raster', source: 'basemap'}}, __basemapBeforeId || undefined);
      __bmCurrentSourceIds = ['basemap'];
      __bmCurrentLayerIds = ['basemap'];
      __bmCurrentType = 'raster';
      __bmApplyOpacityBw();
    }}

    function __bmApplyVectorStyle(bm, styleJson) {{
      const srcIds = [];
      const sourceNames = Object.keys(styleJson.sources || {{}});
      sourceNames.forEach(function(sid, i) {{
        const nsid = '__bm_' + sid;
        const src = Object.assign({{}}, styleJson.sources[sid]);
        // OpenFreeMap style JSON doesn't embed a per-source `attribution`
        // field the way CARTO/OSM raster tiles do -- attach the
        // hand-supplied one (see `BASEMAP_STYLES`) to the first source so
        // MapLibre's default AttributionControl still surfaces it exactly
        // like the raster path does.
        if (i === 0 && bm.attribution) src.attribution = bm.attribution;
        map.addSource(nsid, src);
        srcIds.push(nsid);
      }});
      try {{
        if (styleJson.glyphs && typeof map.setGlyphs === 'function') map.setGlyphs(styleJson.glyphs);
        if (styleJson.sprite && typeof map.setSprite === 'function') map.setSprite(styleJson.sprite);
      }} catch (err) {{ console.warn('basemap: could not set sprite/glyphs', err); }}
      const layerIds = [];
      (styleJson.layers || []).forEach(function(l) {{
        const nl = Object.assign({{}}, l);
        nl.id = '__bm_' + l.id;
        if (l.source) nl.source = '__bm_' + l.source;
        map.addLayer(nl, __basemapBeforeId || undefined);
        layerIds.push(nl.id);
      }});
      __bmCurrentSourceIds = srcIds;
      __bmCurrentLayerIds = layerIds;
      __bmCurrentType = 'vector-style';
      __bmApplyOpacityBw();
    }}

    function __applyBasemap(key) {{
      const bm = __basemapStyles[key];
      if (!bm) return;
      const doApply = function() {{
        __bmRemoveCurrent();
        if (bm.type === 'vector-style') {{
          if (__bmVectorCache[bm.url]) {{
            __bmApplyVectorStyle(bm, __bmVectorCache[bm.url]);
          }} else {{
            fetch(bm.url).then(function(r) {{ return r.json(); }}).then(function(styleJson) {{
              __bmVectorCache[bm.url] = styleJson;
              __bmApplyVectorStyle(bm, styleJson);
            }}).catch(function(err) {{ console.error('basemap: failed to load vector style', bm.url, err); }});
          }}
        }} else {{
          __bmApplyRaster(bm);
        }}
      }};
      if (map.isStyleLoaded()) doApply(); else map.on('load', doApply);
    }}

    __applyBasemap({json.dumps(initial_key)});
"""
    if with_controls:
        js += """
    document.getElementById('basemapSelect').addEventListener('change', function(e) {
      __applyBasemap(e.target.value);
    });
    document.getElementById('basemapOpacitySlider').addEventListener('input', function(e) {
      const pct = parseInt(e.target.value, 10);
      document.getElementById('basemapOpacityValue').textContent = pct + '%';
      __bmApplyOpacityBw();
    });
    document.getElementById('basemapBwCheckbox').addEventListener('change', function(e) {
      __bmApplyOpacityBw();
    });
"""
    return js


def _default_popup_js(popup_fields: Optional[List[str]]) -> str:
    fields_json = json.dumps(popup_fields or [])
    return (
        "function(properties) {\n"
        f"  var fields = {fields_json};\n"
        "  if (!fields.length) { fields = Object.keys(properties).slice(0, 12); }\n"
        "  var html = '<table>';\n"
        "  fields.forEach(function(f) {\n"
        "    html += '<tr><td style=\"font-weight:600;padding-right:6px;\">' + f + "
        "'</td><td>' + properties[f] + '</td></tr>';\n"
        "  });\n"
        "  html += '</table>';\n"
        "  return html;\n"
        "}"
    )


def _custom_select_js() -> str:
    """Replace every native ``<select>`` with a custom dropdown that only
    closes on an explicit option click or an outside click -- never on the
    mouse merely leaving the box (explicit user request: "Dropout boxes...
    do not close them until user clicks on the desired option... if user
    moved mouse outside the box they close and this is annoying" -- for
    ALL dropdowns: opacity by, basemap, place, scenario, distribute by,
    compare with). A plain native ``<select>``'s own open dropdown list is
    rendered by the OS, not this page, so its close-on-mouse-out behavior
    can't be overridden from JS at all -- this is why every affected select
    is instead replaced with a fully custom (page-rendered) equivalent.

    Runs generically over every ``<select>`` in the document, wherever it
    came from (this module's own layer-switcher, or `transitlos.map.build`'s
    stats panel, injected separately) -- a `MutationObserver` also catches
    any `<select>` added to the DOM later, so this doesn't depend on script
    execution order relative to other injected HTML/JS.

    The original `<select>` is kept in the DOM (hidden, not removed) so
    every existing `.value`/`.addEventListener('change', ...)` call site
    elsewhere keeps working unchanged -- `.value = x` assignments are
    caught automatically (no call site needs to change) by overriding the
    `value` property's setter on each enhanced element to also refresh the
    custom widget's displayed label.
    """
    return r"""
<style>
.__csel-wrap { position: relative; display: inline-block; width: 100%; vertical-align: middle; }
.__csel-wrap-auto { width: auto; }
.__csel-btn {
  width: 100%; box-sizing: border-box; padding: 3px 22px 3px 6px;
  border: 1px solid #bbb; border-radius: 3px; background: #fff;
  cursor: pointer; font: inherit; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; position: relative;
}
.__csel-btn:after {
  content: ''; position: absolute; right: 7px; top: 50%; width: 0; height: 0;
  border-left: 4px solid transparent; border-right: 4px solid transparent;
  border-top: 5px solid #666; transform: translateY(-2px);
}
.__csel-list {
  display: none; position: absolute; top: 100%; left: 0; min-width: 100%;
  z-index: 10000; background: #fff; border: 1px solid #bbb; border-radius: 3px;
  max-height: 260px; overflow-y: auto; box-shadow: 0 2px 10px rgba(0,0,0,.18);
  margin-top: 2px;
}
.__csel-list.__csel-open { display: block; }
.__csel-item { padding: 4px 8px; cursor: pointer; white-space: nowrap; }
.__csel-item.__csel-sel { background: #eef2ff; font-weight: 600; }
.__csel-item:hover { background: #e5edff; }
</style>
<script>
(function() {
  var OPEN_CLASS = '__csel-open';
  var nativeValueDesc = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value');

  function closeAll(except) {
    document.querySelectorAll('.__csel-list.' + OPEN_CLASS).forEach(function(l) {
      if (l !== except) l.classList.remove(OPEN_CLASS);
    });
  }

  function enhance(sel) {
    if (!sel || sel.__cselDone || sel.disabled) return;
    sel.__cselDone = true;

    // Bug fix (2026-09-04, live user report -- the top-center bar's score
    // reading squeezed into too little space): `.__csel-wrap`/`.__csel-btn`
    // used to force `width:100%` on every enhanced select unconditionally.
    // That's right for a stats-panel select (explicitly `style="width:100%"`
    // in its own markup, meant to fill its row), but wrong for a compact
    // inline select like the top-center bar's scenario/place pickers (no
    // width in their own style -- meant to size to their content) -- 100%
    // of a `display:flex` row's available space inflated them and left
    // little room for their flex sibling (the score reading) next to them.
    // Only fill width when the ORIGINAL select's own inline style actually
    // asked for it; otherwise size to content, like a native select would.
    var origStyle = sel.getAttribute('style') || '';
    var wantsFullWidth = /width\s*:\s*100%/.test(origStyle);
    var fontMatch = origStyle.match(/font\s*:\s*[^;]+/);
    var colorMatch = origStyle.match(/(?:^|;)\s*color\s*:\s*[^;]+/);

    var wrap = document.createElement('span');
    wrap.className = '__csel-wrap' + (wantsFullWidth ? '' : ' __csel-wrap-auto');
    sel.parentNode.insertBefore(wrap, sel);
    wrap.appendChild(sel);
    sel.style.position = 'absolute';
    sel.style.opacity = '0';
    sel.style.pointerEvents = 'none';
    sel.style.width = '1px';
    sel.style.height = '1px';

    var btn = document.createElement('div');
    btn.className = '__csel-btn';
    if (fontMatch) btn.style.font = fontMatch[0].split(':').slice(1).join(':').trim();
    if (colorMatch) btn.style.color = colorMatch[0].replace(/^;/, '').split(':').slice(1).join(':').trim();
    if (!wantsFullWidth) {
      btn.style.width = 'auto';
      btn.style.paddingRight = '20px';
      btn.style.border = 'none';
      btn.style.background = 'transparent';
    }
    var list = document.createElement('div');
    list.className = '__csel-list';
    wrap.appendChild(btn);
    wrap.appendChild(list);

    function syncLabel() {
      var opt = sel.options[sel.selectedIndex];
      btn.textContent = opt ? opt.textContent : '';
    }

    function buildList() {
      list.innerHTML = '';
      Array.prototype.forEach.call(sel.options, function(opt, i) {
        var item = document.createElement('div');
        item.className = '__csel-item' + (i === sel.selectedIndex ? ' __csel-sel' : '');
        item.textContent = opt.textContent;
        item.addEventListener('click', function(e) {
          e.stopPropagation();
          if (sel.selectedIndex !== i) {
            nativeValueDesc.set.call(sel, opt.value);
            syncLabel();
            sel.dispatchEvent(new Event('change', {bubbles: true}));
          }
          list.classList.remove(OPEN_CLASS);
        });
        list.appendChild(item);
      });
    }

    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      var willOpen = !list.classList.contains(OPEN_CLASS);
      closeAll(null);
      if (willOpen) {
        buildList();
        list.classList.add(OPEN_CLASS);
      }
    });
    list.addEventListener('click', function(e) { e.stopPropagation(); });

    // Catch every `.value = x` assignment anywhere in the page (this
    // codebase sets stats-tab defaults this way) without touching each
    // call site -- override the instance's own `value` accessor.
    Object.defineProperty(sel, 'value', {
      configurable: true,
      get: function() { return nativeValueDesc.get.call(sel); },
      set: function(v) { nativeValueDesc.set.call(sel, v); syncLabel(); },
    });

    syncLabel();
  }

  function enhanceAll(root) {
    (root || document).querySelectorAll('select').forEach(enhance);
  }

  document.addEventListener('click', function(e) {
    // Bug fix (2026-09-04, live user report -- "opacity by and basemap
    // dropdown boxes do not work, I click and nothing opens"): this
    // listener closes every open dropdown on ANY click outside it, but a
    // click on the toggle BUTTON itself was still reaching here and
    // immediately re-closing what that same click had just opened --
    // `e.stopPropagation()` inside the button's own handler didn't
    // reliably prevent it in every browser context this page runs in
    // (confirmed live via headless Chrome + a real DOM click: the class
    // was added, then removed again, all within the same synchronous
    // click). Checking the click's real target here, defensively, instead
    // of relying solely on propagation being stopped upstream.
    if (e.target && e.target.closest && e.target.closest('.__csel-wrap')) return;
    closeAll(null);
  });

  function boot() {
    enhanceAll(document);
    new MutationObserver(function(mutations) {
      mutations.forEach(function(m) {
        m.addedNodes && m.addedNodes.forEach(function(node) {
          if (node.nodeType !== 1) return;
          if (node.tagName === 'SELECT') enhance(node);
          else if (node.querySelectorAll) enhanceAll(node);
        });
      });
    }).observe(document.body, {childList: true, subtree: true});
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
</script>
"""


def _coarsest_level_zoom(hmap, fallback: int = 12) -> int:
    """The zoom to open a map at so only its coarsest (lowest-resolution)
    level's tiles load initially, instead of a fixed mid-zoom that can pull
    in fine-grained census/H3 tiles immediately -- the slow-to-load-a-city
    complaint this fixes. ``hmap.levels`` is coarse-to-fine (see callers'
    "coarse -> fine order" convention), so ``levels[0]`` is the coarsest
    level; its ``set_resolution``-assigned band's upper bound is the last
    zoom at which ONLY that level (and nothing finer) is visible.

    Superseded as the actual initial-zoom choice by `_finest_level_zoom`
    below (2026-09-05, explicit user follow-up: the "zoomed out to just the
    coarsest level" default was reported as "a very low zoom level" that
    "does not correspond" to what's wanted) -- kept as a still-correct,
    still-used building block/fallback, not removed.
    """
    if not hmap.levels:
        return fallback
    band = hmap._manual_resolutions.get(hmap.levels[0])
    if not band:
        return fallback
    # One zoom level below the coarsest level's own upper bound (explicit
    # follow-up request: "activate the lowest census level on one lower
    # zoom level") -- a safety margin so the initial view sits solidly
    # inside the coarsest-only band rather than right on its edge, where a
    # sub-pixel zoom/rounding difference could pull in the next-finer
    # level's tiles too.
    return max(band[1] - 1, band[0])


def _bbox_fit_zoom(
    bounds: tuple[float, float, float, float],
    viewport_width: float = 1024,
    viewport_height: float = 768,
    max_zoom: int = 18,
    min_zoom: int = 0,
    pad_factor: float = 1.0,
) -> int:
    """Zoom level at which a lon/lat bounding box just fits a viewport.

    2026-09-06, explicit user request: the map's own startup zoom (and the
    combined map's zoom-in-to-enter-this-city threshold) should be "a zoom
    level that allows to cover the complete aoi but is high" -- i.e. as
    zoomed IN as possible while the whole AOI still fits on screen, not a
    fixed value tied to whichever census level happens to be finest
    (`_finest_level_zoom`, which a tiny city like Andorra and a huge one
    like Boston metro would get the exact same treatment under, even though
    their AOIs differ by orders of magnitude in extent).

    Same "fit bounds" formula MapLibre/Leaflet/Google Maps use internally
    (Google's public `getBoundsZoomLevel` algorithm): finds the zoom where
    the box's fractional width and height of the whole world map both fit
    within the given pixel viewport, then floors to the last whole zoom
    where it still fits.

    Args:
        bounds: ``(minx, miny, maxx, maxy)`` in EPSG:4326 degrees.
        viewport_width/height: Assumed on-screen map size in pixels --
            deliberately smaller than a full desktop viewport so the fit
            holds up on the narrower panel this map is often embedded in
            (a smaller assumed viewport biases toward a SAFER/lower zoom,
            never one so tight the AOI's edges clip off-screen).
        max_zoom: Never return a zoom finer than this real tile ceiling
            (see `MAX_NATIVE_TILE_ZOOM`).
        pad_factor: Multiplies the box's extent (about its own center)
            before fitting -- pass >1 to zoom OUT further than "just
            fits", e.g. for a companion "covers the AOI and much more
            area" overview-exit zoom (see that call site's own comment).
    """
    minx, miny, maxx, maxy = bounds
    if pad_factor != 1.0:
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        hw, hh = (maxx - minx) / 2 * pad_factor, (maxy - miny) / 2 * pad_factor
        minx, maxx = cx - hw, cx + hw
        miny, maxy = max(cy - hh, -85.0), min(cy + hh, 85.0)

    def lat_rad(lat: float) -> float:
        s = max(min(math.sin(math.radians(lat)), 0.9999), -0.9999)
        rad_x2 = math.log((1 + s) / (1 - s)) / 2
        return max(min(rad_x2, math.pi), -math.pi) / 2

    def zoom_for(map_px: float, world_px: float, fraction: float) -> float:
        if fraction <= 0:
            return float(max_zoom)
        return math.log2(map_px / world_px / fraction)

    lat_fraction = (lat_rad(maxy) - lat_rad(miny)) / math.pi
    lng_fraction = (maxx - minx) / 360 if maxx > minx else 1.0

    lat_zoom = zoom_for(viewport_height, 256, lat_fraction)
    lng_zoom = zoom_for(viewport_width, 256, lng_fraction)
    return int(max(min_zoom, min(math.floor(lat_zoom), math.floor(lng_zoom), max_zoom)))


def _finest_level_zoom(hmap, fallback: int = 12) -> int:
    """The zoom to open a map at so its FINEST (highest-resolution) level
    is already visible, right at the threshold where zooming back out one
    more step would switch to the next-coarser level instead.

    2026-09-05, explicit user request: "The map startup zoom level should
    be the zoom level just before you change from the lowest census level
    to the next one that is higher. Right now it is a very low zoom level
    that does not correspond with that" -- "lowest" here means lowest IN
    THE HIERARCHY (block/blockgroup, the most granular geography, as
    opposed to `_coarsest_level_zoom`'s "lowest resolution"/coarsest
    reading), and "the next one that is higher" means the next COARSER
    level up the hierarchy (blockgroup -> tract, etc.) -- i.e. start
    zoomed IN, not out. ``hmap.levels`` is coarse-to-fine, so
    ``levels[-1]`` is the finest level; its band's LOWER bound is the
    first zoom at which it activates, so opening exactly there shows the
    finest level immediately without being any more zoomed in than
    necessary.
    """
    if not hmap.levels:
        return fallback
    band = hmap._manual_resolutions.get(hmap.levels[-1])
    if not band:
        return fallback
    return band[0]


class MapLibreHierarchyMap:
    """Renders an already-built :class:`HierarchyMap` via MapLibre GL JS + PMTiles.

    Args:
        hmap: A :class:`~geohierarchy.maps.folium.render.HierarchyMap` whose
            ``.build()`` has already been called (its per-level
            ``.pmtiles`` files must exist under ``hmap.tiles_dir``).
    """

    def __init__(self, hmap: HierarchyMap):
        if not hmap.tiles_dir:
            raise ValueError("hmap.tiles_dir must be set (and build() already run)")
        self.hmap = hmap

    # ------------------------------------------------------------------
    def _level_pmtiles_chunks(self, name: str) -> List[str]:
        """Return chunk file basenames (e.g. ``chunk_842a001ffffffff.pmtiles``)
        for ``name`` if it was tiled chunked (see
        ``geohierarchy.maps.folium.tiles.H3_CHUNK_ROW_THRESHOLD``), sorted
        for deterministic source ids across rebuilds. Empty list means the
        level was tiled as a single ``{name}.pmtiles`` file (the common
        case for every level under the chunking threshold).
        """
        level_dir = Path(self.hmap.tiles_dir) / name
        if not level_dir.is_dir():
            return []
        return sorted(p.name for p in level_dir.glob("chunk_*.pmtiles"))

    # ------------------------------------------------------------------
    def _sources_and_layers(
        self, tiles_dir_rel: str
    ) -> tuple[
        Dict[str, Any],
        List[Dict[str, Any]],
        Dict[str, str],
        Dict[str, List[str]],
        Dict[str, List[str]],
    ]:
        """Build MapLibre sources/layers for every level.

        A level tiled as a single ``{name}.pmtiles`` file (the default,
        common path) gets exactly one source (id == level name) and one
        layer per shape kind, unchanged from before.

        A level tiled in H3-res4 chunks (oversized levels only -- see
        `geohierarchy.maps.folium.tiles.H3_CHUNK_ROW_THRESHOLD`) instead
        gets one source *per chunk file* (id ``{name}__chunk{i}``) and one
        layer per chunk, all sharing the identical style/paint/filter
        config and the same ``source-layer: name`` (the MVT layer name
        inside every chunk file is still the plain level name -- so
        feature-state code keyed by source-layer doesn't need to change).
        N chunk sources/layers rendered together look like one continuous
        layer to the end user.

        Returns:
            ``(sources, layers, popup_js_by_level, level_source_ids, level_layer_ids)``
            where `level_source_ids`/`level_layer_ids` map each level name
            to *every* MapLibre source id / layer id generated for it
            (length 1 for an unchunked level, length N for a chunked one)
            -- callers that used to assume "the level name IS the source
            id" (recolor feature-state, popup/hover layer lists) now loop
            over these instead.
        """
        hmap = self.hmap
        zoom_ranges = hmap.resolve_zoom_ranges()
        sources: Dict[str, Any] = {}
        layers: List[Dict[str, Any]] = []
        popup_js_by_level: Dict[str, str] = {}
        level_source_ids: Dict[str, List[str]] = {}
        level_layer_ids: Dict[str, List[str]] = {}

        for name in hmap.levels:
            layer = hmap._layers[name]
            spec = get_layer_type(layer.layer_type)

            gdf = None
            if layer.style is not None:
                gdf = hmap.hierarchy.get_level(name)

            paint: Dict[str, Any] = dict(spec.default_paint)
            if layer.maplibre_paint is not None:
                paint.update(layer.maplibre_paint)
            elif layer.style is not None:
                paint.update(layer.style.maplibre_paint(kind=spec.kind, gdf=gdf))
            elif spec.kind == "polygon":
                paint.setdefault("fill-color", "#3388ff")
                paint.setdefault("fill-opacity", 0.4)
                paint.setdefault("fill-outline-color", "#333333")
            elif spec.kind == "circle":
                paint.setdefault("circle-color", "#3388ff")
                paint.setdefault("circle-opacity", 0.7)
            else:
                paint.setdefault("line-color", "#3388ff")
                paint.setdefault("line-width", 1)

            # Live-recolor plumbing: whenever a `computeAccess()`-style pass
            # sets a `access_override` feature-state entry (a color string)
            # on a fill/circle/line feature via `window.__setAccessOverride()`
            # (added below), it wins over the static/style-derived color --
            # with no source-data regeneration and no tile rebuild. Line
            # layers (streets) have no numeric access-score ramp of their
            # own (see `_maplibre_compute_access_js`'s docstring in
            # transitlos/map/build.py -- streets are highlighted with a
            # fixed color when "affected", not recolored along a ramp), but
            # they still need the coalesce-wrap so that highlight actually
            # has somewhere to land.
            color_prop = {
                "fill": "fill-color",
                "circle": "circle-color",
                "line": "line-color",
            }.get({"polygon": "fill", "circle": "circle", "line": "line"}[spec.kind])
            if color_prop and color_prop in paint:
                paint[color_prop] = [
                    "coalesce",
                    ["feature-state", "access_override"],
                    paint[color_prop],
                ]

            min_z, max_z = zoom_ranges[name]
            # Real bug fix (2026-09-05, live report: "with high zoom levels
            # circles deactivate"): the comment below (and `HierarchyMap`'s
            # own tile-build code) has always assumed MapLibre "transparently
            # overzooms" a vector source past its real tiled max -- true in
            # general, but MapLibre only does this when the SOURCE's own
            # `maxzoom` is declared; undeclared, it defaults to MapLibre's
            # own (much higher) internal default and requests tiles that
            # were never generated past `MAX_NATIVE_TILE_ZOOM` (18), getting
            # nothing back -- the layer visually disappears instead of
            # reusing/scaling the deepest real tile. `native_max_z` (the
            # same clamp `HierarchyMap.build()` itself applies before
            # tiling, see that module's `min(max_z, MAX_NATIVE_TILE_ZOOM)`)
            # is what actually got tiled, captured here BEFORE the
            # fallback-level widening below changes `max_z` for the STYLE
            # only (tiles for a fallback level are still only built for its
            # own narrow band, per `HierarchyMap.fallback_level`'s docstring).
            native_max_z = min(max_z, MAX_NATIVE_TILE_ZOOM)
            if name == hmap.fallback_level:
                # Widen the STYLE layer's zoom filter only -- real tiles are
                # still only generated for this level's own narrow
                # `resolve_zoom_ranges()` band (`build()` is untouched);
                # MapLibre transparently overzooms (reuses/scales) the
                # deepest available tile for any zoom beyond a vector
                # source's real max, the same mechanism a raster basemap's
                # `maxzoom` relies on, so this needs no extra tiling. See
                # `HierarchyMap.fallback_level`'s docstring.
                max_z = 24
            gl_type = {"polygon": "fill", "circle": "circle", "line": "line"}[spec.kind]

            id_col = (
                hmap.hierarchy.id_cols.get(name)
                if hasattr(hmap.hierarchy, "id_cols")
                else None
            )

            chunk_files = self._level_pmtiles_chunks(name)
            if chunk_files:
                pairs = [
                    (f"{name}__chunk{i}", f"{name}/{fname}")
                    for i, fname in enumerate(chunk_files)
                ]
            else:
                pairs = [(name, f"{name}.pmtiles")]

            source_ids: List[str] = []
            layer_ids: List[str] = []
            for source_id, rel_pmtiles in pairs:
                pmtiles_url = f"pmtiles://{tiles_dir_rel}/{rel_pmtiles}"
                # `promoteId` swaps MapLibre's per-tile-local auto id
                # (unstable across tile/zoom boundaries -- the same
                # real-world feature can get a different id in each tile
                # it's clipped into) for the hierarchy's own stable id
                # column, which `write_level_tiles` already bakes into
                # every tile's properties (see `HierarchyMap.build()`).
                # This is what makes `setFeatureState()`-based recolor
                # (computeAccess() overrides, live hover) address the
                # *same* feature consistently no matter which tile/zoom
                # (or, for a chunked level, which chunk source) it's
                # currently rendered from.
                source_cfg: Dict[str, Any] = {
                    "type": "vector",
                    "url": pmtiles_url,
                    "maxzoom": native_max_z,
                }
                if id_col:
                    source_cfg["promoteId"] = {name: id_col}
                sources[source_id] = source_cfg
                source_ids.append(source_id)

                gl_layer_id = (
                    f"{name}_{gl_type}"
                    if source_id == name
                    else f"{name}_{gl_type}__{source_id.rsplit('__', 1)[-1]}"
                )
                gl_layer = {
                    "id": gl_layer_id,
                    "type": gl_type,
                    "source": source_id,
                    # Same plain level name in every chunk's MVT layer --
                    # NOT chunk-specific -- so existing feature-state code
                    # keyed by source-layer doesn't need to change.
                    "source-layer": name,
                    "minzoom": min_z,
                    "maxzoom": min(
                        max_z + 1, 24
                    ),  # MapLibre style maxzoom must be <=24, and is exclusive vs. Folium's inclusive
                    "paint": paint,
                }
                layers.append(gl_layer)
                layer_ids.append(gl_layer_id)

            level_source_ids[name] = source_ids
            level_layer_ids[name] = layer_ids

            # 2026-08-xx bug fix (user report, live Concepcion/Guadalajara
            # maps): `fallback_level` used to mean a SYNTHETIC residual
            # shape (the coarsest level's geometry minus every finer
            # level's polygons, via `set_fallback_level` -- see its
            # docstring at the time), so clicking it opened a nonsensical
            # popup for a sliver that was never a real administrative or h3
            # unit. That synthetic-geometry substitution was removed
            # 2026-08-30 (see `transitlos.map.build.build_city_map`'s
            # census-map construction comment) and `set_fallback_level`
            # went unused until 2026-09-01, when it was restored for a
            # different purpose: `name` here is now always a REAL,
            # unmodified census level (e.g. Mexico's ageb) reused as a
            # gap-filling backdrop for the next-finer level's real coverage
            # holes, not a synthetic shape -- every pixel it draws, whether
            # in its own native zoom band or overzoomed as a backdrop, is a
            # real administrative unit with real attributes. Popups are
            # therefore left enabled unconditionally again (`spec.interactive`
            # alone decides, matching every other level).
            if spec.interactive:
                popup_js_by_level[name] = layer.popup_js or _default_popup_js(
                    layer.popup_fields
                )

        return sources, layers, popup_js_by_level, level_source_ids, level_layer_ids

    # ------------------------------------------------------------------
    def build_html(
        self,
        basemap: Union[str, Dict[str, Any]] = DEFAULT_BASEMAP,
        title: str = "Map",
    ) -> str:
        """Return the complete self-contained MapLibre HTML document (as a string).

        The PMTiles ``pmtiles://`` protocol is registered via
        ``maplibregl.addProtocol`` per the user's spec; tile URLs are
        relative to the output HTML file's own directory (same convention
        Folium's output uses), so this HTML must be saved next to
        ``hmap.tiles_dir`` (see :meth:`save`).
        """
        hmap = self.hmap
        tiles_dir_name = Path(hmap.tiles_dir).name
        sources, layers, popup_js_by_level, level_source_ids, level_layer_ids = (
            self._sources_and_layers(tiles_dir_name)
        )

        bm = (
            BASEMAP_STYLES.get(basemap, basemap)
            if isinstance(basemap, str)
            else basemap
        )
        if "type" not in bm:
            bm = dict(bm)
            bm["type"] = "raster" if "tiles" in bm else "vector-style"
        # The basemap itself is no longer baked into the initial style
        # object -- both raster and vector-style basemaps are now added by
        # `_client_basemap_js`'s `__applyBasemap()` right after the map is
        # constructed (a vector style's sources/layers can't be represented
        # by this single dict-literal `sources`/`layers` pair anyway; see
        # that function's docstring). `all_layers[0]`'s id (if any) is
        # still the insertion point so the basemap always ends up stacked
        # below every real data layer.
        all_layers = layers

        bounds = hmap.hierarchy.level_bounds(hmap.levels[0])
        center = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
        # 2026-09-06, explicit user request -- see `_bbox_fit_zoom`'s own
        # docstring: as zoomed IN as possible while the whole AOI still
        # fits, not a fixed value tied to whichever census level is finest.
        initial_zoom = _bbox_fit_zoom(bounds)
        # Companion "covers the AOI and much more area" zoom (explicit user
        # request) -- `code.combined_map` reads this (via `__overviewExitZoom`
        # below) as the threshold for dropping back from this city's own map
        # to the all-cities overview when the user zooms out; `pad_factor=8`
        # fits a box 8x this AOI's own extent, i.e. a deliberately much
        # lower zoom than `initial_zoom`, not just one step lower.
        overview_exit_zoom = _bbox_fit_zoom(bounds, pad_factor=8.0)

        style_obj = {"version": 8, "sources": sources, "layers": all_layers}
        basemap_before_id_json = (
            json.dumps(all_layers[0]["id"]) if all_layers else "null"
        )
        basemap_bootstrap_js = _client_basemap_js(
            {"__initial": bm},
            "__initial",
            basemap_before_id_json,
            with_controls=False,
        )

        interactive_layer_ids = [
            layer_id for name in popup_js_by_level for layer_id in level_layer_ids[name]
        ]
        # Build a JS lookup: layer id -> popup fn, and register click/hover handlers.
        # Every chunk source's layer id (if the level is chunked) maps to
        # the same popup fn, since the level's popup config doesn't vary
        # per chunk.
        popup_lookup_entries = []
        for name, popup_js in popup_js_by_level.items():
            for layer_id in level_layer_ids[name]:
                popup_lookup_entries.append(f"{json.dumps(layer_id)}: {popup_js}")
        popup_lookup_js = "{\n" + ",\n".join(popup_lookup_entries) + "\n}"

        # Level -> list of MapLibre source ids map, used by the recolor API
        # below. Normally a single-element list (source id == level name);
        # for a level tiled in H3-res4 chunks (see
        # `geohierarchy.maps.folium.tiles.H3_CHUNK_ROW_THRESHOLD`), one
        # entry per chunk source -- the override/clear calls below loop
        # over all of them so a chunked level still behaves like one
        # continuous recolorable layer to callers.
        recolorable_source_ids = {
            name: level_source_ids[name]
            for name in hmap.levels
            if get_layer_type(hmap._layers[name].layer_type).kind
            in ("polygon", "circle")
        }
        recolor_js = f"""
    // Live recolor API (scenario editor / computeAccess() hook): set or
    // clear a per-feature color override via MapLibre feature-state, so a
    // recompute can repaint affected hexagons/geoids without touching
    // source data or rebuilding tiles. `level` is a hierarchy level name
    // (e.g. 'hexagons_r8', 'census_blockgroup'); `featureId` must match the
    // level's promoted id column (see `hierarchy.id_cols`). A chunked
    // level's feature lives in exactly one chunk source, but since we
    // don't know which one without a lookup, the override is applied to
    // every chunk source for that level -- harmless no-ops on sources that
    // don't have the feature loaded.
    window.__mapLevelSources = {json.dumps(recolorable_source_ids)};
    window.__hasMapLevel = function(level) {{
      return level in window.__mapLevelSources;
    }};
    window.__setAccessOverride = function(level, featureId, colorHex) {{
      if (!(level in window.__mapLevelSources)) return false;
      window.__mapLevelSources[level].forEach(function(srcId) {{
        map.setFeatureState({{source: srcId, sourceLayer: level, id: featureId}}, {{access_override: colorHex}});
      }});
      return true;
    }};
    window.__clearAccessOverride = function(level, featureId) {{
      if (!(level in window.__mapLevelSources)) return false;
      window.__mapLevelSources[level].forEach(function(srcId) {{
        map.removeFeatureState({{source: srcId, sourceLayer: level, id: featureId}}, 'access_override');
      }});
      return true;
    }};
    window.__clearAllAccessOverrides = function(level) {{
      if (!(level in window.__mapLevelSources)) return;
      window.__mapLevelSources[level].forEach(function(srcId) {{
        map.removeFeatureState({{source: srcId, sourceLayer: level}}, 'access_override');
      }});
    }};
"""

        interaction_js = f"""
    {recolor_js}
    const popupFns = {popup_lookup_js};
    const interactiveLayerIds = {json.dumps(interactive_layer_ids)};
    // `maxWidth` explicit (2026-09-01, live user report): MapLibre's own
    // default (240px) was clamping this popup well below the 480px the
    // shape popup's own inner content (`_shape_popup_js` in transitLOS,
    // widened for exactly this table) already asks for -- the click popup
    // for hexagons/circles/census shapes was cramped regardless of the
    // inner CSS. 600px leaves margin around the 560px content.
    const popup = new maplibregl.Popup({{closeButton: true, closeOnClick: true, maxWidth: '600px'}});

    map.on('click', function(e) {{
      // Bug fix (user report): clicking a stop marker must never ALSO pop
      // open this shape (hex/circle/census) popup underneath it. MapLibre
      // dispatches every registered 'click' listener for a click
      // regardless of layer order or `preventDefault`/`stopPropagation`
      // (see `Evented.fire` -- it loops every listener unconditionally),
      // so a plain ordering/`stopPropagation` fix can't work here across
      // this module and whatever else (e.g. transitlos.map.build's stop
      // markers) also registers its own 'click' listener on the same
      // `map`. Instead, callers that own a higher-priority point layer
      // (stop markers, drawn-route vertices, ...) publish their layer ids
      // on `window.__mapClickPriorityLayers`; checked here, at click time,
      // so it works regardless of which module's JS happens to load/run
      // first.
      const priorityLayers = (window.__mapClickPriorityLayers || []).filter((id) => map.getLayer(id));
      if (priorityLayers.length && map.queryRenderedFeatures(e.point, {{layers: priorityLayers}}).length) {{
        return;
      }}
      const features = map.queryRenderedFeatures(e.point, {{layers: interactiveLayerIds}});
      if (!features.length) return;
      const f = features[0];
      const fn = popupFns[f.layer.id];
      const html = fn ? fn(f.properties) : JSON.stringify(f.properties);
      popup.setLngLat(e.lngLat).setHTML(html).addTo(map);
    }});

    let hoveredId = null;
    let hoveredLayer = null;
    map.on('mousemove', function(e) {{
      if (!map.isStyleLoaded()) return;
      const features = map.queryRenderedFeatures(e.point, {{layers: interactiveLayerIds}});
      map.getCanvas().style.cursor = features.length ? 'pointer' : '';
      if (hoveredId !== null && hoveredLayer) {{
        map.setFeatureState({{source: hoveredLayer, id: hoveredId}}, {{hover: false}});
      }}
      if (features.length) {{
        hoveredLayer = features[0].source;
        hoveredId = features[0].id;
        if (hoveredId !== undefined && hoveredId !== null) {{
          map.setFeatureState({{source: hoveredLayer, id: hoveredId}}, {{hover: true}});
        }}
      }} else {{
        hoveredId = null;
        hoveredLayer = null;
      }}
    }});
"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="stylesheet" href="{MAPLIBRE_CSS_CDN}">
  <script src="{MAPLIBRE_CDN}"></script>
  <script src="{PMTILES_CDN}"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; }}
    #map {{ width: 100%; height: 100%; }}
    .maplibregl-popup-content table {{ font-size: 12px; }}
  </style>
</head>
<body>
  {_custom_select_js()}
  <div id="map"></div>
  <script>
    const protocol = new pmtiles.Protocol();
    maplibregl.addProtocol('pmtiles', protocol.tile);

    const map = new maplibregl.Map({{
      container: 'map',
      style: {json.dumps(style_obj)},
      center: {json.dumps(center)},
      zoom: {initial_zoom},
      maxZoom: 24,
    }});
    // Exposed on `window` (2026-09-01, for `code.combined_map`'s zoom-based
    // overview<->city auto-switch): `const map` alone is only reachable
    // from code inside this same `<script>` block. The combined page reads
    // a per-city `map.html`'s live zoom via same-origin
    // `iframe.contentWindow.__mainMap.getZoom()`/`.on('zoomend', ...)` to
    // know when to drop back to its own all-cities overview map.
    window.__mainMap = map;
    // 2026-09-06, explicit user request -- `code.combined_map` reads this
    // (regex-parsed straight out of the saved HTML, same as `zoom: N`
    // above) as the zoom threshold for dropping back to the all-cities
    // overview when zooming out of this city; see `_bbox_fit_zoom`'s
    // `pad_factor` comment for why it's deliberately much lower than the
    // map's own startup zoom, not just one step lower.
    window.__overviewExitZoom = {overview_exit_zoom};
    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    {basemap_bootstrap_js}
    {interaction_js}
  </script>
</body>
</html>
"""
        return html

    # ------------------------------------------------------------------
    def save(
        self,
        path: str,
        basemap: Union[str, Dict[str, Any]] = DEFAULT_BASEMAP,
        title: str = "Map",
    ) -> str:
        """Write the MapLibre HTML to ``path``.

        ``path`` must be a sibling of ``hmap.tiles_dir`` (same directory),
        matching the relative ``pmtiles://<tiles_dir_name>/<level>.pmtiles``
        URLs baked into the generated HTML -- exactly the convention
        Folium's own ``HierarchyMap.save()`` output already relies on for
        its XYZ tile URLs.
        """
        html = self.build_html(basemap=basemap, title=title)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out)


def save_maplibre(
    hmap: HierarchyMap,
    path: str,
    basemap: Union[str, Dict[str, Any]] = DEFAULT_BASEMAP,
    title: str = "Map",
) -> str:
    """Convenience wrapper: ``save_maplibre(hmap, "map_maplibre.html")``."""
    return MapLibreHierarchyMap(hmap).save(path, basemap=basemap, title=title)


def save_multi_maplibre(
    multi,
    path: str,
    basemap: Union[str, Dict[str, Any]] = DEFAULT_BASEMAP,
    title: str = "Map",
    radius_field_domains: Optional[Dict[str, tuple]] = None,
    opacity_field_domains: Optional[Dict[str, tuple]] = None,
    circle_fields: Optional[List[str]] = None,
    opacity_fields: Optional[List[str]] = None,
    default_circle_field: Optional[str] = None,
    field_labels: Optional[Dict[str, str]] = None,
    radius_field_domains_by_res: Optional[Dict[int, Dict[str, tuple]]] = None,
    circle_zoom_bands: Optional[Dict[int, tuple]] = None,
) -> str:
    """Render a :class:`~geohierarchy.maps.folium.render.MultiHierarchyMap` via MapLibre.

    Combines every named/overlay group's sources+layers into one MapLibre
    style (each group's layer ids namespaced ``{group}:{level}_{gltype}``,
    matching :class:`MapLibreHierarchyMap`'s per-level id convention so
    popup/hover lookups stay correct within a group), and adds a small
    HTML/JS layer switcher: named groups as radio buttons (mutually
    exclusive, mirroring Folium's ``FeatureGroup(overlay=False)``
    semantics), overlay groups as checkboxes.

    `radius_field_domains`/`opacity_field_domains`/`circle_fields`/
    `opacity_fields`/`default_circle_field` (all optional, round-11 port of
    Folium's `_control_panel_html`'s "Circle size by"/"Hex/circle/census
    opacity"/"Opacity by"/"Contrast" controls -- see
    `transitlos.map.build._control_panel_js`/`_opacity_helper_js` for the
    formulas being matched):
      - when `radius_field_domains`/`circle_fields` are given, a "Circle
        size by" dropdown is added; changing it rebuilds the
        `_circle_radius_expr`-shaped interpolate expression client-side and
        applies it via `map.setPaintProperty(id, 'circle-radius', ...)` to
        every circle-type layer this call produced (only circle/point
        levels have a `circle-radius` paint property at all, so this never
        touches fill/line layers).
      - `opacity_field_domains`/`opacity_fields` add an "Opacity by"
        dropdown + "Contrast" slider, applying a data-driven
        `fill-opacity`/`circle-opacity` expression (gamma-shaped by
        contrast, exactly mirroring `opacityFromField`'s formula) to every
        fill/circle layer this call produced.
      - a "Hex/circle/census opacity" slider is always added (even with no
        fields passed) as a flat multiplier on top of whichever opacity
        rule (flat 0.75 baseline, or the opacity-by-field expression above)

    `field_labels`: optional `{column: display text}` map for the "Circle
    size by"/"Opacity by" `<option>` text -- e.g. `{"inegi_unemployment_rate":
    "Unemployment rate (%)"}`. This module is renderer-agnostic and has no
    concept of any study's naming conventions, so it never derives a display
    name itself; a field missing from this map (or the map being omitted
    entirely) falls back to the raw column name, unchanged from before.
        is currently active -- same "boost/fade on top of, not instead of"
        relationship Folium's own slider has.

    `radius_field_domains_by_res`/`circle_zoom_bands` (2026-09-05, real bug
    fix -- "circle size legend does not change with zoom"): `radius_field_domains`
    above is a single FLAT domain, and `__applyCircleRadius` used to apply
    it to EVERY circle layer id -- every H3 resolution's circles -- with
    the SAME expression, unconditionally, on page load. This silently
    overwrote each resolution's own correctly-domain-calibrated static
    paint (`transitlos.map.build._score_maplibre_paint`, already computed
    per resolution) with one flat domain sized for whichever resolution's
    data happened to be passed in, the moment the page loaded -- so circle
    sizes (and the legend numbers) never actually varied by resolution/zoom
    in practice, confirmed live on Boston. When given, each circle layer's
    id is parsed for its H3 resolution (`h3_(\\d+)` in the layer id) and
    looked up in `radius_field_domains_by_res` for ITS OWN domain instead
    of the flat one; `circle_zoom_bands` (resolution -> `(minZoom, maxZoom)`)
    lets the legend TEXT pick the domain matching the map's current zoom.
    Both optional and additive -- omitting them keeps the old flat-domain
    behavior for a caller that doesn't pass them.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_dir = out.parent

    all_sources: Dict[str, Any] = {}
    all_layers: List[Dict[str, Any]] = []
    popup_lookup_entries: List[str] = []
    group_layer_ids: Dict[str, List[str]] = {}
    interactive_layer_ids: List[str] = []
    # source name ("group:level") -> (column, stops) for every layer whose
    # color is a numeric-column interpolate expression -- lets the
    # scenario-editor computeAccess() port (transitLOS-side) turn a
    # recomputed score back into the exact tile color via
    # `window.__colorForValue`, see `_extract_score_interpolator`.
    score_interpolators: Dict[str, tuple[str, List[List[Any]]]] = {}
    # Logical "{group}:{level}" -> every real MapLibre source id backing it
    # (length 1 unless the level was tiled in H3-res4 chunks -- see
    # `geohierarchy.maps.folium.tiles.H3_CHUNK_ROW_THRESHOLD`). The
    # recolor API below (and `CENSUS_LEVEL`/`STREETS_LEVEL`-style JS
    # constants in transitlos/map/build.py) address levels by this logical
    # name, so a chunked level still behaves like one continuous
    # recolorable layer to callers.
    group_level_source_ids: Dict[str, List[str]] = {}

    def _add_group(group_name: str, hmap: HierarchyMap, visible: bool) -> None:
        rel_tiles_dir = os.path.relpath(Path(hmap.tiles_dir).resolve(), out_dir)
        sub = MapLibreHierarchyMap(hmap)
        sources, layers, popup_js_by_level, level_source_ids, level_layer_ids = (
            sub._sources_and_layers(rel_tiles_dir)
        )
        layer_ids: List[str] = []
        for src_name, src in sources.items():
            all_sources[f"{group_name}:{src_name}"] = src
        for name, sids in level_source_ids.items():
            group_level_source_ids[f"{group_name}:{name}"] = [
                f"{group_name}:{sid}" for sid in sids
            ]
        seen_interp_for_level: set = set()
        for gl_layer in layers:
            gl_layer = dict(gl_layer)
            source_id = gl_layer["source"]
            full_source = f"{group_name}:{source_id}"
            # Plain level name -- NOT chunk-suffixed, since `source-layer`
            # is deliberately the same across every chunk of a level (see
            # `_sources_and_layers`).
            level_name = gl_layer["source-layer"]
            logical = f"{group_name}:{level_name}"
            # Round 18: `line-color` included too -- streets (`street_overlay`
            # layers) ARE colored by a real numeric score
            # (`_score_maplibre_paint("line", score_col, ...)`, see
            # `build_city_map`), same coalesce(feature-state, interpolate)
            # shape as fill/circle layers now that `_sources_and_layers` also
            # wraps line-color (see that method) -- so streets get a genuine
            # value-derived override color via `window.__colorForValue`,
            # not just a flat highlight.
            for color_prop in ("fill-color", "circle-color", "line-color"):
                paint_val = gl_layer.get("paint", {}).get(color_prop)
                if (
                    isinstance(paint_val, list)
                    and paint_val[:1] == ["coalesce"]
                    and len(paint_val) == 3
                ):
                    found = _extract_score_interpolator(paint_val[2])
                    # Every chunk of a level shares the identical paint
                    # expression, so only the first chunk's interpolator is
                    # kept per logical level (avoids redundant overwrites).
                    if found and logical not in seen_interp_for_level:
                        score_interpolators[logical] = found
                        seen_interp_for_level.add(logical)
            gl_layer["id"] = f"{group_name}:{gl_layer['id']}"
            gl_layer["source"] = full_source
            gl_layer["layout"] = {"visibility": "visible" if visible else "none"}
            all_layers.append(gl_layer)
            layer_ids.append(gl_layer["id"])
        group_layer_ids[group_name] = layer_ids
        for level_name, popup_js in popup_js_by_level.items():
            for layer_id in level_layer_ids[level_name]:
                full_layer_id = f"{group_name}:{layer_id}"
                popup_lookup_entries.append(f"{json.dumps(full_layer_id)}: {popup_js}")
                interactive_layer_ids.append(full_layer_id)

    for group_name, hmap in multi.named_hierarchy_maps.items():
        _add_group(group_name, hmap, visible=(group_name == multi.default))
    for group_name, hmap in multi.overlay_hierarchy_maps.items():
        _add_group(group_name, hmap, visible=multi.overlay_show.get(group_name, True))

    # Item 3/4 (round 11): every fill/circle layer this call produced (hex,
    # circle, census levels -- never streets/development, which are lines,
    # so `line` layers are naturally excluded) is a target for the "Hex/
    # circle/census opacity" slider and, when a field is selected, the
    # "Opacity by" data-driven expression. Circle-type layers additionally
    # get "Circle size by". Built here (not inline in the f-string below)
    # since `all_layers` is only fully populated after both `_add_group`
    # loops above finish.
    shape_layer_id_types: Dict[str, str] = {
        gl_layer["id"]: gl_layer["type"]
        for gl_layer in all_layers
        if gl_layer["type"] in ("fill", "circle")
    }
    circle_layer_ids: List[str] = [
        gl_layer["id"] for gl_layer in all_layers if gl_layer["type"] == "circle"
    ]

    # The basemap (raster OR vector-style) is no longer baked into the
    # initial style object -- it's added client-side by
    # `_client_basemap_js`'s `__applyBasemap()` right after map creation,
    # same as the single-group `build_html` path above (see that function's
    # comment and `_client_basemap_js`'s docstring for why: an OpenFreeMap
    # vector style can't be represented as this dict-literal's single
    # `sources`/`layers` pair).
    first = next(iter(multi.named_hierarchy_maps.values()))
    bounds = first.hierarchy.level_bounds(first.levels[0])
    center = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
    # The DEFAULT group (not necessarily `first`, which is just whichever
    # group was added first) decides the initial zoom, since that's the
    # group actually visible on load -- census when available (coarsest
    # level = simplest choropleth, fastest to load), else hexagons.
    default_hmap = multi.named_hierarchy_maps.get(multi.default, first)
    default_bounds = default_hmap.hierarchy.level_bounds(default_hmap.levels[0])
    # 2026-09-06, explicit user request -- see `_bbox_fit_zoom`'s docstring
    # and the matching comment at the single-group call site above.
    initial_zoom = _bbox_fit_zoom(default_bounds)
    overview_exit_zoom = _bbox_fit_zoom(default_bounds, pad_factor=8.0)

    style_obj = {"version": 8, "sources": all_sources, "layers": all_layers}
    popup_lookup_js = "{\n" + ",\n".join(popup_lookup_entries) + "\n}"

    # Small per-group icons so the switcher isn't plain radio/checkbox text --
    # real reported gap ("layer control icons ... are different from the
    # original"). Folium itself has no per-layer-type icon set to match (its
    # own layer control is native-Leaflet text, see round-8-followup notes),
    # so these are a lightweight, recognizable mnemonic per shape/overlay
    # kind (matched by substring on the group name) rather than a port of
    # anything Folium renders.
    _GROUP_ICONS = [
        ("hex", "⬡"),  # ⬡ hexagon
        ("circ", "●"),  # ● circle
        ("census", "▦"),  # ▦ tiled square (census/GEOID polygons)
        ("street", "\U0001f6e3️"),  # 🛣️ road
        ("develop", "\U0001f3d7️"),  # 🏗️ construction/development overlay
    ]

    def _group_icon(group_name: str) -> str:
        lower = group_name.lower()
        for key, icon in _GROUP_ICONS:
            if key in lower:
                return icon
        return "□"  # ▢ generic fallback

    named_radio_html = "\n".join(
        f'<label><input type="radio" name="__base_group" value="{g}"'
        f'{" checked" if g == multi.default else ""}> {_group_icon(g)} {g}</label><br>'
        for g in multi.named_hierarchy_maps
    )
    # 2026-09-01 (explicit user request: "an option to select none so no
    # hexagons no circles and no census just nothing"): an extra radio with
    # no matching key in `namedGroupLayers` -- the change handler below
    # already does `setGroupVisible(namedGroupLayers[g], g === el.value)`
    # for every real group, so `el.value === 'none'` alone (true for no
    # real group) hides all of them with no special-case JS needed.
    named_radio_html += '\n<label><input type="radio" name="__base_group" value="none"> □ None</label><br>'
    overlay_checkbox_html = "\n".join(
        f'<label><input type="checkbox" name="__overlay_group" value="{g}"'
        f'{" checked" if multi.overlay_show.get(g, True) else ""}> {_group_icon(g)} {g}</label><br>'
        for g in multi.overlay_hierarchy_maps
    )

    # Basemap switcher/opacity/B&W controls (deferred item 1/2 from round-10
    # audit -- Folium's `_control_panel_html` has a background-map dropdown,
    # a basemap-opacity slider, and a "Black & white basemap" checkbox
    # (`transitlos/map/build.py` lines ~1495-1560); MapLibre had one fixed
    # basemap and no controls at all. Now also handles OpenFreeMap-style
    # vector basemaps, not just raster ones -- see `_client_basemap_js`'s
    # docstring for the removal-tracking/opacity-approximation/B&W-disable
    # design that makes both basemap shapes swappable in every direction
    # without breaking the overlay layers stacked on top of "the basemap".
    _first_overlay_layer_id = all_layers[0]["id"] if all_layers else None
    basemap_options_html = "\n".join(
        f'<option value="{key}"{" selected" if key == basemap else ""}>{BASEMAP_LABELS.get(key, key)}</option>'
        for key in BASEMAP_STYLES
    )
    basemap_controls_html = f"""
    <hr>
    <div style="margin-top:4px;">
      <label style="display:block;">Basemap<br>
        <select id="basemapSelect" style="width:100%;">{basemap_options_html}</select>
      </label>
      <label style="display:block;margin-top:6px;">Basemap opacity <span id="basemapOpacityValue">100%</span><br>
        <input type="range" id="basemapOpacitySlider" min="0" max="100" value="100" style="width:100%;">
      </label>
      <label style="display:block;margin-top:4px;">
        <input type="checkbox" id="basemapBwCheckbox"> Black &amp; white basemap
        <span style="color:#888;">(raster basemaps only)</span>
      </label>
    </div>
"""
    if isinstance(basemap, str):
        _initial_basemap_key = basemap
        _basemap_styles_for_js = BASEMAP_STYLES
    else:
        _initial_basemap_key = "__initial"
        _bm_custom = dict(basemap)
        if "type" not in _bm_custom:
            _bm_custom["type"] = "raster" if "tiles" in _bm_custom else "vector-style"
        _basemap_styles_for_js = {**BASEMAP_STYLES, "__initial": _bm_custom}
    basemap_controls_js = _client_basemap_js(
        _basemap_styles_for_js,
        _initial_basemap_key,
        json.dumps(_first_overlay_layer_id),
        with_controls=True,
    )

    # Circle-size-by / hex-circle-census-opacity / opacity-by-field /
    # contrast controls (round-11 port of Folium's `_control_panel_html`'s
    # "Circle size by"/"Hex/circle/census opacity"/"Opacity by"/"Contrast"
    # rows -- see `transitlos.map.build._opacity_helper_js`'s docstring for
    # the exact formula this JS mirrors). All optional except the shape-
    # opacity slider, which is always shown (it needs no field data, just a
    # flat multiplier).
    circle_fields = circle_fields or []
    opacity_fields = opacity_fields or []
    radius_field_domains = radius_field_domains or {}
    opacity_field_domains = opacity_field_domains or {}
    field_labels = field_labels or {}
    circle_field_options_html = "\n".join(
        f'<option value="{f}"{" selected" if f == default_circle_field else ""}>{field_labels.get(f, f)}</option>'
        for f in circle_fields
        if f in radius_field_domains
    )
    opacity_field_options_html = "\n".join(
        f'<option value="{f}">{field_labels.get(f, f)}</option>'
        for f in opacity_fields
        if f in opacity_field_domains
    )
    style_controls_html = f"""
    <hr>
    <div style="margin-top:4px;">
      {"<div id='circleSizeByRow'><label style='display:block;'>Circle size by<br><select id='circleFieldSelect' style='width:100%;'>" + circle_field_options_html + "</select></label></div>" if circle_field_options_html else ""}
      <label style="display:block;margin-top:6px;">Hex/circle/census opacity <span id="shapeOpacityValue">100%</span><br>
        <input type="range" id="shapeOpacitySlider" min="0" max="100" value="100" style="width:100%;">
      </label>
      <label style="display:block;margin-top:6px;">Opacity by<br>
        <select id="opacitySelect" style="width:100%;">
          <option value="none">None (fixed)</option>
          {opacity_field_options_html}
        </select>
      </label>
      <label style="display:block;margin-top:6px;">Contrast <span id="opacityContrastValue">50%</span><br>
        <input type="range" id="opacityContrastSlider" min="0" max="100" value="50" style="width:100%;">
      </label>
    </div>
"""
    radius_field_domains_by_res = radius_field_domains_by_res or {}
    circle_zoom_bands = circle_zoom_bands or {}
    style_controls_js = f"""
    const __radiusFieldDomains = {json.dumps({k: list(v) for k, v in radius_field_domains.items()})};
    const __radiusFieldDomainsByRes = {json.dumps({str(res): {k: list(v) for k, v in d.items()} for res, d in radius_field_domains_by_res.items()})};
    const __circleZoomBands = {json.dumps({str(res): list(band) for res, band in circle_zoom_bands.items()})};
    const __opacityFieldDomains = {json.dumps({k: list(v) for k, v in opacity_field_domains.items()})};
    const __shapeLayerIdTypes = {json.dumps(shape_layer_id_types)};
    const __circleLayerIds = {json.dumps(circle_layer_ids)};
    // Resolution each circle layer id belongs to (parsed from the id's
    // own "h3_<N>" segment, the same convention every level name uses) --
    // lets per-resolution domains be looked up per layer instead of one
    // flat domain applied to every resolution at once (see this
    // function's docstring, "real bug fix" note).
    function __resFromLayerId(id) {{
      var m = /h3_(\\d+)/.exec(id);
      return m ? m[1] : null;
    }}
    function __resForZoom(z) {{
      var best = null;
      Object.keys(__circleZoomBands).forEach(function(resKey) {{
        var band = __circleZoomBands[resKey];
        if (z >= band[0] && z <= band[1]) best = resKey;
      }});
      if (best != null) return best;
      var keys = Object.keys(__circleZoomBands).map(Number);
      if (!keys.length) return null;
      // 2026-09-06 bug fix (see the matching comment in transitlos.map.build's
      // own __resForZoom -- same bug, same fix, both implementations kept
      // in lockstep): `z` past every band means either zoomed in past the
      // finest resolution's band, or zoomed OUT past the coarsest
      // resolution's lower bound (the common low-zoom case) -- H3
      // resolution numbers increase with granularity, so the coarsest
      // resolution is the lowest key, not the highest.
      var minKey = Math.min.apply(null, keys), maxKey = Math.max.apply(null, keys);
      var belowAll = z < __circleZoomBands[String(minKey)][0];
      return String(belowAll ? minKey : maxKey);
    }}
    // Exposed on `window` (not just this script's local scope) for
    // test/automation use, matching `window.__mapLevelSources`'s own
    // precedent above.
    window.__circleLayerIds = __circleLayerIds;
    window.__shapeLayerIdTypes = __shapeLayerIdTypes;
    window.__opacityField = 'none';
    window.__opacityContrast = 0.5;
    window.__shapeOpacity = 1;
    window.__circleField = {json.dumps(default_circle_field)};

    function __domainsForLayer(id) {{
      var res = __resFromLayerId(id);
      return (res != null && __radiusFieldDomainsByRes[res]) ? __radiusFieldDomainsByRes[res] : __radiusFieldDomains;
    }}
    function __circleRadiusExpr(field, domains) {{
      const d = (domains || __radiusFieldDomains)[field];
      if (!d) return 5;
      const d0 = d[0], d1 = (d[1] > d[0] ? d[1] : d[0] + 1e-9);
      // Item 8 null-guard (same fix as `_circle_radius_expr`/
      // `_score_maplibre_paint` in transitlos.map.build): a feature missing
      // `field` would otherwise throw "Expected value to be of type
      // number, but found null instead."
      return ['interpolate', ['linear'], ['coalesce', ['get', field], d0], d0, 3, d1, 18];
    }}
    function __applyCircleRadius() {{
      __circleLayerIds.forEach(function(id) {{
        if (map.getLayer(id)) {{
          map.setPaintProperty(id, 'circle-radius', __circleRadiusExpr(window.__circleField, __domainsForLayer(id)));
        }}
      }});
    }}
    // Legend sub-sections (`#circleLegend`/`#opacityLegend`/`#devLegend` --
    // see `transitlos.map.build._legend_html`, shared verbatim by both
    // renderers via `_inject_maplibre_legend_into_saved_html`) start
    // `display:none` and were never wired up for MapLibre -- only Folium's
    // `_control_panel_js` toggled/populated them. `__fmtLegendNum` is
    // defined by the stats-panel injection (`_inject_maplibre_stats_panel_into_saved_html`),
    // which always runs before a user can interact with these controls; the
    // inline fallback only matters for a page with no census/stats data.
    // Real bug: a share/rate field (e.g. `worldpop_male_share`) is stored
    // as a raw 0..1 fraction on the grid, same as everywhere else on this
    // map -- the click-popup (`__isPercentField`/`__shapePopupNum` in
    // `transitlos.map.build._shape_popup_js`) already rescales it to a
    // percentage for display, but this legend formatter didn't, so
    // "Opacity by: Worldpop male share (%)" showed a raw "0.5" instead of
    // "50%" even though the field's own label already promises "%".
    // `field` is optional (only `__updateOpacityLegend` passes it; circle
    // size/other legends calling `__fmtLeg(v)` alone keep their old plain
    // formatting) since not every `__fmtLeg` call site has a field name to
    // check. Falls back to plain formatting if `__isPercentField` isn't
    // defined yet (script load order) rather than throwing.
    function __fmtLeg(v, field, refAbs) {{
      if (field && window.__isPercentField && window.__isPercentField(field)) {{
        return (Number(v) * 100).toFixed(1) + '%';
      }}
      return window.__fmtLegendNum ? window.__fmtLegendNum(v, refAbs) : String(v);
    }}
    function __activeLegendDomains() {{
      // Legend text tracks whichever resolution is actually on screen at
      // the current zoom, not a flat/finest-only domain -- see this
      // function's docstring's "real bug fix" note.
      var z = (typeof map !== 'undefined' && map.getZoom) ? map.getZoom() : null;
      if (z != null && Object.keys(__circleZoomBands).length) {{
        var res = __resForZoom(z);
        if (res != null && __radiusFieldDomainsByRes[res]) return __radiusFieldDomainsByRes[res];
      }}
      return __radiusFieldDomains;
    }}
    function __updateCircleLegend() {{
      var el = document.getElementById('circleLegend');
      if (!el) return;
      var activeGroup = document.querySelector('input[name="__base_group"]:checked');
      var field = window.__circleField;
      var domains = __activeLegendDomains();
      var d = field ? domains[field] : null;
      var show = !!(activeGroup && activeGroup.value === 'circles' && field && d);
      el.style.display = show ? 'block' : 'none';
      if (!show) return;
      var opt = __circleFieldSelect ? __circleFieldSelect.options[__circleFieldSelect.selectedIndex] : null;
      document.getElementById('circleLegendField').textContent = opt ? opt.textContent : field;
      var refAbs = Math.max(Math.abs(d[0]), Math.abs(d[1]));
      document.getElementById('circleLegendMin').textContent = __fmtLeg(d[0], field, refAbs);
      document.getElementById('circleLegendMid').textContent = __fmtLeg((d[0] + d[1]) / 2, field, refAbs);
      document.getElementById('circleLegendMax').textContent = __fmtLeg(d[1], field, refAbs);
    }}
    var __circleFieldSelect = document.getElementById('circleFieldSelect');
    if (__circleFieldSelect) {{
      __circleFieldSelect.addEventListener('change', function(e) {{
        window.__circleField = e.target.value;
        __applyCircleRadius();
        __updateCircleLegend();
      }});
      __applyCircleRadius();
    }}
    if (typeof map !== 'undefined' && map.on && Object.keys(__circleZoomBands).length) {{
      map.on('zoomend', __updateCircleLegend);
    }}
    __updateCircleLegend();

    // Mirrors `opacityFromField` in `transitlos/map/build.py`'s
    // `_opacity_helper_js` exactly: percentile-clipped-domain normalize,
    // gamma-shape by contrast, then multiply by the flat shape-opacity
    // slider. Built as a MapLibre expression (not a JS function) since
    // paint properties are per-feature and MapLibre evaluates expressions
    // per-feature on the GPU/tile-worker side, not via a JS callback.
    function __shapeOpacityExpr(baseOp) {{
      var field = window.__opacityField;
      var shape = window.__shapeOpacity;
      if (!field || field === 'none' || !__opacityFieldDomains[field]) {{
        return Math.max(0, Math.min(1, baseOp * shape));
      }}
      var d = __opacityFieldDomains[field];
      var span = Math.max(d[1] - d[0], 1e-9);
      var c = Math.max(0, Math.min(1, window.__opacityContrast));
      var minOp = 1 - 0.95 * c, maxOp = 1;
      var gamma = 1 + 2 * c;
      // Bug fix: a bare `['get', field]` is `null` on any feature/level that
      // simply doesn't carry that property (e.g. coarse census levels --
      // state/municipality -- only carry the fields aggregated up to them,
      // while a finer field like an INEGI rate lives only on ageb/block; a
      // circle-source h3 chunk missing a column for some other reason hits
      // the same case). `to-number(null)` on a real MapLibre GL evaluation
      // throws ("Expected value to be of type number, but found null
      // instead"), which aborts the WHOLE style's paint-property update --
      // not just that one feature -- so `setPaintProperty` silently left the
      // PREVIOUS opacity in place for every layer this was applied to after
      // the failing one. Layer application order made hexagons (added
      // first, and whose fields are always present on every h3 cell) look
      // like the only shape the dropdown affected, while circles/census
      // (added later, and far more likely to hit a level missing the field)
      // never got their new opacity. `coalesce` to the field's own domain
      // minimum before the numeric cast removes the null entirely, matching
      // the same null-safety `_score_maplibre_paint`'s color expressions
      // already use via `coalesce`.
      return ['*',
        ['+', minOp, ['*', (maxOp - minOp),
          ['^', ['max', 0, ['min', 1, ['/', ['-', ['to-number', ['coalesce', ['get', field], d[0]]], d[0]], span]]], gamma],
        ]],
        shape,
      ];
    }}
    function __applyShapeOpacity() {{
      Object.keys(__shapeLayerIdTypes).forEach(function(id) {{
        if (!map.getLayer(id)) return;
        var type = __shapeLayerIdTypes[id];
        var prop = type === 'circle' ? 'circle-opacity' : 'fill-opacity';
        map.setPaintProperty(id, prop, __shapeOpacityExpr(0.75));
      }});
    }}
    document.getElementById('shapeOpacitySlider').addEventListener('input', function(e) {{
      var pct = parseInt(e.target.value, 10);
      window.__shapeOpacity = pct / 100;
      document.getElementById('shapeOpacityValue').textContent = pct + '%';
      __applyShapeOpacity();
    }});
    function __updateOpacityLegend() {{
      var el = document.getElementById('opacityLegend');
      if (!el) return;
      var field = window.__opacityField;
      var d = (field && field !== 'none') ? __opacityFieldDomains[field] : null;
      el.style.display = d ? 'block' : 'none';
      if (!d) return;
      var sel = document.getElementById('opacitySelect');
      var opt = sel ? sel.options[sel.selectedIndex] : null;
      document.getElementById('opacityLegendField').textContent = opt ? opt.textContent : field;
      document.getElementById('opacityLegendMin').textContent = __fmtLeg(d[0], field);
      document.getElementById('opacityLegendMax').textContent = __fmtLeg(d[1], field);
    }}
    document.getElementById('opacitySelect').addEventListener('change', function(e) {{
      window.__opacityField = e.target.value;
      __applyShapeOpacity();
      __updateOpacityLegend();
    }});
    document.getElementById('opacityContrastSlider').addEventListener('input', function(e) {{
      var pct = parseInt(e.target.value, 10);
      window.__opacityContrast = pct / 100;
      document.getElementById('opacityContrastValue').textContent = pct + '%';
      if (window.__opacityField && window.__opacityField !== 'none') __applyShapeOpacity();
    }});
"""

    switcher_js = f"""
    const namedGroupLayers = {json.dumps({g: ids for g, ids in group_layer_ids.items() if g in multi.named_hierarchy_maps})};
    const overlayGroupLayers = {json.dumps({g: ids for g, ids in group_layer_ids.items() if g in multi.overlay_hierarchy_maps})};

    function setGroupVisible(layerIds, visible) {{
      layerIds.forEach(function(id) {{
        map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none');
      }});
    }}

    // "Circle size by" only makes sense while the "circles" named group is
    // the active shape -- `circle-radius` is a paint property circle-TYPE
    // layers have, and every other named group (hexagons/census) is a
    // fill/line layer with no such property, so the control was previously
    // shown (and editable) even while pointing at layers it could never
    // actually affect.
    var __circleSizeByRow = document.getElementById('circleSizeByRow');
    function __updateCircleSizeByVisibility(activeGroup) {{
      if (__circleSizeByRow) __circleSizeByRow.style.display = (activeGroup === 'circles') ? '' : 'none';
    }}
    document.querySelectorAll('input[name="__base_group"]').forEach(function(el) {{
      el.addEventListener('change', function() {{
        Object.keys(namedGroupLayers).forEach(function(g) {{
          setGroupVisible(namedGroupLayers[g], g === el.value);
        }});
        __updateCircleSizeByVisibility(el.value);
        if (typeof __updateCircleLegend === 'function') __updateCircleLegend();
      }});
    }});
    (function() {{
      var __checkedGroup = document.querySelector('input[name="__base_group"]:checked');
      __updateCircleSizeByVisibility(__checkedGroup ? __checkedGroup.value : null);
    }})();
    document.querySelectorAll('input[name="__overlay_group"]').forEach(function(el) {{
      el.addEventListener('change', function() {{
        setGroupVisible(overlayGroupLayers[el.value], el.checked);
        // "development" is the one overlay group `_legend_html` has its own
        // sub-legend for (`#devLegend`) -- see `transitlos.map.build`'s
        // `_legend_html`/`show_development`.
        if (el.value === 'development') {{
          var devEl = document.getElementById('devLegend');
          if (devEl) devEl.style.display = el.checked ? 'block' : 'none';
        }}
      }});
    }});
"""

    # Live recolor API (scenario editor / computeAccess() hook), mirroring
    # `MapLibreHierarchyMap.build_html`'s single-group version -- see that
    # method's comment for the rationale. Here sources are namespaced
    # "{group}:{level}" (e.g. "hexagons:h3_8", "census:census_blockgroup"),
    # so callers pass that combined string as `level`.
    recolorable_source_map = {
        logical: sids
        for logical, sids in group_level_source_ids.items()
        if any(
            isinstance(all_sources.get(sid), dict) and "promoteId" in all_sources[sid]
            for sid in sids
        )
    }
    recolor_js = f"""
    // Real MapLibre source ids can outnumber logical levels 1:N when a
    // level was tiled in H3-res4 chunks (see
    // `geohierarchy.maps.folium.tiles.H3_CHUNK_ROW_THRESHOLD`); `level`
    // here is always the logical "{{group}}:{{level}}" name callers already
    // use (e.g. `CENSUS_LEVEL`/`STREETS_LEVEL` in transitlos/map/build.py),
    // and every function below loops over that level's real source ids so
    // a chunked level keeps behaving like one continuous layer.
    window.__mapLevelSources = {json.dumps(recolorable_source_map)};
    // Membership test replacing the old `window.__mapLevelSources.indexOf(level) !== -1`
    // (back when `__mapLevelSources` was a flat array of source ids, before
    // chunked levels needed a level -> [source id, ...] map instead).
    window.__hasMapLevel = function(level) {{
      return level in window.__mapLevelSources;
    }};
    window.__setAccessOverride = function(level, featureId, colorHex) {{
      if (!(level in window.__mapLevelSources)) return false;
      const sourceLayer = level.indexOf(':') === -1 ? level : level.slice(level.indexOf(':') + 1);
      window.__mapLevelSources[level].forEach(function(srcId) {{
        map.setFeatureState({{source: srcId, sourceLayer: sourceLayer, id: featureId}}, {{access_override: colorHex}});
      }});
      return true;
    }};
    window.__clearAccessOverride = function(level, featureId) {{
      if (!(level in window.__mapLevelSources)) return false;
      const sourceLayer = level.indexOf(':') === -1 ? level : level.slice(level.indexOf(':') + 1);
      window.__mapLevelSources[level].forEach(function(srcId) {{
        map.removeFeatureState({{source: srcId, sourceLayer: sourceLayer, id: featureId}}, 'access_override');
      }});
      return true;
    }};
    window.__clearAllAccessOverrides = function(level) {{
      if (!(level in window.__mapLevelSources)) return;
      const sourceLayer = level.indexOf(':') === -1 ? level : level.slice(level.indexOf(':') + 1);
      window.__mapLevelSources[level].forEach(function(srcId) {{
        map.removeFeatureState({{source: srcId, sourceLayer: sourceLayer}}, 'access_override');
      }});
    }};
    // Aggregates `map.querySourceFeatures` across every real source id
    // backing a logical level -- replaces direct
    // `map.getSource(LEVEL)`/`map.querySourceFeatures(LEVEL, ...)` calls,
    // which assumed one source per level (broken once a level can be
    // tiled in H3-res4 chunks -- see `H3_CHUNK_ROW_THRESHOLD`). Pass an
    // explicit `sourceLayer` in `opts` when it differs from the plain
    // level name (e.g. a group namespace prefix already stripped).
    window.__mapLevelQueryFeatures = function(level, opts) {{
      if (!(level in window.__mapLevelSources)) return [];
      const sourceLayer = (opts && opts.sourceLayer) ||
        (level.indexOf(':') === -1 ? level : level.slice(level.indexOf(':') + 1));
      let out = [];
      window.__mapLevelSources[level].forEach(function(srcId) {{
        if (!map.getSource(srcId)) return;
        out = out.concat(map.querySourceFeatures(srcId, {{sourceLayer: sourceLayer}}));
      }});
      return out;
    }};

    // Same-ramp color lookup for a recomputed numeric score (scenario-editor
    // computeAccess() port) -- linear RGB interpolation between the exact
    // sample stops baked into this source's own fill/circle color
    // expression, so an override color matches the static ramp exactly at
    // the sample points and closely between them (16 samples => plenty
    // dense for this purpose).
    window.__scoreInterpolators = {json.dumps(score_interpolators)};
    function __hexToRgb(h) {{
      h = h.replace('#', '');
      if (h.length === 3) h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
      var n = parseInt(h, 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }}
    function __rgbToHex(rgb) {{
      return '#' + rgb.map(function(v) {{
        return Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
      }}).join('');
    }}
    window.__colorForValue = function(level, value) {{
      const stops = window.__scoreInterpolators[level];
      if (!stops || value == null || !isFinite(value)) return null;
      const pts = stops[1];
      if (!pts.length) return null;
      if (value <= pts[0][0]) return pts[0][1];
      if (value >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
      for (let i = 0; i < pts.length - 1; i++) {{
        const [v0, c0] = pts[i], [v1, c1] = pts[i + 1];
        if (value >= v0 && value <= v1) {{
          const t = v1 > v0 ? (value - v0) / (v1 - v0) : 0;
          const a = __hexToRgb(c0), b = __hexToRgb(c1);
          return __rgbToHex([a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]);
        }}
      }}
      return pts[pts.length - 1][1];
    }};
"""

    interaction_js = f"""
    {recolor_js}
    const popupFns = {popup_lookup_js};
    const interactiveLayerIds = {json.dumps(interactive_layer_ids)};
    // `maxWidth` explicit (2026-09-01, live user report): MapLibre's own
    // default (240px) was clamping this popup well below the 480px the
    // shape popup's own inner content (`_shape_popup_js` in transitLOS,
    // widened for exactly this table) already asks for -- the click popup
    // for hexagons/circles/census shapes was cramped regardless of the
    // inner CSS. 600px leaves margin around the 560px content.
    const popup = new maplibregl.Popup({{closeButton: true, closeOnClick: true, maxWidth: '600px'}});

    map.on('click', function(e) {{
      // Bug fix (user report): clicking a stop marker must never ALSO pop
      // open this shape (hex/circle/census) popup underneath it. MapLibre
      // dispatches every registered 'click' listener for a click
      // regardless of layer order or `preventDefault`/`stopPropagation`
      // (see `Evented.fire` -- it loops every listener unconditionally),
      // so a plain ordering/`stopPropagation` fix can't work here across
      // this module and whatever else (e.g. transitlos.map.build's stop
      // markers) also registers its own 'click' listener on the same
      // `map`. Instead, callers that own a higher-priority point layer
      // (stop markers, drawn-route vertices, ...) publish their layer ids
      // on `window.__mapClickPriorityLayers`; checked here, at click time,
      // so it works regardless of which module's JS happens to load/run
      // first.
      const priorityLayers = (window.__mapClickPriorityLayers || []).filter((id) => map.getLayer(id));
      if (priorityLayers.length && map.queryRenderedFeatures(e.point, {{layers: priorityLayers}}).length) {{
        return;
      }}
      const features = map.queryRenderedFeatures(e.point, {{layers: interactiveLayerIds}});
      if (!features.length) return;
      const f = features[0];
      const fn = popupFns[f.layer.id];
      const html = fn ? fn(f.properties) : JSON.stringify(f.properties);
      popup.setLngLat(e.lngLat).setHTML(html).addTo(map);
    }});

    // Hover feature-state (parity with `MapLibreHierarchyMap.build_html`) --
    // this group-switcher build previously had no hover handling at all.
    let hoveredId = null;
    let hoveredLayer = null;
    let hoveredSourceLayer = null;
    map.on('mousemove', function(e) {{
      if (!map.isStyleLoaded()) return;
      const features = map.queryRenderedFeatures(e.point, {{layers: interactiveLayerIds}});
      map.getCanvas().style.cursor = features.length ? 'pointer' : '';
      if (hoveredId !== null && hoveredLayer) {{
        map.setFeatureState({{source: hoveredLayer, sourceLayer: hoveredSourceLayer, id: hoveredId}}, {{hover: false}});
      }}
      if (features.length) {{
        hoveredLayer = features[0].source;
        hoveredSourceLayer = features[0].sourceLayer;
        hoveredId = features[0].id;
        if (hoveredId !== undefined && hoveredId !== null) {{
          map.setFeatureState({{source: hoveredLayer, sourceLayer: hoveredSourceLayer, id: hoveredId}}, {{hover: true}});
        }}
      }} else {{
        hoveredId = null;
        hoveredLayer = null;
        hoveredSourceLayer = null;
      }}
    }});
"""

    draw_html, draw_js = _route_draw_html_js()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="stylesheet" href="{MAPLIBRE_CSS_CDN}">
  <script src="{MAPLIBRE_CDN}"></script>
  <script src="{PMTILES_CDN}"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; }}
    #map {{ width: 100%; height: 100%; }}
    #layer-switcher {{
      position: absolute; top: 10px; right: 50px; z-index: 5;
      background: white; padding: 8px 10px; border-radius: 4px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.3); font: 12px sans-serif;
      max-height: 90vh; overflow-y: auto; width: 220px;
    }}
    #layer-switcher-head {{
      display: flex; align-items: center; justify-content: space-between;
      gap: 8px;
    }}
    #layer-switcher-toggle {{
      border: 1px solid #ccc; background: #f4f4f4; border-radius: 3px;
      font: 11px sans-serif; padding: 2px 6px; cursor: pointer;
    }}
    #layer-switcher-toggle:hover {{ background: #e8e8e8; }}
    #layer-switcher-body {{ margin-top: 6px; }}
    #layer-switcher-body.collapsed {{ display: none; }}
    #legend-slot:empty {{ display: none; }}
    .maplibregl-popup-content table {{ font-size: 12px; }}
  </style>
</head>
<body>
  {_custom_select_js()}
  <div id="map"></div>
  <div id="layer-switcher">
    <div id="layer-switcher-head">
      <strong>Legend</strong>
      <button id="layer-switcher-toggle" type="button" title="Layers" onclick="(function(){{
        var b = document.getElementById('layer-switcher-body');
        var collapsed = b.classList.toggle('collapsed');
        document.getElementById('layer-switcher-toggle').textContent = collapsed ? '🗺️ ▾' : '🗺️ ▴';
      }})()">&#x1F5FA;&#xFE0F; &#9652;</button>
    </div>
    <div id="legend-slot"><!--LEGEND_PLACEHOLDER--></div>
    <div id="layer-switcher-body">
      <strong>Layers</strong><br>
      {named_radio_html}
      <!-- 2026-09-01 (explicit user request: "the opacity slider for
           hexagons/census/circles to be just below the selector between
           hexagons census circles") -- `style_controls_html` (circle-size-by,
           the flat hex/circle/census opacity slider, opacity-by-field,
           contrast) moved to sit directly under the shape radios above,
           instead of after the streets/development overlay checkboxes and
           the basemap switcher. -->
      {style_controls_html}
      {"<hr>" + overlay_checkbox_html if overlay_checkbox_html else ""}
      {basemap_controls_html}
    </div>
  </div>
  {draw_html}
  <script>
    const protocol = new pmtiles.Protocol();
    maplibregl.addProtocol('pmtiles', protocol.tile);

    const map = new maplibregl.Map({{
      container: 'map',
      style: {json.dumps(style_obj)},
      center: {json.dumps(center)},
      zoom: {initial_zoom},
      maxZoom: 24,
    }});
    // Exposed on `window` (2026-09-01, for `code.combined_map`'s zoom-based
    // overview<->city auto-switch): `const map` alone is only reachable
    // from code inside this same `<script>` block. The combined page reads
    // a per-city `map.html`'s live zoom via same-origin
    // `iframe.contentWindow.__mainMap.getZoom()`/`.on('zoomend', ...)` to
    // know when to drop back to its own all-cities overview map.
    window.__mainMap = map;
    // 2026-09-06, explicit user request -- `code.combined_map` reads this
    // (regex-parsed straight out of the saved HTML, same as `zoom: N`
    // above) as the zoom threshold for dropping back to the all-cities
    // overview when zooming out of this city; see `_bbox_fit_zoom`'s
    // `pad_factor` comment for why it's deliberately much lower than the
    // map's own startup zoom, not just one step lower.
    window.__overviewExitZoom = {overview_exit_zoom};
    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    {switcher_js}
    {basemap_controls_js}
    {style_controls_js}
    {interaction_js}
    {draw_js}
  </script>
</body>
</html>
"""
    out.write_text(html)
    return str(out)


def _route_draw_html_js() -> tuple[str, str]:
    """Minimal click-to-add-point route drawing tool (item 1 of the scenario-editor port).

    Deliberately NOT `maplibre-gl-draw`: that library's MapLibre-compatible
    fork is not reliably loadable from a CDN alongside this project's pinned
    `maplibre-gl@5.13.0`, and the UX this needs (draw a fresh line, extend it
    from either endpoint only, undo last point, clear) is small enough that a
    ~100-line custom tool avoids an extra third-party dependency entirely --
    matching the "no new dependency" philosophy `pop_chunks.py` documents for
    the client-side compute step this feeds (see that module's docstring).

    UX, matched to `transitlos/map/build.py`'s `_editor_js` Folium/Geoman tool:
      - "Draw Route" button arms draw mode; each map click appends a point
        and the line redraws live; double-click (or the button again) ends
        the draw.
      - Once a route exists, clicking within `EXTEND_PX` of its FIRST or LAST
        node arms "extend" mode from that end; subsequent clicks append there
        (`unshift` at the start, `push` at the end) -- no mid-route branching,
        matching Folium's "extend from an endpoint only" rule.
      - Undo removes the most-recently-added point (from whichever end is
        active); Clear resets the route to empty.
      - Drag-to-move: whenever the tool isn't mid-draw/add/delete, mousedown
        within `EXTEND_PX` of an existing point grabs it; dragging moves it
        live (map panning is suspended for the gesture); mouseup drops it.
        Matches Folium/Geoman's "any vertex is always draggable" behavior --
        round 6 (Item B, sub-part 1) of the port.
      - "Delete Node" button arms delete mode; clicking a point removes it
        and the line reconnects around the gap (round 6, sub-part 2).
      - "Add Node" button arms mid-route insertion; clicking anywhere on the
        route line projects onto the nearest segment and splices a new point
        in there, splitting that segment -- ported from Folium's
        `closestOnRoute`/`insertOnSegment` (round 6, sub-part 3).

    Exposes `window.__routeState()` (`{points: [[lng,lat],...]}`) and
    `window.__setRoutePoints(points)` so `computeAccess()`
    (`transitlos/map/build.py`'s `_maplibre_compute_access_js`) can
    read/replace the drawn route and this tool's `window.__setAccessOverride()`
    counterpart in `render.py`'s live-recolor plumbing can repaint affected
    features once that recompute lands. Also exposes
    `window.__dragRoutePoint(idx, lng, lat)` / `window.__deleteRoutePoint(idx)`
    / `window.__insertRoutePoint(afterIdx, lng, lat)` as direct-call
    test/automation hooks for the drag/delete/insert gestures, since
    Playwright can't reliably synthesize real mousedown/mousemove/mouseup
    sequences against a WebGL canvas -- each hook runs the exact same code
    the real DOM handlers use. Not ported from Folium: snapping to nearby
    existing stops (`nearbyStop`/`STOP_SNAP_M`) and per-point station
    toggling (every point is already treated as a station in this port, see
    `_maplibre_compute_access_js`'s docstring) -- deferred, matches this
    port's already-documented scope.
    """
    html = """
  <style>
    /* Item 3 (verbatim user request): the route-editor box should NOT show
       by default -- only once edit mode is toggled on via the standalone
       edit-mode icon button (#routeEditToggleBtn, rendered outside this box
       so it stays clickable while the box itself is hidden). */
    #route-toolbar { display:none; position:absolute; top:56px; left:10px; z-index:5;
       background:#ffffff; padding:12px 14px; border-radius:10px;
       box-shadow:0 4px 16px rgba(0,0,0,0.18); font:12px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       min-width:210px; }
    #route-toolbar.rt-open { display:block; }
    #routeEditToggleBtn { position:absolute; top:10px; left:10px; z-index:6;
       font-size:18px; width:34px; height:34px; padding:0; border:1px solid #d8dbe2;
       background:#ffffff; border-radius:8px; cursor:pointer; box-shadow:0 2px 8px rgba(0,0,0,0.18); }
    #routeEditToggleBtn:hover { background:#eef1f5; }
    #routeEditToggleBtn.rt-active { background:#2563eb; border-color:#2563eb; color:#fff; }
    #route-toolbar .rt-title { font-size:13px; font-weight:600; color:#1a1a2e; display:flex;
       align-items:center; gap:6px; margin-bottom:8px; }
    #route-toolbar .rt-badge { font-size:10px; font-weight:600; padding:2px 6px; border-radius:10px;
       background:#eef0f4; color:#666; }
    #route-toolbar .rt-badge.unlocked { background:#e3f6e8; color:#1e8a4c; }
    #route-toolbar button { font:inherit; border:1px solid #d8dbe2; background:#fafbfc; color:#333;
       border-radius:6px; padding:6px 9px; margin:0 4px 4px 0; cursor:pointer; transition:background .12s,opacity .12s; }
    #route-toolbar button:hover:not(:disabled) { background:#eef1f5; }
    #route-toolbar button:disabled { opacity:0.4; cursor:not-allowed; }
    #route-toolbar button.rt-active { background:#2563eb; border-color:#2563eb; color:#fff; }
    #route-toolbar button.rt-primary { background:#2563eb; border-color:#2563eb; color:#fff; font-weight:600; width:100%; margin:0 0 8px 0; padding:8px 10px; }
    #route-toolbar button.rt-primary:hover { background:#1d4ed8; }
    #route-toolbar .rt-group { border-top:1px solid #eee; margin-top:8px; padding-top:8px; }
    #routeHint { margin-top:6px; color:#666; max-width:220px; }
    #routeLockMsg { margin-top:6px; color:#8a5a00; background:#fff6e0; border-radius:6px; padding:6px 8px; }
    /* Round 17 (item 4): compact +/- (add/delete node) + an edit-mode emoji
       button, top-left of the toolbar -- the emoji button is the SOLE entry
       point into the scenario panel (list existing scenarios / create new),
       replacing the old standalone "+ New Scenario" primary button. */
    #rt-icon-row { display:flex; gap:4px; margin-bottom:8px; }
    #rt-icon-row button { font:14px/1 inherit; width:30px; height:30px; padding:0; margin:0;
       border:1px solid #d8dbe2; background:#fafbfc; border-radius:6px; cursor:pointer; }
    #rt-icon-row button:hover { background:#eef1f5; }
    #rt-icon-row button.rt-active { background:#2563eb; border-color:#2563eb; color:#fff; }
    #rtEditModeBtn { font-size:16px; }
    /* Scenario panel: list of saved (read-only) scenarios + "create new". */
    #routeScenarioPanel { display:none; }
    #routeScenarioList { max-height:160px; overflow-y:auto; margin-bottom:6px; }
    .rt-scenario-row { display:flex; align-items:center; justify-content:space-between;
       padding:4px 6px; border:1px solid #eee; border-radius:6px; margin-bottom:4px; font-size:11.5px; }
    .rt-scenario-row .rt-scenario-name { font-weight:600; color:#1a1a2e; }
    .rt-scenario-row .rt-scenario-lock { color:#8a5a00; font-size:10px; }
    .rt-scenario-row button { margin:0; padding:3px 7px; }
    #routeScenarioEmpty { color:#888; font-style:italic; margin-bottom:6px; }
    #routeAddStopsRow { display:flex; align-items:center; gap:5px; margin:6px 0; }
    #routeConfigSection { border-top:1px solid #eee; margin-top:8px; padding-top:8px; }
  </style>
  <button id="routeEditToggleBtn" type="button" title="Edit mode: pick or create a scenario">&#9998;&#65039;</button>
  <div id="route-toolbar">
    <div class="rt-title">Route editor <span id="routeLockBadge" class="rt-badge">read-only</span></div>
    <div id="rt-icon-row">
      <button id="rtAddNodeIconBtn" type="button" title="Add node">&#10133;</button>
      <button id="rtDeleteNodeIconBtn" type="button" title="Delete node">&#10134;</button>
    </div>
    <div id="routeScenarioPanel">
      <div id="routeScenarioList"></div>
      <div id="routeScenarioEmpty">No saved scenarios yet.</div>
      <button id="routeCreateNewBtn" type="button" class="rt-primary">+ Create new route</button>
    </div>
    <div id="routeLockMsg">The default network is read-only. Pick or create a scenario to enable editing.</div>
    <div id="routeToolButtons" style="display:none;">
      <button id="routeDrawBtn" type="button">Draw Route</button>
      <button id="routeAddNodeBtn" type="button">Add Node</button>
      <button id="routeDeleteNodeBtn" type="button">Delete Node</button>
      <button id="routeUndoBtn" type="button">Undo</button>
      <button id="routeClearBtn" type="button">Clear</button>
      <div id="routeAddStopsRow">
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;">
          <input id="routeAddStopsChk" type="checkbox"> Add stops while drawing
        </label>
      </div>
      <div>
        <button id="routeAddStopBtn" type="button">Add Stop</button>
        <button id="routeDeleteStopBtn" type="button">Delete Stop</button>
      </div>
      <div id="routeHint"></div>
      <div id="routeConfigSection"></div>
    </div>
  </div>
"""
    js = r"""
    (function() {
      var EXTEND_PX = 18; // click-tolerance (px) for grabbing an endpoint to extend from
      var points = [];      // [[lng,lat], ...] in route order
      var stations = [];     // parallel bool array: stations[i] true means points[i] is a transit
                              // stop (round 17, item 5) -- a route node is just line geometry, a
                              // stop is a distinct entity that may or may not coincide with one.
      var mode = 'idle';    // 'idle' | 'draw' | 'extend' | 'addnode' | 'delete' | 'addstop' | 'delstop'
      var extendFrom = null; // 'start' | 'end' | null
      var dragIdx = -1;      // index of the point currently being drag-moved, or -1

      // --- edit lock: the default/baseline map is read-only. Editing (draw,
      // add/delete node, drag, undo/clear) is only allowed once a scenario
      // has been started (see the "+ New Scenario" button wired in
      // `_maplibre_compute_access_js`, which calls `window.__setRouteEditLocked(false)`).
      window.__routeEditLocked = true;
      window.__setRouteEditLocked = function(locked) {
        locked = !!locked;
        var wasLocked = window.__routeEditLocked;
        window.__routeEditLocked = locked;
        var toolButtons = document.getElementById('routeToolButtons');
        var badge = document.getElementById('routeLockBadge');
        var lockMsg = document.getElementById('routeLockMsg');
        if (toolButtons) toolButtons.style.display = locked ? 'none' : '';
        if (lockMsg) lockMsg.style.display = locked ? '' : 'none';
        if (badge) {
          badge.textContent = locked ? 'read-only' : 'editing';
          badge.className = 'rt-badge' + (locked ? '' : ' unlocked');
        }
        if (locked && !wasLocked) { setMode('idle'); }
      };

      function addRouteLayers() {
        map.addSource('__route_draw', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
        });
        map.addLayer({
          id: '__route_draw_line',
          type: 'line',
          source: '__route_draw',
          paint: { 'line-color': '#e6194b', 'line-width': 4, 'line-opacity': 0.9 },
        });
        map.addLayer({
          id: '__route_draw_points',
          type: 'circle',
          source: '__route_draw',
          filter: ['all', ['==', ['geometry-type'], 'Point'], ['!=', ['get', 'is_station'], true]],
          paint: { 'circle-color': '#ffffff', 'circle-stroke-color': '#e6194b',
                    'circle-stroke-width': 2, 'circle-radius': 5 },
        });
        // Stations (stops) render as a distinct, larger filled marker so a
        // "route node" and a "transit stop" are visually different entities
        // (round 17, item 5).
        map.addLayer({
          id: '__route_draw_stations',
          type: 'circle',
          source: '__route_draw',
          filter: ['all', ['==', ['geometry-type'], 'Point'], ['==', ['get', 'is_station'], true]],
          paint: { 'circle-color': '#e6194b', 'circle-stroke-color': '#ffffff',
                    'circle-stroke-width': 2, 'circle-radius': 7 },
        });
      }
      // `map.addSource`/`addLayer` throw ("Style is not done loading") if
      // called before the initial style finishes loading (a real race: this
      // script runs synchronously right after `new maplibregl.Map(...)`,
      // before any tiles/style JSON have arrived), which would abort this
      // whole IIFE partway through and leave `window.__routeState` etc.
      // undefined. Match the rest of this file's `map.isStyleLoaded()` guard
      // convention instead of assuming synchronous readiness.
      if (map.isStyleLoaded()) { addRouteLayers(); } else { map.on('load', addRouteLayers); }

      function render() {
        if (!map.getSource('__route_draw')) return;
        var features = [];
        if (points.length > 1) {
          features.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: points }, properties: {} });
        }
        points.forEach(function(p, i) {
          features.push({ type: 'Feature', geometry: { type: 'Point', coordinates: p },
                           properties: { endpoint: (i === 0 || i === points.length - 1),
                                         is_station: !!stations[i] } });
        });
        map.getSource('__route_draw').setData({ type: 'FeatureCollection', features: features });
      }

      function setHint(text) { document.getElementById('routeHint').textContent = text; }

      function setMode(newMode) {
        mode = newMode;
        var drawBtn = document.getElementById('routeDrawBtn');
        var addBtn = document.getElementById('routeAddNodeBtn');
        var delBtn = document.getElementById('routeDeleteNodeBtn');
        drawBtn.textContent = (mode === 'draw') ? 'Finish Draw' : 'Draw Route';
        addBtn.textContent = (mode === 'addnode') ? 'Adding...' : 'Add Node';
        delBtn.textContent = (mode === 'delete') ? 'Deleting...' : 'Delete Node';
        var addStopBtn = document.getElementById('routeAddStopBtn');
        var delStopBtn = document.getElementById('routeDeleteStopBtn');
        if (addStopBtn) addStopBtn.textContent = (mode === 'addstop') ? 'Click a node...' : 'Add Stop';
        if (delStopBtn) delStopBtn.textContent = (mode === 'delstop') ? 'Click a stop...' : 'Delete Stop';
        var addIconBtn = document.getElementById('rtAddNodeIconBtn');
        var delIconBtn = document.getElementById('rtDeleteNodeIconBtn');
        if (addIconBtn) addIconBtn.className = (mode === 'addnode') ? 'rt-active' : '';
        if (delIconBtn) delIconBtn.className = (mode === 'delete') ? 'rt-active' : '';
        if (mode === 'draw') {
          setHint('Click the map to add points. Double-click or "Finish Draw" to stop.');
        } else if (mode === 'addnode') {
          setHint('Click on the route line to insert a new point there.');
        } else if (mode === 'delete') {
          setHint('Click a point to remove it. The route reconnects around it.');
        } else if (mode === 'addstop') {
          setHint('Click an existing node to mark it as a stop.');
        } else if (mode === 'delstop') {
          setHint('Click a stop to unmark it (the node itself stays).');
        } else {
          extendFrom = null;
          setHint(points.length ? 'Click the first or last point to extend, drag any point to move it.' : '');
        }
      }

      document.getElementById('routeDrawBtn').addEventListener('click', function() {
        if (window.__routeEditLocked) return;
        setMode(mode === 'draw' ? 'idle' : 'draw');
      });
      document.getElementById('routeAddNodeBtn').addEventListener('click', function() {
        if (window.__routeEditLocked) return;
        setMode(mode === 'addnode' ? 'idle' : 'addnode');
      });
      document.getElementById('routeDeleteNodeBtn').addEventListener('click', function() {
        if (window.__routeEditLocked) return;
        setMode(mode === 'delete' ? 'idle' : 'delete');
      });
      // Round 17 (item 4): the +/- icons top-left are shortcuts for the same
      // add-node/delete-node modes as the labeled buttons below (once the
      // "Create new route" flow reveals the toolbox).
      document.getElementById('rtAddNodeIconBtn').addEventListener('click', function() {
        if (window.__routeEditLocked) return;
        setMode(mode === 'addnode' ? 'idle' : 'addnode');
      });
      document.getElementById('rtDeleteNodeIconBtn').addEventListener('click', function() {
        if (window.__routeEditLocked) return;
        setMode(mode === 'delete' ? 'idle' : 'delete');
      });
      // Round 17 (item 5): distinct add-stop/delete-stop modes -- a stop is a
      // transit-stop entity flagged on an EXISTING node, not a new geometric
      // point (that's Add Node's job).
      document.getElementById('routeAddStopBtn').addEventListener('click', function() {
        if (window.__routeEditLocked) return;
        setMode(mode === 'addstop' ? 'idle' : 'addstop');
      });
      document.getElementById('routeDeleteStopBtn').addEventListener('click', function() {
        if (window.__routeEditLocked) return;
        setMode(mode === 'delstop' ? 'idle' : 'delstop');
      });
      document.getElementById('routeUndoBtn').addEventListener('click', function() {
        if (window.__routeEditLocked || !points.length) return;
        if (extendFrom === 'start') { points.shift(); stations.shift(); }
        else { points.pop(); stations.pop(); }
        render();
      });
      document.getElementById('routeClearBtn').addEventListener('click', function() {
        if (window.__routeEditLocked) return;
        points = [];
        stations = [];
        extendFrom = null;
        setMode('idle');
        render();
      });

      map.on('dblclick', function(e) {
        if (window.__routeEditLocked) return;
        if (mode === 'draw') { e.preventDefault(); setMode('idle'); }
      });

      // --- point hit-testing, shared by delete/drag ------------------------
      function nearestPointIdx(pt, tolerancePx) {
        var best = -1, bestD = tolerancePx;
        for (var i = 0; i < points.length; i++) {
          var px = map.project(points[i]);
          var d = Math.hypot(pt.x - px.x, pt.y - px.y);
          if (d <= bestD) { bestD = d; best = i; }
        }
        return best;
      }

      // Closest point on the closest segment of the route (for mid-route
      // insertion, matching Folium's `closestOnRoute`/`insertOnSegment`: the
      // new point SPLITS the nearest segment rather than being appended).
      function closestOnRoute(lngLat) {
        if (points.length < 2) return {idx: -1, lngLat: null};
        var p = map.project(lngLat);
        var best = -1, bestD = Infinity, bestLngLat = null;
        for (var i = 0; i < points.length - 1; i++) {
          var a = map.project(points[i]), b = map.project(points[i + 1]);
          var abx = b.x - a.x, aby = b.y - a.y;
          var len2 = abx * abx + aby * aby;
          var t = len2 > 0 ? Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2)) : 0;
          var cx = a.x + t * abx, cy = a.y + t * aby;
          var d = Math.hypot(p.x - cx, p.y - cy);
          if (d < bestD) {
            bestD = d; best = i;
            var ll = map.unproject([cx, cy]);
            bestLngLat = [ll.lng, ll.lat];
          }
        }
        return {idx: best, lngLat: bestLngLat};
      }

      // --- drag-to-move an existing point -----------------------------------
      // Active whenever the tool isn't mid-draw/add/delete (i.e. 'idle' or
      // 'extend', which a plain click-near-endpoint can arm) -- matches
      // Folium/Geoman's "any existing vertex is always draggable" behavior.
      map.on('mousedown', function(e) {
        if (window.__routeEditLocked) return;
        if (mode === 'draw' || mode === 'addnode' || mode === 'delete') return;
        var idx = nearestPointIdx(e.point, EXTEND_PX);
        if (idx === -1) return;
        e.preventDefault();
        dragIdx = idx;
        map.dragPan.disable();
        map.getCanvas().style.cursor = 'grabbing';
      });
      map.on('mousemove', function(e) {
        if (dragIdx === -1) return;
        points[dragIdx] = [e.lngLat.lng, e.lngLat.lat];
        render();
      });
      function endDrag() {
        if (dragIdx === -1) return;
        dragIdx = -1;
        map.dragPan.enable();
        map.getCanvas().style.cursor = '';
      }
      map.on('mouseup', endDrag);
      map.on('mouseout', endDrag);

      map.on('click', function(e) {
        if (window.__routeEditLocked) return;
        // A drag that just ended fires a trailing 'click' on some browsers;
        // suppress it so a drag-move never ALSO adds/deletes a point.
        if (dragIdx !== -1) return;
        var lngLat = [e.lngLat.lng, e.lngLat.lat];

        if (mode === 'draw') {
          points.push(lngLat);
          // Live-checked on every click (round 17, item 5): the "Add stops
          // while drawing" checkbox can be toggled mid-draw, and clicks
          // after that point must stop/start adding stops immediately --
          // never just latched at draw-start.
          var addStopsChk = document.getElementById('routeAddStopsChk');
          stations.push(!!(addStopsChk && addStopsChk.checked));
          render();
          return;
        }

        if (mode === 'delete') {
          var delIdx = nearestPointIdx(e.point, EXTEND_PX);
          if (delIdx !== -1) { points.splice(delIdx, 1); stations.splice(delIdx, 1); render(); }
          return;
        }

        if (mode === 'addnode') {
          var hit = closestOnRoute(lngLat);
          if (hit.idx >= 0 && hit.lngLat) { points.splice(hit.idx + 1, 0, hit.lngLat); stations.splice(hit.idx + 1, 0, false); render(); }
          return;
        }

        if (mode === 'addstop') {
          var asIdx = nearestPointIdx(e.point, EXTEND_PX);
          if (asIdx !== -1) { stations[asIdx] = true; render(); }
          return;
        }

        if (mode === 'delstop') {
          var dsIdx = nearestPointIdx(e.point, EXTEND_PX);
          if (dsIdx !== -1) { stations[dsIdx] = false; render(); }
          return;
        }

        if (mode === 'extend' && extendFrom) {
          if (extendFrom === 'start') { points.unshift(lngLat); stations.unshift(false); }
          else { points.push(lngLat); stations.push(false); }
          render();
          return;
        }

        // idle: clicking near an endpoint arms extend-from-that-end.
        if (points.length) {
          var startPx = map.project(points[0]);
          var endPx = map.project(points[points.length - 1]);
          var dStart = Math.hypot(e.point.x - startPx.x, e.point.y - startPx.y);
          var dEnd = Math.hypot(e.point.x - endPx.x, e.point.y - endPx.y);
          if (Math.min(dStart, dEnd) <= EXTEND_PX) {
            extendFrom = dStart <= dEnd ? 'start' : 'end';
            mode = 'extend';
            setHint('Extending from the ' + extendFrom + '. Click the map to add points.');
          }
        }
      });

      // Exposed for a future computeAccess() pass (item 2) to read/replace
      // the drawn route and for tests/verification.
      // `stations` (round 17, item 5): parallel bool array, `stations[i]`
      // true iff `points[i]` is also a transit stop, not just line geometry.
      window.__routeState = function() { return { points: points.slice(), stations: stations.slice(), mode: mode }; };
      window.__setRoutePoints = function(newPoints, newStations) {
        points = (newPoints || []).map(function(p) { return [p[0], p[1]]; });
        stations = (newStations || []).map(function(s) { return !!s; });
        while (stations.length < points.length) stations.push(false);
        stations.length = points.length;
        render();
      };
      window.__setRouteStopFlag = function(idx, isStop) {
        if (idx < 0 || idx >= points.length) return false;
        stations[idx] = !!isStop;
        render();
        return true;
      };
      // Test/automation hooks for the drag-move and delete-node gestures,
      // which Playwright can't easily fire as real DOM mouse events against a
      // WebGL canvas -- these call the exact same code paths the real
      // mousedown/mousemove/mouseup and delete-mode click handlers use.
      window.__dragRoutePoint = function(idx, lng, lat) {
        if (idx < 0 || idx >= points.length) return false;
        points[idx] = [lng, lat];
        render();
        return true;
      };
      window.__deleteRoutePoint = function(idx) {
        if (idx < 0 || idx >= points.length) return false;
        points.splice(idx, 1);
        stations.splice(idx, 1);
        render();
        return true;
      };
      window.__insertRoutePoint = function(afterIdx, lng, lat) {
        if (afterIdx < -1 || afterIdx >= points.length) return false;
        points.splice(afterIdx + 1, 0, [lng, lat]);
        stations.splice(afterIdx + 1, 0, false);
        render();
        return true;
      };
    })();
"""
    return html, js
