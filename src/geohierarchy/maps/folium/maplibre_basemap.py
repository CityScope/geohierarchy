"""MapLibre GL vector-tile background for Folium maps (via ``maplibre-gl-leaflet``)."""

from __future__ import annotations

from branca.element import MacroElement
from jinja2 import Template

MAPLIBRE_JS = "https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.js"
MAPLIBRE_CSS = "https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.css"
MAPLIBRE_LEAFLET_JS = (
    "https://unpkg.com/@maplibre/maplibre-gl-leaflet@0.1.3/leaflet-maplibre-gl.js"
)


class MapLibreBasemap(MacroElement):
    """A MapLibre GL style used as the background layer of a Folium map.

    Leaflet draws it in ``pane`` (default: the tile pane, below overlays);
    it is not part of the layer control.
    """

    _template = Template(
        """
        {% macro header(this, kwargs) %}
        <link rel="stylesheet" href="{{ this.css }}"/>
        <script src="{{ this.js }}"></script>
        <script src="{{ this.leaflet_js }}"></script>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            // Leaflet needs a max zoom for fitBounds/zoom controls; a vector style does not provide one.
            if (map.options.maxZoom === undefined) { map.options.maxZoom = 20; }
            function hasWebGL() {
                try { var c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); }
                catch (e) { return false; }
            }
            // Plain OpenStreetMap tiles always go underneath: they are what you see if the vector background cannot
            // draw (blocked scripts or workers, no WebGL, notebook sandboxes). The MapLibre vector style goes on top.
            L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19, attribution: '&copy; OpenStreetMap contributors', zIndex: -1
            }).addTo(map);
            if (typeof L.maplibreGL === 'function' && typeof maplibregl !== 'undefined' && hasWebGL()) {
                try {
                    var gl = L.maplibreGL({
                        style: {{ this.style|tojson }},
                        attribution: {{ this.attribution|tojson }},
                        pane: {{ this.pane|tojson }}
                    }).addTo(map);
                    // A style paints a solid background even if its tiles never load (e.g. blocked workers): if no
                    // vector tile arrives in a few seconds, drop the vector layer so the OpenStreetMap tiles show.
                    var tilesLoaded = false, dropGl = function() {
                        if (!tilesLoaded && map.hasLayer(gl)) { map.removeLayer(gl); console.warn('MapLibre tiles did not load, showing OpenStreetMap tiles'); }
                    };
                    var mgl = gl.getMaplibreMap();
                    mgl.on('sourcedata', function(e) { if (e.tile) { tilesLoaded = true; } });
                    mgl.on('error', function() { setTimeout(dropGl, 1500); });
                    setTimeout(dropGl, 8000);
                } catch (e) { console.warn('MapLibre background failed, showing OpenStreetMap tiles', e); }
            }
        })();
        {% endmacro %}
        """
    )

    def __init__(self, style: str, attribution: str = "", pane: str = "tilePane"):
        super().__init__()
        self._name = "MapLibreBasemap"
        self.style = style
        self.attribution = attribution
        self.pane = pane
        self.css = MAPLIBRE_CSS
        self.js = MAPLIBRE_JS
        self.leaflet_js = MAPLIBRE_LEAFLET_JS
