"""PMTiles v3 header (zoom range, bounds) read locally, so a MapLibre page can declare each vector source as a tile template with its
own bounds and zooms instead of a ``pmtiles://`` TileJSON URL.

Why: MapLibre fetches the TileJSON (the archive header) of every ``url`` source as soon as the source is added, visible or not. A nationwide
page with thousands of chunk sources (one per state and level, ~4,000 voting-district chunks) then reads thousands of headers at start-up.
A ``tiles: ["pmtiles://<file>/{z}/{x}/{y}"]`` source with ``bounds`` / ``minzoom`` / ``maxzoom`` is not touched until a tile inside its bounds
and zoom range is needed (the pmtiles protocol reads the header then, once).
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Optional


def read_header(path) -> Optional[Dict]:
    """``{"minzoom", "maxzoom", "bounds": [w, s, e, n]}`` of a PMTiles v3 file, or None when it is not one (missing / older version).

    Args:
        path: Path of the .pmtiles file.
    """
    p = Path(path)
    try:
        with open(p, "rb") as f:
            h = f.read(127)
    except OSError:
        return None
    if len(h) < 127 or h[:7] != b"PMTiles" or h[7] != 3:
        return None
    minz, maxz = h[100], h[101]
    w, s, e, n = (v / 1e7 for v in struct.unpack_from("<iiii", h, 102))
    out = {"minzoom": int(minz), "maxzoom": int(maxz)}
    if e > w and n > s:
        out["bounds"] = [round(w, 6), round(s, 6), round(e, 6), round(n, 6)]
    return out


def lazy_source(url_path: str, file_path, maxzoom: Optional[int] = None) -> Optional[Dict]:
    """The source fields for a lazily read archive: ``tiles`` template (+ minzoom / maxzoom / bounds from its header), or None if unreadable.

    Args:
        url_path: Path of the archive as the page requests it (relative URL, no ``pmtiles://``).
        file_path: The archive on disk (its header is read).
        maxzoom: Optional cap on the source's maximum zoom.
    """
    hdr = read_header(file_path)
    if hdr is None:
        return None
    out = {"tiles": [f"pmtiles://{url_path}/{{z}}/{{x}}/{{y}}"], "minzoom": hdr["minzoom"],
           "maxzoom": min(hdr["maxzoom"], maxzoom) if maxzoom is not None else hdr["maxzoom"]}
    if "bounds" in hdr:
        out["bounds"] = hdr["bounds"]
    return out
