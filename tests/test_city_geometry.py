"""get_city_geometry: the first polygon among Nominatim's answers (points skipped), cached by name."""

import pytest

from geohierarchy import geocoding


class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


POINT = {"display_name": "Marker", "geojson": {"type": "Point", "coordinates": [0, 0]}}
SQUARE = {"display_name": "Town, Country", "geojson": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}


def test_first_polygon_and_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(geocoding, "_nominatim_get", lambda *a, **k: calls.append(1) or _Resp([POINT, SQUARE]))
    g = geocoding.get_city_geometry("Town", cache_dir=tmp_path)
    assert g["display_name"].iloc[0] == "Town, Country" and g.geometry.iloc[0].geom_type == "Polygon"
    again = geocoding.get_city_geometry("Town", cache_dir=tmp_path)   # from the cache: no new request
    assert len(calls) == 1 and again["name"].iloc[0] == "Town, Country"


def test_no_polygon(monkeypatch):
    monkeypatch.setattr(geocoding, "_nominatim_get", lambda *a, **k: _Resp([POINT]))
    monkeypatch.setattr(geocoding.time, "sleep", lambda s: None)
    with pytest.raises(ValueError):
        geocoding.get_city_geometry("Nowhere", attempts=2)
