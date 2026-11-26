# -*- coding: utf-8 -*-
"""Geocoding helpers (one implementation for every package of the project): a place's boundary, place-name suggestions for a
query or an area, and the country / subdivision of a point.

Nominatim (OpenStreetMap) is asked with a timeout and retries on rate limits; reverse-geocoding answers are cached on disk
(``<cache>/geocoding/reverse.json``; cache folder from ``cache_dir``, else ``GEOHIERARCHY_CACHE_DIR``, else the temp folder), since
the country of a point never changes and the analysis date of a feed must not depend on a busy geocoder.
"""

import json
import logging
import os
import tempfile
import time
import warnings
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pycountry
import requests
from shapely.geometry import MultiPolygon, Point, Polygon, shape as shapely_shape

logger = logging.getLogger(__name__)

NOMINATIM = "https://nominatim.openstreetmap.org"
USER_AGENT = "geohierarchy (https://github.com/CityScope/geohierarchy)"
CACHE_ENV_VAR = "GEOHIERARCHY_CACHE_DIR"


def get_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Folder of geohierarchy's caches: ``cache_dir``, else ``GEOHIERARCHY_CACHE_DIR``, else ``<tempdir>/geohierarchy-cache``.

    Args:
        cache_dir: Explicit folder.

    Returns:
        The folder (not created).
    """
    return Path(cache_dir or os.environ.get(CACHE_ENV_VAR) or Path(tempfile.gettempdir()) / "geohierarchy-cache")


def _nominatim_get(path: str, params: dict, user_agent: str = USER_AGENT, timeout: float = 30) -> requests.Response:
    """One Nominatim request with retries on rate limits, server errors and dropped connections."""
    last: Optional[Exception] = None
    for attempt in range(6):
        try:
            resp = requests.get(f"{NOMINATIM}/{path}", headers={"User-Agent": user_agent}, params=params, timeout=timeout)
            blocked = resp.status_code >= 600 or (resp.ok and resp.text[:30].lstrip().startswith(("<?xml", "<!DOCTYPE", "<html")))   # a throttling page, not an answer
            if resp.status_code in (429, 500, 502, 503, 504) or blocked:
                wait = float(resp.headers.get("Retry-After") or 0) or 2.0 * (attempt + 1)
                time.sleep(min(wait, 60))
                last = requests.HTTPError(f"{resp.status_code} from Nominatim")
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise last if last else RuntimeError("Nominatim request failed")


def reverse_geocode(lat: float, lon: float, cache_dir: Optional[Union[str, Path]] = None) -> dict:
    """Nominatim's address of a point (zoom 10: city level), cached on disk by the point rounded to 0.01 degree.

    Args:
        lat: Latitude.
        lon: Longitude.
        cache_dir: Cache folder (see :func:`get_cache_dir`).

    Returns:
        The ``address`` dict of the answer (``country_code``, ``state``, ``city`` ...).
    """
    f = get_cache_dir(cache_dir) / "geocoding" / "reverse.json"
    key = f"{round(lat, 2)},{round(lon, 2)}"
    try:
        cache = json.loads(f.read_text()) if f.is_file() else {}
    except Exception:  # noqa: BLE001
        cache = {}
    if key in cache:
        return cache[key]
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 10, "addressdetails": 1}
    address = _nominatim_get("reverse", params).json().get("address", {})
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        cache[key] = address
        f.write_text(json.dumps(cache))
    except Exception:  # noqa: BLE001
        pass
    return address


def get_country_region(
    lat: float,
    lon: float,
    code_format: str = "alpha_2",
    get_region: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Union[str, Tuple[Optional[str], Optional[str]]]:
    """Country (and ISO 3166-2 subdivision) of a point, by reverse geocoding.

    Args:
        lat: Latitude.
        lon: Longitude.
        code_format: Country code returned: ``"alpha_2"`` (default), ``"alpha_3"``, ``"numeric"`` or ``"name"``.
        get_region: Also return the subdivision code (the address' state / region matched to ``pycountry``, fuzzy if needed).
        cache_dir: Cache folder (see :func:`get_cache_dir`).

    Returns:
        ``(country code, subdivision code or None)``; with ``get_region=False`` the country code alone (``""`` if unknown).
    """
    address = reverse_geocode(lat, lon, cache_dir)
    iso2 = address.get("country_code", "").upper()
    region_name = address.get("state") or address.get("region")
    country_code = iso2
    if iso2:
        country = pycountry.countries.get(alpha_2=iso2)
        if country is not None and code_format != "alpha_2":
            country_code = {"alpha_3": country.alpha_3, "numeric": country.numeric, "name": country.name}.get(code_format, iso2)
    if not get_region:
        return country_code
    # Nominatim's own ISO 3166-2 code of the first-level subdivision, when it gives one (e.g. "ES-PV" for Euskadi)
    subdivision_code = address.get("ISO3166-2-lvl4") or None
    if subdivision_code is None and iso2 and region_name:
        try:
            subdivisions = list(pycountry.subdivisions.get(country_code=iso2))
            subdivision_code = next((s.code for s in subdivisions if s.name.lower() == region_name.lower()), None)
            if subdivision_code is None:
                match = get_close_matches(region_name, [s.name for s in subdivisions], n=1, cutoff=0.6)
                if match:
                    subdivision_code = next(s.code for s in subdivisions if s.name == match[0])
                    warnings.warn(f"Fuzzy match used for region '{region_name}' -> '{match[0]}'")
        except (LookupError, TypeError):
            subdivision_code = None
    return country_code, subdivision_code


def get_city_geometry(
    city_name: str,
    user_agent: str = USER_AGENT,
    cache_dir: Optional[Union[str, Path]] = None,
    limit: int = 10,
    attempts: int = 4,
) -> gpd.GeoDataFrame:
    """A place's boundary polygon from Nominatim: the first Polygon / MultiPolygon among its answers (a bare point -- a place
    marker -- is skipped).

    Nominatim is rate limited and sometimes answers differently under load: answers without a polygon are asked again
    (``attempts`` times, with a growing wait). With ``cache_dir`` the boundary is saved there and read back next time, so the
    same name always gives the same polygon (delete the file to search again).

    Args:
        city_name: Place name.
        user_agent: User agent sent to Nominatim.
        cache_dir: Folder of cached boundaries (``<sha1 of the name>.geojson``); None = no cache.
        limit: Answers requested per search.
        attempts: Searches before giving up.

    Returns:
        One-row GeoDataFrame (EPSG:4326) with ``name`` / ``display_name`` (the place's full name).

    Raises:
        ValueError: If no boundary polygon is found.
    """
    import hashlib

    cached = Path(cache_dir) / (hashlib.sha1(city_name.encode("utf-8")).hexdigest()[:16] + ".geojson") if cache_dir else None
    if cached is not None and cached.is_file():
        gdf = gpd.read_file(cached).to_crs(4326)
        if "name" not in gdf.columns and "display_name" in gdf.columns:
            gdf.insert(0, "name", gdf["display_name"])
        return gdf
    params = {"q": city_name, "format": "jsonv2", "polygon_geojson": 1, "limit": limit}
    for attempt in range(1, attempts + 1):
        for result in _nominatim_get("search", params, user_agent=user_agent, timeout=30).json():
            geojson = result.get("geojson") or {}
            if geojson.get("type") in ("Polygon", "MultiPolygon"):
                name = result.get("display_name", city_name)
                gdf = gpd.GeoDataFrame({"name": [name], "display_name": [name]}, geometry=[shapely_shape(geojson)], crs="EPSG:4326")
                if cached is not None:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    gdf.to_file(cached, driver="GeoJSON")
                return gdf
        if attempt < attempts:
            time.sleep(5 * attempt)
    raise ValueError(f"No boundary polygon found for {city_name!r}.")


def get_geographic_suggestions_from_string(
    query: str,
    user_agent: str = "MobilityDatabaseClient",
    max_results: int = 25
) -> Dict[str, List[str]]:
    """Suggest country codes, subdivisions, and municipalities from a query string.

    Args:
        query: Place name.
        user_agent: User agent sent to Nominatim.
        max_results: Maximum number of results.

    Returns:
        ``{"country_codes": [...], "subdivisions": [...], "municipalities": [...]}``.
    """
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    try:
        locations = geolocator.geocode(
            query, addressdetails=True, language='en', exactly_one=False, limit=max_results
        )
        if locations:
            for loc in locations:
                address = loc.raw.get('address', {})
                if country_code := address.get('country_code'):
                    suggested_country_codes.add(country_code.upper())
                for key in ['state', 'province', 'region', 'county']:
                    if value := address.get(key):
                        suggested_subdivision_names.add(value)
                for key in ['city', 'town', 'village', 'county']:
                    if value := address.get(key):
                        suggested_municipalities.add(value)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.warning(f"Geocoding failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected geocoding error: {e}")

    return {
        'country_codes': sorted(suggested_country_codes),
        'subdivision_names': sorted(suggested_subdivision_names),
        'municipalities': sorted(suggested_municipalities)
    }


def get_geographic_suggestions_from_aoi(
    aoi: Union[Polygon, MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries],
    num_points: int = 1,
    user_agent: str = "MobilityDatabaseClient"
) -> Dict[str, List[str]]:
    """Reverse-geocode AOI geometry to suggest country, subdivision, and municipality.

    Args:
        aoi: Area (polygon or GeoDataFrame / GeoSeries).
        num_points: Random points inside it to reverse-geocode.
        user_agent: User agent sent to Nominatim.

    Returns:
        ``{"country_codes": [...], "subdivisions": [...], "municipalities": [...]}``.

    Raises:
        ValueError: If the area is empty.
        TypeError: For an unsupported area type.
    """
    import random
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if aoi.empty:
            raise ValueError("GeoDataFrame/GeoSeries is empty.")
        target_geometry = aoi.to_crs(4326).unary_union
    elif isinstance(aoi, (Polygon, MultiPolygon)):
        target_geometry = aoi
    else:
        raise TypeError("AOI must be Polygon, MultiPolygon, GeoDataFrame, or GeoSeries.")

    if target_geometry.is_empty:
        raise ValueError("AOI geometry is empty.")

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    points_to_geocode: List[Point] = []
    min_lon, min_lat, max_lon, max_lat = target_geometry.bounds

    if num_points <= 0:
        num_points = 1
    if num_points == 1:
        points_to_geocode.append(target_geometry.representative_point())
    else:
        for _ in range(num_points):
            points_to_geocode.append(Point(random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat)))

    for i, point in enumerate(points_to_geocode):
        lat, lon = point.y, point.x
        logger.debug(f"Reverse geocoding point {i+1}/{len(points_to_geocode)}: ({lat}, {lon})")
        try:
            location = geolocator.reverse((lat, lon), language='en')
            if location and location.raw:
                address = location.raw.get('address', {})
                if cc := address.get('country_code'):
                    suggested_country_codes.add(cc.upper())
                if subdivision := address.get('state') or address.get('province') or address.get('region') or address.get('county'):
                    suggested_subdivision_names.add(subdivision)
                if municipality := address.get('city') or address.get('town') or address.get('village') or address.get('county'):
                    suggested_municipalities.add(municipality)
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning(f"Geocoding failed for point ({lat}, {lon}): {e}")
        except Exception as e:
            logger.error(f"Unexpected error for point ({lat}, {lon}): {e}")

    return {
        'country_codes': sorted(list(suggested_country_codes)),
        'subdivision_names': sorted(list(suggested_subdivision_names)),
        'municipalities': sorted(list(suggested_municipalities))
    }
