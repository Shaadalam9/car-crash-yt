"""Strict evidence based location resolution and optional geocoding."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import settings
from .shared import (
    clean_text,
    load_json,
    normalise_string_list,
    optional_text,
    recover_json,
    save_state,
    write_json_atomic,
)


CONTINENT_CODES = {
    "Africa": set("DZ AO BJ BW BF BI CV CM CF TD KM CG CD CI DJ EG GQ ER SZ ET GA GM GH GN GW KE LS LR LY MG MW ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA SS SD TZ TG TN UG ZM ZW".split()),
    "Asia": set("AF AM AZ BH BD BT BN KH CN CY GE IN ID IR IQ IL JP JO KZ KW KG LA LB MY MV MN MM NP KP OM PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE".split()),
    "Europe": set("AL AD AT BY BE BA BG HR CZ DK EE FI FR DE GR HU IS IE IT XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH UA GB VA".split()),
    "North America": set("AG BS BB BZ CA CR CU DM DO SV GD GT HT HN JM MX NI PA KN LC VC US".split()),
    "South America": set("AR BO BR CL CO EC GY PY PE SR UY VE".split()),
    "Oceania": set("AU FJ KI MH FM NR NZ PW PG WS SB TO TV VU".split()),
}
LOCATION_RESOLUTION_VERSION = "segment_evidence_location_v3"


def _continent(iso2: Optional[str]) -> Optional[str]:
    if not iso2:
        return None
    for name, codes in CONTINENT_CODES.items():
        if iso2.upper() in codes:
            return name
    return None


def _iso3(iso2: Optional[str]) -> Optional[str]:
    if not iso2:
        return None


def _coordinate(value: Any, minimum: float, maximum: float) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if minimum <= result <= maximum else None


def _apply_nominatim_item(
    result: Dict[str, Any],
    item: Dict[str, Any],
    locality: Optional[str],
    state_name: Optional[str],
    country_name: Optional[str],
    *,
    preserve_coordinates: bool,
) -> None:
    address = item.get("address", {}) if isinstance(item.get("address"), dict) else {}
    result["locality"] = next(
        (
            optional_text(address.get(name))
            for name in ("city", "town", "village", "municipality", "borough")
            if optional_text(address.get(name))
        ),
        locality,
    )
    result["state"] = optional_text(address.get("state")) or state_name
    result["country"] = optional_text(address.get("country")) or country_name
    iso2 = optional_text(address.get("country_code"))
    subdivision = optional_text(
        address.get("ISO3166-2-lvl4") or address.get("ISO3166-2-lvl3")
    )
    if iso2 and iso2.upper() in {"US", "CA"} and subdivision and "-" in subdivision:
        result["state"] = subdivision.rsplit("-", 1)[-1]
    result["iso3"] = _iso3(iso2)
    result["continent"] = _continent(iso2)
    if not preserve_coordinates:
        try:
            result["lat"] = round(float(item.get("lat")), 7)
            result["lon"] = round(float(item.get("lon")), 7)
        except (TypeError, ValueError):
            pass
    namedetails = item.get("namedetails", {})
    alternatives = list(result["locality_aka"])
    if isinstance(namedetails, dict):
        for name in ("name:en", "official_name", "alt_name"):
            value = optional_text(namedetails.get(name))
            if value:
                alternatives.extend(re.split(r"[;,]", value))
    result["locality_aka"] = [
        name
        for name in normalise_string_list(alternatives)
        if not result["locality"]
        or name.casefold() != str(result["locality"]).casefold()
    ]
    try:
        import pycountry

        country = pycountry.countries.get(alpha_2=iso2.upper())
        return str(country.alpha_3) if country else None
    except Exception:
        return None


def geocode(fields: Dict[str, Any], cache: Dict[str, Any]) -> Dict[str, Any]:
    locality = optional_text(fields.get("locality"))
    state_name = optional_text(fields.get("state"))
    country_name = optional_text(fields.get("country"))
    lat = _coordinate(fields.get("lat"), -90.0, 90.0)
    lon = _coordinate(fields.get("lon"), -180.0, 180.0)
    if (lat is None) != (lon is None):
        lat = lon = None
    result = {
        "locality": locality,
        "locality_aka": normalise_string_list(fields.get("locality_aka")),
        "state": state_name,
        "country": country_name,
        "iso3": None,
        "continent": None,
        "lat": lat,
        "lon": lon,
        "geocode_status": "not_attempted",
        "location_resolution_version": LOCATION_RESOLUTION_VERSION,
    }
    reverse_lookup = lat is not None and lon is not None
    parts = [part for part in (locality, state_name, country_name) if part]
    query = optional_text(fields.get("_location_query"))
    if query is None and parts:
        query = ", ".join(parts)
    if not reverse_lookup and not query:
        return result
    if not settings.ENABLE_GEOCODING:
        return result
    key = (
        f"coordinates:{lat:.7f},{lon:.7f}"
        if reverse_lookup
        else str(query).casefold()
    )
    if isinstance(cache.get(key), dict):
        cached = dict(cache[key])
        cached.setdefault("locality_aka", [])
        cached["location_resolution_version"] = LOCATION_RESOLUTION_VERSION
        return cached

    if reverse_lookup:
        endpoint = "https://nominatim.openstreetmap.org/reverse?"
        parameters = {
            "lat": f"{lat:.7f}",
            "lon": f"{lon:.7f}",
            "format": "jsonv2",
            "addressdetails": 1,
            "namedetails": 1,
            "accept-language": "en",
        }
    else:
        endpoint = "https://nominatim.openstreetmap.org/search?"
        parameters = {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "namedetails": 1,
            "accept-language": "en",
        }
    url = endpoint + urlencode(parameters)
    request = Request(url, headers={"User-Agent": settings.GEOCODER_USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        time.sleep(settings.GEOCODER_DELAY_SECONDS)
    except Exception as exc:
        result["geocode_status"] = f"failed: {exc}"
        return result
    if reverse_lookup:
        item = payload if isinstance(payload, dict) and not payload.get("error") else {}
    else:
        item = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], dict) else {}
    if not item:
        result["geocode_status"] = "not_found"
        cache[key] = result
        return result
    _apply_nominatim_item(
        result,
        item,
        locality,
        state_name,
        country_name,
        preserve_coordinates=reverse_lookup,
    )
    result["geocode_status"] = "resolved"
    cache[key] = result
    return result


def _location_candidates(
    record: Dict[str, Any], segment: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Collect explicit location evidence in descending reliability order."""
    candidates: List[Dict[str, Any]] = []
    seen = set()

    def add(fields: Dict[str, Any]) -> None:
        cleaned = {
            key: value
            for key, value in fields.items()
            if value is not None and value != "" and value != []
        }
        if not cleaned:
            return
        key = json.dumps(cleaned, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            candidates.append(cleaned)

    def add_structured(source: Dict[str, Any]) -> None:
        fields = {
            name: source.get(name)
            for name in (
                "locality",
                "locality_aka",
                "state",
                "country",
                "lat",
                "lon",
            )
        }
        has_place = any(
            fields.get(name) for name in ("locality", "state", "country")
        )
        has_coordinates = (
            _coordinate(fields.get("lat"), -90.0, 90.0) is not None
            and _coordinate(fields.get("lon"), -180.0, 180.0) is not None
        )
        if has_place or has_coordinates:
            add(fields)

    def add_text(value: Any) -> None:
        text = clean_text(value)
        if not text or len(text) > 180:
            return
        lowered = text.casefold()
        if lowered.startswith("youtube title:") or lowered.startswith("youtube description:"):
            return
        add({"_location_query": text})
        for part in re.split(r"\s*[/|;]\s*", text):
            if part and part != text:
                add({"_location_query": part})

    add_structured(segment)
    raw = recover_json(str(segment.get("raw_response") or ""))
    if isinstance(raw, dict):
        add_structured(raw)

    for source in (segment, raw if isinstance(raw, dict) else {}):
        for text in normalise_string_list(source.get("embedded_location_text")):
            add_text(text)

    labels = segment.get("timestamp_labels", [])
    if isinstance(labels, list):
        for label in labels:
            if isinstance(label, dict):
                add_text(label.get("label"))
            else:
                add_text(label)

    segments = record.get("segments", [])
    if isinstance(segments, list) and len(segments) == 1:
        text_decision = record.get("text_decision", {})
        if isinstance(text_decision, dict):
            add_structured(text_decision)
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict):
            add_text(metadata.get("title"))

    return candidates


def _unknown_location(status: str = "not_found") -> Dict[str, Any]:
    return {
        "locality": None,
        "locality_aka": [],
        "state": None,
        "country": None,
        "iso3": None,
        "continent": None,
        "lat": None,
        "lon": None,
        "geocode_status": status,
        "location_resolution_version": LOCATION_RESOLUTION_VERSION,
    }


def run_location_stage(state: Dict[str, Any]) -> int:
    cache = load_json(settings.GEOCODE_CACHE, {})
    if not isinstance(cache, dict):
        cache = {}
    processed = 0
    for record in state.get("videos", {}).values():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue
        for segment in record.get("segments", []):
            if not isinstance(segment, dict):
                continue
            existing_location = segment.get("location")
            if isinstance(existing_location, dict):
                status = clean_text(existing_location.get("geocode_status"))
                current_version = (
                    existing_location.get("location_resolution_version")
                    == LOCATION_RESOLUTION_VERSION
                )
                if status == "resolved" or (
                    current_version
                    and (not settings.ENABLE_GEOCODING or status != "not_attempted")
                ):
                    continue

            candidates = _location_candidates(record, segment)
            resolved = _unknown_location(
                "not_attempted" if not settings.ENABLE_GEOCODING else "not_found"
            )
            if candidates and not settings.ENABLE_GEOCODING:
                candidate = geocode(candidates[0], cache)
                if candidate.get("locality"):
                    resolved = candidate
            elif candidates:
                for fields in candidates:
                    candidate = geocode(fields, cache)
                    if (
                        candidate.get("geocode_status") == "resolved"
                        and candidate.get("locality")
                    ):
                        resolved = candidate
                        break
                    if candidate.get("locality"):
                        resolved = candidate
            segment["location"] = resolved
            processed += 1
    if processed:
        write_json_atomic(settings.GEOCODE_CACHE, cache)
        save_state(settings.STATE_JSON, state)
    return processed
