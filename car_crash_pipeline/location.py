"""Strict evidence based location resolution and optional geocoding."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import settings
from .shared import load_json, optional_text, save_state, write_json_atomic


CONTINENT_CODES = {
    "Africa": set("DZ AO BJ BW BF BI CV CM CF TD KM CG CD CI DJ EG GQ ER SZ ET GA GM GH GN GW KE LS LR LY MG MW ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA SS SD TZ TG TN UG ZM ZW".split()),
    "Asia": set("AF AM AZ BH BD BT BN KH CN CY GE IN ID IR IQ IL JP JO KZ KW KG LA LB MY MV MN MM NP KP OM PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE".split()),
    "Europe": set("AL AD AT BY BE BA BG HR CZ DK EE FI FR DE GR HU IS IE IT XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH UA GB VA".split()),
    "North America": set("AG BS BB BZ CA CR CU DM DO SV GD GT HT HN JM MX NI PA KN LC VC US".split()),
    "South America": set("AR BO BR CL CO EC GY PY PE SR UY VE".split()),
    "Oceania": set("AU FJ KI MH FM NR NZ PW PG WS SB TO TV VU".split()),
}


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
    result = {
        "locality": locality,
        "state": state_name,
        "country": country_name,
        "iso3": None,
        "continent": None,
        "lat": None,
        "lon": None,
        "geocode_status": "not_attempted",
    }
    parts = [part for part in (locality, state_name, country_name) if part]
    if not parts or not settings.ENABLE_GEOCODING:
        return result
    query = ", ".join(parts)
    key = query.casefold()
    if isinstance(cache.get(key), dict):
        return cache[key]

    url = "https://nominatim.openstreetmap.org/search?" + urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "accept-language": "en",
        }
    )
    request = Request(url, headers={"User-Agent": settings.GEOCODER_USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        time.sleep(settings.GEOCODER_DELAY_SECONDS)
    except Exception as exc:
        result["geocode_status"] = f"failed: {exc}"
        return result
    if not isinstance(payload, list) or not payload:
        result["geocode_status"] = "not_found"
        cache[key] = result
        return result

    item = payload[0] if isinstance(payload[0], dict) else {}
    address = item.get("address", {}) if isinstance(item.get("address"), dict) else {}
    result["locality"] = next(
        (
            optional_text(address.get(name))
            for name in ("city", "town", "village", "municipality")
            if optional_text(address.get(name))
        ),
        locality,
    )
    result["state"] = optional_text(address.get("state")) or state_name
    result["country"] = optional_text(address.get("country")) or country_name
    iso2 = optional_text(address.get("country_code"))
    result["iso3"] = _iso3(iso2)
    result["continent"] = _continent(iso2)
    try:
        result["lat"] = round(float(item.get("lat")), 7)
        result["lon"] = round(float(item.get("lon")), 7)
    except (TypeError, ValueError):
        pass
    result["geocode_status"] = "resolved"
    cache[key] = result
    return result


def run_location_stage(state: Dict[str, Any]) -> int:
    cache = load_json(settings.GEOCODE_CACHE, {})
    if not isinstance(cache, dict):
        cache = {}
    processed = 0
    for record in state.get("videos", {}).values():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue
        text_decision = record.get("text_decision", {})
        for segment in record.get("segments", []):
            if not isinstance(segment, dict) or isinstance(segment.get("location"), dict):
                continue
            evidence = segment.get("location_evidence", "none")
            if evidence in {"embedded_text", "both"}:
                fields = segment
            elif evidence == "metadata" and isinstance(text_decision, dict):
                fields = text_decision
            else:
                fields = {}
            segment["location"] = geocode(fields, cache)
            processed += 1
    if processed:
        write_json_atomic(settings.GEOCODE_CACHE, cache)
        save_state(settings.STATE_JSON, state)
    return processed
