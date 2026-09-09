"""Strict evidence based location resolution and optional geocoding."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import settings
from .shared import (
    clean_text,
    coordinates_with_hemispheres,
    load_json,
    normalise_string_list,
    optional_text,
    recover_json,
    save_state,
    write_json_atomic,
)


CONTINENT_CODES = {
    "Africa": set(
        "DZ AO BJ BW BF BI CV CM CF TD KM CG CD CI DJ EG GQ ER SZ ET GA GM GH "
        "GN GW KE LS LR LY MG MW ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA "
        "SS SD TZ TG TN UG ZM ZW".split()
    ),
    "Asia": set(
        "AF AM AZ BH BD BT BN KH CN CY GE IN ID IR IQ IL JP JO KZ KW KG LA LB "
        "MY MV MN MM NP KP OM PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE "
        "UZ VN YE".split()
    ),
    "Europe": set(
        "AL AD AT BY BE BA BG HR CZ DK EE FI FR DE GR HU IS IE IT XK LV LI LT "
        "LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH UA GB VA".split()
    ),
    "North America": set(
        "AG BS BB BZ CA CR CU DM DO SV GD GT HT HN JM MX NI PA KN LC VC US".split()
    ),
    "South America": set("AR BO BR CL CO EC GY PY PE SR UY VE".split()),
    "Oceania": set("AU FJ KI MH FM NR NZ PW PG WS SB TO TV VU".split()),
}

# v7 resolves every accepted result to the locality entity itself. Earlier
# versions are automatically reprocessed because their version differs.
LOCATION_RESOLUTION_VERSION = "segment_evidence_location_v7"

NULL_PLACE_VALUES = {"none", "null", "unknown", "n/a", "na"}

LOCALITY_ADDRESS_FIELDS = (
    "city",
    "town",
    "village",
    "municipality",
    "borough",
    "hamlet",
    "suburb",
    "neighbourhood",
    "quarter",
    "city_district",
)

LOCALITY_ENTITY_TYPES = {
    "city",
    "town",
    "village",
    "municipality",
    "borough",
    "hamlet",
    "suburb",
    "neighbourhood",
    "quarter",
    "locality",
    "city_district",
}

TERMINAL_LOCATION_STATUSES = {
    "resolved",
    "not_found",
    "rejected_result",
    "no_evidence",
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


def _looks_like_coordinate_text(value: Any) -> bool:
    text = clean_text(value).upper().replace("−", "-").replace("–", "-")
    if not text:
        return False

    number = r"[+-]?\d+(?:\.\d+)?"
    north_south = re.search(
        rf"(?:^|[^A-Z])(?:[NS]\s*[:=]?\s*{number}|{number}\s*°?\s*[NS])",
        text,
    )
    east_west = re.search(
        rf"(?:^|[^A-Z])(?:[EW]\s*[:=]?\s*{number}|{number}\s*°?\s*[EW])",
        text,
    )
    return bool(north_south and east_west)


def _place_text(value: Any) -> Optional[str]:
    text = optional_text(value)
    if (
        text is None
        or text.casefold() in NULL_PLACE_VALUES
        or _looks_like_coordinate_text(text)
    ):
        return None
    return text


def _coordinate(value: Any, minimum: float, maximum: float) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if minimum <= result <= maximum else None


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).casefold()).strip()


def _strip_name_prefix(value: str) -> str:
    for prefix in (
        "city of ",
        "town of ",
        "village of ",
        "municipality of ",
        "borough of ",
    ):
        if value.startswith(prefix):
            return value[len(prefix) :].strip()
    return value


def _names_equivalent(left: Any, right: Any) -> bool:
    left_value = _strip_name_prefix(_normalise_name(left))
    right_value = _strip_name_prefix(_normalise_name(right))
    return bool(left_value and right_value and left_value == right_value)


def _address(item: Dict[str, Any]) -> Dict[str, Any]:
    value = item.get("address")
    return value if isinstance(value, dict) else {}


def _namedetails(item: Dict[str, Any]) -> Dict[str, Any]:
    value = item.get("namedetails")
    return value if isinstance(value, dict) else {}


def _entity_names(item: Dict[str, Any]) -> List[str]:
    names: List[str] = []

    for value in (
        item.get("name"),
        _namedetails(item).get("name:en"),
        _namedetails(item).get("official_name"),
        _namedetails(item).get("short_name"),
        _namedetails(item).get("alt_name"),
        _namedetails(item).get("old_name"),
    ):
        text = optional_text(value)
        if not text:
            continue
        names.extend(part.strip() for part in re.split(r"[;,]", text) if part.strip())

    return normalise_string_list(names)


def _address_locality(item: Dict[str, Any]) -> Optional[str]:
    address = _address(item)
    for field in LOCALITY_ADDRESS_FIELDS:
        value = _place_text(address.get(field))
        if value:
            return value
    return None


def _is_locality_entity(item: Dict[str, Any]) -> bool:
    result_class = clean_text(item.get("class") or item.get("category")).lower()
    result_type = clean_text(item.get("type")).lower()
    address_type = clean_text(item.get("addresstype")).lower()

    if result_type in LOCALITY_ENTITY_TYPES or address_type in LOCALITY_ENTITY_TYPES:
        return True

    # "place" results are only accepted when their type/addresstype is a
    # locality type. This deliberately excludes counties, states, stations,
    # businesses, roads, amenities, and other POIs.
    return result_class == "place" and (
        result_type in LOCALITY_ENTITY_TYPES or address_type in LOCALITY_ENTITY_TYPES
    )


def _locality_entity_name(item: Dict[str, Any]) -> Optional[str]:
    if not _is_locality_entity(item):
        return None

    value = _place_text(item.get("name"))
    if value:
        return value

    locality = _address_locality(item)
    if locality:
        return locality

    display_name = optional_text(item.get("display_name"))
    if display_name:
        return _place_text(display_name.split(",", 1)[0])

    return None


def _country_aliases(item: Dict[str, Any]) -> set[str]:
    address = _address(item)
    aliases: set[str] = set()

    country = _place_text(address.get("country"))
    if country:
        aliases.add(_normalise_name(country))

    iso2 = optional_text(address.get("country_code"))
    if iso2:
        aliases.add(iso2.casefold())
        iso3 = _iso3(iso2)
        if iso3:
            aliases.add(iso3.casefold())

    return {value for value in aliases if value}


def _state_aliases(item: Dict[str, Any]) -> set[str]:
    address = _address(item)
    aliases: set[str] = set()

    state = _place_text(address.get("state"))
    if state:
        aliases.add(_normalise_name(state))

    for key in (
        "ISO3166-2-lvl4",
        "ISO3166-2-lvl3",
        "ISO3166-2-lvl6",
    ):
        subdivision = optional_text(address.get(key))
        if subdivision:
            aliases.add(_normalise_name(subdivision))
            if "-" in subdivision:
                aliases.add(_normalise_name(subdivision.rsplit("-", 1)[-1]))

    return {value for value in aliases if value}


def _candidate_locality_parts(value: Any) -> List[str]:
    text = _place_text(value)
    if not text:
        return []
    return [
        part.strip()
        for part in re.split(r"\s*[,/|;]\s*", text)
        if part.strip()
    ]


def _candidate_matches_locality(value: Any, item: Dict[str, Any]) -> bool:
    """
    Match a structured locality candidate to a locality entity.

    "JACKSON, GEORGIA" may match the locality entity "Jackson" when the
    trailing component agrees with the returned state. A county named Jackson
    is never considered because counties are not locality entities.
    """
    if not _is_locality_entity(item):
        return False

    parts = _candidate_locality_parts(value)
    if not parts:
        return True

    result_names = _entity_names(item)
    entity_name = _locality_entity_name(item)
    if entity_name:
        result_names.append(entity_name)

    if any(_names_equivalent(value, name) for name in result_names):
        return True

    first = parts[0]
    if not any(_names_equivalent(first, name) for name in result_names):
        return False

    if len(parts) == 1:
        return True

    admin_aliases = _state_aliases(item) | _country_aliases(item)
    return all(_normalise_name(part) in admin_aliases for part in parts[1:])


def _admin_matches(expected: Optional[str], aliases: set[str]) -> bool:
    if not expected:
        return True
    value = _normalise_name(expected)
    return bool(value and value in aliases)


def _query_mentions_locality(query: Optional[str], locality: Optional[str]) -> bool:
    if not query or not locality:
        return False

    query_tokens = f" {_normalise_name(query)} "
    locality_tokens = _normalise_name(locality)
    return bool(locality_tokens and f" {locality_tokens} " in query_tokens)


def _strip_redundant_suffixes(
    locality: Optional[str],
    item: Dict[str, Any],
) -> Optional[str]:
    """
    Keep locality, state, and country as separate fields.

    For example, "JACKSON, GEORGIA" becomes "JACKSON" when Georgia is already
    represented by the state returned for the same Nominatim entity.
    """
    value = _place_text(locality)
    if not value:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) <= 1:
        return value

    removable = _state_aliases(item) | _country_aliases(item)

    while len(parts) > 1 and _normalise_name(parts[-1]) in removable:
        parts.pop()

    return ", ".join(parts).strip() or value


def _request_json(
    endpoint: str,
    parameters: Dict[str, Any],
) -> Any:
    request = Request(
        endpoint + urlencode(parameters),
        headers={"User-Agent": settings.GEOCODER_USER_AGENT},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    time.sleep(settings.GEOCODER_DELAY_SECONDS)
    return payload


def _search_nominatim(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    payload = _request_json(
        "https://nominatim.openstreetmap.org/search?",
        {
            "q": query,
            "format": "jsonv2",
            "limit": max(1, int(limit)),
            "addressdetails": 1,
            "namedetails": 1,
            "accept-language": "en",
        },
    )
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _reverse_nominatim(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    payload = _request_json(
        "https://nominatim.openstreetmap.org/reverse?",
        {
            "lat": f"{lat:.7f}",
            "lon": f"{lon:.7f}",
            "format": "jsonv2",
            "addressdetails": 1,
            "namedetails": 1,
            "accept-language": "en",
        },
    )
    if isinstance(payload, dict) and not payload.get("error"):
        return payload
    return None


def _canonical_query(
    locality: str,
    state: Optional[str],
    country: Optional[str],
) -> str:
    return ", ".join(part for part in (locality, state, country) if part)


def _select_locality_entity(
    items: Sequence[Dict[str, Any]],
    *,
    expected_locality: Optional[str],
    expected_state: Optional[str],
    expected_country: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Select a locality entity rather than the first Nominatim result.

    This is the main protection against cases such as Jackson County being
    selected for Jackson city or Oku station being selected for Kita.
    """
    best: Optional[Dict[str, Any]] = None
    best_score = -1

    for item in items:
        if not _is_locality_entity(item):
            continue

        locality_name = _locality_entity_name(item)
        if not locality_name:
            continue

        if expected_locality and not _candidate_matches_locality(
            expected_locality, item
        ):
            continue

        if expected_state and not _admin_matches(
            expected_state, _state_aliases(item)
        ):
            continue

        if expected_country and not _admin_matches(
            expected_country, _country_aliases(item)
        ):
            continue

        if _coordinate(item.get("lat"), -90.0, 90.0) is None:
            continue
        if _coordinate(item.get("lon"), -180.0, 180.0) is None:
            continue

        score = 10
        if expected_locality:
            if any(
                _names_equivalent(expected_locality, name)
                for name in _entity_names(item)
            ):
                score += 6
            elif _candidate_matches_locality(expected_locality, item):
                score += 4
        if expected_state:
            score += 2
        if expected_country:
            score += 2
        if item.get("osm_id") is not None:
            score += 1

        if score > best_score:
            best = item
            best_score = score

    return best


def _canonicalise_from_locality(
    locality: str,
    *,
    state: Optional[str],
    country: Optional[str],
) -> Optional[Dict[str, Any]]:
    query = _canonical_query(locality, state, country)
    items = _search_nominatim(query)
    return _select_locality_entity(
        items,
        expected_locality=locality,
        expected_state=state,
        expected_country=country,
    )


def _canonicalise_raw_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Resolve free text only when it directly supports a locality.

    A station, road, venue, username, watermark, or other POI is not itself a
    locality. If such a result is returned, its containing locality is used
    only when that locality is explicitly present in the original evidence
    text; the locality is then looked up again as its own geographic entity.
    """
    for item in _search_nominatim(query):
        if _is_locality_entity(item):
            locality = _locality_entity_name(item)
            if locality and (
                _query_mentions_locality(query, locality)
                or any(
                    _query_mentions_locality(query, name)
                    for name in _entity_names(item)
                )
            ):
                return item
            continue

        locality = _address_locality(item)
        if not locality or not _query_mentions_locality(query, locality):
            continue

        address = _address(item)
        state = _place_text(address.get("state"))
        country = _place_text(address.get("country"))

        canonical = _canonicalise_from_locality(
            locality,
            state=state,
            country=country,
        )
        if canonical is not None:
            return canonical

    return None


def _canonicalise_reverse(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Use visible coordinates as evidence, but publish the containing locality's
    canonical coordinates rather than the road/POI/point coordinates.
    """
    item = _reverse_nominatim(lat, lon)
    if item is None:
        return None

    locality = _address_locality(item)
    if not locality and _is_locality_entity(item):
        locality = _locality_entity_name(item)

    if not locality:
        return None

    address = _address(item)
    state = _place_text(address.get("state"))
    country = _place_text(address.get("country"))

    return _canonicalise_from_locality(
        locality,
        state=state,
        country=country,
    )


def _canonical_aliases(item: Dict[str, Any], locality: str) -> List[str]:
    """
    Keep only names attached to the same canonical OSM entity.

    Containing entities such as counties, wards, stations, roads, and nearby
    places are intentionally not added to locality_aka.
    """
    aliases: List[str] = []

    for value in _entity_names(item):
        cleaned = _strip_redundant_suffixes(value, item)
        if (
            cleaned
            and not _names_equivalent(cleaned, locality)
        ):
            aliases.append(cleaned)

    return normalise_string_list(aliases)


def _apply_canonical_item(
    result: Dict[str, Any],
    item: Dict[str, Any],
) -> None:
    locality = _locality_entity_name(item)
    locality = _strip_redundant_suffixes(locality, item)

    address = _address(item)
    state = _place_text(address.get("state"))
    country = _place_text(address.get("country"))
    iso2 = optional_text(address.get("country_code"))

    subdivision = optional_text(
        address.get("ISO3166-2-lvl4")
        or address.get("ISO3166-2-lvl3")
    )
    if (
        iso2
        and iso2.upper() in {"US", "CA"}
        and subdivision
        and "-" in subdivision
    ):
        state = subdivision.rsplit("-", 1)[-1]

    lat = _coordinate(item.get("lat"), -90.0, 90.0)
    lon = _coordinate(item.get("lon"), -180.0, 180.0)

    result["locality"] = locality
    result["locality_aka"] = (
        _canonical_aliases(item, locality) if locality else []
    )
    result["state"] = state
    result["country"] = country
    result["iso3"] = _iso3(iso2)
    result["continent"] = _continent(iso2)
    result["lat"] = round(lat, 7) if lat is not None else None
    result["lon"] = round(lon, 7) if lon is not None else None

    result["osm_type"] = optional_text(item.get("osm_type"))
    result["osm_id"] = item.get("osm_id")
    result["place_id"] = item.get("place_id")
    result["nominatim_class"] = optional_text(item.get("class") or item.get("category"))
    result["nominatim_type"] = optional_text(item.get("type"))
    result["nominatim_addresstype"] = optional_text(item.get("addresstype"))


def _clear_public_location(result: Dict[str, Any]) -> None:
    for key, value in {
        "locality": None,
        "locality_aka": [],
        "state": None,
        "country": None,
        "iso3": None,
        "continent": None,
        "lat": None,
        "lon": None,
        "osm_type": None,
        "osm_id": None,
        "place_id": None,
        "nominatim_class": None,
        "nominatim_type": None,
        "nominatim_addresstype": None,
    }.items():
        result[key] = value


def _usable_resolved_location(result: Dict[str, Any]) -> bool:
    return (
        _place_text(result.get("locality")) is not None
        and _place_text(result.get("country")) is not None
        and _coordinate(result.get("lat"), -90.0, 90.0) is not None
        and _coordinate(result.get("lon"), -180.0, 180.0) is not None
        and result.get("osm_id") is not None
        and optional_text(result.get("osm_type")) is not None
    )


def geocode(fields: Dict[str, Any], cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve evidence derived candidates to canonical locality entities.

    The public location fields are populated only after the locality itself has
    been verified. Candidate evidence remains available separately for audit.
    """
    locality = _place_text(fields.get("locality"))
    locality_aka = normalise_string_list(fields.get("locality_aka"))
    state_name = _place_text(fields.get("state"))
    country_name = _place_text(fields.get("country"))

    lat = _coordinate(fields.get("lat"), -90.0, 90.0)
    lon = _coordinate(fields.get("lon"), -180.0, 180.0)
    if (lat is None) != (lon is None):
        lat = lon = None

    query = optional_text(fields.get("_location_query"))
    if query is None and locality:
        query = _canonical_query(locality, state_name, country_name)

    result = _unknown_location("not_attempted")
    result.update(
        {
            "candidate_locality": locality,
            "candidate_locality_aka": locality_aka,
            "candidate_state": state_name,
            "candidate_country": country_name,
            "candidate_query": query,
            "candidate_lat": lat,
            "candidate_lon": lon,
        }
    )

    reverse_lookup = lat is not None and lon is not None

    if not reverse_lookup and not query:
        result["geocode_status"] = "no_evidence"
        return result

    if not settings.ENABLE_GEOCODING:
        return result

    cache_key = LOCATION_RESOLUTION_VERSION + ":" + (
        f"coordinates:{lat:.7f},{lon:.7f}"
        if reverse_lookup
        else str(query).casefold()
    )

    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        value = dict(cached)
        value["location_resolution_version"] = LOCATION_RESOLUTION_VERSION
        return value

    try:
        if reverse_lookup:
            canonical = _canonicalise_reverse(lat, lon)
        elif locality:
            items = _search_nominatim(str(query))
            canonical = _select_locality_entity(
                items,
                expected_locality=locality,
                expected_state=state_name,
                expected_country=country_name,
            )

            # If the structured search found only a POI/road result whose
            # address explicitly agrees with the candidate locality, resolve
            # the containing locality as a second step.
            if canonical is None:
                for item in items:
                    address_locality = _address_locality(item)
                    if (
                        address_locality
                        and _candidate_matches_locality(locality, {
                            **item,
                            "class": "place",
                            "type": "city",
                            "addresstype": "city",
                            "name": address_locality,
                        })
                    ):
                        address = _address(item)
                        canonical = _canonicalise_from_locality(
                            address_locality,
                            state=_place_text(address.get("state")),
                            country=_place_text(address.get("country")),
                        )
                        if canonical is not None:
                            break
        else:
            canonical = _canonicalise_raw_query(str(query))
    except Exception as exc:
        # Network/rate limit failures are intentionally not cached.
        result["geocode_status"] = f"failed: {exc}"
        return result

    if canonical is None:
        result["geocode_status"] = "not_found"
        cache[cache_key] = result
        return result

    _apply_canonical_item(result, canonical)

    if not _usable_resolved_location(result):
        _clear_public_location(result)
        result["geocode_status"] = "rejected_result"
        result["geocode_rejection_reason"] = "incomplete_locality_entity"
        cache[cache_key] = result
        return result

    result["geocode_status"] = "resolved"
    result["geocode_rejection_reason"] = None
    cache[cache_key] = result
    return result


def _location_candidates(
    record: Dict[str, Any],
    segment: Dict[str, Any],
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

        key = json.dumps(
            cleaned,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if key not in seen:
            seen.add(key)
            candidates.append(cleaned)

    def add_structured(
        source: Dict[str, Any],
        coordinate_evidence: Any = (),
    ) -> None:
        lat = _coordinate(source.get("lat"), -90.0, 90.0)
        lon = _coordinate(source.get("lon"), -180.0, 180.0)
        lat, lon = coordinates_with_hemispheres(
            lat,
            lon,
            coordinate_evidence,
        )

        fields = {
            "locality": _place_text(source.get("locality")),
            "locality_aka": source.get("locality_aka"),
            "state": _place_text(source.get("state")),
            "country": _place_text(source.get("country")),
            "lat": lat,
            "lon": lon,
        }

        if any(fields.get(name) for name in ("locality", "state", "country")):
            add(fields)
            return

        if lat is not None and lon is not None:
            add({"lat": lat, "lon": lon})

    def add_audit_candidate(source: Dict[str, Any]) -> None:
        fields = {
            "locality": _place_text(source.get("candidate_locality")),
            "locality_aka": source.get("candidate_locality_aka"),
            "state": _place_text(source.get("candidate_state")),
            "country": _place_text(source.get("candidate_country")),
            "lat": _coordinate(source.get("candidate_lat"), -90.0, 90.0),
            "lon": _coordinate(source.get("candidate_lon"), -180.0, 180.0),
        }

        if any(
            fields.get(name)
            for name in ("locality", "state", "country", "lat", "lon")
        ):
            add(fields)

        query = optional_text(source.get("candidate_query"))
        if query:
            add({"_location_query": query})

    def add_text(value: Any) -> None:
        text = clean_text(value)
        if (
            not text
            or len(text) > 180
            or text.casefold() in NULL_PLACE_VALUES
            or _looks_like_coordinate_text(text)
        ):
            return

        lowered = text.casefold()
        if (
            lowered.startswith("youtube title:")
            or lowered.startswith("youtube description:")
        ):
            return

        add({"_location_query": text})

        for part in re.split(r"\s*[/|;]\s*", text):
            if (
                part
                and part != text
                and part.casefold() not in NULL_PLACE_VALUES
                and not _looks_like_coordinate_text(part)
            ):
                add({"_location_query": part})

    review = segment.get("location_visual_review", {})
    review_evidence = (
        review.get("visible_location_text", [])
        if isinstance(review, dict)
        else []
    )

    coordinate_evidence = [
        *normalise_string_list(segment.get("embedded_location_text")),
        *normalise_string_list(review_evidence),
    ]

    add_structured(segment, coordinate_evidence)

    existing_location = segment.get("location")
    if isinstance(existing_location, dict):
        # Reprocess older versions from their old public fields and recover
        # retryable v7 attempts from their audit fields.
        add_structured(existing_location, coordinate_evidence)
        add_audit_candidate(existing_location)

    raw = recover_json(str(segment.get("raw_response") or ""))
    if isinstance(raw, dict):
        add_structured(raw, raw.get("embedded_location_text", []))

    for source in (segment, raw if isinstance(raw, dict) else {}):
        for text in normalise_string_list(
            source.get("embedded_location_text")
        ):
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
        "osm_type": None,
        "osm_id": None,
        "place_id": None,
        "nominatim_class": None,
        "nominatim_type": None,
        "nominatim_addresstype": None,
        "candidate_locality": None,
        "candidate_locality_aka": [],
        "candidate_state": None,
        "candidate_country": None,
        "candidate_query": None,
        "candidate_lat": None,
        "candidate_lon": None,
        "geocode_status": status,
        "geocode_rejection_reason": None,
        "location_resolution_version": LOCATION_RESOLUTION_VERSION,
    }


def _is_retryable_status(status: str) -> bool:
    return status == "not_attempted" or status.startswith("failed:")


def run_location_stage(
    state: Dict[str, Any],
    max_segments: Optional[int] = None,
) -> int:
    """Resolve a bounded number of stale/retryable segment locations.

    ``max_segments`` is deliberately optional so direct callers and existing
    tests retain the historical "process everything" behaviour. The
    continuous pipeline supplies a small limit, which makes large v4 -> v7
    migrations resumable and prevents geocoding from starving GPU taxonomy
    work for hours before the first checkpoint.
    """
    cache = load_json(settings.GEOCODE_CACHE, {})
    if not isinstance(cache, dict):
        cache = {}

    limit: Optional[int]
    if max_segments is None:
        limit = None
    else:
        try:
            limit = max(1, int(max_segments))
        except (TypeError, ValueError):
            raise ValueError("max_segments must be a positive integer")

    processed = 0

    try:
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

                    if current_version:
                        if status in TERMINAL_LOCATION_STATUSES:
                            continue
                        if not settings.ENABLE_GEOCODING:
                            continue

                candidates = _location_candidates(record, segment)

                if not candidates:
                    segment["location"] = _unknown_location(
                        "not_attempted"
                        if not settings.ENABLE_GEOCODING
                        else "no_evidence"
                    )
                    processed += 1
                else:
                    resolved = _unknown_location(
                        "not_attempted"
                        if not settings.ENABLE_GEOCODING
                        else "not_found"
                    )

                    if not settings.ENABLE_GEOCODING:
                        resolved = geocode(candidates[0], cache)
                    else:
                        first_retryable: Optional[Dict[str, Any]] = None
                        first_terminal: Optional[Dict[str, Any]] = None

                        for fields in candidates:
                            candidate = geocode(fields, cache)
                            status = clean_text(candidate.get("geocode_status"))

                            if status == "resolved":
                                resolved = candidate
                                break

                            if _is_retryable_status(status):
                                if first_retryable is None:
                                    first_retryable = candidate
                            elif first_terminal is None:
                                first_terminal = candidate
                        else:
                            if first_retryable is not None:
                                resolved = first_retryable
                            elif first_terminal is not None:
                                resolved = first_terminal

                    segment["location"] = resolved
                    processed += 1

                if limit is not None and processed >= limit:
                    break

            if limit is not None and processed >= limit:
                break
    except KeyboardInterrupt:
        if processed:
            write_json_atomic(settings.GEOCODE_CACHE, cache)
            save_state(settings.STATE_JSON, state)
        raise

    if processed:
        # One atomic checkpoint per bounded batch. With the production pipeline
        # limit this caps the amount of geocoding work that can be lost on a
        # pod restart without repeatedly rewriting a ~hundreds-of-MB state file.
        write_json_atomic(settings.GEOCODE_CACHE, cache)
        save_state(settings.STATE_JSON, state)

    return processed
