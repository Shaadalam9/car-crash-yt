"""One atomic CSV row for every accepted full crash source clip."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from . import settings
from .location import LOCATION_RESOLUTION_VERSION
from .shared import replace_file_with_retry


COLUMNS = [
    "segment_id",
    "video_id",
    "youtube_url",
    "title",
    "channel",
    "upload_date",
    "segment_index",
    "start_time",
    "impact_time_in_video",
    "end_time",
    "duration_seconds",
    "confidence",
    "short_description",
    "crash_type",
    "event_kind",
    "manner_of_collision_code",
    "manner_of_collision",
    "first_harmful_event_code",
    "first_harmful_event",
    "sequence_of_events",
    "crash_taxonomy_standard",
    "crash_taxonomy_status",
    "crash_taxonomy_version",
    "camera_view",
    "road_user_count",
    "road_users",
    "road_environment",
    "time_of_day",
    "weather",
    "road_condition",
    "visible_outcomes",
    "timestamp_labels",
    "embedded_location_text",
    "location_evidence",
    "locality",
    "state",
    "country",
    "iso3",
    "continent",
    "lat",
    "lon",
    "model_version",
]

# Mapping keeps the established geographic columns first and appends the
# video-aligned MMUCC taxonomy fields. OSM identifiers remain in state.json.
MAPPING_COLUMNS = [
    "id",
    "locality",
    "locality_aka",
    "state",
    "country",
    "iso3",
    "continent",
    "lat",
    "lon",
    "videos",
    "time_of_day",
    "start_time",
    "end_time",
    "vehicle_type",
    "event_kind",
    "manner_of_collision_code",
    "manner_of_collision",
    "first_harmful_event_code",
    "first_harmful_event",
    "sequence_of_events",
    "crash_taxonomy_standard",
    "crash_taxonomy_status",
    "crash_taxonomy_version",
]

US_STATE_CODES = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

CANADIAN_PROVINCE_CODES = {
    "alberta": "AB",
    "british columbia": "BC",
    "manitoba": "MB",
    "new brunswick": "NB",
    "newfoundland and labrador": "NL",
    "northwest territories": "NT",
    "nova scotia": "NS",
    "nunavut": "NU",
    "ontario": "ON",
    "prince edward island": "PE",
    "quebec": "QC",
    "saskatchewan": "SK",
    "yukon": "YT",
}


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _bracket_text(value: Any) -> str:
    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace(",", ";")
        .strip()
    )


def bracket_cell(value: Any) -> str:
    """Encode list values using the reference CSV's unquoted bracket style."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(bracket_cell(item) for item in value) + "]"
    return _bracket_text(value)


def _blank_if_missing(value: Any) -> Any:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.casefold() == "unknown":
        return ""
    return value


def _normalise_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value or "").casefold(),
    ).strip()


def _time_of_day(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text == "dawn_dusk":
        return "dusk/dawn"
    return text if text in {"day", "night", "dusk/dawn"} else "unknown"


def _country_is_us(location: Dict[str, Any]) -> bool:
    country = str(location.get("country") or "").strip().casefold()
    iso3 = str(location.get("iso3") or "").strip().upper()
    return iso3 == "USA" or country in {
        "united states",
        "united states of america",
        "usa",
        "us",
    }


def _country_is_canada(location: Dict[str, Any]) -> bool:
    country = str(location.get("country") or "").strip().casefold()
    iso3 = str(location.get("iso3") or "").strip().upper()
    return iso3 == "CAN" or country == "canada"


def _canonical_subdivision(location: Dict[str, Any]) -> str:
    """
    Normalise US/Canadian subdivisions for output and grouping.

    Missing states/provinces remain blank. Other countries keep the returned
    administrative region exactly as supplied by the canonical locality.
    """
    state = str(location.get("state") or "").strip()

    if not state or state.casefold() == "unknown":
        return ""

    key = state.casefold().replace(".", "")

    if _country_is_us(location):
        if len(key) == 2:
            return key.upper()
        return US_STATE_CODES.get(key, state)

    if _country_is_canada(location):
        if len(key) == 2:
            return key.upper()
        return CANADIAN_PROVINCE_CODES.get(key, state)

    return state


def _subdivision_aliases(location: Dict[str, Any]) -> set[str]:
    aliases: set[str] = set()

    raw_state = str(location.get("state") or "").strip()
    canonical = _canonical_subdivision(location)

    for value in (raw_state, canonical):
        normalised = _normalise_name(value)
        if normalised:
            aliases.add(normalised)

    if _country_is_us(location):
        reverse = {
            code.casefold(): name
            for name, code in US_STATE_CODES.items()
        }
        raw_key = raw_state.casefold().replace(".", "")
        canonical_key = canonical.casefold()

        if raw_key in US_STATE_CODES:
            aliases.add(_normalise_name(raw_key))
            aliases.add(_normalise_name(US_STATE_CODES[raw_key]))

        if canonical_key in reverse:
            aliases.add(_normalise_name(reverse[canonical_key]))
            aliases.add(_normalise_name(canonical))

    if _country_is_canada(location):
        reverse = {
            code.casefold(): name
            for name, code in CANADIAN_PROVINCE_CODES.items()
        }
        raw_key = raw_state.casefold().replace(".", "")
        canonical_key = canonical.casefold()

        if raw_key in CANADIAN_PROVINCE_CODES:
            aliases.add(_normalise_name(raw_key))
            aliases.add(
                _normalise_name(CANADIAN_PROVINCE_CODES[raw_key])
            )

        if canonical_key in reverse:
            aliases.add(_normalise_name(reverse[canonical_key]))
            aliases.add(_normalise_name(canonical))

    return aliases


def _clean_locality(value: Any, location: Dict[str, Any]) -> str:
    """
    Defensively keep locality separate from state/country.

    location.py v7 already stores the canonical locality name, but this cleanup
    prevents older state values from creating rows such as
    "JACKSON, GEORGIA".
    """
    locality = str(value or "").strip()
    if not locality or locality.casefold() == "unknown":
        return ""

    parts = [
        part.strip()
        for part in re.split(r"\s*,\s*", locality)
        if part.strip()
    ]
    if len(parts) <= 1:
        return locality

    removable = _subdivision_aliases(location)

    country = _normalise_name(location.get("country"))
    if country:
        removable.add(country)

    while len(parts) > 1 and _normalise_name(parts[-1]) in removable:
        parts.pop()

    return ", ".join(parts).strip()


def _coordinate(value: Any, minimum: float, maximum: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return minimum <= number <= maximum


def _resolved_location(value: Any) -> Dict[str, Any]:
    """
    Return only verified canonical locality records.

    Unresolved/rejected locations stay in state.json and crash_segments.csv
    as blank geographic fields. They are deliberately excluded from
    mapping.csv.
    """
    if not isinstance(value, dict):
        return {}

    if str(value.get("geocode_status") or "").strip() != "resolved":
        return {}

    # Never publish legacy geographic resolutions through the new schema.
    # A v4 row is allowed back into CSV output only after the v7 resolver has
    # revalidated it as the canonical locality entity.
    if value.get("location_resolution_version") != LOCATION_RESOLUTION_VERSION:
        return {}

    if not _clean_locality(value.get("locality"), value):
        return {}

    if not str(value.get("country") or "").strip():
        return {}

    if not _coordinate(value.get("lat"), -90.0, 90.0):
        return {}

    if not _coordinate(value.get("lon"), -180.0, 180.0):
        return {}

    return value


def _location_group_key(location: Dict[str, Any]) -> Tuple[str, ...]:
    """
    Prefer the canonical OSM entity identity for grouping.

    This prevents different places sharing the same display name from being
    merged. Coordinates and names are only a fallback for legacy resolved
    records.
    """
    osm_type = str(location.get("osm_type") or "").strip().casefold()
    osm_id = location.get("osm_id")

    if osm_type and osm_id is not None:
        return ("osm", osm_type, str(osm_id))

    place_id = location.get("place_id")
    if place_id is not None:
        return ("place_id", str(place_id))

    return (
        "fallback",
        _clean_locality(location.get("locality"), location).casefold(),
        _canonical_subdivision(location).casefold(),
        str(location.get("iso3") or location.get("country") or "")
        .strip()
        .casefold(),
        f"{float(location.get('lat')):.7f}",
        f"{float(location.get('lon')):.7f}",
    )


def _append_unique(values: List[str], additions: Any) -> None:
    seen = {value.casefold() for value in values}
    source = additions if isinstance(additions, (list, tuple)) else [additions]

    for value in source:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            values.append(text)


def _mapping_groups(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Group only geographically resolved segments.

    There is intentionally no synthetic "unknown" locality row.
    """
    groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for video_id, record in state.get("videos", {}).items():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue

        segments = [
            segment
            for segment in record.get("segments", [])
            if isinstance(segment, dict)
        ]
        segments.sort(
            key=lambda item: float(item.get("start_time", 0.0))
        )

        for segment in segments:
            location = _resolved_location(segment.get("location"))
            if not location:
                continue

            key = _location_group_key(location)

            if key not in groups:
                locality = _clean_locality(
                    location.get("locality"),
                    location,
                )

                groups[key] = {
                    "locality": locality,
                    "locality_aka": [],
                    "state": _canonical_subdivision(location),
                    "country": _blank_if_missing(
                        location.get("country")
                    ),
                    "iso3": _blank_if_missing(
                        location.get("iso3")
                    ),
                    "continent": _blank_if_missing(
                        location.get("continent")
                    ),
                    "lat": _blank_if_missing(location.get("lat")),
                    "lon": _blank_if_missing(location.get("lon")),
                    "videos": {},
                }

            group = groups[key]

            # locality_aka comes only from names attached to the canonical OSM
            # entity in location.py v7. Defensively clean repeated suffixes.
            aliases = []
            for alias in location.get("locality_aka", []):
                cleaned = _clean_locality(alias, location)
                if (
                    cleaned
                    and cleaned.casefold()
                    != str(group["locality"]).casefold()
                ):
                    aliases.append(cleaned)

            _append_unique(group["locality_aka"], aliases)

            ranges = group["videos"].setdefault(video_id, [])

            start = float(segment.get("start_time", 0.0))
            end = float(segment.get("end_time", start))

            ranges.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "time_of_day": _time_of_day(
                        segment.get("time_of_day")
                    ),
                    "vehicle_type": [
                        str(value).strip()
                        for value in (
                            segment.get("road_users")
                            or ["unknown"]
                        )
                        if str(value).strip()
                    ],
                    "event_kind": _blank_if_missing(
                        segment.get("event_kind")
                    ),
                    "manner_of_collision_code": _blank_if_missing(
                        segment.get("manner_of_collision_code")
                    ),
                    "manner_of_collision": _blank_if_missing(
                        segment.get("manner_of_collision")
                    ),
                    "first_harmful_event_code": _blank_if_missing(
                        segment.get("first_harmful_event_code")
                    ),
                    "first_harmful_event": _blank_if_missing(
                        segment.get("first_harmful_event")
                    ),
                    "sequence_of_events": (
                        segment.get("sequence_of_events")
                        if isinstance(
                            segment.get("sequence_of_events"),
                            list,
                        )
                        else []
                    ),
                    "crash_taxonomy_standard": _blank_if_missing(
                        segment.get("crash_taxonomy_standard")
                    ),
                    "crash_taxonomy_status": _blank_if_missing(
                        segment.get("crash_taxonomy_status")
                    ),
                    "crash_taxonomy_version": _blank_if_missing(
                        segment.get("crash_taxonomy_version")
                    ),
                }
            )

    return list(groups.values())


def iter_rows(state: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Segment CSV keeps every accepted crash segment.

    Geographic columns are populated only for verified resolved localities;
    unresolved evidence remains available in the richer state.json audit log.
    """
    for video_id, record in state.get("videos", {}).items():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue

        metadata = record.get("metadata", {})

        for segment in record.get("segments", []):
            if not isinstance(segment, dict):
                continue

            index = int(segment.get("segment_index", 0))
            location = _resolved_location(segment.get("location"))

            yield {
                "segment_id": f"{video_id}_{index:05d}",
                "video_id": video_id,
                "youtube_url": metadata.get("youtube_url"),
                "title": metadata.get("title"),
                "channel": metadata.get("channel"),
                "upload_date": metadata.get("upload_date"),
                "segment_index": index,
                "start_time": segment.get("start_time"),
                "impact_time_in_video": segment.get(
                    "impact_time_in_video"
                ),
                "end_time": segment.get("end_time"),
                "duration_seconds": segment.get("duration_seconds"),
                "confidence": segment.get("confidence"),
                "short_description": segment.get(
                    "short_description"
                ),
                "crash_type": segment.get("crash_type"),
                "event_kind": segment.get("event_kind"),
                "manner_of_collision_code": segment.get(
                    "manner_of_collision_code"
                ),
                "manner_of_collision": segment.get(
                    "manner_of_collision"
                ),
                "first_harmful_event_code": segment.get(
                    "first_harmful_event_code"
                ),
                "first_harmful_event": segment.get(
                    "first_harmful_event"
                ),
                "sequence_of_events": _json_cell(
                    segment.get("sequence_of_events", [])
                ),
                "crash_taxonomy_standard": segment.get(
                    "crash_taxonomy_standard"
                ),
                "crash_taxonomy_status": segment.get(
                    "crash_taxonomy_status"
                ),
                "crash_taxonomy_version": segment.get(
                    "crash_taxonomy_version"
                ),
                "camera_view": segment.get("camera_view"),
                "road_user_count": segment.get(
                    "road_user_count"
                ),
                "road_users": _json_cell(
                    segment.get("road_users", [])
                ),
                "road_environment": segment.get(
                    "road_environment"
                ),
                "time_of_day": segment.get("time_of_day"),
                "weather": segment.get("weather"),
                "road_condition": segment.get(
                    "road_condition"
                ),
                "visible_outcomes": _json_cell(
                    segment.get("visible_outcomes", [])
                ),
                "timestamp_labels": _json_cell(
                    segment.get("timestamp_labels", [])
                ),
                "embedded_location_text": _json_cell(
                    segment.get("embedded_location_text", [])
                ),
                "location_evidence": segment.get(
                    "location_evidence"
                ),
                "locality": (
                    _clean_locality(
                        location.get("locality"),
                        location,
                    )
                    if location
                    else None
                ),
                "state": (
                    _canonical_subdivision(location) or None
                    if location
                    else None
                ),
                "country": (
                    location.get("country") if location else None
                ),
                "iso3": location.get("iso3") if location else None,
                "continent": (
                    location.get("continent") if location else None
                ),
                "lat": location.get("lat") if location else None,
                "lon": location.get("lon") if location else None,
                "model_version": record.get(
                    "visual_review_version"
                ),
            }


def iter_mapping_rows(
    state: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    """Yield one row per verified canonical locality."""
    for row_id, group in enumerate(
        _mapping_groups(state),
        start=1,
    ):
        videos = list(group["videos"])
        ranges = [
            group["videos"][video_id]
            for video_id in videos
        ]

        time_values = [
            [item["time_of_day"] for item in video_ranges]
            for video_ranges in ranges
        ]
        start_values = [
            [item["start_time"] for item in video_ranges]
            for video_ranges in ranges
        ]
        end_values = [
            [item["end_time"] for item in video_ranges]
            for video_ranges in ranges
        ]
        vehicle_values = [
            vehicle
            for video_ranges in ranges
            for item in video_ranges
            for vehicle in item["vehicle_type"]
        ]

        event_kind_values = [
            [item["event_kind"] for item in video_ranges]
            for video_ranges in ranges
        ]
        manner_code_values = [
            [item["manner_of_collision_code"] for item in video_ranges]
            for video_ranges in ranges
        ]
        manner_values = [
            [item["manner_of_collision"] for item in video_ranges]
            for video_ranges in ranges
        ]
        first_harmful_code_values = [
            [item["first_harmful_event_code"] for item in video_ranges]
            for video_ranges in ranges
        ]
        first_harmful_values = [
            [item["first_harmful_event"] for item in video_ranges]
            for video_ranges in ranges
        ]
        sequence_values = [
            [item["sequence_of_events"] for item in video_ranges]
            for video_ranges in ranges
        ]
        taxonomy_standard_values = [
            [item["crash_taxonomy_standard"] for item in video_ranges]
            for video_ranges in ranges
        ]
        taxonomy_status_values = [
            [item["crash_taxonomy_status"] for item in video_ranges]
            for video_ranges in ranges
        ]
        taxonomy_version_values = [
            [item["crash_taxonomy_version"] for item in video_ranges]
            for video_ranges in ranges
        ]

        if len(videos) == 1:
            time_values = time_values[0]
            start_values = start_values[0]
            end_values = end_values[0]
            event_kind_values = event_kind_values[0]
            manner_code_values = manner_code_values[0]
            manner_values = manner_values[0]
            first_harmful_code_values = first_harmful_code_values[0]
            first_harmful_values = first_harmful_values[0]
            sequence_values = sequence_values[0]
            taxonomy_standard_values = taxonomy_standard_values[0]
            taxonomy_status_values = taxonomy_status_values[0]
            taxonomy_version_values = taxonomy_version_values[0]

        yield {
            "id": row_id,
            "locality": group["locality"],
            "locality_aka": bracket_cell(
                group["locality_aka"]
            ),
            "state": group["state"],
            "country": group["country"],
            "iso3": group["iso3"],
            "continent": group["continent"],
            "lat": group["lat"],
            "lon": group["lon"],
            "videos": bracket_cell(videos),
            "time_of_day": bracket_cell(time_values),
            "start_time": bracket_cell(start_values),
            "end_time": bracket_cell(end_values),
            "vehicle_type": bracket_cell(
                vehicle_values or ["unknown"]
            ),
            "event_kind": bracket_cell(event_kind_values),
            "manner_of_collision_code": bracket_cell(
                manner_code_values
            ),
            "manner_of_collision": bracket_cell(
                manner_values
            ),
            "first_harmful_event_code": bracket_cell(
                first_harmful_code_values
            ),
            "first_harmful_event": bracket_cell(
                first_harmful_values
            ),
            "sequence_of_events": bracket_cell(
                sequence_values
            ),
            "crash_taxonomy_standard": bracket_cell(
                taxonomy_standard_values
            ),
            "crash_taxonomy_status": bracket_cell(
                taxonomy_status_values
            ),
            "crash_taxonomy_version": bracket_cell(
                taxonomy_version_values
            ),
        }


def _write_csv_atomic(
    path: Path,
    columns: list[str],
    rows: Iterable[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

        handle.flush()
        os.fsync(handle.fileno())

    replace_file_with_retry(temporary, path)


def write_output_csv(state: Dict[str, Any]) -> None:
    _write_csv_atomic(
        settings.OUTPUT_CSV,
        COLUMNS,
        iter_rows(state),
    )
    _write_csv_atomic(
        settings.MAPPING_CSV,
        MAPPING_COLUMNS,
        iter_mapping_rows(state),
    )
