"""One atomic CSV row for every accepted full crash source clip."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from . import settings
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


def _known_or_unknown(value: Any) -> Any:
    return value if value is not None and value != "" else "unknown"


def _time_of_day(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text == "dawn_dusk":
        return "dusk/dawn"
    return text if text in {"day", "night", "dusk/dawn"} else "unknown"


def _canonical_subdivision(location: Dict[str, Any]) -> str:
    state = str(location.get("state") or "unknown").strip()
    if state.casefold() == "unknown":
        return "unknown"
    country = str(location.get("country") or "").strip().casefold()
    iso3 = str(location.get("iso3") or "").strip().upper()
    state_key = state.casefold().replace(".", "")
    if iso3 == "USA" or country in {
        "united states",
        "united states of america",
        "usa",
        "us",
    }:
        if len(state_key) == 2:
            return state_key.upper()
        return US_STATE_CODES.get(state_key, state_key)
    if iso3 == "CAN" or country == "canada":
        if len(state_key) == 2:
            return state_key.upper()
        return CANADIAN_PROVINCE_CODES.get(state_key, state_key)
    return state_key


def _location_group_key(location: Dict[str, Any]) -> Tuple[str, ...]:
    locality = str(location.get("locality") or "unknown").strip().casefold()
    country = str(location.get("country") or "unknown").strip().casefold()
    iso3 = str(location.get("iso3") or "unknown").strip().upper()
    if locality == "unknown" and country == "unknown" and iso3 == "UNKNOWN":
        return ("unknown",)
    country_key = iso3 if iso3 != "UNKNOWN" else country
    return locality, _canonical_subdivision(location), country_key


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
    """Group mapping data by place while retaining every video time range."""
    groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for video_id, record in state.get("videos", {}).items():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue
        segments = [
            segment
            for segment in record.get("segments", [])
            if isinstance(segment, dict)
        ]
        segments.sort(key=lambda item: float(item.get("start_time", 0.0)))
        for segment in segments:
            location = segment.get("location", {})
            if not isinstance(location, dict):
                location = {}
            key = _location_group_key(location)
            if key not in groups:
                state_value = _known_or_unknown(location.get("state"))
                canonical_state = _canonical_subdivision(location)
                if canonical_state != "unknown" and canonical_state.isupper():
                    state_value = canonical_state
                groups[key] = {
                    "locality": _known_or_unknown(location.get("locality")),
                    "locality_aka": [],
                    "state": state_value,
                    "country": _known_or_unknown(location.get("country")),
                    "iso3": _known_or_unknown(location.get("iso3")),
                    "continent": _known_or_unknown(location.get("continent")),
                    "lat": _known_or_unknown(location.get("lat")),
                    "lon": _known_or_unknown(location.get("lon")),
                    "videos": {},
                }
            group = groups[key]
            _append_unique(group["locality_aka"], location.get("locality_aka", []))

            video_ranges = group["videos"].setdefault(video_id, [])
            start = float(segment.get("start_time", 0.0))
            end = float(segment.get("end_time", start))
            time_of_day = _time_of_day(segment.get("time_of_day"))
            video_ranges.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "time_of_day": time_of_day,
                    "vehicle_type": [
                        str(value).strip()
                        for value in (segment.get("road_users") or ["unknown"])
                        if str(value).strip()
                    ],
                }
            )
    return list(groups.values())


def iter_rows(state: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for video_id, record in state.get("videos", {}).items():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue
        metadata = record.get("metadata", {})
        for segment in record.get("segments", []):
            if not isinstance(segment, dict):
                continue
            index = int(segment.get("segment_index", 0))
            location = segment.get("location", {})
            if not isinstance(location, dict):
                location = {}
            yield {
                "segment_id": f"{video_id}_{index:05d}",
                "video_id": video_id,
                "youtube_url": metadata.get("youtube_url"),
                "title": metadata.get("title"),
                "channel": metadata.get("channel"),
                "upload_date": metadata.get("upload_date"),
                "segment_index": index,
                "start_time": segment.get("start_time"),
                "impact_time_in_video": segment.get("impact_time_in_video"),
                "end_time": segment.get("end_time"),
                "duration_seconds": segment.get("duration_seconds"),
                "confidence": segment.get("confidence"),
                "short_description": segment.get("short_description"),
                "crash_type": segment.get("crash_type"),
                "camera_view": segment.get("camera_view"),
                "road_user_count": segment.get("road_user_count"),
                "road_users": _json_cell(segment.get("road_users", [])),
                "road_environment": segment.get("road_environment"),
                "time_of_day": segment.get("time_of_day"),
                "weather": segment.get("weather"),
                "road_condition": segment.get("road_condition"),
                "visible_outcomes": _json_cell(segment.get("visible_outcomes", [])),
                "timestamp_labels": _json_cell(segment.get("timestamp_labels", [])),
                "embedded_location_text": _json_cell(segment.get("embedded_location_text", [])),
                "location_evidence": segment.get("location_evidence"),
                "locality": location.get("locality"),
                "state": location.get("state"),
                "country": location.get("country"),
                "iso3": location.get("iso3"),
                "continent": location.get("continent"),
                "lat": location.get("lat"),
                "lon": location.get("lon"),
                "model_version": record.get("visual_review_version"),
            }


def iter_mapping_rows(state: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield one reference compatible row per locality, including unknown."""
    for row_id, group in enumerate(_mapping_groups(state), start=1):
        videos = list(group["videos"])
        ranges = [group["videos"][video_id] for video_id in videos]
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
        if len(videos) == 1:
            time_values = time_values[0]
            start_values = start_values[0]
            end_values = end_values[0]
        yield {
            "id": row_id,
            "locality": group["locality"],
            "locality_aka": bracket_cell(group["locality_aka"]),
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
            "vehicle_type": bracket_cell(vehicle_values or ["unknown"]),
        }


def _write_csv_atomic(
    path: Path, columns: list[str], rows: Iterable[Dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    replace_file_with_retry(temporary, path)


def write_output_csv(state: Dict[str, Any]) -> None:
    _write_csv_atomic(settings.OUTPUT_CSV, COLUMNS, iter_rows(state))
    _write_csv_atomic(
        settings.MAPPING_CSV,
        MAPPING_COLUMNS,
        iter_mapping_rows(state),
    )
