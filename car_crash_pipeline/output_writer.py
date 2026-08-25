"""One atomic CSV row for every accepted full crash source clip."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable

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


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def write_output_csv(state: Dict[str, Any]) -> None:
    settings.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.OUTPUT_CSV.with_suffix(
        settings.OUTPUT_CSV.suffix + f".tmp.{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in iter_rows(state):
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    replace_file_with_retry(temporary, settings.OUTPUT_CSV)
