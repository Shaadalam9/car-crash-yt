"""Paper-oriented analysis for the current car-crash dataset outputs.

This module deliberately treats ``state.json`` as the authoritative source for
segment-level counts, taxonomy fields, road users, time of day, and location
status. ``mapping.csv`` is used as a derived geographic publication table and
for consistency checks.

Why this split matters
----------------------
``mapping.csv`` contains only segments with a current, resolved canonical
locality. It is therefore not a valid denominator for whole-dataset taxonomy or
road-user distributions. Some mapping columns are also aggregated across
multiple videos or segments. Using the authoritative state avoids accidental
under-counting and invalid cross-field alignment.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CRASH_TAXONOMY_VERSION = "nhtsa_mmucc6_video_v1"
LOCATION_RESOLUTION_VERSION = "segment_evidence_location_v7"

MAPPING_REQUIRED_COLUMNS = {
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
}


@dataclass
class CrashAnalysisResult:
    """All computed paper-facing results."""

    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    segments: pd.DataFrame
    resolved_segments: pd.DataFrame
    mapping: pd.DataFrame


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "none", "null", "nan", "unknown"}:
        return ""
    return text


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _segment_duration_seconds(segment: dict[str, Any]) -> float:
    """Return the retained full-clip duration without the legacy one-second cut.

    The current crash pipeline stores complete source-clip start and end times
    and already stores ``duration_seconds``. The old detector/tracker analysis
    subtracted one second from every endpoint; that assumption does not apply to
    the current crash dataset.
    """

    duration = _finite_float(segment.get("duration_seconds"))
    if duration is not None and duration >= 0:
        return duration

    start = _finite_float(segment.get("start_time"))
    end = _finite_float(segment.get("end_time"))
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def _normalise_geocode_status(location: Any) -> str:
    if not isinstance(location, dict):
        return "missing_location_record"
    status = _clean_text(location.get("geocode_status"))
    if not status:
        return "missing_location_status"
    if status.startswith("failed:"):
        return "failed"
    return status


def _is_current_resolved_location(location: Any) -> bool:
    if not isinstance(location, dict):
        return False
    if location.get("geocode_status") != "resolved":
        return False
    if location.get("location_resolution_version") != LOCATION_RESOLUTION_VERSION:
        return False
    if not _clean_text(location.get("locality")):
        return False
    if not _clean_text(location.get("country")):
        return False
    lat = _finite_float(location.get("lat"))
    lon = _finite_float(location.get("lon"))
    return lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180


def _canonical_locality_key(location: dict[str, Any]) -> str:
    """Return a stable key for a verified canonical locality entity."""

    osm_type = _clean_text(location.get("osm_type")).casefold()
    osm_id = location.get("osm_id")
    if osm_type and osm_id is not None:
        return f"osm:{osm_type}:{osm_id}"

    place_id = location.get("place_id")
    if place_id is not None:
        return f"place_id:{place_id}"

    locality = _clean_text(location.get("locality")).casefold()
    state = _clean_text(location.get("state")).casefold()
    iso3 = _clean_text(location.get("iso3")).upper()
    country = _clean_text(location.get("country")).casefold()
    lat = _finite_float(location.get("lat"))
    lon = _finite_float(location.get("lon"))
    return "fallback:" + "|".join(
        [
            locality,
            state,
            iso3 or country,
            "" if lat is None else f"{lat:.7f}",
            "" if lon is None else f"{lon:.7f}",
        ]
    )


def _flatten_numeric_list(value: Any) -> list[float]:
    """Parse mapping numeric list cells such as ``[1,2]`` or ``[[1],[2,3]]``."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []

    out: list[float] = []

    def walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        number = _finite_float(item)
        if number is not None:
            out.append(number)

    walk(parsed)
    return out


def _parse_bare_bracket_list(value: Any) -> list[str]:
    """Parse flat mapping cells such as ``[_E4jvaos-sM,abc123]``."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [_clean_text(item) for item in value if _clean_text(item)]

    text = str(value).strip().strip('"').strip("'")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return []
    return [
        token.strip().strip('"').strip("'")
        for token in text.split(",")
        if token.strip().strip('"').strip("'")
    ]


def _count_table(series: pd.Series, *, value_name: str = "category") -> pd.DataFrame:
    values = series.fillna("unknown").astype(str)
    counts = values.value_counts(dropna=False).rename_axis(value_name).reset_index(name="count")
    total = int(counts["count"].sum())
    counts["percentage"] = 0.0 if total == 0 else counts["count"] / total * 100
    counts["percentage"] = counts["percentage"].round(2)
    return counts


def _human_label(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return "Unknown"
    return text.replace("_", " ").replace("/", " / ").strip().title()


class CrashResultsAnalysis:
    """Compute the crash-corpus Results values and paper tables."""

    def __init__(self, state_path: str | Path, mapping_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.mapping_path = Path(mapping_path)

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise FileNotFoundError(f"State file not found: {self.state_path}")
        with self.state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict) or not isinstance(state.get("videos", {}), dict):
            raise ValueError("state.json does not contain a valid videos object")
        return state

    def load_mapping(self) -> pd.DataFrame:
        if not self.mapping_path.is_file():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")
        mapping = pd.read_csv(self.mapping_path, dtype=str, keep_default_na=False)
        missing = sorted(MAPPING_REQUIRED_COLUMNS - set(mapping.columns))
        if missing:
            raise ValueError(f"mapping.csv is missing required columns: {missing}")
        for column in ("lat", "lon"):
            mapping[column] = pd.to_numeric(mapping[column], errors="coerce")
        return mapping

    def _extract_segments(self, state: dict[str, Any]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for video_id, record in state.get("videos", {}).items():
            if not isinstance(record, dict) or record.get("status") != "complete":
                continue
            segments = record.get("segments", [])
            if not isinstance(segments, list):
                continue

            for position, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    continue
                location = segment.get("location") if isinstance(segment.get("location"), dict) else {}
                resolved = _is_current_resolved_location(location)
                rows.append(
                    {
                        "video_id": str(video_id),
                        "segment_index": segment.get("segment_index", position),
                        "start_time": _finite_float(segment.get("start_time")),
                        "end_time": _finite_float(segment.get("end_time")),
                        "duration_seconds": _segment_duration_seconds(segment),
                        "time_of_day": _clean_text(segment.get("time_of_day")) or "unknown",
                        "road_users": segment.get("road_users") if isinstance(segment.get("road_users"), list) else [],
                        "event_kind": _clean_text(segment.get("event_kind")),
                        "manner_of_collision_code": _clean_text(segment.get("manner_of_collision_code")),
                        "manner_of_collision": _clean_text(segment.get("manner_of_collision")),
                        "first_harmful_event_code": _clean_text(segment.get("first_harmful_event_code")),
                        "first_harmful_event": _clean_text(segment.get("first_harmful_event")),
                        "sequence_of_events": (
                            segment.get("sequence_of_events")
                            if isinstance(segment.get("sequence_of_events"), list)
                            else []
                        ),
                        "crash_taxonomy_standard": _clean_text(segment.get("crash_taxonomy_standard")),
                        "crash_taxonomy_status": _clean_text(segment.get("crash_taxonomy_status")) or "missing",
                        "crash_taxonomy_version": _clean_text(segment.get("crash_taxonomy_version")),
                        "geocode_status": _normalise_geocode_status(segment.get("location")),
                        "location_resolution_version": _clean_text(location.get("location_resolution_version")),
                        "resolved_location": resolved,
                        "locality_key": _canonical_locality_key(location) if resolved else "",
                        "locality": _clean_text(location.get("locality")) if resolved else "",
                        "state": _clean_text(location.get("state")) if resolved else "",
                        "country": _clean_text(location.get("country")) if resolved else "",
                        "iso3": _clean_text(location.get("iso3")).upper() if resolved else "",
                        "continent": _clean_text(location.get("continent")) if resolved else "",
                        "lat": _finite_float(location.get("lat")) if resolved else None,
                        "lon": _finite_float(location.get("lon")) if resolved else None,
                    }
                )

        columns = [
            "video_id",
            "segment_index",
            "start_time",
            "end_time",
            "duration_seconds",
            "time_of_day",
            "road_users",
            "event_kind",
            "manner_of_collision_code",
            "manner_of_collision",
            "first_harmful_event_code",
            "first_harmful_event",
            "sequence_of_events",
            "crash_taxonomy_standard",
            "crash_taxonomy_status",
            "crash_taxonomy_version",
            "geocode_status",
            "location_resolution_version",
            "resolved_location",
            "locality_key",
            "locality",
            "state",
            "country",
            "iso3",
            "continent",
            "lat",
            "lon",
        ]
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _video_status_table(state: dict[str, Any]) -> pd.DataFrame:
        statuses = []
        for record in state.get("videos", {}).values():
            if not isinstance(record, dict):
                statuses.append("invalid_record")
                continue
            statuses.append(_clean_text(record.get("status")) or "missing")
        table = _count_table(pd.Series(statuses, dtype="object"), value_name="status")
        table["label"] = table["status"].map(_human_label)
        return table[["status", "label", "count", "percentage"]]

    @staticmethod
    def _taxonomy_coverage_table(segments: pd.DataFrame) -> pd.DataFrame:
        if segments.empty:
            return pd.DataFrame(columns=["status", "label", "count", "percentage"])

        def coverage_status(row: pd.Series) -> str:
            if row.get("crash_taxonomy_version") != CRASH_TAXONOMY_VERSION:
                return "pending"
            status = _clean_text(row.get("crash_taxonomy_status"))
            return status or "current_unlabelled"

        values = segments.apply(coverage_status, axis=1)
        table = _count_table(values, value_name="status")
        table["label"] = table["status"].map(_human_label)
        return table[["status", "label", "count", "percentage"]]

    @staticmethod
    def _road_user_table(segments: pd.DataFrame) -> pd.DataFrame:
        total = len(segments)
        counter: dict[str, int] = {}
        for users in segments.get("road_users", pd.Series(dtype=object)):
            if not isinstance(users, list):
                continue
            for value in {_clean_text(item) for item in users if _clean_text(item)}:
                counter[value] = counter.get(value, 0) + 1
        rows = [
            {
                "road_user": key,
                "label": _human_label(key),
                "segments": count,
                "segment_prevalence_pct": round(count / total * 100, 2) if total else 0.0,
            }
            for key, count in counter.items()
        ]
        return pd.DataFrame(rows).sort_values(["segments", "road_user"], ascending=[False, True], ignore_index=True)

    @staticmethod
    def _sequence_table(classified_collisions: pd.DataFrame) -> pd.DataFrame:
        denominator = len(classified_collisions)
        counter: dict[str, int] = {}
        for sequence in classified_collisions.get("sequence_of_events", pd.Series(dtype=object)):
            if not isinstance(sequence, list):
                continue
            for value in {_clean_text(item) for item in sequence if _clean_text(item)}:
                counter[value] = counter.get(value, 0) + 1
        rows = [
            {
                "sequence_event": key,
                "label": _human_label(key),
                "segments": count,
                "classified_collision_prevalence_pct": round(count / denominator * 100, 2) if denominator else 0.0,
            }
            for key, count in counter.items()
        ]
        return pd.DataFrame(rows).sort_values(["segments", "sequence_event"], ascending=[False, True], ignore_index=True)

    @staticmethod
    def _mapping_summary(mapping: pd.DataFrame) -> dict[str, Any]:
        segment_records = 0
        duration_seconds = 0.0
        unique_resolved_uploads: set[str] = set()
        invalid_coordinate_rows = 0

        for _, row in mapping.iterrows():
            starts = _flatten_numeric_list(row.get("start_time"))
            ends = _flatten_numeric_list(row.get("end_time"))
            segment_records += min(len(starts), len(ends))
            duration_seconds += sum(max(0.0, end - start) for start, end in zip(starts, ends))
            unique_resolved_uploads.update(_parse_bare_bracket_list(row.get("videos")))

            lat = _finite_float(row.get("lat"))
            lon = _finite_float(row.get("lon"))
            if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                invalid_coordinate_rows += 1

        return {
            "rows": int(len(mapping)),
            "expanded_segment_records": int(segment_records),
            "expanded_duration_seconds": round(duration_seconds, 3),
            "expanded_duration_hours": round(duration_seconds / 3600, 2),
            "unique_uploads_with_resolved_mapping": int(len(unique_resolved_uploads)),
            "invalid_coordinate_rows": int(invalid_coordinate_rows),
        }

    @staticmethod
    def _mapping_geography_table(mapping: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, row in mapping.iterrows():
            starts = _flatten_numeric_list(row.get("start_time"))
            ends = _flatten_numeric_list(row.get("end_time"))
            segment_count = min(len(starts), len(ends))
            duration_seconds = sum(max(0.0, end - start) for start, end in zip(starts, ends))
            rows.append(
                {
                    "mapping_id": row.get("id", ""),
                    "locality": _clean_text(row.get("locality")),
                    "state": _clean_text(row.get("state")),
                    "country": _clean_text(row.get("country")),
                    "iso3": _clean_text(row.get("iso3")).upper(),
                    "continent": _clean_text(row.get("continent")),
                    "lat": _finite_float(row.get("lat")),
                    "lon": _finite_float(row.get("lon")),
                    "resolved_segment_count": int(segment_count),
                    "resolved_duration_seconds": round(duration_seconds, 3),
                    "resolved_duration_hours": round(duration_seconds / 3600, 2),
                    "unique_uploads": len(set(_parse_bare_bracket_list(row.get("videos")))),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _geography_tables(resolved: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if resolved.empty:
            empty = pd.DataFrame()
            return {"continents": empty, "countries": empty, "localities": empty}

        continent = (
            resolved.groupby("continent", dropna=False)
            .agg(
                resolved_segments=("video_id", "size"),
                unique_uploads=("video_id", "nunique"),
                resolved_duration_seconds=("duration_seconds", "sum"),
                canonical_localities=("locality_key", "nunique"),
            )
            .reset_index()
        )
        continent["resolved_duration_hours"] = (continent["resolved_duration_seconds"] / 3600).round(2)
        continent["segment_share_pct"] = (continent["resolved_segments"] / len(resolved) * 100).round(2)
        continent = continent.sort_values(["resolved_segments", "continent"], ascending=[False, True], ignore_index=True)

        country = (
            resolved.groupby(["country", "iso3", "continent"], dropna=False)
            .agg(
                resolved_segments=("video_id", "size"),
                unique_uploads=("video_id", "nunique"),
                canonical_localities=("locality_key", "nunique"),
                resolved_duration_seconds=("duration_seconds", "sum"),
            )
            .reset_index()
        )
        country["resolved_duration_hours"] = (country["resolved_duration_seconds"] / 3600).round(2)
        country["segment_share_pct"] = (country["resolved_segments"] / len(resolved) * 100).round(2)
        country = country.sort_values(["resolved_segments", "country"], ascending=[False, True], ignore_index=True)

        locality = (
            resolved.groupby(
                ["locality_key", "locality", "state", "country", "iso3", "continent", "lat", "lon"],
                dropna=False,
            )
            .agg(
                resolved_segments=("video_id", "size"),
                unique_uploads=("video_id", "nunique"),
                resolved_duration_seconds=("duration_seconds", "sum"),
            )
            .reset_index()
        )
        locality["resolved_duration_hours"] = (locality["resolved_duration_seconds"] / 3600).round(2)
        locality["segment_share_pct"] = (locality["resolved_segments"] / len(resolved) * 100).round(2)
        locality = locality.sort_values(
            ["resolved_segments", "country", "locality"],
            ascending=[False, True, True],
            ignore_index=True,
        )

        return {"continents": continent, "countries": country, "localities": locality}

    def analyse(self) -> CrashAnalysisResult:
        state = self.load_state()
        mapping = self.load_mapping()
        segments = self._extract_segments(state)
        resolved = segments.loc[segments["resolved_location"]].copy() if not segments.empty else segments.copy()

        video_status = self._video_status_table(state)
        location_status = _count_table(segments["geocode_status"], value_name="status") if not segments.empty else pd.DataFrame()
        if not location_status.empty:
            location_status["label"] = location_status["status"].map(_human_label)
            location_status = location_status[["status", "label", "count", "percentage"]]

        taxonomy_coverage = self._taxonomy_coverage_table(segments)

        classified = segments.loc[
            (segments["crash_taxonomy_version"] == CRASH_TAXONOMY_VERSION)
            & (segments["crash_taxonomy_status"] == "classified")
        ].copy()
        classified_collisions = classified.loc[classified["event_kind"] == "collision"].copy()

        event_kind = _count_table(classified["event_kind"].replace("", "unknown"), value_name="event_kind") if not classified.empty else pd.DataFrame()
        if not event_kind.empty:
            event_kind["label"] = event_kind["event_kind"].map(_human_label)

        manner = _count_table(
            classified_collisions["manner_of_collision"].replace("", "unknown"),
            value_name="manner_of_collision",
        ) if not classified_collisions.empty else pd.DataFrame()
        if not manner.empty:
            manner["label"] = manner["manner_of_collision"].map(_human_label)

        first_harmful = _count_table(
            classified_collisions["first_harmful_event"].replace("", "unknown"),
            value_name="first_harmful_event",
        ) if not classified_collisions.empty else pd.DataFrame()
        if not first_harmful.empty:
            first_harmful["label"] = first_harmful["first_harmful_event"].map(_human_label)

        sequence = self._sequence_table(classified_collisions)
        road_users = self._road_user_table(segments)
        time_of_day = _count_table(segments["time_of_day"].replace("", "unknown"), value_name="time_of_day") if not segments.empty else pd.DataFrame()
        if not time_of_day.empty:
            time_of_day["label"] = time_of_day["time_of_day"].map(_human_label)

        geography = self._geography_tables(resolved)
        mapping_geography = self._mapping_geography_table(mapping)
        mapping_summary = self._mapping_summary(mapping)

        status_counts = {
            row["status"]: int(row["count"])
            for row in video_status.to_dict(orient="records")
        }
        location_counts = {
            row["status"]: int(row["count"])
            for row in location_status.to_dict(orient="records")
        } if not location_status.empty else {}
        taxonomy_counts = {
            row["status"]: int(row["count"])
            for row in taxonomy_coverage.to_dict(orient="records")
        } if not taxonomy_coverage.empty else {}

        accepted_segments = int(len(segments))
        resolved_segments = int(len(resolved))
        classified_segments = int(len(classified))
        taxonomy_current_segments = int(
            (segments["crash_taxonomy_version"] == CRASH_TAXONOMY_VERSION).sum()
        ) if not segments.empty else 0

        all_records = [
            record for record in state.get("videos", {}).values() if isinstance(record, dict)
        ]
        metadata_included = sum(
            isinstance(record.get("text_decision"), dict)
            and bool(record["text_decision"].get("include"))
            for record in all_records
        )

        unique_localities = int(resolved["locality_key"].nunique()) if not resolved.empty else 0
        unique_countries = int(resolved.loc[resolved["iso3"] != "", "iso3"].nunique()) if not resolved.empty else 0
        unique_continents = int(resolved.loc[resolved["continent"] != "", "continent"].nunique()) if not resolved.empty else 0

        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "state_json": str(self.state_path),
                "mapping_csv": str(self.mapping_path),
            },
            "videos": {
                "records": int(len(state.get("videos", {}))),
                "metadata_included": int(metadata_included),
                "statuses": status_counts,
                "complete": int(status_counts.get("complete", 0)),
                "text_rejected": int(status_counts.get("text_rejected", 0)),
                "visual_rejected": int(status_counts.get("visual_rejected", 0)),
                "visual_error": int(status_counts.get("visual_error", 0)),
            },
            "segments": {
                "accepted": accepted_segments,
                "retained_duration_seconds": round(float(segments["duration_seconds"].sum()), 3) if not segments.empty else 0.0,
                "retained_duration_hours": round(float(segments["duration_seconds"].sum()) / 3600, 2) if not segments.empty else 0.0,
            },
            "location": {
                "statuses": location_counts,
                "resolved": resolved_segments,
                "resolved_pct": round(resolved_segments / accepted_segments * 100, 2) if accepted_segments else 0.0,
                "unresolved": accepted_segments - resolved_segments,
                "unresolved_pct": round((accepted_segments - resolved_segments) / accepted_segments * 100, 2) if accepted_segments else 0.0,
                "current_resolution_version": LOCATION_RESOLUTION_VERSION,
            },
            "geography": {
                "canonical_locality_entities": unique_localities,
                "countries_or_territories": unique_countries,
                "continents": unique_continents,
                "coordinate_semantics": "canonical locality reference coordinates, not event coordinates",
            },
            "taxonomy": {
                "current_version": CRASH_TAXONOMY_VERSION,
                "current_version_segments": taxonomy_current_segments,
                "classified_segments": classified_segments,
                "classified_pct_of_accepted": round(classified_segments / accepted_segments * 100, 2) if accepted_segments else 0.0,
                "pending_segments": accepted_segments - taxonomy_current_segments,
                "pending_pct_of_accepted": round((accepted_segments - taxonomy_current_segments) / accepted_segments * 100, 2) if accepted_segments else 0.0,
                "coverage_statuses": taxonomy_counts,
            },
            "mapping": {
                **mapping_summary,
                "state_resolved_segments": resolved_segments,
                "segment_count_matches_state": mapping_summary["expanded_segment_records"] == resolved_segments,
                "note": "mapping.csv rows are not used as a substitute for the canonical locality-entity count",
            },
        }

        tables = {
            "video_status": video_status,
            "location_status": location_status,
            "taxonomy_coverage": taxonomy_coverage,
            "taxonomy_event_kind": event_kind,
            "taxonomy_manner": manner,
            "taxonomy_first_harmful_event": first_harmful,
            "taxonomy_sequence_events": sequence,
            "road_users": road_users,
            "time_of_day": time_of_day,
            "continents": geography["continents"],
            "countries": geography["countries"],
            "localities": geography["localities"],
            "mapping_geography": mapping_geography,
        }

        return CrashAnalysisResult(
            summary=summary,
            tables=tables,
            segments=segments,
            resolved_segments=resolved,
            mapping=mapping,
        )

    @staticmethod
    def write_outputs(result: CrashAnalysisResult, output_dir: str | Path) -> None:
        output = Path(output_dir)
        tables_dir = output / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        (output / "summary.json").write_text(
            json.dumps(result.summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for name, table in result.tables.items():
            table.to_csv(tables_dir / f"{name}.csv", index=False)

        # Compact segment table for audit, downstream statistics, and plotting.
        result.segments.to_pickle(output / "segments_analysis.pkl")

        CrashResultsAnalysis._write_results_markdown(result, output / "results_summary.md")

    @staticmethod
    def _write_results_markdown(result: CrashAnalysisResult, path: Path) -> None:
        summary = result.summary
        videos = summary["videos"]
        segments = summary["segments"]
        location = summary["location"]
        geography = summary["geography"]
        taxonomy = summary["taxonomy"]
        mapping = summary["mapping"]

        lines = [
            "# Crash corpus analysis summary",
            "",
            "## Corpus construction",
            f"* Video records: {videos['records']:,}",
            f"* Complete videos: {videos['complete']:,}",
            f"* Metadata rejected: {videos['text_rejected']:,}",
            f"* Visual rejected: {videos['visual_rejected']:,}",
            f"* Visual errors: {videos['visual_error']:,}",
            f"* Accepted segments: {segments['accepted']:,}",
            f"* Retained full-clip duration: {segments['retained_duration_hours']:.2f} h",
            "",
            "## Geographic coverage",
            f"* Resolved segments: {location['resolved']:,} ({location['resolved_pct']:.2f}%)",
            f"* Unresolved segments: {location['unresolved']:,} ({location['unresolved_pct']:.2f}%)",
            f"* Canonical locality entities: {geography['canonical_locality_entities']:,}",
            f"* Countries or territories: {geography['countries_or_territories']:,}",
            f"* Continents: {geography['continents']:,}",
            "* Coordinates denote canonical locality reference points, not crash-event coordinates.",
            "",
            "## MMUCC taxonomy",
            f"* Current taxonomy version: {taxonomy['current_version']}",
            f"* Classified segments: {taxonomy['classified_segments']:,} ({taxonomy['classified_pct_of_accepted']:.2f}% of accepted segments)",
            f"* Pending segments: {taxonomy['pending_segments']:,} ({taxonomy['pending_pct_of_accepted']:.2f}% of accepted segments)",
            "",
            "## Mapping consistency",
            f"* mapping.csv rows: {mapping['rows']:,}",
            f"* Expanded resolved segment records in mapping.csv: {mapping['expanded_segment_records']:,}",
            f"* Resolved segments in state.json: {mapping['state_resolved_segments']:,}",
            f"* Counts match: {mapping['segment_count_matches_state']}",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def append_snapshot(
        result: CrashAnalysisResult,
        output_dir: str | Path,
        *,
        label: str = "",
    ) -> Path:
        """Append one explicitly requested longitudinal snapshot.

        This is intentionally opt-in so repeated analysis runs are not silently
        misrepresented as dataset collection rounds.
        """

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "snapshots.csv"
        summary = result.summary
        row = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "video_records": summary["videos"]["records"],
            "complete_videos": summary["videos"]["complete"],
            "accepted_segments": summary["segments"]["accepted"],
            "resolved_segments": summary["location"]["resolved"],
            "taxonomy_classified_segments": summary["taxonomy"]["classified_segments"],
        }

        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return path
