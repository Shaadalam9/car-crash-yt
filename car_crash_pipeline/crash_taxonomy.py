"""NHTSA MMUCC 6th Edition crash taxonomy classification and backfill.

This module adds a video-observable taxonomy alongside the existing crash review.

The semantic category names follow the NHTSA Model Minimum Uniform Crash
Criteria (MMUCC), Sixth Edition, for:

* C7 First Harmful Event
* C9 Manner of Collision of the First Harmful Event
* V37 Sequence of Events

``event_kind`` is a dataset-specific extension because near collisions are not
reportable crashes and therefore fall outside MMUCC crash coding.

The ``*_code`` fields below are stable explicit identifiers made from the MMUCC
element identifier plus the canonical attribute name. They are deliberately not
presented as official MMUCC numeric codes.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import settings
from .shared import (
    clean_text,
    log,
    normalise_string_list,
    recover_json,
    save_state,
    unload_model,
)


CRASH_TAXONOMY_VERSION = "nhtsa_mmucc6_video_v1"
CRASH_TAXONOMY_STANDARD = "NHTSA MMUCC 6th Edition (2024)"
TAXONOMY_REVIEW_ATTEMPTS = 2

EVENT_KINDS = {
    "collision",
    "near_collision",
}

MANNER_OF_COLLISION_VALUES = {
    "not_collision_with_motor_vehicle_in_transport",
    "angle",
    "front_to_front",
    "front_to_rear_or_rear_to_front",
    "rear_to_rear",
    "rear_to_side_or_side_to_rear",
    "sideswipe_opposite_direction",
    "sideswipe_same_direction",
    "other",
    "unknown",
}

MANNER_OF_COLLISION_CODES = {
    value: "C9_" + value.upper()
    for value in MANNER_OF_COLLISION_VALUES
}

FIRST_HARMFUL_EVENT_VALUES = {
    # Group 1: Non-Collision Harmful Events
    "rollover_overturn",
    "cargo_equipment_loss_shift_or_damage_harmful",
    "fell_jumped_from_motor_vehicle",
    "fire_explosion",
    "immersion_full_or_partial",
    "jackknife_harmful",
    "thrown_or_falling_object",
    "pavement_surface_irregularity",
    "injured_in_vehicle_non_collision",
    "gas_inhalation",
    "other_non_collision",
    # Group 2: Collision with Motor Vehicle
    "motor_vehicle_in_transport",
    "parked_motor_vehicle",
    "working_motor_vehicle",
    # Group 3: Collision with Non-Fixed Object
    "non_motorist",
    "live_animal",
    "ridden_animal_or_animal_drawn_conveyance",
    "railroad_vehicle",
    "road_vehicle_on_rails",
    "strikes_object_at_rest_that_had_fallen_from_motor_vehicle_in_transport",
    "striking_struck_by_object_cargo_person_from_other_motor_vehicle_in_transport",
    "other_object_not_fixed",
    "unknown_object_not_fixed",
    # Group 4: Collision with Fixed Object
    "bridge_overhead_structure",
    "bridge_pier_or_support",
    "bridge_rail_includes_parapet",
    "building",
    "wall",
    "cable_barrier",
    "concrete_traffic_barrier",
    "guardrail_face",
    "guardrail_end",
    "guardrail_end_treatment",
    "impact_attenuator_crash_cushion",
    "other_traffic_barrier",
    "traffic_sign_support",
    "traffic_signal_support",
    "utility_pole_light_support",
    "other_post_pole_or_other_supports",
    "culvert",
    "curb",
    "ditch",
    "embankment",
    "boulder",
    "ground",
    "tree_standing_only",
    "shrubbery",
    "snowbank",
    "fence",
    "mailbox",
    "fire_hydrant",
    "other_fixed_object",
    "unknown_fixed_object",
    "unknown",
}

FIRST_HARMFUL_EVENT_CODES = {
    value: "C7_" + value.upper()
    for value in FIRST_HARMFUL_EVENT_VALUES
}

SEQUENCE_OF_EVENT_VALUES = {
    # Group 1: Non-Harmful Events
    "cross_centerline",
    "cross_median",
    "end_departure",
    "downhill_runaway",
    "equipment_failure",
    "ran_off_roadway_left",
    "ran_off_roadway_right",
    "ran_off_roadway_direction_unknown",
    "non_harmful_swaying_trailer_jackknife",
    "cargo_equipment_loss_or_shift_non_harmful",
    "reentering_roadway",
    "separation_of_units",
    "vehicle_went_airborne",
    # Group 2: Non-Collision Harmful Events
    "rollover_overturn",
    "cargo_equipment_loss_shift_or_damage_harmful",
    "fell_jumped_from_motor_vehicle",
    "fire_explosion",
    "immersion_full_or_partial",
    "jackknife_harmful",
    "thrown_or_falling_object",
    "pavement_surface_irregularity",
    "injured_in_vehicle_non_collision",
    "gas_inhalation",
    "other_non_collision",
    # Group 3: Collision with Motor Vehicle
    "motor_vehicle_in_transport",
    "parked_motor_vehicle",
    "working_motor_vehicle",
    # Group 4: Collision with Non-Fixed Object
    "non_motorist",
    "live_animal",
    "ridden_animal_or_animal_drawn_conveyance",
    "railroad_vehicle",
    "road_vehicle_on_rails",
    "strikes_object_at_rest_that_had_fallen_from_motor_vehicle_in_transport",
    "striking_struck_by_object_cargo_person_from_other_motor_vehicle_in_transport",
    "other_object_not_fixed",
    "unknown_object_not_fixed",
    # Group 5: Collision with Fixed Object
    "bridge_overhead_structure",
    "bridge_pier_or_support",
    "bridge_rail_includes_parapet",
    "building",
    "wall",
    "cable_barrier",
    "concrete_traffic_barrier",
    "guardrail_face",
    "guardrail_end",
    "guardrail_end_treatment",
    "impact_attenuator_crash_cushion",
    "other_traffic_barrier",
    "traffic_sign_support",
    "traffic_signal_support",
    "utility_pole_light_support",
    "other_post_pole_or_other_supports",
    "culvert",
    "curb",
    "ditch",
    "embankment",
    "boulder",
    "ground",
    "tree_standing_only",
    "shrubbery",
    "snowbank",
    "fence",
    "mailbox",
    "fire_hydrant",
    "other_fixed_object",
    "unknown_fixed_object",
    # Group 6
    "unknown",
}

NON_HARMFUL_SEQUENCE_VALUES = {
    "cross_centerline",
    "cross_median",
    "end_departure",
    "downhill_runaway",
    "equipment_failure",
    "ran_off_roadway_left",
    "ran_off_roadway_right",
    "ran_off_roadway_direction_unknown",
    "non_harmful_swaying_trailer_jackknife",
    "cargo_equipment_loss_or_shift_non_harmful",
    "reentering_roadway",
    "separation_of_units",
    "vehicle_went_airborne",
}


def _normalised_enum(value: Any) -> str:
    return clean_text(value).strip().lower().replace("-", "_").replace(" ", "_")


def _taxonomy_empty_fields(
    status: str,
    *,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "event_kind": None,
        "manner_of_collision_code": None,
        "manner_of_collision": None,
        "first_harmful_event_code": None,
        "first_harmful_event": None,
        "sequence_of_events": [],
        "crash_taxonomy_standard": CRASH_TAXONOMY_STANDARD,
        "crash_taxonomy_status": status,
        "crash_taxonomy_version": CRASH_TAXONOMY_VERSION,
        "crash_taxonomy_error": error,
    }


def validate_taxonomy_response(data: Dict[str, Any]) -> Optional[str]:
    """Return a semantic validation error for a taxonomy-only model response."""
    event_kind = _normalised_enum(data.get("event_kind"))
    if event_kind not in EVENT_KINDS:
        return "event_kind_must_be_collision_or_near_collision"

    manner = data.get("manner_of_collision")
    first_harmful = data.get("first_harmful_event")
    sequence = data.get("sequence_of_events")

    if not isinstance(sequence, list):
        return "sequence_of_events_must_be_a_list"
    if len(sequence) > 4:
        return "sequence_of_events_must_have_at_most_four_values"

    sequence_values = [_normalised_enum(value) for value in sequence]
    if len(sequence_values) != len(set(sequence_values)):
        return "sequence_of_events_must_not_contain_duplicates"
    if any(value not in SEQUENCE_OF_EVENT_VALUES for value in sequence_values):
        return "sequence_of_events_contains_unknown_value"

    if event_kind == "near_collision":
        if manner is not None or first_harmful is not None:
            return "near_collision_must_not_have_mmucc_harmful_event_fields"
        if sequence_values:
            return "near_collision_must_have_empty_sequence_of_events"
        return None

    manner_value = _normalised_enum(manner)
    first_value = _normalised_enum(first_harmful)

    if manner_value not in MANNER_OF_COLLISION_VALUES:
        return "manner_of_collision_must_be_one_allowed_value"
    if first_value not in FIRST_HARMFUL_EVENT_VALUES:
        return "first_harmful_event_must_be_one_allowed_value"

    if first_value == "motor_vehicle_in_transport":
        if manner_value == "not_collision_with_motor_vehicle_in_transport":
            return "motor_vehicle_in_transport_requires_collision_manner"
    elif manner_value != "not_collision_with_motor_vehicle_in_transport":
        return "non_motor_vehicle_first_harmful_event_requires_not_collision_manner"

    if not sequence_values:
        return "collision_requires_at_least_one_sequence_event"

    if first_value not in sequence_values:
        return "sequence_of_events_must_include_first_harmful_event"

    first_harmful_index = sequence_values.index(first_value)
    for position, value in enumerate(sequence_values):
        if position < first_harmful_index and value not in NON_HARMFUL_SEQUENCE_VALUES:
            return "harmful_sequence_event_cannot_precede_first_harmful_event"

    return None


def normalise_taxonomy_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a validated model response into stable state and CSV fields."""
    event_kind = _normalised_enum(data.get("event_kind"))

    if event_kind == "near_collision":
        return {
            "event_kind": "near_collision",
            "manner_of_collision_code": None,
            "manner_of_collision": None,
            "first_harmful_event_code": None,
            "first_harmful_event": None,
            "sequence_of_events": [],
            "crash_taxonomy_standard": CRASH_TAXONOMY_STANDARD,
            "crash_taxonomy_status": "classified",
            "crash_taxonomy_version": CRASH_TAXONOMY_VERSION,
            "crash_taxonomy_error": None,
        }

    manner = _normalised_enum(data.get("manner_of_collision"))
    first_harmful = _normalised_enum(data.get("first_harmful_event"))
    sequence = [
        _normalised_enum(value)
        for value in normalise_string_list(data.get("sequence_of_events"))
    ][:4]

    return {
        "event_kind": "collision",
        "manner_of_collision_code": MANNER_OF_COLLISION_CODES[manner],
        "manner_of_collision": manner,
        "first_harmful_event_code": FIRST_HARMFUL_EVENT_CODES[first_harmful],
        "first_harmful_event": first_harmful,
        "sequence_of_events": sequence,
        "crash_taxonomy_standard": CRASH_TAXONOMY_STANDARD,
        "crash_taxonomy_status": "classified",
        "crash_taxonomy_version": CRASH_TAXONOMY_VERSION,
        "crash_taxonomy_error": None,
    }


def taxonomy_prompt(
    segment: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the taxonomy-only prompt used to backfill accepted old segments."""
    metadata = metadata if isinstance(metadata, dict) else {}
    duration = float(segment.get("duration_seconds") or 0.0)
    title = clean_text(metadata.get("title"))
    previous_description = clean_text(segment.get("short_description"))
    previous_type = clean_text(segment.get("crash_type"))

    manner_values = ", ".join(sorted(MANNER_OF_COLLISION_VALUES))
    first_values = ", ".join(sorted(FIRST_HARMFUL_EVENT_VALUES))
    sequence_values = ", ".join(sorted(SEQUENCE_OF_EVENT_VALUES))

    return f"""
You are classifying an already accepted road safety video segment using the
NHTSA Model Minimum Uniform Crash Criteria, MMUCC, Sixth Edition.

Use only events that are visibly supported by this video. Do not infer fault,
injury severity, intent, legal responsibility, or an event that happens outside
the sampled clip.

The clip duration is {duration:.3f} seconds.

Dataset extension:
event_kind is collision when visible contact, damage-producing impact,
rollover, fire, or another harmful crash event occurs.
event_kind is near_collision only when impact is narrowly avoided through
visible evasive action. A near collision is outside MMUCC crash coding, so
manner_of_collision and first_harmful_event must both be null and
sequence_of_events must be [].

For a collision:
1. C7 First Harmful Event is the first visible injury-producing or
   damage-producing event.
2. C9 Manner of Collision of the First Harmful Event describes orientation
   only when the first harmful event is a collision with another motor vehicle
   in transport.
3. V37 Sequence of Events contains up to four visually observable events in
   chronological order. Include non-harmful precursor events only when visible.
   The first harmful event must appear in the sequence.

If first_harmful_event is not motor_vehicle_in_transport, set
manner_of_collision to not_collision_with_motor_vehicle_in_transport.
If the first harmful event is motor_vehicle_in_transport, select its visible
manner. Use unknown rather than guessing.

Previous machine-generated context may help you find the relevant event but is
not authoritative:
previous short description: {previous_description or "none"}
previous crash_type: {previous_type or "none"}
upload title: {title or "none"}

Allowed manner_of_collision:
{manner_values}

Allowed first_harmful_event:
{first_values}

Allowed sequence_of_events:
{sequence_values}

Return JSON only and use the canonical snake_case values exactly:
{{
  "event_kind": "collision",
  "manner_of_collision": "front_to_rear_or_rear_to_front",
  "first_harmful_event": "motor_vehicle_in_transport",
  "sequence_of_events": ["motor_vehicle_in_transport"]
}}
""".strip()


def crash_taxonomy_pending(record: Any) -> bool:
    """Whether a completed record still needs taxonomy backfill."""
    if not isinstance(record, dict) or record.get("status") != "complete":
        return False

    segments = record.get("segments", [])
    if not isinstance(segments, list):
        return False

    return any(
        isinstance(segment, dict)
        and segment.get("crash_taxonomy_version") != CRASH_TAXONOMY_VERSION
        for segment in segments
    )


def _pending_segments(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        segment
        for segment in record.get("segments", [])
        if isinstance(segment, dict)
        and segment.get("crash_taxonomy_version") != CRASH_TAXONOMY_VERSION
    ]


def _mark_video_unavailable(
    record: Dict[str, Any],
    segments: List[Dict[str, Any]],
    error: str,
) -> None:
    for segment in segments:
        segment.update(
            _taxonomy_empty_fields(
                "video_unavailable",
                error=error,
            )
        )

    record["crash_taxonomy_status"] = "video_unavailable"
    record["crash_taxonomy_error"] = error
    record["crash_taxonomy_version"] = CRASH_TAXONOMY_VERSION


def _existing_video_path(video_id: str, record: Dict[str, Any]) -> Optional[Path]:
    from .crash_review import _valid_video

    candidates: List[Path] = []

    stored = record.get("downloaded_path")
    if stored:
        candidates.append(Path(str(stored)))

    if settings.VIDEO_DIR.exists():
        candidates.extend(
            path
            for path in settings.VIDEO_DIR.glob(f"{video_id}.*")
            if path.is_file()
        )

    seen = set()
    for path in candidates:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file() and _valid_video(path):
            return path

    return None


def _obtain_video(video_id: str, record: Dict[str, Any]) -> Path:
    from .crash_review import download_video

    existing = _existing_video_path(video_id, record)
    if existing is not None:
        record["downloaded_path"] = str(existing)
        log(f"Reusing downloaded video for taxonomy backfill: {video_id}")
        return existing

    path = download_video(video_id)
    record["downloaded_path"] = str(path)
    return path


def _review_one_segment(
    judge: Any,
    video_path: Path,
    video_id: str,
    segment: Dict[str, Any],
    metadata: Dict[str, Any],
    work: Path,
) -> Dict[str, Any]:
    from .crash_review import _create_segment_sample
    from .cut_detection import FullSegment

    full_segment = FullSegment(
        start_time=float(segment.get("start_time")),
        end_time=float(segment.get("end_time")),
    )

    sample_path, sample_fps = _create_segment_sample(
        video_path,
        full_segment,
        work / f"taxonomy-{int(segment.get('segment_index', 0)):05d}.mp4",
    )

    base_prompt = taxonomy_prompt(segment, metadata)
    prompt = base_prompt
    answer = ""
    last_error = "model_did_not_return_usable_taxonomy"

    for attempt in range(TAXONOMY_REVIEW_ATTEMPTS):
        try:
            answer = judge._generate(sample_path, prompt, sample_fps)
            data = recover_json(answer)
            if data is None:
                last_error = "json_recovery_failed"
            else:
                error = validate_taxonomy_response(data)
                if error is None:
                    result = normalise_taxonomy_response(data)
                    result["crash_taxonomy_raw_response"] = answer
                    return result
                last_error = f"semantic_error: {error}"
        except Exception as exc:
            last_error = f"model_error: {exc}"

        if attempt + 1 < TAXONOMY_REVIEW_ATTEMPTS:
            prompt = (
                base_prompt
                + "\n\nYour previous answer was invalid because: "
                + last_error
                + ". Reinspect the video and return one corrected JSON object."
            )

    return {
        "status": "model_error",
        "error": last_error,
        "raw_response": answer,
    }


def _store_model_error(
    segment: Dict[str, Any],
    result: Dict[str, Any],
) -> bool:
    """Store retry state. Return True when the failure became terminal."""
    try:
        previous_cycles = max(
            0,
            int(segment.get("crash_taxonomy_review_cycles", 0)),
        )
    except (TypeError, ValueError):
        previous_cycles = 0

    cycles = previous_cycles + 1
    segment["crash_taxonomy_review_cycles"] = cycles
    segment["crash_taxonomy_status"] = "model_error"
    segment["crash_taxonomy_error"] = str(result.get("error") or "model_error")
    segment["crash_taxonomy_raw_response"] = str(
        result.get("raw_response") or ""
    )
    segment["crash_taxonomy_standard"] = CRASH_TAXONOMY_STANDARD

    if cycles < settings.MAX_REVIEW_CYCLES:
        return False

    segment.update(
        _taxonomy_empty_fields(
            "model_error_terminal",
            error=segment["crash_taxonomy_error"],
        )
    )
    segment["crash_taxonomy_review_cycles"] = cycles
    return True


def _cleanup_taxonomy_video_if_safe(
    record: Dict[str, Any],
    video_path: Path,
) -> None:
    """
    Delete a retained/downloaded source only when no location review still
    needs the video. This preserves the existing pipeline's deletion policy.
    """
    if not settings.DELETE_VIDEO_AFTER_PROCESSING:
        return

    from .crash_review import (
        _needs_location_visual_review,
        has_location_visual_review_errors,
    )

    if has_location_visual_review_errors(record):
        return

    if any(
        isinstance(segment, dict)
        and _needs_location_visual_review(segment)
        for segment in record.get("segments", [])
    ):
        return

    video_path.unlink(missing_ok=True)
    record["downloaded_path"] = None
    log(f"Deleted taxonomy-reviewed video: {video_path}")


def run_crash_taxonomy_stage(state: Dict[str, Any]) -> int:
    """
    Backfill MMUCC taxonomy for completed accepted segments.

    Old state is upgraded on restart. If an old YouTube source can no longer be
    downloaded, its accepted segment remains intact and is terminally marked
    ``video_unavailable`` so the continuous queue is never blocked.
    """
    pending_records = [
        (video_id, record)
        for video_id, record in state.get("videos", {}).items()
        if crash_taxonomy_pending(record)
    ][: settings.MAX_VIDEOS_PER_RUN]

    if not pending_records:
        return 0

    judge: Any = None
    processed = 0

    try:
        for video_id, record in pending_records:
            segments = _pending_segments(record)
            if not segments:
                continue

            try:
                video_path = _obtain_video(video_id, record)
            except KeyboardInterrupt:
                save_state(settings.STATE_JSON, state)
                raise
            except Exception as exc:
                error = clean_text(exc) or "video_download_failed"
                _mark_video_unavailable(record, segments, error)
                save_state(settings.STATE_JSON, state)
                log(
                    f"Skipping taxonomy backfill for unavailable video "
                    f"{video_id}: {error}"
                )
                processed += len(segments)
                continue

            if judge is None:
                from .crash_review import CosmosCrashJudge

                judge = CosmosCrashJudge()

            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            classified = 0
            retryable_errors = 0
            terminal_errors = 0

            with tempfile.TemporaryDirectory(
                prefix="crash-taxonomy-",
                dir=settings.DATA_DIR,
            ) as temporary:
                work = Path(temporary)

                for position, segment in enumerate(segments, start=1):
                    try:
                        result = _review_one_segment(
                            judge,
                            video_path,
                            video_id,
                            segment,
                            metadata,
                            work,
                        )
                    except KeyboardInterrupt:
                        save_state(settings.STATE_JSON, state)
                        raise
                    except Exception as exc:
                        result = {
                            "status": "model_error",
                            "error": f"sample_error: {exc}",
                            "raw_response": "",
                        }

                    if result.get("status") == "model_error":
                        terminal = _store_model_error(segment, result)
                        if terminal:
                            terminal_errors += 1
                        else:
                            retryable_errors += 1
                    else:
                        segment.update(result)
                        segment["crash_taxonomy_review_cycles"] = 0
                        classified += 1

                    if (
                        position == 1
                        or position % 10 == 0
                        or position == len(segments)
                    ):
                        log(
                            f"Taxonomy progress for {video_id}: "
                            f"{position}/{len(segments)}"
                        )

            if retryable_errors:
                record["crash_taxonomy_status"] = "model_error"
                record["crash_taxonomy_error"] = (
                    f"{retryable_errors} segment taxonomy reviews need retry"
                )
                record.pop("crash_taxonomy_version", None)
            elif terminal_errors and not classified:
                record["crash_taxonomy_status"] = "model_error_terminal"
                record["crash_taxonomy_error"] = (
                    f"{terminal_errors} segment taxonomy reviews were "
                    "terminally skipped"
                )
                record["crash_taxonomy_version"] = CRASH_TAXONOMY_VERSION
            elif terminal_errors:
                record["crash_taxonomy_status"] = "partial"
                record["crash_taxonomy_error"] = (
                    f"{terminal_errors} segment taxonomy reviews were "
                    "terminally skipped"
                )
                record["crash_taxonomy_version"] = CRASH_TAXONOMY_VERSION
            else:
                record["crash_taxonomy_status"] = "classified"
                record["crash_taxonomy_error"] = None
                record["crash_taxonomy_version"] = CRASH_TAXONOMY_VERSION

            save_state(settings.STATE_JSON, state)
            _cleanup_taxonomy_video_if_safe(record, video_path)
            save_state(settings.STATE_JSON, state)
            processed += len(segments)

    finally:
        if judge is not None:
            unload_model(judge)

    return processed
