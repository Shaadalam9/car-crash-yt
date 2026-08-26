"""Download, boundary verification and structured crash segment review."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import settings
from .cut_detection import FullSegment, build_full_segments, detect_candidate_cuts
from .model_loading import load_model_with_fallback
from .shared import (
    clamp_float,
    clean_text,
    log,
    normalise_bool,
    normalise_string_list,
    optional_text,
    recover_json,
    run_command,
    save_state,
    unload_model,
)


CRASH_REVIEW_VERSION = "cosmos3_full_clip_crash_v3"
LOCATION_VISUAL_REVIEW_VERSION = "cosmos3_location_text_v2"
COMPATIBLE_REVIEW_VERSIONS = {
    "cosmos3_full_clip_crash_v1",
    "cosmos3_full_clip_crash_v2",
    CRASH_REVIEW_VERSION,
}
CRASH_TYPES = {
    "rear_end",
    "side_impact",
    "head_on",
    "rollover",
    "pedestrian",
    "cyclist",
    "multi_vehicle",
    "single_vehicle",
    "near_collision",
    "other",
    "unknown",
}
LOCATION_EVIDENCE_VALUES = {"metadata", "embedded_text", "both", "none"}
SEGMENT_REVIEW_ATTEMPTS = 2
LOCATION_REVIEW_ATTEMPTS = 2


def _timestamp_seconds(hours: str | None, minutes: str, seconds: str) -> int:
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)


def extract_description_timestamps(description: Any) -> List[Dict[str, Any]]:
    """Extract author supplied chapter or timestamp labels."""
    result: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"(?<!\d)(?:(?P<hours>\d{1,3}):)?(?P<minutes>\d{1,2}):"
        r"(?P<seconds>\d{2})(?!\d)"
    )
    for line in str(description or "").splitlines():
        for match in pattern.finditer(line):
            minutes = int(match.group("minutes"))
            seconds = int(match.group("seconds"))
            if minutes >= 60 or seconds >= 60:
                continue
            value = _timestamp_seconds(
                match.group("hours"), match.group("minutes"), match.group("seconds")
            )
            label = clean_text(line[match.end() :].lstrip(" -–—|:•"))
            result.append(
                {
                    "timestamp_seconds": value,
                    "timestamp_text": match.group(0),
                    "label": label,
                }
            )
    unique: Dict[tuple[int, str], Dict[str, Any]] = {}
    for item in result:
        unique[(item["timestamp_seconds"], item["label"].casefold())] = item
    return sorted(unique.values(), key=lambda item: item["timestamp_seconds"])


def timestamp_labels_for_segment(
    labels: List[Dict[str, Any]], start_time: float, end_time: float
) -> List[Dict[str, Any]]:
    """Return labels inside the clip plus the chapter active at its start."""
    selected = [
        item
        for item in labels
        if start_time <= float(item.get("timestamp_seconds", -1)) < end_time
    ]
    previous = [
        item
        for item in labels
        if float(item.get("timestamp_seconds", -1)) <= start_time
    ]
    if previous and previous[-1] not in selected:
        selected.insert(0, previous[-1])
    return selected


@dataclass(frozen=True)
class BoundaryDecision:
    cut_time: float
    is_edit_boundary: bool
    confidence: float
    transition_type: str
    short_reason: str
    raw_response: str
    error: Optional[str] = None


@dataclass(frozen=True)
class CrashDecision:
    is_crash: bool
    confidence: float
    impact_time_seconds: Optional[float]
    short_description: str
    crash_type: str
    camera_view: str
    road_user_count: Optional[int]
    road_users: List[str]
    road_environment: str
    time_of_day: str
    weather: str
    road_condition: str
    visible_outcomes: List[str]
    embedded_location_text: List[str]
    locality: Optional[str]
    state: Optional[str]
    country: Optional[str]
    location_evidence: str
    raw_response: str
    error: Optional[str] = None


@dataclass(frozen=True)
class LocationVisualDecision:
    location_found: bool
    confidence: float
    locality: Optional[str]
    locality_aka: List[str]
    state: Optional[str]
    country: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    visible_location_text: List[str]
    raw_response: str
    error: Optional[str] = None


def normalise_location_visual_decision(
    data: Dict[str, Any], raw_response: str
) -> LocationVisualDecision:
    location_found = normalise_bool(data.get("location_found"))
    if not location_found:
        return LocationVisualDecision(
            location_found=False,
            confidence=clamp_float(data.get("confidence")),
            locality=None,
            locality_aka=[],
            state=None,
            country=None,
            lat=None,
            lon=None,
            visible_location_text=normalise_string_list(
                data.get("visible_location_text")
            ),
            raw_response=raw_response,
        )
    return LocationVisualDecision(
        location_found=True,
        confidence=clamp_float(data.get("confidence")),
        locality=optional_text(data.get("locality")),
        locality_aka=normalise_string_list(data.get("locality_aka")),
        state=optional_text(data.get("state")),
        country=optional_text(data.get("country")),
        lat=_normalise_coordinate(data.get("lat"), -90.0, 90.0),
        lon=_normalise_coordinate(data.get("lon"), -180.0, 180.0),
        visible_location_text=normalise_string_list(
            data.get("visible_location_text")
        ),
        raw_response=raw_response,
    )


def invalid_location_visual_decision(
    raw_response: str, error: str
) -> LocationVisualDecision:
    return LocationVisualDecision(
        location_found=False,
        confidence=0.0,
        locality=None,
        locality_aka=[],
        state=None,
        country=None,
        lat=None,
        lon=None,
        visible_location_text=[],
        raw_response=raw_response,
        error=error,
    )


def _normalise_coordinate(
    value: Any, minimum: float, maximum: float
) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or not minimum <= result <= maximum:
        return None
    return result


def _evidence_contains(value: Any, evidence: List[str]) -> bool:
    expected = re.sub(r"[^a-z0-9]+", " ", clean_text(value).casefold()).strip()
    if not expected:
        return True
    visible = re.sub(
        r"[^a-z0-9]+", " ", " ".join(evidence).casefold()
    ).strip()
    return expected in visible


def _coordinate_appears_in_evidence(value: float, evidence: List[str]) -> bool:
    visible = " ".join(evidence).replace("−", "-").replace("–", "-")
    variants = {
        f"{value:g}",
        f"{value:.5f}".rstrip("0").rstrip("."),
        f"{value:.6f}".rstrip("0").rstrip("."),
        f"{value:.7f}".rstrip("0").rstrip("."),
    }
    return any(candidate and candidate in visible for candidate in variants)


def validate_location_visual_response(data: Dict[str, Any]) -> Optional[str]:
    if not isinstance(data.get("location_found"), bool):
        return "location_found_must_be_boolean"
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return "confidence_must_be_numeric"
    if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
        return "confidence_must_be_greater_than_zero_and_at_most_one"
    lat = _normalise_coordinate(data.get("lat"), -90.0, 90.0)
    lon = _normalise_coordinate(data.get("lon"), -180.0, 180.0)
    if (lat is None) != (lon is None):
        return "latitude_and_longitude_must_be_a_valid_pair"
    if data["location_found"]:
        evidence = normalise_string_list(data.get("visible_location_text"))
        if not evidence:
            return "found_location_requires_visible_location_text"
        fields = [
            optional_text(data.get("locality")),
            optional_text(data.get("state")),
            optional_text(data.get("country")),
        ]
        if not any(fields) and lat is None:
            return "found_location_requires_place_fields_or_coordinates"
        for value in fields:
            if value and not _evidence_contains(value, evidence):
                return "structured_location_must_appear_in_visible_text"
        if lat is not None and (
            not _coordinate_appears_in_evidence(lat, evidence)
            or not _coordinate_appears_in_evidence(lon, evidence)
        ):
            return "coordinate_pair_must_appear_in_visible_text"
    return None


def normalise_crash_decision(
    data: Dict[str, Any], raw_response: str, segment_duration: float
) -> CrashDecision:
    confidence = clamp_float(data.get("confidence"))
    is_crash = normalise_bool(data.get("is_crash"))
    impact: Optional[float]
    try:
        candidate = float(data.get("impact_time_seconds"))
        impact = candidate if math.isfinite(candidate) and 0 <= candidate <= segment_duration else None
    except (TypeError, ValueError):
        impact = None

    count: Optional[int]
    try:
        candidate_count = int(data.get("road_user_count"))
        count = candidate_count if candidate_count >= 0 else None
    except (TypeError, ValueError):
        count = None

    location_evidence = clean_text(data.get("location_evidence") or "none").lower()
    if location_evidence not in LOCATION_EVIDENCE_VALUES:
        location_evidence = "none"
    if location_evidence == "none":
        locality = state_name = country = None
    else:
        locality = optional_text(data.get("locality"))
        state_name = optional_text(data.get("state"))
        country = optional_text(data.get("country"))

    return CrashDecision(
        is_crash=is_crash,
        confidence=confidence,
        impact_time_seconds=impact,
        short_description=clean_text(data.get("short_description")),
        crash_type=clean_text(data.get("crash_type") or "unknown").lower(),
        camera_view=clean_text(data.get("camera_view") or "unknown").lower(),
        road_user_count=count,
        road_users=normalise_string_list(data.get("road_users")),
        road_environment=clean_text(data.get("road_environment") or "unknown").lower(),
        time_of_day=clean_text(data.get("time_of_day") or "unknown").lower(),
        weather=clean_text(data.get("weather") or "unknown").lower(),
        road_condition=clean_text(data.get("road_condition") or "unknown").lower(),
        visible_outcomes=normalise_string_list(data.get("visible_outcomes")),
        embedded_location_text=normalise_string_list(data.get("embedded_location_text")),
        locality=locality,
        state=state_name,
        country=country,
        location_evidence=location_evidence,
        raw_response=raw_response,
        error=None,
    )


def invalid_crash_decision(raw_response: str, error: str) -> CrashDecision:
    return CrashDecision(
        is_crash=False,
        confidence=0.0,
        impact_time_seconds=None,
        short_description="The visual model did not produce a usable decision.",
        crash_type="unknown",
        camera_view="unknown",
        road_user_count=None,
        road_users=[],
        road_environment="unknown",
        time_of_day="unknown",
        weather="unknown",
        road_condition="unknown",
        visible_outcomes=[],
        embedded_location_text=[],
        locality=None,
        state=None,
        country=None,
        location_evidence="none",
        raw_response=raw_response,
        error=error,
    )


def validate_crash_response(data: Dict[str, Any]) -> Optional[str]:
    """Return a retry reason when a model answer is syntactically valid but unusable."""
    if not isinstance(data.get("is_crash"), bool):
        return "is_crash_must_be_boolean"

    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return "confidence_must_be_numeric"
    if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
        return "confidence_must_be_greater_than_zero_and_at_most_one"

    crash_type = clean_text(data.get("crash_type") or "unknown").lower()
    if crash_type not in CRASH_TYPES:
        return "crash_type_must_be_one_allowed_value"
    if not data["is_crash"] and crash_type != "unknown":
        return "non_crash_response_must_use_unknown_crash_type"

    impact_time = data.get("impact_time_seconds")
    if not data["is_crash"] and impact_time is not None:
        return "non_crash_response_must_not_have_impact_time"

    return None


def saved_segment_retry_reason(review: Dict[str, Any]) -> Optional[str]:
    """Return why a saved segment review must be regenerated, if applicable."""
    if normalise_bool(review.get("retry_exhausted")):
        return None
    error = optional_text(review.get("error"))
    if error:
        return error

    raw_response = str(review.get("raw_response") or "")
    data = recover_json(raw_response)
    if data is None:
        return "missing_or_invalid_raw_response"
    return validate_crash_response(data)


def is_accepted_crash_review(review: Dict[str, Any]) -> bool:
    """Apply the acceptance threshold without changing the model's Boolean answer."""
    return normalise_bool(review.get("is_crash")) and clamp_float(
        review.get("confidence")
    ) >= settings.MIN_CRASH_CONFIDENCE


def has_visual_review_errors(record: Dict[str, Any]) -> bool:
    """Return whether a saved visual review contains retryable errors."""
    boundary_reviews = record.get("boundary_reviews", [])
    if isinstance(boundary_reviews, list) and any(
        isinstance(item, dict) and item.get("error") for item in boundary_reviews
    ):
        return True

    segment_reviews = record.get("segment_reviews", [])
    if isinstance(segment_reviews, list) and any(
        isinstance(item, dict) and saved_segment_retry_reason(item)
        for item in segment_reviews
    ):
        return True
    return False


def _terminalise_visual_review_errors(record: Dict[str, Any]) -> int:
    """Convert persistent review errors into nonblocking terminal warnings."""
    skipped = 0
    for field in ("boundary_reviews", "segment_reviews"):
        reviews = record.get(field, [])
        if not isinstance(reviews, list):
            continue
        for review in reviews:
            if not isinstance(review, dict) or not review.get("error"):
                continue
            review["terminal_error"] = str(review["error"])
            review["error"] = None
            review["retry_exhausted"] = True
            skipped += 1
    return skipped


def _set_visual_review_status(
    record: Dict[str, Any],
    *,
    accepted_count: int,
    boundary_error_count: int,
    segment_error_count: int,
) -> Optional[str]:
    """Update status and stop persistent model errors from blocking the queue."""
    if not boundary_error_count and not segment_error_count:
        record["visual_retry_cycles"] = 0
        record["status"] = "complete" if accepted_count else "visual_rejected"
        record["error"] = None
        return None

    try:
        retry_cycles = max(0, int(record.get("visual_retry_cycles", 0))) + 1
    except (TypeError, ValueError):
        retry_cycles = 1
    record["visual_retry_cycles"] = retry_cycles
    if retry_cycles < settings.MAX_REVIEW_CYCLES:
        record["status"] = "visual_error"
        record["error"] = (
            f"Retry required for {boundary_error_count} boundary reviews and "
            f"{segment_error_count} segment reviews "
            f"(cycle {retry_cycles}/{settings.MAX_REVIEW_CYCLES})"
        )
        return None

    skipped = _terminalise_visual_review_errors(record)
    record["status"] = "complete" if accepted_count else "visual_rejected"
    record["error"] = None
    warning = (
        f"Skipped {skipped} persistently invalid reviews after "
        f"{retry_cycles} cycles"
    )
    record["visual_review_warning"] = warning
    return warning


def _boundary_decision_from_record(value: Dict[str, Any]) -> BoundaryDecision:
    return BoundaryDecision(
        cut_time=float(value.get("cut_time", 0.0)),
        is_edit_boundary=normalise_bool(value.get("is_edit_boundary")),
        confidence=clamp_float(value.get("confidence")),
        transition_type=clean_text(value.get("transition_type") or "none"),
        short_reason=clean_text(value.get("short_reason")),
        raw_response=str(value.get("raw_response") or ""),
        error=optional_text(value.get("error")),
    )


def _review_key(start_time: Any, end_time: Any) -> tuple[float, float]:
    return round(float(start_time), 3), round(float(end_time), 3)


class CosmosCrashJudge:
    def __init__(self) -> None:
        from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

        log(f"Loading visual model: {settings.VISUAL_MODEL}")
        self.processor = AutoProcessor.from_pretrained(settings.VISUAL_MODEL)

        def loader(checkpoint: str, **kwargs: Any) -> Any:
            dtype = kwargs.pop("torch_dtype", None)
            if dtype is not None:
                kwargs["dtype"] = dtype
            return Cosmos3OmniForConditionalGeneration.from_pretrained(
                checkpoint, **kwargs
            )

        self.model = load_model_with_fallback(
            loader,
            settings.VISUAL_MODEL,
            device=settings.VISUAL_DEVICE,
            load_in_4bit=settings.VISUAL_MODEL_4BIT,
            label="visual model",
        ).eval()

    def _generate(self, video_path: Path, prompt: str, fps: float) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(video_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"fps": fps},
        ).to(self.model.device, torch.bfloat16)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=settings.VISUAL_MAX_NEW_TOKENS,
            )
        input_length = inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    def verify_boundary(self, sample_path: Path, cut_time: float, boundary_in_clip: float) -> BoundaryDecision:
        prompt = f"""
The video is sampled immediately before and after a proposed compilation edit.
The proposed boundary is {boundary_in_clip:.3f} seconds after this sample starts.

Decide whether the boundary is a real edit separating two source clips. A real
edit includes a hard cut, fade, wipe, title card transition, or a discontinuous
change of camera, place, time, or recording. Rapid ego motion, impact shake,
exposure change, occlusion, or a pan within one continuous recording is not an
edit. Be conservative: accept only direct temporal evidence of an edit.

Return JSON only:
{{
  "is_edit_boundary": true,
  "confidence": 0.95,
  "transition_type": "hard_cut",
  "short_reason": "one factual sentence"
}}
""".strip()
        try:
            answer = self._generate(sample_path, prompt, min(settings.SAMPLE_MAX_FPS, 4.0))
            data = recover_json(answer)
            if data is None:
                raise ValueError("json_recovery_failed")
            confidence = clamp_float(data.get("confidence"))
            return BoundaryDecision(
                cut_time=cut_time,
                is_edit_boundary=normalise_bool(data.get("is_edit_boundary"))
                and confidence >= settings.MIN_BOUNDARY_CONFIDENCE,
                confidence=confidence,
                transition_type=clean_text(data.get("transition_type") or "none").lower(),
                short_reason=clean_text(data.get("short_reason")),
                raw_response=answer,
            )
        except Exception as exc:
            return BoundaryDecision(
                cut_time=cut_time,
                is_edit_boundary=False,
                confidence=0.0,
                transition_type="none",
                short_reason=str(exc),
                raw_response="",
                error="model_error",
            )

    def review_segment(
        self,
        sample_path: Path,
        metadata: Dict[str, Any],
        segment: FullSegment,
        sample_fps: float,
    ) -> CrashDecision:
        base_prompt = self._segment_prompt(metadata, segment.duration)
        prompt = base_prompt
        answer = ""
        last_error = "model_did_not_return_a_usable_decision"
        for attempt in range(SEGMENT_REVIEW_ATTEMPTS):
            try:
                answer = self._generate(sample_path, prompt, sample_fps)
                data = recover_json(answer)
                if data is None:
                    last_error = "json_recovery_failed"
                else:
                    validation_error = validate_crash_response(data)
                    if validation_error is None:
                        return normalise_crash_decision(
                            data, answer, segment.duration
                        )
                    last_error = f"semantic_error: {validation_error}"
            except Exception as exc:
                last_error = f"model_error: {exc}"

            if attempt + 1 < SEGMENT_REVIEW_ATTEMPTS:
                prompt = (
                    base_prompt
                    + "\n\nYour previous answer was invalid because: "
                    + last_error
                    + ". Reconsider the video and return one corrected JSON object."
                )
        return invalid_crash_decision(answer, last_error)

    def review_location(
        self,
        sample_path: Path,
        sample_fps: float,
    ) -> LocationVisualDecision:
        base_prompt = self._location_prompt()
        prompt = base_prompt
        answer = ""
        last_error = "model_did_not_return_a_usable_location_decision"
        for attempt in range(LOCATION_REVIEW_ATTEMPTS):
            try:
                answer = self._generate(sample_path, prompt, sample_fps)
                data = recover_json(answer)
                if data is None:
                    last_error = "json_recovery_failed"
                else:
                    validation_error = validate_location_visual_response(data)
                    if validation_error is None:
                        return normalise_location_visual_decision(data, answer)
                    last_error = f"semantic_error: {validation_error}"
            except Exception as exc:
                last_error = f"model_error: {exc}"

            if attempt + 1 < LOCATION_REVIEW_ATTEMPTS:
                prompt = (
                    base_prompt
                    + "\n\nYour previous answer was invalid because: "
                    + last_error
                    + ". Reinspect the video and return one corrected JSON object."
                )
        return invalid_location_visual_decision(answer, last_error)

    @staticmethod
    def _location_prompt() -> str:
        return """
Inspect the complete source clip only for explicit geographic information.
Carefully read text overlays in every corner, title cards, captions, creator
annotations, street signs, motorway signs, business signs, and other readable
text inside the video. Location overlays can be small and may appear for only
part of the clip.

Return a location only when it is directly supported by readable text. Do not
infer a place from scenery, language, flags, number plates, road design,
architecture, weather, or general appearance. Transcribe the supporting text
exactly in visible_location_text. Separate a city or locality from its state or
province when possible. Leave country null when the visible text does not name
it. locality_aka contains only alternative locality names that are explicitly
visible.

If a latitude and longitude pair is visibly written in the clip, transcribe it
as decimal numbers in lat and lon and also preserve the exact coordinate text
in visible_location_text. Do not estimate coordinates from the scene. Both
coordinates must be clearly readable; otherwise set both to null.

The confidence value is certainty that the location reading is correct. If no
explicit geographic text is clearly readable, or if any letters or digits are
ambiguous, use location_found=false with high confidence and null location
fields. Prefer an unknown result over guessing.

Return JSON only:
{
  "location_found": false,
  "confidence": 0.99,
  "locality": null,
  "locality_aka": [],
  "state": null,
  "country": null,
  "lat": null,
  "lon": null,
  "visible_location_text": []
}
""".strip()

    @staticmethod
    def _segment_prompt(metadata: Dict[str, Any], duration: float) -> str:
        title = clean_text(metadata.get("title"))
        return f"""
This sample represents the full temporal span of one edit-bounded source clip
from a YouTube video. The source clip lasts {duration:.3f} seconds. Frames may
be temporally sparse when the clip is long.

Decide whether the clip visibly contains a real road traffic collision or a
near collision where evasive action narrowly prevents impact. Reject video
games, simulations, staged scenes, crash tests, motorsport incidents, title
cards, commentary-only clips, emergency response without the event, and clips
that only show damage after an unseen collision.

Describe only visible evidence. Never infer fault, injuries, fatalities,
intent, identity, or legal responsibility. Use "unknown" when an attribute is
not visible. impact_time_seconds is relative to this source clip, not the full
YouTube upload. Use null when an impact is absent or cannot be timed.

Location fields may use only readable text visibly embedded in this source
clip. Metadata location evidence is processed separately. Do not infer a place
from architecture, language, plates, road design, or general appearance. Set
location_evidence to none if there is no explicit visible support.

The confidence value is your certainty that the is_crash Boolean decision is
correct. It is not the probability that a crash occurred. A clearly visible
non-crash clip should therefore use is_crash=false with high confidence. Never
use zero confidence; zero means the answer is unusable and will be retried.

The fields must be logically consistent:
* Use is_crash=true and crash_type=near_collision when evasive action narrowly
  prevents an impact.
* Use crash_type=unknown and impact_time_seconds=null when is_crash=false.
* Select exactly one allowed value for every categorical field. Never copy a
  list of choices or the | character into a value.
* embedded_location_text contains only short geographic text visibly present
  inside the video frames. Never copy titles, descriptions, URLs, email
  addresses, channel names, promotional text, or attribution lists into it.
* Keep the complete JSON response under 500 tokens. road_users must contain
  unique road user types rather than one repeated entry per vehicle. Include at
  most five short embedded_location_text items.

Upload title for crash context only: {title}

Return JSON only:
{{
  "is_crash": true,
  "confidence": 0.95,
  "impact_time_seconds": null,
  "short_description": "one factual sentence",
  "crash_type": "near_collision",
  "camera_view": "dashcam",
  "road_user_count": null,
  "road_users": ["car"],
  "road_environment": "motorway",
  "time_of_day": "day",
  "weather": "clear",
  "road_condition": "dry",
  "visible_outcomes": ["evasive manoeuvre"],
  "embedded_location_text": [],
  "locality": null,
  "state": null,
  "country": null,
  "location_evidence": "none"
}}

Allowed crash_type values: rear_end, side_impact, head_on, rollover,
pedestrian, cyclist, multi_vehicle, single_vehicle, near_collision, other,
unknown.
Allowed camera_view values: dashcam, cctv, handheld, action_camera, broadcast,
other, unknown.
Allowed road_environment values: urban, rural, motorway, junction, parking,
other, unknown.
Allowed time_of_day values: day, night, dawn_dusk, unknown.
Allowed weather values: clear, rain, snow, fog, other, unknown.
Allowed road_condition values: dry, wet, snow_ice, other, unknown.
Allowed location_evidence values: metadata, embedded_text, both, none.
""".strip()


def reset_temp_directory() -> None:
    if settings.TEMP_DIR.exists():
        shutil.rmtree(settings.TEMP_DIR)
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _cookie_arguments() -> List[str]:
    return ["--cookies", settings.COOKIE_FILE] if settings.COOKIE_FILE else []


def _valid_video(path: Path) -> bool:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout=60,
    )
    return result.returncode == 0 and "video" in result.stdout


def download_video(video_id: str) -> Path:
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    settings.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    output = settings.TEMP_DIR / f"{video_id}.%(ext)s"
    command = [
        "yt-dlp",
        "--no-playlist",
        "--newline",
        "--retries",
        str(settings.DOWNLOAD_RETRIES),
        "--fragment-retries",
        str(settings.DOWNLOAD_RETRIES),
        "--merge-output-format",
        "mp4",
        "-f",
        settings.VIDEO_FORMAT,
        "-o",
        str(output),
        *_cookie_arguments(),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = run_command(command, timeout=settings.DOWNLOAD_TIMEOUT_SECONDS)
    if result.returncode != 0:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        raise RuntimeError(lines[-1] if lines else "yt-dlp failed")
    candidates = [path for path in settings.TEMP_DIR.glob(f"{video_id}.*") if path.is_file()]
    candidates = [path for path in candidates if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not candidates:
        raise RuntimeError("yt-dlp produced no media file")
    source = max(candidates, key=lambda path: path.stat().st_size)
    if not _valid_video(source):
        raise RuntimeError("downloaded file has no readable video stream")
    destination = settings.VIDEO_DIR / f"{video_id}{source.suffix.lower()}"
    os.replace(source, destination)
    return destination


def get_duration(video_path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe could not determine video duration")
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("video duration is invalid")
    return duration


def _run_sample_command(
    command: List[str], destination: Path, *, timeout: float, label: str
) -> None:
    """Run transient FFmpeg sample creation with bounded retries."""
    last_error = "unknown FFmpeg failure"
    for attempt in range(5):
        destination.unlink(missing_ok=True)
        try:
            result = run_command(command, timeout=timeout)
            if (
                result.returncode == 0
                and destination.is_file()
                and destination.stat().st_size > 0
            ):
                return
            details = [
                line.strip()
                for line in (result.stderr or result.stdout).splitlines()
                if line.strip()
            ]
            last_error = details[-1] if details else f"FFmpeg exit code {result.returncode}"
        except Exception as exc:
            last_error = str(exc)
        if attempt < 4:
            time.sleep(min(0.25 * (2**attempt), 2.0))
    raise RuntimeError(f"{label} after 5 attempts: {last_error}")


def _create_boundary_sample(video_path: Path, cut_time: float, duration: float, destination: Path) -> tuple[Path, float]:
    start = max(0.0, cut_time - settings.BOUNDARY_CONTEXT_SECONDS)
    end = min(duration, cut_time + settings.BOUNDARY_CONTEXT_SECONDS)
    boundary_in_clip = cut_time - start
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{max(0.001, end - start):.6f}",
        "-vf",
        f"scale={settings.SAMPLE_WIDTH}:-2:flags=fast_bilinear",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "25",
        "-y",
        str(destination),
    ]
    _run_sample_command(
        command,
        destination,
        timeout=300,
        label="failed to create boundary sample",
    )
    return destination, boundary_in_clip


def _create_segment_sample(video_path: Path, segment: FullSegment, destination: Path) -> tuple[Path, float]:
    sample_fps = min(
        settings.SAMPLE_MAX_FPS,
        max(0.01, settings.SAMPLE_FRAME_COUNT / max(segment.duration, 0.001)),
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{segment.start_time:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{segment.duration:.6f}",
        "-vf",
        (
            f"fps={sample_fps:.8g},"
            f"scale={settings.SAMPLE_WIDTH}:-2:flags=fast_bilinear,"
            "setpts=PTS-STARTPTS"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-bf",
        "0",
        "-preset",
        "veryfast",
        "-crf",
        "27",
        "-y",
        str(destination),
    ]
    _run_sample_command(
        command,
        destination,
        timeout=600,
        label="failed to create full segment sample",
    )
    return destination, sample_fps


def _create_location_sample(
    video_path: Path, segment: FullSegment, destination: Path
) -> tuple[Path, float]:
    """Create a higher resolution full segment sample for reading small text."""
    sample_fps = min(
        settings.LOCATION_SAMPLE_MAX_FPS,
        max(
            0.01,
            settings.LOCATION_SAMPLE_FRAME_COUNT / max(segment.duration, 0.001),
        ),
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{segment.start_time:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{segment.duration:.6f}",
        "-vf",
        (
            f"fps={sample_fps:.8g},"
            f"scale={settings.LOCATION_SAMPLE_WIDTH}:-2:flags=lanczos,"
            "setpts=PTS-STARTPTS"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-bf",
        "0",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-y",
        str(destination),
    ]
    _run_sample_command(
        command,
        destination,
        timeout=600,
        label="failed to create location sample",
    )
    return destination, sample_fps


def _location_is_resolved(segment: Dict[str, Any]) -> bool:
    location = segment.get("location")
    return (
        isinstance(location, dict)
        and location.get("geocode_status") == "resolved"
        and optional_text(location.get("locality")) is not None
    )


def _has_structured_location(segment: Dict[str, Any]) -> bool:
    return optional_text(segment.get("locality")) is not None


def _location_review_is_current(segment: Dict[str, Any]) -> bool:
    review = segment.get("location_visual_review")
    return (
        segment.get("location_visual_review_version")
        == LOCATION_VISUAL_REVIEW_VERSION
        and isinstance(review, dict)
        and not review.get("error")
    )


def _needs_location_visual_review(segment: Dict[str, Any]) -> bool:
    saved_version = optional_text(segment.get("location_visual_review_version"))
    if saved_version and saved_version != LOCATION_VISUAL_REVIEW_VERSION:
        return True
    review = segment.get("location_visual_review")
    if (
        saved_version == LOCATION_VISUAL_REVIEW_VERSION
        and isinstance(review, dict)
        and review.get("error")
    ):
        return True
    if _location_is_resolved(segment) or _has_structured_location(segment):
        return False
    return not _location_review_is_current(segment)


def has_location_visual_review_errors(record: Dict[str, Any]) -> bool:
    return any(
        isinstance(segment, dict)
        and isinstance(segment.get("location_visual_review"), dict)
        and bool(segment["location_visual_review"].get("error"))
        for segment in record.get("segments", [])
    )


def _delete_downloaded_video_when_finished(
    record: Dict[str, Any], video_path: Path
) -> bool:
    if (
        not settings.DELETE_VIDEO_AFTER_PROCESSING
        or record.get("status") == "visual_error"
        or has_location_visual_review_errors(record)
    ):
        return False
    video_path.unlink(missing_ok=True)
    record["downloaded_path"] = None
    log(f"Deleted reviewed video: {video_path}")
    return True


def _restore_pre_location_visual_evidence(segment: Dict[str, Any]) -> None:
    """Remove old visual location data and restore the crash review evidence."""
    raw = recover_json(str(segment.get("raw_response") or "")) or {}
    evidence = clean_text(raw.get("location_evidence") or "none").lower()
    if evidence not in LOCATION_EVIDENCE_VALUES:
        evidence = "none"
    segment["embedded_location_text"] = normalise_string_list(
        raw.get("embedded_location_text")
    )
    segment["location_evidence"] = evidence
    if evidence == "none":
        segment["locality"] = None
        segment["state"] = None
        segment["country"] = None
    else:
        segment["locality"] = optional_text(raw.get("locality"))
        segment["state"] = optional_text(raw.get("state"))
        segment["country"] = optional_text(raw.get("country"))
    segment["locality_aka"] = normalise_string_list(raw.get("locality_aka"))
    segment["lat"] = _normalise_coordinate(raw.get("lat"), -90.0, 90.0)
    segment["lon"] = _normalise_coordinate(raw.get("lon"), -180.0, 180.0)
    segment.pop("location", None)


def _apply_location_visual_decision(
    segment: Dict[str, Any], decision: LocationVisualDecision
) -> None:
    segment["location_visual_review"] = asdict(decision)
    segment["location_visual_review_version"] = LOCATION_VISUAL_REVIEW_VERSION
    if (
        decision.error
        or not decision.location_found
        or decision.confidence < settings.MIN_LOCATION_CONFIDENCE
    ):
        return

    segment["embedded_location_text"] = normalise_string_list(
        [
            *normalise_string_list(segment.get("embedded_location_text")),
            *decision.visible_location_text,
        ]
    )
    segment["locality"] = decision.locality
    segment["locality_aka"] = decision.locality_aka
    segment["state"] = decision.state
    segment["country"] = decision.country
    segment["lat"] = decision.lat
    segment["lon"] = decision.lon
    previous_evidence = clean_text(segment.get("location_evidence")).lower()
    segment["location_evidence"] = (
        "both" if previous_evidence == "metadata" else "embedded_text"
    )
    segment.pop("location", None)


def _store_location_visual_decision(
    segment: Dict[str, Any],
    decision: LocationVisualDecision,
    previous_cycles: int,
) -> bool:
    """Store a location decision and terminalise it after the retry budget."""
    _apply_location_visual_decision(segment, decision)
    current_review = segment["location_visual_review"]
    current_review["review_cycles"] = previous_cycles + 1
    if (
        current_review.get("error")
        and current_review["review_cycles"] >= settings.MAX_REVIEW_CYCLES
    ):
        current_review["terminal_error"] = str(current_review["error"])
        current_review["error"] = None
        current_review["retry_exhausted"] = True
        return True
    return False


def review_missing_segment_locations(
    video_id: str,
    record: Dict[str, Any],
    judge: CosmosCrashJudge,
    video_path: Path,
) -> int:
    """Read location overlays from accepted segments that remain unresolved."""
    pending = [
        segment
        for segment in record.get("segments", [])
        if isinstance(segment, dict) and _needs_location_visual_review(segment)
    ]
    if not pending:
        return 0

    log(f"Reviewing visible location text in {len(pending)} segments for {video_id}")
    reviewed = 0
    with tempfile.TemporaryDirectory(
        prefix="location-review-", dir=settings.DATA_DIR
    ) as temporary:
        work = Path(temporary)
        for position, segment in enumerate(pending, start=1):
            previous_review = segment.get("location_visual_review")
            previous_cycles = 0
            if isinstance(previous_review, dict):
                try:
                    previous_cycles = max(
                        0, int(previous_review.get("review_cycles", 0))
                    )
                except (TypeError, ValueError):
                    previous_cycles = 0
            saved_version = optional_text(
                segment.get("location_visual_review_version")
            )
            if saved_version and saved_version != LOCATION_VISUAL_REVIEW_VERSION:
                _restore_pre_location_visual_evidence(segment)
            try:
                full_segment = FullSegment(
                    start_time=float(segment.get("start_time")),
                    end_time=float(segment.get("end_time")),
                )
                sample, sample_fps = _create_location_sample(
                    video_path,
                    full_segment,
                    work
                    / f"location-{int(segment.get('segment_index', position)):05d}.mp4",
                )
                decision = judge.review_location(sample, sample_fps)
            except Exception as exc:
                decision = invalid_location_visual_decision(
                    "", f"sample_error: {exc}"
                )
            if _store_location_visual_decision(
                segment, decision, previous_cycles
            ):
                log(
                    "Skipping location review after persistent failures for "
                    f"{video_id} segment {segment.get('segment_index')}"
                )
            reviewed += 1
            if position == 1 or position % 10 == 0 or position == len(pending):
                log(
                    f"Location review progress for {video_id}: "
                    f"{position}/{len(pending)}"
                )
    return reviewed


def analyse_video(video_id: str, record: Dict[str, Any], judge: CosmosCrashJudge) -> Dict[str, Any]:
    stored_path = optional_text(record.get("downloaded_path"))
    existing_path = Path(stored_path) if stored_path else None
    if existing_path is not None and existing_path.is_file() and _valid_video(existing_path):
        video_path = existing_path
        log(f"Reusing downloaded video for {video_id}")
    else:
        video_path = download_video(video_id)
    record["downloaded_path"] = str(video_path)
    try:
        duration = float(record.get("duration_seconds"))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError
    except (TypeError, ValueError):
        duration = get_duration(video_path)
    record["duration_seconds"] = duration

    boundary_reviews: List[BoundaryDecision] = []
    saved_boundaries = record.get("boundary_reviews")
    can_resume = (
        record.get("visual_review_version") in COMPATIBLE_REVIEW_VERSIONS
        and isinstance(saved_boundaries, list)
        and bool(saved_boundaries)
    )
    if can_resume:
        boundary_reviews = [
            _boundary_decision_from_record(item)
            for item in saved_boundaries
            if isinstance(item, dict)
        ]
        log(
            f"Resuming {sum(bool(item.error) for item in boundary_reviews)} failed "
            f"boundary reviews for {video_id}"
        )
    else:
        detection = detect_candidate_cuts(video_path, duration)
        record["cut_detection"] = {
            "backend": detection.backend,
            "candidate_count": len(detection.candidates),
            "error": detection.error,
        }
        if detection.error:
            raise RuntimeError(f"Cut detection failed: {detection.error}")
        boundary_reviews = [
            BoundaryDecision(
                cut_time=cut_time,
                is_edit_boundary=False,
                confidence=0.0,
                transition_type="none",
                short_reason="Pending visual verification",
                raw_response="",
                error="pending",
            )
            for cut_time in detection.candidates
        ]

    with tempfile.TemporaryDirectory(prefix="crash-review-", dir=settings.DATA_DIR) as temporary:
        work = Path(temporary)
        pending_boundary_count = sum(bool(item.error) for item in boundary_reviews)
        completed_boundaries = 0
        for index, existing_boundary in enumerate(boundary_reviews):
            if not existing_boundary.error:
                continue
            cut_time = existing_boundary.cut_time
            try:
                sample, local_time = _create_boundary_sample(
                    video_path, cut_time, duration, work / f"boundary-{index:05d}.mp4"
                )
                decision = judge.verify_boundary(sample, cut_time, local_time)
            except Exception as exc:
                decision = BoundaryDecision(
                    cut_time=cut_time,
                    is_edit_boundary=False,
                    confidence=0.0,
                    transition_type="none",
                    short_reason=str(exc),
                    raw_response="",
                    error="sample_error",
                )
            boundary_reviews[index] = decision
            completed_boundaries += 1
            if (
                completed_boundaries == 1
                or completed_boundaries % 10 == 0
                or completed_boundaries == pending_boundary_count
            ):
                log(
                    f"Boundary review progress for {video_id}: "
                    f"{completed_boundaries}/{pending_boundary_count}"
                )

        record["boundary_reviews"] = [asdict(item) for item in boundary_reviews]
        record["visual_review_version"] = CRASH_REVIEW_VERSION

        verified = [item.cut_time for item in boundary_reviews if item.is_edit_boundary]
        full_segments = build_full_segments(duration, verified)
        accepted: List[Dict[str, Any]] = []
        all_reviews: List[Dict[str, Any]] = []
        reusable_reviews: Dict[tuple[float, float], Dict[str, Any]] = {}
        if can_resume:
            saved_segments = record.get("segment_reviews", [])
            if not isinstance(saved_segments, list):
                saved_segments = []
            for item in saved_segments:
                if (
                    not isinstance(item, dict)
                    or saved_segment_retry_reason(item) is not None
                ):
                    continue
                try:
                    reusable_reviews[
                        _review_key(item.get("start_time"), item.get("end_time"))
                    ] = item
                except (TypeError, ValueError):
                    continue
            log(
                f"Reusing {len(reusable_reviews)} successful segment reviews for "
                f"{video_id}"
            )
        timestamp_labels = extract_description_timestamps(
            record.get("metadata", {}).get("description")
        )
        pending_segment_count = sum(
            _review_key(segment.start_time, segment.end_time) not in reusable_reviews
            for segment in full_segments
        )
        if pending_segment_count:
            log(f"Reviewing {pending_segment_count} remaining segments for {video_id}")
        completed_segments = 0
        for index, segment in enumerate(full_segments):
            key = _review_key(segment.start_time, segment.end_time)
            saved_review = reusable_reviews.get(key)
            if saved_review is not None:
                review = dict(saved_review)
                review.update(
                    {
                        "segment_index": index,
                        "start_time": round(segment.start_time, 3),
                        "end_time": round(segment.end_time, 3),
                        "duration_seconds": round(segment.duration, 3),
                        "timestamp_labels": timestamp_labels_for_segment(
                            timestamp_labels, segment.start_time, segment.end_time
                        ),
                    }
                )
            else:
                try:
                    sample, sample_fps = _create_segment_sample(
                        video_path, segment, work / f"segment-{index:05d}.mp4"
                    )
                    decision = judge.review_segment(
                        sample, record.get("metadata", {}), segment, sample_fps
                    )
                except Exception as exc:
                    decision = invalid_crash_decision("", f"sample_error: {exc}")
                review = {
                    "segment_index": index,
                    "start_time": round(segment.start_time, 3),
                    "end_time": round(segment.end_time, 3),
                    "duration_seconds": round(segment.duration, 3),
                    "timestamp_labels": timestamp_labels_for_segment(
                        timestamp_labels, segment.start_time, segment.end_time
                    ),
                    **asdict(decision),
                }
                completed_segments += 1
                if (
                    completed_segments == 1
                    or completed_segments % 10 == 0
                    or completed_segments == pending_segment_count
                ):
                    log(
                        f"Segment review progress for {video_id}: "
                        f"{completed_segments}/{pending_segment_count}"
                    )
            all_reviews.append(review)
            if is_accepted_crash_review(review):
                if review.get("impact_time_seconds") is not None:
                    review["impact_time_in_video"] = round(
                        segment.start_time + float(review["impact_time_seconds"]), 3
                    )
                else:
                    review["impact_time_in_video"] = None
                accepted.append(review)
            record["segment_reviews"] = all_reviews
            record["segments"] = accepted

    record["boundary_reviews"] = [asdict(item) for item in boundary_reviews]
    record["segment_reviews"] = all_reviews
    record["segments"] = accepted
    record["visual_review_version"] = CRASH_REVIEW_VERSION
    boundary_error_count = sum(bool(item.error) for item in boundary_reviews)
    segment_error_count = sum(bool(item.get("error")) for item in all_reviews)
    warning = _set_visual_review_status(
        record,
        accepted_count=len(accepted),
        boundary_error_count=boundary_error_count,
        segment_error_count=segment_error_count,
    )
    if warning:
        log(f"{video_id}: {warning}")
    if record["status"] == "complete":
        review_missing_segment_locations(video_id, record, judge, video_path)
    _delete_downloaded_video_when_finished(record, video_path)
    return record


def run_visual_stage(state: Dict[str, Any], after_video: Optional[Any] = None) -> int:
    pending = [
        (video_id, record)
        for video_id, record in state.get("videos", {}).items()
        if isinstance(record, dict)
        and isinstance(record.get("text_decision"), dict)
        and record["text_decision"].get("include")
        and (
            record.get("status") not in {"complete", "visual_rejected"}
            or record.get("visual_review_version") != CRASH_REVIEW_VERSION
            or has_visual_review_errors(record)
        )
    ][: settings.MAX_VIDEOS_PER_RUN]
    if not pending:
        return 0

    judge = CosmosCrashJudge()
    processed = 0
    try:
        for video_id, record in pending:
            log(f"Analysing full crash clips in {video_id}")
            try:
                analyse_video(video_id, record, judge)
            except KeyboardInterrupt:
                save_state(settings.STATE_JSON, state)
                raise
            except Exception as exc:
                record["status"] = "visual_error"
                record["error"] = str(exc)
                log(f"Visual processing failed for {video_id}: {exc}")
            save_state(settings.STATE_JSON, state)
            if after_video is not None:
                after_video(state)
            processed += 1
    finally:
        unload_model(judge)
    return processed


def run_location_visual_stage(state: Dict[str, Any]) -> int:
    """Reopen retained videos to recover missed location overlays."""
    pending_records: List[tuple[str, Dict[str, Any], Path]] = []
    finished_records: List[tuple[Dict[str, Any], Path]] = []
    for video_id, record in state.get("videos", {}).items():
        if (
            not isinstance(record, dict)
            or record.get("status") not in {"complete", "visual_rejected"}
        ):
            continue
        stored_path = optional_text(record.get("downloaded_path"))
        if not stored_path:
            continue
        video_path = Path(stored_path)
        if not video_path.is_file():
            continue
        needs_location_review = record.get("status") == "complete" and any(
            isinstance(segment, dict) and _needs_location_visual_review(segment)
            for segment in record.get("segments", [])
        )
        if needs_location_review:
            pending_records.append((video_id, record, video_path))
        else:
            finished_records.append((record, video_path))

    deleted_finished_video = False
    for record, video_path in finished_records:
        deleted_finished_video = (
            _delete_downloaded_video_when_finished(record, video_path)
            or deleted_finished_video
        )
    if deleted_finished_video:
        save_state(settings.STATE_JSON, state)

    if not pending_records:
        return 0

    judge = CosmosCrashJudge()
    reviewed = 0
    try:
        for video_id, record, video_path in pending_records:
            try:
                reviewed += review_missing_segment_locations(
                    video_id, record, judge, video_path
                )
            except KeyboardInterrupt:
                save_state(settings.STATE_JSON, state)
                raise
            except Exception as exc:
                log(f"Location visual processing failed for {video_id}: {exc}")
            _delete_downloaded_video_when_finished(record, video_path)
            save_state(settings.STATE_JSON, state)
    finally:
        unload_model(judge)
    return reviewed
