"""Download, boundary verification and structured crash segment review."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
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


CRASH_REVIEW_VERSION = "cosmos3_full_clip_crash_v1"


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


def normalise_crash_decision(
    data: Dict[str, Any], raw_response: str, segment_duration: float
) -> CrashDecision:
    confidence = clamp_float(data.get("confidence"))
    is_crash = normalise_bool(data.get("is_crash")) and confidence >= settings.MIN_CRASH_CONFIDENCE
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
    if location_evidence not in {"metadata", "embedded_text", "both", "none"}:
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
  "confidence": 0.0,
  "transition_type": "hard_cut|fade|wipe|title_card|other|none",
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
        prompt = self._segment_prompt(metadata, segment.duration)
        try:
            answer = self._generate(sample_path, prompt, sample_fps)
            data = recover_json(answer)
            if data is None:
                return invalid_crash_decision(answer, "json_recovery_failed")
            return normalise_crash_decision(data, answer, segment.duration)
        except Exception as exc:
            return invalid_crash_decision("", f"model_error: {exc}")

    @staticmethod
    def _segment_prompt(metadata: Dict[str, Any], duration: float) -> str:
        title = clean_text(metadata.get("title"))
        description = clean_text(metadata.get("description"))[:2500]
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

Location fields may use the supplied title or description or readable embedded
text. Do not infer a place from architecture, language, plates, road design, or
general appearance. Set location_evidence to none if there is no explicit
support.

YouTube title: {title}
YouTube description: {description}

Return JSON only:
{{
  "is_crash": true,
  "confidence": 0.0,
  "impact_time_seconds": null,
  "short_description": "one factual sentence",
  "crash_type": "rear_end|side_impact|head_on|rollover|pedestrian|cyclist|multi_vehicle|single_vehicle|near_collision|other|unknown",
  "camera_view": "dashcam|cctv|handheld|action_camera|broadcast|other|unknown",
  "road_user_count": null,
  "road_users": ["car"],
  "road_environment": "urban|rural|motorway|junction|parking|other|unknown",
  "time_of_day": "day|night|dawn_dusk|unknown",
  "weather": "clear|rain|snow|fog|other|unknown",
  "road_condition": "dry|wet|snow_ice|other|unknown",
  "visible_outcomes": ["vehicle stopped"],
  "embedded_location_text": [],
  "locality": null,
  "state": null,
  "country": null,
  "location_evidence": "metadata|embedded_text|both|none"
}}
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
    result = run_command(command, timeout=300)
    if result.returncode != 0:
        raise RuntimeError("failed to create boundary sample")
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
        f"fps={sample_fps:.8g},scale={settings.SAMPLE_WIDTH}:-2:flags=fast_bilinear",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "27",
        "-y",
        str(destination),
    ]
    result = run_command(command, timeout=600)
    if result.returncode != 0:
        raise RuntimeError("failed to create full segment sample")
    return destination, sample_fps


def analyse_video(video_id: str, record: Dict[str, Any], judge: CosmosCrashJudge) -> Dict[str, Any]:
    video_path = download_video(video_id)
    record["downloaded_path"] = str(video_path)
    duration = get_duration(video_path)
    record["duration_seconds"] = duration
    detection = detect_candidate_cuts(video_path, duration)
    record["cut_detection"] = {
        "backend": detection.backend,
        "candidate_count": len(detection.candidates),
        "error": detection.error,
    }
    if detection.error:
        raise RuntimeError(f"Cut detection failed: {detection.error}")

    boundary_reviews: List[BoundaryDecision] = []
    with tempfile.TemporaryDirectory(prefix="crash-review-", dir=settings.DATA_DIR) as temporary:
        work = Path(temporary)
        for index, cut_time in enumerate(detection.candidates):
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
            boundary_reviews.append(decision)

        verified = [item.cut_time for item in boundary_reviews if item.is_edit_boundary]
        full_segments = build_full_segments(duration, verified)
        accepted: List[Dict[str, Any]] = []
        all_reviews: List[Dict[str, Any]] = []
        timestamp_labels = extract_description_timestamps(
            record.get("metadata", {}).get("description")
        )
        for index, segment in enumerate(full_segments):
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
            all_reviews.append(review)
            if decision.is_crash:
                if decision.impact_time_seconds is not None:
                    review["impact_time_in_video"] = round(
                        segment.start_time + decision.impact_time_seconds, 3
                    )
                else:
                    review["impact_time_in_video"] = None
                accepted.append(review)

    record["boundary_reviews"] = [asdict(item) for item in boundary_reviews]
    record["segment_reviews"] = all_reviews
    record["segments"] = accepted
    record["visual_review_version"] = CRASH_REVIEW_VERSION
    record["status"] = "complete" if accepted else "visual_rejected"
    record["error"] = None
    if settings.DELETE_VIDEO_AFTER_PROCESSING:
        video_path.unlink(missing_ok=True)
        record["downloaded_path"] = None
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
