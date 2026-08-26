"""Continuous resumable crash video orchestration."""

from __future__ import annotations

import time
from typing import Any, Dict

from . import settings
from .crash_review import (
    CRASH_REVIEW_VERSION,
    has_location_visual_review_errors,
    has_visual_review_errors,
    reset_temp_directory,
    run_location_visual_stage,
    run_visual_stage,
)
from .location import run_location_stage
from .metadata_filter import run_text_stage
from .output_writer import write_output_csv
from .shared import empty_state, load_json, log, require_binary, save_state
from .youtube_discovery import YouTubeDiscovery, load_api_keys


FINAL_STATUSES = {"complete", "text_rejected", "visual_rejected"}


def validate_environment() -> None:
    for command in ("ffmpeg", "ffprobe", "yt-dlp"):
        require_binary(command)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> Dict[str, Any]:
    state = load_json(settings.STATE_JSON, empty_state())
    if not isinstance(state, dict):
        raise RuntimeError("State JSON must contain an object")
    state.setdefault("videos", {})
    state.setdefault("discovery", {})
    return state


def requires_processing(record: Any) -> bool:
    if not isinstance(record, dict):
        return True
    decision = record.get("text_decision")
    if not isinstance(decision, dict):
        return True
    if not decision.get("include"):
        return False
    if has_visual_review_errors(record):
        return True
    if has_location_visual_review_errors(record):
        return True
    if record.get("status") not in FINAL_STATUSES:
        return True
    return record.get("visual_review_version") != CRASH_REVIEW_VERSION


def unfinished_count(state: Dict[str, Any]) -> int:
    return sum(1 for record in state.get("videos", {}).values() if requires_processing(record))


def discover_when_ready(state: Dict[str, Any]) -> int:
    unfinished = unfinished_count(state)
    if unfinished:
        log(f"Continuing {unfinished} unfinished videos before new discovery")
        return 0
    try:
        discovered = YouTubeDiscovery(load_api_keys()).discover(state)
    except RuntimeError as exc:
        discovery = state.setdefault("discovery", {})
        discovery["last_error"] = str(exc)
        save_state(settings.STATE_JSON, state)
        log(f"YouTube discovery deferred: {exc}")
        return 0
    discovery = state.setdefault("discovery", {})
    discovery["last_error"] = None
    return discovered


def update_outputs(state: Dict[str, Any]) -> None:
    run_location_stage(state)
    write_output_csv(state)
    save_state(settings.STATE_JSON, state)


def run_cycle(state: Dict[str, Any], cycle: int) -> tuple[int, int, int, int]:
    log(f"Starting cycle {cycle}")
    run_location_stage(state)
    location_visual_processed = run_location_visual_stage(state)
    if location_visual_processed:
        update_outputs(state)
    discovered = discover_when_ready(state)
    text_processed = run_text_stage(state)
    visual_processed = run_visual_stage(state, after_video=update_outputs)
    update_outputs(state)
    unfinished = unfinished_count(state)
    accepted_segments = sum(
        len(record.get("segments", []))
        for record in state.get("videos", {}).values()
        if isinstance(record, dict)
    )
    log(
        f"Cycle {cycle}: discovered={discovered}, text={text_processed}, "
        f"visual={visual_processed}, location_visual={location_visual_processed}, "
        f"accepted_segments={accepted_segments}, "
        f"unfinished={unfinished}"
    )
    return discovered, text_processed, visual_processed, unfinished


def cycle_pause_seconds(
    discovered: int,
    text_processed: int,
    visual_processed: int,
    unfinished: int,
) -> int:
    """Use the short pause whenever the current batch still has work."""
    active = bool(
        discovered or text_processed or visual_processed or unfinished
    )
    return (
        settings.ACTIVE_PAUSE_SECONDS
        if active
        else settings.IDLE_PAUSE_SECONDS
    )


def main() -> None:
    validate_environment()
    reset_temp_directory()
    state = load_state()
    cycle = 0
    try:
        while True:
            cycle += 1
            discovered, text_count, visual_count, unfinished = run_cycle(state, cycle)
            if not settings.CONTINUOUS_MODE:
                return
            pause = cycle_pause_seconds(
                discovered, text_count, visual_count, unfinished
            )
            if unfinished:
                log(
                    f"Pausing {pause} seconds before continuing the current "
                    f"batch ({unfinished} unfinished)"
                )
            else:
                log(f"Pausing {pause} seconds before the next discovery batch")
            time.sleep(pause)
    except KeyboardInterrupt:
        update_outputs(state)
        log("Stopped by the user; current state and CSV were saved")