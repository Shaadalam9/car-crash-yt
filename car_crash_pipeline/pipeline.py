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
from .crash_taxonomy import (
    crash_taxonomy_pending,
    run_crash_taxonomy_stage,
)
from .location import run_location_stage
from .metadata_filter import run_text_stage
from .output_writer import write_output_csv
from .shared import empty_state, load_json, log, require_binary, save_state
from .youtube_discovery import YouTubeDiscovery, load_api_keys


FINAL_STATUSES = {"complete", "text_rejected", "visual_rejected"}

# Historical migrations are deliberately interleaved. One taxonomy video gets
# GPU time first, followed by a small Nominatim batch. This keeps both backlogs
# moving and, critically, gives each stage a frequent persistence boundary.
TAXONOMY_VIDEOS_PER_CYCLE = 1
LOCATION_SEGMENTS_PER_CYCLE = 25


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

    # Completed accepted records remain part of the unfinished batch until
    # their MMUCC taxonomy has been classified or terminally skipped.
    if crash_taxonomy_pending(record):
        return True

    if record.get("status") not in FINAL_STATUSES:
        return True

    return record.get("visual_review_version") != CRASH_REVIEW_VERSION


def unfinished_count(state: Dict[str, Any]) -> int:
    return sum(
        1
        for record in state.get("videos", {}).values()
        if requires_processing(record)
    )


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


def update_outputs(
    state: Dict[str, Any],
    *,
    save_state_file: bool = True,
) -> None:
    """Regenerate derived CSVs without starting another migration pass."""
    write_output_csv(state)
    if save_state_file:
        save_state(settings.STATE_JSON, state)


def _run_location_visual_without_premature_delete(
    state: Dict[str, Any],
) -> int:
    """
    Keep retained videos long enough for a pending taxonomy backfill.

    The taxonomy stage cleans them after classification when no location review
    still needs the source.
    """
    taxonomy_waiting = any(
        crash_taxonomy_pending(record)
        for record in state.get("videos", {}).values()
    )

    if not taxonomy_waiting:
        return run_location_visual_stage(state)

    delete_after_processing = settings.DELETE_VIDEO_AFTER_PROCESSING
    settings.DELETE_VIDEO_AFTER_PROCESSING = False
    try:
        return run_location_visual_stage(state)
    finally:
        settings.DELETE_VIDEO_AFTER_PROCESSING = delete_after_processing


def _run_visual_without_premature_delete(
    state: Dict[str, Any],
) -> int:
    """
    Prevent a newly reviewed source from being deleted before the MMUCC
    taxonomy stage can classify its accepted segments in the same cycle.
    """
    delete_after_processing = settings.DELETE_VIDEO_AFTER_PROCESSING
    settings.DELETE_VIDEO_AFTER_PROCESSING = False
    try:
        return run_visual_stage(state, after_video=update_outputs)
    finally:
        settings.DELETE_VIDEO_AFTER_PROCESSING = delete_after_processing


def run_cycle(state: Dict[str, Any], cycle: int) -> tuple[int, int, int, int]:
    log(f"Starting cycle {cycle}")

    # Historical MMUCC classification gets first access to the GPU. Keeping
    # this batch intentionally small means state and CSV output become visibly
    # newer after every processed historical video.
    taxonomy_processed = run_crash_taxonomy_stage(
        state,
        max_videos=TAXONOMY_VIDEOS_PER_CYCLE,
    )
    if taxonomy_processed:
        # The taxonomy stage already checkpointed state for its video.
        update_outputs(state, save_state_file=False)

    # Migrate only a bounded number of old location records per cycle. v4
    # locations are excluded from public CSVs until this v7 pass revalidates
    # them, so stale rows disappear immediately and verified rows return
    # progressively.
    location_processed = run_location_stage(
        state,
        max_segments=LOCATION_SEGMENTS_PER_CYCLE,
    )
    if location_processed:
        # run_location_stage() already checkpointed state and geocode cache.
        update_outputs(state, save_state_file=False)

    taxonomy_waiting = any(
        crash_taxonomy_pending(record)
        for record in state.get("videos", {}).values()
    )

    # Taxonomy backfill now performs location visual recovery on the same video
    # while that source and Cosmos are already available. Defer the broad legacy
    # location-only sweep until the taxonomy backlog has cleared so it cannot
    # monopolise the GPU ahead of MMUCC work.
    location_visual_processed = 0
    if not taxonomy_waiting:
        location_visual_processed = (
            _run_location_visual_without_premature_delete(state)
        )
        if location_visual_processed:
            # run_location_visual_stage() checkpoints each reviewed record.
            update_outputs(state, save_state_file=False)

    # crash_taxonomy_pending() participates in unfinished_count(), so new
    # discovery remains paused while historical accepted videos still require
    # taxonomy. Existing partially processed records may still advance below.
    discovered = discover_when_ready(state)
    text_processed = run_text_stage(state)

    # Keep newly downloaded sources until their taxonomy pass can use them.
    visual_processed = _run_visual_without_premature_delete(state)

    # New visual work may have produced accepted segments after the historical
    # taxonomy batch above. Their taxonomy is intentionally picked up at the
    # start of the next cycle, preserving the one-video checkpoint cadence.
    update_outputs(state)

    unfinished = unfinished_count(state)
    accepted_segments = sum(
        len(record.get("segments", []))
        for record in state.get("videos", {}).values()
        if isinstance(record, dict)
    )

    log(
        f"Cycle {cycle}: discovered={discovered}, text={text_processed}, "
        f"visual={visual_processed}, taxonomy={taxonomy_processed}, "
        f"location={location_processed}, "
        f"location_visual={location_visual_processed}, "
        f"accepted_segments={accepted_segments}, unfinished={unfinished}"
    )

    # Preserve the existing public return signature.
    return discovered, text_processed, visual_processed, unfinished


def cycle_pause_seconds(
    discovered: int,
    text_processed: int,
    visual_processed: int,
    unfinished: int,
) -> int:
    """Use the short pause whenever the current batch still has work."""
    active = bool(
        discovered
        or text_processed
        or visual_processed
        or unfinished
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

    # Replace any legacy 14-column CSV immediately. The new writer emits the
    # 23-column taxonomy schema and suppresses stale pre-v7 geographic rows even
    # before the first migration batch has finished.
    update_outputs(state, save_state_file=False)

    cycle = 0

    try:
        while True:
            cycle += 1
            discovered, text_count, visual_count, unfinished = run_cycle(
                state,
                cycle,
            )

            if not settings.CONTINUOUS_MODE:
                return

            pause = cycle_pause_seconds(
                discovered,
                text_count,
                visual_count,
                unfinished,
            )

            if unfinished:
                log(
                    f"Pausing {pause} seconds before continuing the current "
                    f"batch ({unfinished} unfinished)"
                )
            else:
                log(
                    f"Pausing {pause} seconds before the next discovery batch"
                )

            time.sleep(pause)

    except KeyboardInterrupt:
        update_outputs(state)
        log("Stopped by the user; current state and CSV were saved")
