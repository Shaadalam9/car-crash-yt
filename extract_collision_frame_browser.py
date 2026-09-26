#!/usr/bin/env python3
"""
Interactively choose one frame from a random traffic collision.

Workflow
--------
1. Randomly choose a geographically resolved collision from mapping.csv.
2. Cross reference crash_segments.csv to recover the exact impact timestamp.
3. Show the collision metadata and ask before downloading anything.
4. Download only a short section around the collision.
5. Open a frame browser centred on the impact frame.
6. Use Previous / Next or the left / right arrow keys to inspect nearby frames.
7. Save exactly one selected frame, or choose a different random collision.

Requirements
------------
yt-dlp
ffmpeg
ffprobe
Python tkinter

Run
---
python3 extract_collision_frame_browser_v4.py
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATCH_TOLERANCE_SECONDS = 0.05

# Download a little more video than we browse so frame decoding near the edges
# remains reliable.
DOWNLOAD_CONTEXT_SECONDS = 2.0

# Frames shown in the browser are restricted to this interval around the
# recorded impact timestamp.
BROWSE_RADIUS_SECONDS = 1.0

# Maximum preview size. The final saved frame is extracted at full resolution.
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 800

# Remember proposed collision candidates across runs so rerunning the script
# does not immediately show the same collision again.
DEFAULT_HISTORY_FILE = Path(".collision_frame_history_v4.txt")


@dataclass(frozen=True)
class MappingSegment:
    mapping_row_id: str
    locality: str
    state: str
    country: str
    continent: str
    video_id: str
    start_time: float
    end_time: float
    event_kind: str
    manner_of_collision: str
    first_harmful_event: str


@dataclass(frozen=True)
class Candidate:
    mapping: MappingSegment
    youtube_url: str
    impact_time: float
    segment_id: str
    short_description: str
    road_users: str
    time_of_day: str
    event_kind: str
    manner_of_collision: str
    first_harmful_event: str


@dataclass(frozen=True)
class PreviewFrame:
    path: Path
    source_time: float
    source_frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose a random traffic collision, browse nearby frames, "
            "and save exactly one selected frame."
        )
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("mapping.csv"),
        help="Path to mapping.csv",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("crash_segments.csv"),
        help="Path to crash_segments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("collision_frames"),
        help="Directory for the one frame you finally choose",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY_FILE,
        help=(
            "Persistent history of collision candidates already shown. "
            "This prevents the same collision from being selected again "
            "on the next run."
        ),
    )
    parser.add_argument(
        "--reset-history",
        action="store_true",
        help=(
            "Explicitly clear the collision selection history before "
            "choosing a candidate. History is never reset automatically."
        ),
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help=(
            "Number of decoded frames moved by Previous/Next. "
            "Default: 5, which is about 0.17 seconds at 30 fps."
        ),
    )
    return parser.parse_args()


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required command '{name}' was not found on PATH."
        )


def parse_bracket_cell(raw: Any) -> Any:
    """Parse the repository's unquoted nested bracket cell format."""
    text = str(raw or "").strip()
    if not text:
        return []

    index = 0

    def skip_space() -> None:
        nonlocal index
        while index < len(text) and text[index].isspace():
            index += 1

    def parse_value() -> Any:
        nonlocal index
        skip_space()

        if index < len(text) and text[index] == "[":
            index += 1
            values = []
            skip_space()

            if index < len(text) and text[index] == "]":
                index += 1
                return values

            while index < len(text):
                values.append(parse_value())
                skip_space()

                if index >= len(text):
                    break

                if text[index] == ",":
                    index += 1
                    continue

                if text[index] == "]":
                    index += 1
                    break

                raise ValueError(
                    f"Unexpected character at position {index}: "
                    f"{text[index]!r}"
                )

            return values

        start = index
        while index < len(text) and text[index] not in ",]":
            index += 1

        return text[start:index].strip()

    value = parse_value()
    skip_space()

    if index != len(text):
        raise ValueError(
            f"Could not fully parse bracket cell: {text!r}"
        )

    return value


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value

    if value in (None, ""):
        return []

    return [value]


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def per_video_values(raw: str, video_count: int) -> list[list[Any]]:
    """
    A mapping value is flattened one level when the locality has one video,
    but remains nested by video when several videos belong to the locality.
    """
    parsed = parse_bracket_cell(raw)

    if video_count == 1:
        return [as_list(parsed)]

    if (
        isinstance(parsed, list)
        and len(parsed) == video_count
        and all(isinstance(item, list) for item in parsed)
    ):
        return parsed

    return [[] for _ in range(video_count)]


def item_at(values: list[Any], index: int) -> str:
    if index >= len(values):
        return ""

    value = values[index]

    if isinstance(value, list):
        return ""

    return str(value or "").strip()


def expand_mapping_row(row: dict[str, str]) -> list[MappingSegment]:
    videos = [
        str(value).strip()
        for value in as_list(
            parse_bracket_cell(row.get("videos", ""))
        )
        if str(value).strip()
    ]

    if not videos:
        return []

    starts = per_video_values(
        row.get("start_time", ""),
        len(videos),
    )
    ends = per_video_values(
        row.get("end_time", ""),
        len(videos),
    )
    event_kinds = per_video_values(
        row.get("event_kind", ""),
        len(videos),
    )
    manners = per_video_values(
        row.get("manner_of_collision", ""),
        len(videos),
    )
    first_events = per_video_values(
        row.get("first_harmful_event", ""),
        len(videos),
    )

    result: list[MappingSegment] = []

    for video_index, video_id in enumerate(videos):
        for segment_index, start_raw in enumerate(
            starts[video_index]
        ):
            start = as_float(start_raw)

            end = as_float(
                ends[video_index][segment_index]
                if segment_index < len(ends[video_index])
                else None
            )

            if start is None or end is None or end < start:
                continue

            result.append(
                MappingSegment(
                    mapping_row_id=str(
                        row.get("id", "")
                    ).strip(),
                    locality=str(
                        row.get("locality", "")
                    ).strip(),
                    state=str(
                        row.get("state", "")
                    ).strip(),
                    country=str(
                        row.get("country", "")
                    ).strip(),
                    continent=str(
                        row.get("continent", "")
                    ).strip(),
                    video_id=video_id,
                    start_time=start,
                    end_time=end,
                    event_kind=item_at(
                        event_kinds[video_index],
                        segment_index,
                    ),
                    manner_of_collision=item_at(
                        manners[video_index],
                        segment_index,
                    ),
                    first_harmful_event=item_at(
                        first_events[video_index],
                        segment_index,
                    ),
                )
            )

    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def build_crash_index(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    by_video: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        video_id = str(
            row.get("video_id", "")
        ).strip()

        if video_id:
            by_video.setdefault(
                video_id,
                [],
            ).append(row)

    return by_video


def find_exact_crash_segment(
    mapped: MappingSegment,
    crash_index: dict[str, list[dict[str, str]]],
) -> dict[str, str] | None:
    best: dict[str, str] | None = None
    best_error = float("inf")

    for row in crash_index.get(mapped.video_id, []):
        start = as_float(row.get("start_time"))
        end = as_float(row.get("end_time"))

        if start is None or end is None:
            continue

        error = (
            abs(start - mapped.start_time)
            + abs(end - mapped.end_time)
        )

        if error < best_error:
            best_error = error
            best = row

    if best is None:
        return None

    start = as_float(best.get("start_time"))
    end = as_float(best.get("end_time"))

    if start is None or end is None:
        return None

    if (
        abs(start - mapped.start_time)
        > MATCH_TOLERANCE_SECONDS
        or abs(end - mapped.end_time)
        > MATCH_TOLERANCE_SECONDS
    ):
        return None

    return best


def candidate_from_match(
    mapped: MappingSegment,
    crash_row: dict[str, str],
) -> Candidate | None:
    """Build one collision candidate from a mapped accepted segment.

    IMPORTANT: candidate selection must not depend on the MMUCC backfill.
    `event_kind` is a taxonomy field and can legitimately be blank while the
    historical taxonomy migration is still running.  The original visual
    review already gives us the signal needed for frame extraction:

    * a real impact has `impact_time_in_video`;
    * near collisions use `crash_type=near_collision` and normally have no
      impact timestamp.

    Requiring `event_kind == collision` therefore incorrectly shrinks the
    pool to only the small taxonomy-complete subset.
    """
    impact_time = as_float(
        crash_row.get("impact_time_in_video")
    )

    # No physical impact timestamp means there is no exact collision frame to
    # centre the browser on.  This also removes the normal near-collision case.
    if impact_time is None:
        return None

    crash_type = str(
        crash_row.get("crash_type") or ""
    ).strip().casefold().replace(" ", "_")

    event_kind = str(
        crash_row.get("event_kind")
        or mapped.event_kind
        or ""
    ).strip().casefold().replace(" ", "_")

    # Defensively reject anything explicitly marked as a near collision by
    # either the original visual review or the later taxonomy.
    if crash_type == "near_collision" or event_kind == "near_collision":
        return None

    if not (
        mapped.start_time
        <= impact_time
        <= mapped.end_time
    ):
        return None

    youtube_url = str(
        crash_row.get("youtube_url", "")
    ).strip()

    if not youtube_url:
        youtube_url = (
            "https://www.youtube.com/watch?v="
            f"{mapped.video_id}"
        )

    return Candidate(
        mapping=mapped,
        youtube_url=youtube_url,
        impact_time=impact_time,
        segment_id=str(
            crash_row.get("segment_id", "")
        ).strip(),
        short_description=str(
            crash_row.get("short_description", "")
        ).strip(),
        road_users=str(
            crash_row.get("road_users", "")
        ).strip(),
        time_of_day=str(
            crash_row.get("time_of_day", "")
        ).strip(),
        event_kind=(
            str(crash_row.get("event_kind") or "").strip()
            or "collision (visual review)"
        ),
        manner_of_collision=str(
            crash_row.get("manner_of_collision")
            or mapped.manner_of_collision
            or ""
        ).strip(),
        first_harmful_event=str(
            crash_row.get("first_harmful_event")
            or mapped.first_harmful_event
            or ""
        ).strip(),
    )


def build_candidates_by_mapping_row(
    mapping_rows: list[dict[str, str]],
    crash_rows: list[dict[str, str]],
) -> list[list[Candidate]]:
    crash_index = build_crash_index(crash_rows)
    candidate_rows: list[list[Candidate]] = []

    for mapping_row in mapping_rows:
        row_candidates: list[Candidate] = []

        for mapped in expand_mapping_row(mapping_row):
            crash_row = find_exact_crash_segment(
                mapped,
                crash_index,
            )

            if crash_row is None:
                continue

            candidate = candidate_from_match(
                mapped,
                crash_row,
            )

            if candidate is not None:
                row_candidates.append(candidate)

        if row_candidates:
            candidate_rows.append(row_candidates)

    return candidate_rows


def candidate_key(candidate: Candidate) -> str:
    """Stable identifier used only to avoid repeating the same collision."""
    if candidate.segment_id:
        return candidate.segment_id

    return (
        f"{candidate.mapping.video_id}|"
        f"{candidate.mapping.start_time:.3f}|"
        f"{candidate.mapping.end_time:.3f}"
    )


def seen_video_ids(
    candidate_rows: list[list[Candidate]],
    seen: set[str],
) -> set[str]:
    """Recover video IDs represented by already shown collision keys."""
    videos: set[str] = set()

    for row in candidate_rows:
        for candidate in row:
            if candidate_key(candidate) in seen:
                videos.add(candidate.mapping.video_id)

    return videos


def load_seen_candidates(path: Path) -> set[str]:
    if not path.exists():
        return set()

    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def remember_candidate(path: Path, candidate: Candidate) -> None:
    """Persist a candidate as soon as it is shown to the user."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(candidate_key(candidate) + "\n")


def unseen_candidate_rows(
    candidate_rows: list[list[Candidate]],
    seen: set[str],
) -> list[list[Candidate]]:
    rows: list[list[Candidate]] = []

    for row in candidate_rows:
        remaining = [
            candidate
            for candidate in row
            if candidate_key(candidate) not in seen
        ]
        if remaining:
            rows.append(remaining)

    return rows


def choose_random_candidate(
    candidate_rows: list[list[Candidate]],
    seen_videos: set[str] | None = None,
) -> Candidate:
    """Choose uniformly from individual mapped collision segments.

    We still prefer a source video that has not been shown before, but once
    every source video has appeared we continue with unseen segments from the
    existing videos.  We do not treat a whole mapping row or whole source video
    as exhausted just because one segment from it was displayed.
    """
    rng = random.SystemRandom()
    seen_videos = seen_videos or set()

    flat = [
        candidate
        for row in candidate_rows
        for candidate in row
    ]

    if not flat:
        raise RuntimeError("No collision candidates are available.")

    new_video_pool = [
        candidate
        for candidate in flat
        if candidate.mapping.video_id not in seen_videos
    ]

    return rng.choice(new_video_pool or flat)



def candidate_diagnostics(
    mapping_rows: list[dict[str, str]],
    crash_rows: list[dict[str, str]],
    candidate_rows: list[list[Candidate]],
) -> dict[str, int]:
    """Return transparent counts so a tiny candidate pool is never silent."""
    expanded_mapping = sum(
        len(expand_mapping_row(row))
        for row in mapping_rows
    )

    crash_with_impact = sum(
        1
        for row in crash_rows
        if as_float(row.get("impact_time_in_video")) is not None
    )

    taxonomy_collisions = sum(
        1
        for row in crash_rows
        if str(row.get("event_kind") or "").strip().casefold()
        == "collision"
    )

    visual_near_collisions = sum(
        1
        for row in crash_rows
        if str(row.get("crash_type") or "")
        .strip()
        .casefold()
        .replace(" ", "_")
        == "near_collision"
    )

    eligible = sum(len(row) for row in candidate_rows)
    eligible_videos = {
        candidate.mapping.video_id
        for row in candidate_rows
        for candidate in row
    }

    return {
        "expanded_mapping_segments": expanded_mapping,
        "crash_rows": len(crash_rows),
        "crash_rows_with_impact": crash_with_impact,
        "taxonomy_collision_rows": taxonomy_collisions,
        "visual_near_collision_rows": visual_near_collisions,
        "eligible_mapped_collisions": eligible,
        "eligible_source_videos": len(eligible_videos),
    }

def format_seconds(value: float) -> str:
    whole_minutes, seconds = divmod(
        value,
        60.0,
    )
    hours, minutes = divmod(
        int(whole_minutes),
        60,
    )

    if hours:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:06.3f}"
        )

    return f"{minutes:02d}:{seconds:06.3f}"


def print_candidate(candidate: Candidate) -> None:
    mapped = candidate.mapping

    print("\n" + "=" * 72)
    print("RANDOM COLLISION CANDIDATE")
    print("=" * 72)
    print(
        f"Mapping row:          "
        f"{mapped.mapping_row_id}"
    )
    print(
        f"Locality:             "
        f"{mapped.locality}"
    )
    print(
        f"State/region:         "
        f"{mapped.state or 'N/A'}"
    )
    print(
        f"Country:              "
        f"{mapped.country}"
    )
    print(
        f"Continent:            "
        f"{mapped.continent}"
    )
    print(
        f"Video ID:             "
        f"{mapped.video_id}"
    )
    print(
        f"Segment ID:           "
        f"{candidate.segment_id or 'N/A'}"
    )
    print(
        "Retained clip:        "
        f"{format_seconds(mapped.start_time)} "
        "to "
        f"{format_seconds(mapped.end_time)}"
    )
    print(
        "RECORDED IMPACT TIME: "
        f"{format_seconds(candidate.impact_time)} "
        f"({candidate.impact_time:.3f} s)"
    )
    print(
        f"Event kind:           "
        f"{candidate.event_kind}"
    )
    print(
        f"Manner:               "
        f"{candidate.manner_of_collision or 'N/A'}"
    )
    print(
        "First harmful event:  "
        f"{candidate.first_harmful_event or 'N/A'}"
    )
    print(
        f"Road users:           "
        f"{candidate.road_users or 'N/A'}"
    )
    print(
        f"Time of day:          "
        f"{candidate.time_of_day or 'N/A'}"
    )

    if candidate.short_description:
        print(
            f"Description:          "
            f"{candidate.short_description}"
        )

    print(
        f"YouTube source:       "
        f"{candidate.youtube_url}"
    )
    print()
    print(
        "If you continue, a frame browser will open centred on the "
        "recorded impact. You will see the actual pixels before saving."
    )
    print("=" * 72)


def ask_candidate_action() -> str:
    while True:
        answer = input(
            "\nOpen frame browser for this collision? "
            "[y = yes, r = another random collision, n = stop]: "
        ).strip().casefold()

        if answer in {"y", "yes"}:
            return "yes"

        if answer in {
            "r",
            "random",
            "another",
        }:
            return "reroll"

        if answer in {
            "n",
            "no",
            "",
        }:
            return "no"

        print("Please enter y, r, or n.")


def download_preview_clip(
    candidate: Candidate,
    temp_dir: Path,
) -> tuple[Path, float]:
    clip_start = max(
        candidate.mapping.start_time,
        candidate.impact_time
        - DOWNLOAD_CONTEXT_SECONDS,
        0.0,
    )

    clip_end = min(
        candidate.mapping.end_time,
        candidate.impact_time
        + DOWNLOAD_CONTEXT_SECONDS,
    )

    if clip_end <= clip_start:
        raise RuntimeError(
            "Invalid preview interval."
        )

    output_template = str(
        temp_dir / "source.%(ext)s"
    )

    command = [
        "yt-dlp",
        "--no-playlist",
        "--download-sections",
        f"*{clip_start:.3f}-{clip_end:.3f}",
        "--force-keyframes-at-cuts",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        candidate.youtube_url,
    ]

    print()
    print(
        f"Downloading only "
        f"{clip_start:.3f} to "
        f"{clip_end:.3f} seconds "
        "around the selected collision..."
    )

    subprocess.run(
        command,
        check=True,
    )

    downloaded = [
        path
        for path in temp_dir.glob("source.*")
        if path.is_file()
        and path.suffix.lower()
        in {
            ".mp4",
            ".mkv",
            ".webm",
            ".mov",
        }
    ]

    if not downloaded:
        raise RuntimeError(
            "yt-dlp completed but no video clip "
            "was found."
        )

    return (
        max(
            downloaded,
            key=lambda path: path.stat().st_size,
        ),
        clip_start,
    )


def probe_frame_times(
    clip_path: Path,
) -> list[float]:
    """
    Return decoded video frame timestamps from the downloaded clip.

    We normalise them to the first decoded frame because a downloaded section
    can retain a nonzero stream timestamp depending on its container.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(clip_path),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    values: list[float] = []

    for line in result.stdout.splitlines():
        value = line.strip().split(",", 1)[0]

        try:
            values.append(float(value))
        except ValueError:
            continue

    if not values:
        raise RuntimeError(
            "ffprobe could not read video frame timestamps."
        )

    first = values[0]

    return [
        value - first
        for value in values
    ]


def extract_preview_frames(
    clip_path: Path,
    clip_start: float,
    candidate: Candidate,
    temp_dir: Path,
) -> list[PreviewFrame]:
    preview_dir = temp_dir / "preview_frames"
    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_times = probe_frame_times(
        clip_path
    )

    # The scale expression only shrinks frames. It never enlarges them.
    scale_filter = (
        "scale="
        f"'min({PREVIEW_MAX_WIDTH},iw)':"
        f"'min({PREVIEW_MAX_HEIGHT},ih)':"
        "force_original_aspect_ratio=decrease"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        scale_filter,
        "-vsync",
        "0",
        "-y",
        str(
            preview_dir / "frame_%06d.png"
        ),
    ]

    print(
        "Decoding nearby frames for the browser..."
    )

    subprocess.run(
        command,
        check=True,
    )

    paths = sorted(
        preview_dir.glob("frame_*.png")
    )

    usable = min(
        len(paths),
        len(frame_times),
    )

    if usable == 0:
        raise RuntimeError(
            "No preview frames were decoded."
        )

    all_frames: list[PreviewFrame] = []

    for index in range(usable):
        source_time = (
            clip_start
            + frame_times[index]
        )

        all_frames.append(
            PreviewFrame(
                path=paths[index],
                source_time=source_time,
                source_frame_index=index,
            )
        )

    lower = (
        candidate.impact_time
        - BROWSE_RADIUS_SECONDS
    )
    upper = (
        candidate.impact_time
        + BROWSE_RADIUS_SECONDS
    )

    nearby = [
        frame
        for frame in all_frames
        if lower
        <= frame.source_time
        <= upper
    ]

    # Extremely short clips can fall outside our approximate source timestamp
    # mapping. In that case retain a small neighbourhood around the closest
    # decoded frame rather than failing.
    if not nearby:
        closest_index = min(
            range(len(all_frames)),
            key=lambda i: abs(
                all_frames[i].source_time
                - candidate.impact_time
            ),
        )

        start = max(
            0,
            closest_index - 15,
        )
        end = min(
            len(all_frames),
            closest_index + 16,
        )
        nearby = all_frames[start:end]

    return nearby


class FrameBrowser:
    def __init__(
        self,
        frames: list[PreviewFrame],
        candidate: Candidate,
        frame_step: int = 5,
    ) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            raise RuntimeError(
                "Python tkinter is required for the "
                "interactive frame browser."
            ) from exc

        self.tk = tk
        self.ttk = ttk
        self.frames = frames
        self.candidate = candidate
        self.frame_step = max(1, int(frame_step))

        self.index = min(
            range(len(frames)),
            key=lambda i: abs(
                frames[i].source_time
                - candidate.impact_time
            ),
        )

        self.result: tuple[
            str,
            PreviewFrame | None,
        ] = ("cancel", None)

        self.root = tk.Tk()
        self.root.title(
            "Traffic collision frame selector"
        )
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.cancel,
        )

        self.image_label = ttk.Label(
            self.root
        )
        self.image_label.pack(
            padx=12,
            pady=(12, 6),
        )

        self.timestamp_label = ttk.Label(
            self.root,
            anchor="center",
            font=(
                "TkDefaultFont",
                13,
                "bold",
            ),
        )
        self.timestamp_label.pack(
            fill="x",
            padx=12,
        )

        self.details_label = ttk.Label(
            self.root,
            anchor="center",
            justify="center",
        )
        self.details_label.pack(
            fill="x",
            padx=12,
            pady=(3, 9),
        )

        controls = ttk.Frame(
            self.root
        )
        controls.pack(
            padx=12,
            pady=(0, 12),
            fill="x",
        )

        self.previous_button = ttk.Button(
            controls,
            text=f"◀ {self.frame_step} frames",
            command=self.previous,
        )
        self.previous_button.pack(
            side="left",
            padx=4,
        )

        self.impact_button = ttk.Button(
            controls,
            text="Impact frame",
            command=self.jump_to_impact,
        )
        self.impact_button.pack(
            side="left",
            padx=4,
        )

        self.next_button = ttk.Button(
            controls,
            text=f"{self.frame_step} frames ▶",
            command=self.next,
        )
        self.next_button.pack(
            side="left",
            padx=4,
        )

        spacer = ttk.Frame(
            controls
        )
        spacer.pack(
            side="left",
            expand=True,
        )

        self.random_button = ttk.Button(
            controls,
            text="Different collision",
            command=self.reroll,
        )
        self.random_button.pack(
            side="left",
            padx=4,
        )

        self.cancel_button = ttk.Button(
            controls,
            text="Cancel",
            command=self.cancel,
        )
        self.cancel_button.pack(
            side="left",
            padx=4,
        )

        self.save_button = ttk.Button(
            controls,
            text="Save this frame",
            command=self.save,
        )
        self.save_button.pack(
            side="left",
            padx=4,
        )

        self.root.bind(
            "<Left>",
            lambda _event: self.previous(),
        )
        self.root.bind(
            "<Right>",
            lambda _event: self.next(),
        )
        self.root.bind(
            "<Shift-Left>",
            lambda _event: self.previous_one(),
        )
        self.root.bind(
            "<Shift-Right>",
            lambda _event: self.next_one(),
        )
        self.root.bind(
            "<Home>",
            lambda _event: self.jump_to_impact(),
        )
        self.root.bind(
            "<Return>",
            lambda _event: self.save(),
        )
        self.root.bind(
            "<Escape>",
            lambda _event: self.cancel(),
        )

        self.photo = None
        self.refresh()

    def refresh(self) -> None:
        frame = self.frames[self.index]

        self.photo = self.tk.PhotoImage(
            file=str(frame.path)
        )

        self.image_label.configure(
            image=self.photo
        )

        offset = (
            frame.source_time
            - self.candidate.impact_time
        )

        sign = "+" if offset >= 0 else ""

        self.timestamp_label.configure(
            text=(
                f"Frame {self.index + 1} / "
                f"{len(self.frames)}    |    "
                f"Source time "
                f"{frame.source_time:.3f} s    |    "
                f"{sign}{offset:.3f} s "
                "from recorded impact"
            )
        )

        mapped = self.candidate.mapping

        self.details_label.configure(
            text=(
                f"{mapped.locality}, "
                f"{mapped.country}    |    "
                f"{self.candidate.manner_of_collision or 'collision'}"
                "\n"
                f"Previous / Next move {self.frame_step} decoded frames. "
                "Use Shift+Left / Shift+Right for one frame at a time. "
                "The displayed pixels are the frame you are selecting."
            )
        )

        self.previous_button.configure(
            state=(
                "normal"
                if self.index > 0
                else "disabled"
            )
        )

        self.next_button.configure(
            state=(
                "normal"
                if self.index
                < len(self.frames) - 1
                else "disabled"
            )
        )

    def previous(self) -> None:
        if self.index > 0:
            self.index = max(0, self.index - self.frame_step)
            self.refresh()

    def next(self) -> None:
        if self.index < len(self.frames) - 1:
            self.index = min(
                len(self.frames) - 1,
                self.index + self.frame_step,
            )
            self.refresh()

    def previous_one(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.refresh()

    def next_one(self) -> None:
        if self.index < len(self.frames) - 1:
            self.index += 1
            self.refresh()

    def jump_to_impact(self) -> None:
        self.index = min(
            range(len(self.frames)),
            key=lambda i: abs(
                self.frames[i].source_time
                - self.candidate.impact_time
            ),
        )
        self.refresh()

    def save(self) -> None:
        self.result = (
            "save",
            self.frames[self.index],
        )
        self.root.destroy()

    def reroll(self) -> None:
        self.result = (
            "reroll",
            None,
        )
        self.root.destroy()

    def cancel(self) -> None:
        self.result = (
            "cancel",
            None,
        )
        self.root.destroy()

    def run(
        self,
    ) -> tuple[str, PreviewFrame | None]:
        self.root.mainloop()
        return self.result


def extract_full_resolution_frame(
    clip_path: Path,
    selected: PreviewFrame,
    destination: Path,
) -> None:
    """
    Extract the exact decoded frame that the user selected in the browser.

    Selection is by frame index, not by a second approximate timestamp seek.
    This makes the final full resolution image correspond to the preview.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    select_filter = (
        "select="
        f"'eq(n\\,{selected.source_frame_index})'"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        select_filter,
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-y",
        str(destination),
    ]

    subprocess.run(
        command,
        check=True,
    )

    if (
        not destination.exists()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            "ffmpeg did not create the selected "
            "full resolution frame."
        )


def output_path_for(
    candidate: Candidate,
    selected: PreviewFrame,
    output_dir: Path,
) -> Path:
    # Requested naming convention: {video_id}_{time}.png
    # The time is the exact selected source-video timestamp in seconds.
    return output_dir / (
        f"{candidate.mapping.video_id}_"
        f"{selected.source_time:.3f}.png"
    )


def browse_candidate(
    candidate: Candidate,
    output_dir: Path,
    frame_step: int,
) -> str:
    """
    Return one of:
        saved
        reroll
        cancelled
    """
    with tempfile.TemporaryDirectory(
        prefix="collision_frame_browser_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        clip_path, clip_start = (
            download_preview_clip(
                candidate,
                temp_dir,
            )
        )

        frames = extract_preview_frames(
            clip_path,
            clip_start,
            candidate,
            temp_dir,
        )

        print(
            f"Opening {len(frames):,} nearby "
            "decoded frames in the selector..."
        )

        browser = FrameBrowser(
            frames,
            candidate,
            frame_step=frame_step,
        )

        action, selected = browser.run()

        if action == "reroll":
            return "reroll"

        if (
            action != "save"
            or selected is None
        ):
            print(
                "Nothing saved. Temporary preview "
                "files were removed."
            )
            return "cancelled"

        destination = output_path_for(
            candidate,
            selected,
            output_dir,
        )

        extract_full_resolution_frame(
            clip_path,
            selected,
            destination,
        )

        offset = (
            selected.source_time
            - candidate.impact_time
        )

        print()
        print("=" * 72)
        print("SAVED ONE FRAME")
        print("=" * 72)
        print(
            f"File:             "
            f"{destination.resolve()}"
        )
        print(
            f"Video ID:         "
            f"{candidate.mapping.video_id}"
        )
        print(
            f"Source time:      "
            f"{selected.source_time:.3f} s"
        )
        print(
            f"Recorded impact:  "
            f"{candidate.impact_time:.3f} s"
        )
        print(
            f"Offset:           "
            f"{offset:+.3f} s"
        )
        print("=" * 72)

        return "saved"


def main() -> int:
    args = parse_args()

    require_program("yt-dlp")
    require_program("ffmpeg")
    require_program("ffprobe")

    print(
        f"Reading {args.mapping}..."
    )
    mapping_rows = read_csv(
        args.mapping
    )

    print(
        f"Reading {args.segments}..."
    )
    crash_rows = read_csv(
        args.segments
    )

    candidate_rows = (
        build_candidates_by_mapping_row(
            mapping_rows,
            crash_rows,
        )
    )

    if not candidate_rows:
        raise RuntimeError(
            "No mapping entries matched classified "
            "collision segments with a valid "
            "impact_time_in_video."
        )

    total_candidates = sum(
        len(row)
        for row in candidate_rows
    )

    diagnostics = candidate_diagnostics(
        mapping_rows,
        crash_rows,
        candidate_rows,
    )

    print()
    print("=== Candidate pool diagnostics ===")
    print(
        "Mapped segment references:       "
        f"{diagnostics['expanded_mapping_segments']:,}"
    )
    print(
        "Rows in crash_segments.csv:      "
        f"{diagnostics['crash_rows']:,}"
    )
    print(
        "Rows with recorded impact time:  "
        f"{diagnostics['crash_rows_with_impact']:,}"
    )
    print(
        "MMUCC collision rows right now:  "
        f"{diagnostics['taxonomy_collision_rows']:,}"
    )
    print(
        "Visual near collisions excluded: "
        f"{diagnostics['visual_near_collision_rows']:,}"
    )
    print(
        "Eligible mapped collisions:      "
        f"{diagnostics['eligible_mapped_collisions']:,}"
    )
    print(
        "Eligible source videos:          "
        f"{diagnostics['eligible_source_videos']:,}"
    )
    print()
    print(
        f"Found {total_candidates:,} eligible collision segments "
        f"across {len(candidate_rows):,} mapping rows."
    )

    if args.reset_history:
        args.history.unlink(missing_ok=True)
        print(f"Selection history reset: {args.history}")

    seen = load_seen_candidates(args.history)
    available_rows = unseen_candidate_rows(
        candidate_rows,
        seen,
    )

    total_unseen = sum(len(row) for row in available_rows)
    seen_videos = seen_video_ids(candidate_rows, seen)
    all_videos = {
        candidate.mapping.video_id
        for row in candidate_rows
        for candidate in row
    }
    unseen_videos = all_videos - seen_videos

    print(
        f"History file: {args.history}"
    )
    print(
        f"Previously shown collisions: {len(seen):,}"
    )
    print(
        f"Unseen eligible collisions: {total_unseen:,}"
    )
    print(
        f"Unseen source videos: {len(unseen_videos):,} / "
        f"{len(all_videos):,}"
    )

    if not available_rows:
        print()
        print(
            f"No unseen collision candidates remain out of {total_candidates:,} eligible mapped collisions. History was NOT reset."
        )
        print(
            "Run again with --reset-history only if you intentionally want "
            "to allow previously shown collisions."
        )
        return 0

    while True:
        seen_videos = seen_video_ids(candidate_rows, seen)
        candidate = choose_random_candidate(
            available_rows,
            seen_videos=seen_videos,
        )

        # Mark only this selected collision as shown. Nothing else in the
        # candidate pool is added to history.
        remember_candidate(args.history, candidate)
        seen.add(candidate_key(candidate))
        available_rows = unseen_candidate_rows(
            candidate_rows,
            seen,
        )

        print_candidate(
            candidate
        )

        action = ask_candidate_action()

        if action == "reroll":
            if not available_rows:
                print(
                    "No unseen collision candidates remain. "
                    "History was NOT reset."
                )
                return 0
            continue

        if action == "no":
            print(
                "Nothing downloaded or saved."
            )
            return 0

        result = browse_candidate(
            candidate,
            args.output_dir,
            frame_step=args.frame_step,
        )

        if result == "reroll":
            if not available_rows:
                print(
                    "No unseen collision candidates remain. "
                    "History was NOT reset."
                )
                return 0
            continue

        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nCancelled. Nothing else will be saved."
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
