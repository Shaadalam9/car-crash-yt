"""FFmpeg candidate cut detection and full clip boundary construction."""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from . import settings
from .shared import log, run_command


@dataclass(frozen=True)
class CutDetectionResult:
    candidates: List[float]
    backend: str
    error: str | None = None


@dataclass(frozen=True)
class FullSegment:
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


def _filter_graph(cuda: bool) -> str:
    if cuda:
        prefix = (
            f"scale_cuda={settings.CUT_WIDTH}:-2,hwdownload,format=nv12,"
            f"format=yuv420p,fps={settings.CUT_FPS:.8g}"
        )
    else:
        prefix = (
            f"fps={settings.CUT_FPS:.8g},"
            f"scale={settings.CUT_WIDTH}:-2:flags=fast_bilinear,format=yuv420p"
        )
    return (
        f"[0:v]{prefix},setpts=PTS-STARTPTS,split=2[scene][adaptive];"
        f"[scene]select='gt(scene,{settings.SCENE_THRESHOLD})',showinfo[out1];"
        f"[adaptive]scdet=t={settings.SCDET_THRESHOLD}:s=1,"
        "metadata=mode=print:key=lavfi.scd.time[out2]"
    )


def _command(video_path: Path, cuda: bool) -> List[str]:
    command = ["ffmpeg", "-hide_banner", "-nostdin"]
    if cuda:
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    command.extend(
        [
            "-i",
            str(video_path),
            "-filter_complex",
            _filter_graph(cuda),
            "-map",
            "[out1]",
            "-map",
            "[out2]",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    return command


def extract_candidate_times(stderr: str, duration: float) -> List[float]:
    values: List[float] = []
    patterns = (
        r"pts_time:([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        r"lavfi\.scd\.time=([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    )
    for line in stderr.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if not match:
                continue
            value = float(match.group(1))
            if math.isfinite(value) and 0.0 < value < duration:
                values.append(value)
    return merge_nearby_times(values, settings.CUT_MERGE_SECONDS)


def merge_nearby_times(times: Sequence[float], gap: float) -> List[float]:
    ordered = sorted(float(value) for value in times if math.isfinite(float(value)))
    if not ordered:
        return []
    groups: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def detect_candidate_cuts(video_path: Path, duration: float) -> CutDetectionResult:
    attempts = []
    if settings.CUT_BACKEND in {"auto", "ffmpeg_cuda"}:
        attempts.append(("ffmpeg_cuda", True))
    if settings.CUT_BACKEND in {"auto", "ffmpeg_cpu"} or settings.CPU_FALLBACK:
        attempts.append(("ffmpeg_cpu", False))

    failures = []
    for name, cuda in attempts:
        log(f"Detecting compilation cuts with {name}")
        try:
            result = run_command(
                _command(video_path, cuda), timeout=settings.CUT_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{name}: timeout")
            continue
        if result.returncode == 0:
            return CutDetectionResult(
                extract_candidate_times(result.stderr, duration), name
            )
        lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        failures.append(f"{name}: {lines[-1] if lines else 'failed'}")
    return CutDetectionResult([], attempts[-1][0] if attempts else "none", " | ".join(failures))


def build_full_segments(duration: float, verified_cuts: Sequence[float]) -> List[FullSegment]:
    """Build complete edit bounded clips without a duration threshold."""
    if not math.isfinite(duration) or duration <= 0:
        return []
    boundaries = [0.0]
    for value in sorted(float(item) for item in verified_cuts):
        if 0.0 < value < duration and value > boundaries[-1]:
            boundaries.append(value)
    boundaries.append(duration)
    return [
        FullSegment(start, end)
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    ]

