"""Small shared utilities with atomic state persistence."""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def replace_file_with_retry(source: Path, destination: Path, attempts: int = 8) -> None:
    """Replace a file while tolerating short lived Windows file locks."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.05 * (2**attempt), 1.0))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    replace_file_with_retry(temporary, path)


def run_command(
    command: Sequence[str], *, timeout: Optional[float] = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command is unavailable: {name}")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def optional_text(value: Any) -> Optional[str]:
    text = clean_text(value)
    return text or None


def clamp_float(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lower
    if not math.isfinite(number):
        return lower
    return max(lower, min(upper, number))


def normalise_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).casefold() in {"1", "true", "yes", "y"}


def normalise_string_list(value: Any) -> List[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    result: List[str] = []
    seen = set()
    for item in values:
        text = clean_text(item)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def coordinates_with_hemispheres(
    lat: Optional[float],
    lon: Optional[float],
    evidence: Any,
) -> tuple[Optional[float], Optional[float]]:
    """Apply visible N/S/E/W markers to an extracted coordinate pair."""
    texts = normalise_string_list(evidence)

    def directed(
        value: Optional[float], positive: str, negative: str
    ) -> Optional[float]:
        if value is None:
            return None
        patterns = (
            rf"(?<![A-Z])([{positive}{negative}])\s*[:=]?\s*"
            r"([+-]?\d+(?:\.\d+)?)",
            r"([+-]?\d+(?:\.\d+)?)\s*°?\s*"
            rf"([{positive}{negative}])(?![A-Z])",
        )
        for text in texts:
            upper = text.upper().replace("−", "-").replace("–", "-")
            for pattern in patterns:
                for match in re.finditer(pattern, upper):
                    first, second = match.groups()
                    if first in {positive, negative}:
                        direction, number_text = first, second
                    else:
                        number_text, direction = first, second
                    try:
                        visible_value = float(number_text)
                    except ValueError:
                        continue
                    tolerance = max(1e-6, abs(value) * 1e-7)
                    if abs(abs(visible_value) - abs(value)) <= tolerance:
                        sign = -1.0 if direction == negative else 1.0
                        return sign * abs(value)
        return value

    return directed(lat, "N", "S"), directed(lon, "E", "W")


def recover_json(text: str) -> Optional[Dict[str, Any]]:
    """Recover the first JSON object from a model response."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(candidate)):
        character = candidate[index]
        if escaped:
            escaped = False
            continue
        if character == "\\" and quoted:
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(candidate[start : index + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def empty_state() -> Dict[str, Any]:
    return {
        "schema_version": "car_crash_segments_v1",
        "videos": {},
        "discovery": {},
    }


def save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(path, state)


def unload_model(owner: Any) -> None:
    for name in ("model", "processor", "tokenizer"):
        if hasattr(owner, name):
            try:
                delattr(owner, name)
            except Exception:
                pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
