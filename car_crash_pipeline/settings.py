"""Typed configuration loaded from ``config`` or ``default.config``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("CAR_CRASH_CONFIG", ROOT / "config"))
if not CONFIG_PATH.exists():
    CONFIG_PATH = ROOT / "default.config"

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    _CONFIG = json.load(handle)


def _value(name: str) -> Any:
    if name not in _CONFIG:
        raise ValueError(f"Missing configuration value: {name}")
    return _CONFIG[name]


def text(name: str) -> str:
    value = _value(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def optional_text(name: str) -> Optional[str]:
    value = _value(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value.strip() or None


def boolean(name: str) -> bool:
    value = _value(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def integer(name: str, minimum: int = 0, maximum: Optional[int] = None) -> int:
    value = _value(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def integer_or(
    name: str,
    default: int,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if name not in _CONFIG:
        value = default
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"Default {name} is outside the allowed range")
        return value
    return integer(name, minimum, maximum)


def number(
    name: str, minimum: float = 0.0, maximum: Optional[float] = None
) -> float:
    value = _value(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return result


def number_or(
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
) -> float:
    if name not in _CONFIG:
        value = float(default)
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"Default {name} is outside the allowed range")
        return value
    return number(name, minimum, maximum)


def text_list(name: str) -> List[str]:
    value = _value(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Every {name} value must be nonempty text")
        if item.strip() not in result:
            result.append(item.strip())
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def optional_text_list(name: str) -> List[str]:
    value = _value(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Every {name} value must be nonempty text")
        if item.strip() not in result:
            result.append(item.strip())
    return result


DATA_DIR = Path(text("DATA_DIR")).expanduser()
if not DATA_DIR.is_absolute():
    DATA_DIR = ROOT / DATA_DIR
STATE_JSON = DATA_DIR / text("STATE_JSON")
OUTPUT_CSV = DATA_DIR / text("OUTPUT_CSV")
_mapping_csv_name = _CONFIG.get("MAPPING_CSV", "mapping.csv")
if not isinstance(_mapping_csv_name, str) or not _mapping_csv_name.strip():
    raise ValueError("MAPPING_CSV must be a nonempty string")
MAPPING_CSV = DATA_DIR / _mapping_csv_name.strip()
VIDEO_DIR = DATA_DIR / text("VIDEO_DIR")
TEMP_DIR = VIDEO_DIR / ".tmp"
GEOCODE_CACHE = DATA_DIR / text("GEOCODE_CACHE")

DISCOVERY_QUERIES = text_list("DISCOVERY_QUERIES")
SEED_VIDEO_IDS = optional_text_list("SEED_VIDEO_IDS")
MAX_PAGES_PER_QUERY = integer("MAX_PAGES_PER_QUERY", 1)
RESULTS_PER_PAGE = integer("RESULTS_PER_PAGE", 1, 50)
MAX_NEW_CANDIDATES = integer("MAX_NEW_CANDIDATES", 1)
MAX_VIDEOS_PER_RUN = integer("MAX_VIDEOS_PER_RUN", 1)
CONTINUOUS_MODE = boolean("CONTINUOUS_MODE")
ACTIVE_PAUSE_SECONDS = integer("ACTIVE_PAUSE_SECONDS", 0)
IDLE_PAUSE_SECONDS = integer("IDLE_PAUSE_SECONDS", 1)

TEXT_MODEL = text("TEXT_MODEL")
TEXT_MODEL_4BIT = boolean("TEXT_MODEL_4BIT")
TEXT_MAX_NEW_TOKENS = integer("TEXT_MAX_NEW_TOKENS", 1)
MIN_TEXT_CONFIDENCE = number("MIN_TEXT_CONFIDENCE", 0.0, 1.0)

VISUAL_MODEL = text("VISUAL_MODEL")
VISUAL_MODEL_4BIT = boolean("VISUAL_MODEL_4BIT")
VISUAL_MAX_NEW_TOKENS = integer("VISUAL_MAX_NEW_TOKENS", 1)
VISUAL_DEVICE = text("VISUAL_DEVICE")
TEXT_DEVICE = text("TEXT_DEVICE")
MIN_CRASH_CONFIDENCE = number("MIN_CRASH_CONFIDENCE", 0.0, 1.0)
MIN_BOUNDARY_CONFIDENCE = number("MIN_BOUNDARY_CONFIDENCE", 0.0, 1.0)
SAMPLE_FRAME_COUNT = integer("SAMPLE_FRAME_COUNT", 4)
SAMPLE_MAX_FPS = number("SAMPLE_MAX_FPS", 0.1)
SAMPLE_WIDTH = integer("SAMPLE_WIDTH", 224)
BOUNDARY_CONTEXT_SECONDS = number("BOUNDARY_CONTEXT_SECONDS", 0.5)
LOCATION_SAMPLE_FRAME_COUNT = integer_or("LOCATION_SAMPLE_FRAME_COUNT", 16, 4)
LOCATION_SAMPLE_WIDTH = integer_or("LOCATION_SAMPLE_WIDTH", 960, 320)
LOCATION_SAMPLE_MAX_FPS = number_or("LOCATION_SAMPLE_MAX_FPS", 2.0, 0.1)
MIN_LOCATION_CONFIDENCE = max(
    0.90,
    number_or("MIN_LOCATION_CONFIDENCE", 0.90, 0.0, 1.0),
)
MAX_REVIEW_CYCLES = integer_or("MAX_REVIEW_CYCLES", 3, 1, 20)

CUT_BACKEND = text("CUT_BACKEND").lower()
if CUT_BACKEND not in {"auto", "ffmpeg_cuda", "ffmpeg_cpu"}:
    raise ValueError("CUT_BACKEND must be auto, ffmpeg_cuda or ffmpeg_cpu")
CUT_FPS = number("CUT_FPS", 1.0)
CUT_WIDTH = integer("CUT_WIDTH", 160)
SCENE_THRESHOLD = number("SCENE_THRESHOLD", 0.0, 1.0)
SCDET_THRESHOLD = number("SCDET_THRESHOLD", 0.0, 100.0)
CUT_MERGE_SECONDS = number("CUT_MERGE_SECONDS", 0.0)
CUT_TIMEOUT_SECONDS = integer("CUT_TIMEOUT_SECONDS", 1)
CPU_FALLBACK = boolean("CPU_FALLBACK")

VIDEO_FORMAT = text("VIDEO_FORMAT")
DOWNLOAD_RETRIES = integer("DOWNLOAD_RETRIES", 0)
DOWNLOAD_TIMEOUT_SECONDS = integer("DOWNLOAD_TIMEOUT_SECONDS", 1)
DELETE_VIDEO_AFTER_PROCESSING = boolean("DELETE_VIDEO_AFTER_PROCESSING")
COOKIE_FILE = optional_text("COOKIE_FILE")
if COOKIE_FILE:
    cookie_path = Path(COOKIE_FILE).expanduser()
    COOKIE_FILE = str(cookie_path if cookie_path.is_absolute() else ROOT / cookie_path)

ENABLE_GEOCODING = boolean("ENABLE_GEOCODING")
GEOCODER_USER_AGENT = text("GEOCODER_USER_AGENT")
GEOCODER_DELAY_SECONDS = number("GEOCODER_DELAY_SECONDS", 0.0)
