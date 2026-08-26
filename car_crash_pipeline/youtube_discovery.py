"""YouTube Data API discovery without an upload duration restriction."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlencode
from urllib.request import urlopen

from . import settings
from .shared import clean_text, log, save_state


API_ROOT = "https://www.googleapis.com/youtube/v3"


def load_api_keys() -> List[str]:
    path = settings.ROOT / "secret"
    if not path.exists():
        raise RuntimeError("Create secret from default.secret and add a YouTube API key")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    keys = payload.get("YOUTUBE_API_KEYS", [])
    if not isinstance(keys, list):
        raise RuntimeError("YOUTUBE_API_KEYS must be a JSON list")
    result = [clean_text(value) for value in keys if clean_text(value)]
    if not result or result[0].startswith("replace-with"):
        raise RuntimeError("No usable YouTube API key was configured")
    return result


def _request(path: str, parameters: Dict[str, Any], api_keys: Iterable[str]) -> Dict[str, Any]:
    failures = []
    for api_key in api_keys:
        query = urlencode({**parameters, "key": api_key})
        try:
            with urlopen(f"{API_ROOT}/{path}?{query}", timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            failures.append(str(exc))
    raise RuntimeError("All YouTube API keys failed: " + " | ".join(failures))


def _chunks(values: List[str], size: int = 50) -> Iterable[List[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


class YouTubeDiscovery:
    def __init__(self, api_keys: List[str]) -> None:
        self.api_keys = api_keys

    def discover(self, state: Dict[str, Any]) -> int:
        existing = state.setdefault("videos", {})
        discovery_state = state.setdefault("discovery", {})
        query_cursors = discovery_state.get("query_cursors")
        if not isinstance(query_cursors, dict):
            query_cursors = {}

        candidate_ids: List[str] = []
        overflow_ids: List[str] = []
        queued_ids = discovery_state.get("pending_candidate_ids", [])
        if not isinstance(queued_ids, list):
            queued_ids = []

        def add_candidate(video_id: Any) -> None:
            value = clean_text(video_id)
            if (
                not re.fullmatch(r"[A-Za-z0-9_-]{11}", value)
                or value in existing
                or value in candidate_ids
                or value in overflow_ids
            ):
                return
            if len(candidate_ids) < settings.MAX_NEW_CANDIDATES:
                candidate_ids.append(value)
            else:
                overflow_ids.append(value)

        for video_id in settings.SEED_VIDEO_IDS:
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                raise ValueError(f"Invalid YouTube video ID in SEED_VIDEO_IDS: {video_id}")
            add_candidate(video_id)
        for video_id in queued_ids:
            add_candidate(video_id)
        queries_used: List[str] = []

        for query in settings.DISCOVERY_QUERIES:
            if len(candidate_ids) >= settings.MAX_NEW_CANDIDATES:
                break
            cursor = query_cursors.get(query)
            if not isinstance(cursor, dict):
                cursor = {}
            token = clean_text(cursor.get("next_page_token")) or None
            pages_fetched = cursor.get("pages_fetched", 0)
            if not isinstance(pages_fetched, int) or pages_fetched < 0:
                pages_fetched = 0
            queries_used.append(query)
            for _ in range(settings.MAX_PAGES_PER_QUERY):
                parameters: Dict[str, Any] = {
                    "part": "id",
                    "type": "video",
                    "q": query,
                    "maxResults": settings.RESULTS_PER_PAGE,
                    "safeSearch": "none",
                }
                if token:
                    parameters["pageToken"] = token
                payload = _request("search", parameters, self.api_keys)
                for item in payload.get("items", []):
                    video_id = item.get("id", {}).get("videoId")
                    add_candidate(video_id)
                token = clean_text(payload.get("nextPageToken")) or None
                pages_fetched += 1
                query_cursors[query] = {
                    "next_page_token": token,
                    "pages_fetched": pages_fetched,
                    "exhausted": token is None,
                }
                if len(candidate_ids) >= settings.MAX_NEW_CANDIDATES:
                    break
                if not token:
                    break
            if len(candidate_ids) >= settings.MAX_NEW_CANDIDATES:
                break

        now = datetime.now(timezone.utc).isoformat()
        for batch in _chunks(candidate_ids):
            payload = _request(
                "videos",
                {
                    "part": "snippet,contentDetails",
                    "id": ",".join(batch),
                    "maxResults": 50,
                },
                self.api_keys,
            )
            for item in payload.get("items", []):
                video_id = item.get("id")
                snippet = item.get("snippet", {})
                details = item.get("contentDetails", {})
                if not video_id or video_id in existing:
                    continue
                existing[video_id] = {
                    "status": "discovered",
                    "discovered_at": now,
                    "metadata": {
                        "title": snippet.get("title"),
                        "description": snippet.get("description"),
                        "channel": snippet.get("channelTitle"),
                        "channel_id": snippet.get("channelId"),
                        "upload_date": snippet.get("publishedAt"),
                        "duration_iso8601": details.get("duration"),
                        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                    },
                }

        state["discovery"] = {
            "last_completed_at": now,
            "queries": queries_used,
            "new_candidates": len(candidate_ids),
            "pending_candidate_ids": overflow_ids,
            "query_cursors": query_cursors,
        }
        save_state(settings.STATE_JSON, state)
        log(f"Discovered {len(candidate_ids)} new videos")
        return len(candidate_ids)
