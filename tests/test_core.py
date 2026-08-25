"""Fast standard library tests that do not load models or access YouTube."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from car_crash_pipeline.crash_review import (
    _run_sample_command,
    extract_description_timestamps,
    has_visual_review_errors,
    normalise_crash_decision,
    timestamp_labels_for_segment,
)
from car_crash_pipeline.cut_detection import build_full_segments, merge_nearby_times
from car_crash_pipeline.location import run_location_stage
from car_crash_pipeline.output_writer import iter_rows
from car_crash_pipeline.shared import replace_file_with_retry
from car_crash_pipeline.youtube_discovery import YouTubeDiscovery


class CorePipelineTests(unittest.TestCase):
    def test_full_segments_have_no_duration_threshold(self) -> None:
        segments = build_full_segments(12.25, [0.10, 2.75, 12.20])
        self.assertEqual(
            [(item.start_time, item.end_time) for item in segments],
            [(0.0, 0.10), (0.10, 2.75), (2.75, 12.20), (12.20, 12.25)],
        )

    def test_no_cuts_keeps_the_complete_upload(self) -> None:
        segments = build_full_segments(91.7, [])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_time, 0.0)
        self.assertEqual(segments[0].end_time, 91.7)

    def test_nearby_cut_proposals_are_combined(self) -> None:
        self.assertEqual(merge_nearby_times([1.0, 1.2, 4.0], 0.4), [1.1, 4.0])

    def test_crash_decision_drops_unsupported_location(self) -> None:
        decision = normalise_crash_decision(
            {
                "is_crash": True,
                "confidence": 0.95,
                "impact_time_seconds": 2.4,
                "locality": "Invented City",
                "location_evidence": "none",
                "road_users": ["car", "car"],
            },
            "{}",
            4.0,
        )
        self.assertTrue(decision.is_crash)
        self.assertEqual(decision.impact_time_seconds, 2.4)
        self.assertIsNone(decision.locality)
        self.assertEqual(decision.road_users, ["car"])

    def test_invalid_impact_time_is_removed(self) -> None:
        decision = normalise_crash_decision(
            {"is_crash": True, "confidence": 0.9, "impact_time_seconds": 20},
            "{}",
            5.0,
        )
        self.assertIsNone(decision.impact_time_seconds)

    def test_output_has_one_row_per_accepted_segment(self) -> None:
        state = {
            "videos": {
                "abc": {
                    "status": "complete",
                    "visual_review_version": "test",
                    "metadata": {"youtube_url": "https://youtu.be/abc"},
                    "segments": [
                        {"segment_index": 0, "start_time": 0.0, "end_time": 1.0},
                        {"segment_index": 4, "start_time": 8.0, "end_time": 20.0},
                    ],
                }
            }
        }
        rows = list(iter_rows(state))
        self.assertEqual(
            [row["segment_id"] for row in rows],
            ["abc_00000", "abc_00004"],
        )

    def test_author_timestamp_labels_follow_full_segments(self) -> None:
        labels = extract_description_timestamps(
            "00:00 Intro\n01:05 First crash\n1:02:03 Later crash"
        )
        self.assertEqual([item["timestamp_seconds"] for item in labels], [0, 65, 3723])
        selected = timestamp_labels_for_segment(labels, 70.0, 100.0)
        self.assertEqual([item["label"] for item in selected], ["First crash"])

    def test_full_seed_batch_skips_search(self) -> None:
        state = {"videos": {}}
        seed_payload = {
            "items": [
                {
                    "id": "jJXT2zGlSc0",
                    "snippet": {"title": "Crash compilation"},
                    "contentDetails": {"duration": "PT1H1M19S"},
                }
            ]
        }
        with (
            patch("car_crash_pipeline.youtube_discovery.settings.SEED_VIDEO_IDS", ["jJXT2zGlSc0"]),
            patch("car_crash_pipeline.youtube_discovery.settings.MAX_NEW_CANDIDATES", 1),
            patch("car_crash_pipeline.youtube_discovery.save_state"),
            patch(
                "car_crash_pipeline.youtube_discovery._request",
                return_value=seed_payload,
            ) as request,
        ):
            discovered = YouTubeDiscovery(["test-key"]).discover(state)

        self.assertEqual(discovered, 1)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0], "videos")

    def test_saved_visual_errors_are_retryable(self) -> None:
        self.assertTrue(
            has_visual_review_errors(
                {"segment_reviews": [{"segment_index": 4, "error": "sample_error"}]}
            )
        )
        self.assertFalse(
            has_visual_review_errors(
                {"segment_reviews": [{"segment_index": 4, "error": None}]}
            )
        )

    def test_sample_creation_retries_transient_ffmpeg_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "sample.mp4"
            attempts = 0

            def run(command, timeout):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    return SimpleNamespace(returncode=1, stderr="temporarily locked", stdout="")
                destination.write_bytes(b"video")
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            with (
                patch("car_crash_pipeline.crash_review.run_command", side_effect=run),
                patch("car_crash_pipeline.crash_review.time.sleep"),
            ):
                _run_sample_command(
                    ["ffmpeg"], destination, timeout=10, label="sample failed"
                )

        self.assertEqual(attempts, 2)

    def test_atomic_replace_retries_windows_lock(self) -> None:
        with (
            patch(
                "car_crash_pipeline.shared.os.replace",
                side_effect=[PermissionError("locked"), None],
            ) as replace,
            patch("car_crash_pipeline.shared.time.sleep"),
        ):
            replace_file_with_retry(Path("source"), Path("destination"))
        self.assertEqual(replace.call_count, 2)

    def test_location_stage_saves_once_for_multiple_segments(self) -> None:
        state = {
            "videos": {
                "abc": {
                    "status": "complete",
                    "text_decision": {},
                    "segments": [
                        {"location_evidence": "none"},
                        {"location_evidence": "none"},
                    ],
                }
            }
        }
        with (
            patch("car_crash_pipeline.location.load_json", return_value={}),
            patch("car_crash_pipeline.location.write_json_atomic") as write_cache,
            patch("car_crash_pipeline.location.save_state") as save,
        ):
            processed = run_location_stage(state)

        self.assertEqual(processed, 2)
        self.assertEqual(write_cache.call_count, 1)
        self.assertEqual(save.call_count, 1)


if __name__ == "__main__":
    unittest.main()
