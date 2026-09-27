"""Regression tests for conditions that previously stalled the pipeline."""

from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from car_crash_pipeline.crash_review import (
    DOWNLOAD_UNAVAILABLE_STATUS,
    MIN_SAMPLE_FRAMES,
    VISUAL_FAILED_STATUS,
    _create_location_sample,
    _create_segment_sample,
    run_location_visual_stage,
    run_visual_stage,
)
from car_crash_pipeline.crash_taxonomy import (
    CRASH_TAXONOMY_VERSION,
    crash_taxonomy_pending,
    run_crash_taxonomy_stage,
)
from car_crash_pipeline.cut_detection import FullSegment
from car_crash_pipeline.location import (
    LOCATION_RESOLUTION_VERSION,
    MAX_GEOCODE_CLIENT_ERROR_ATTEMPTS,
    geocode,
    run_location_stage,
)
from car_crash_pipeline.pipeline import requires_processing, unfinished_count
from car_crash_pipeline import settings
from car_crash_pipeline.metadata_filter import run_text_stage
from car_crash_pipeline.shared import classify_download_failure


AGE_ERROR = "ERROR: [youtube] abc: Sorry, this content is age-restricted"


def _completed_record(**extra):
    record = {
        "status": "complete",
        "text_decision": {"include": True},
        "visual_review_version": "cosmos3_full_clip_crash_v3",
        "segments": [
            {
                "segment_index": 0,
                "start_time": 1.0,
                "end_time": 2.0,
                "duration_seconds": 1.0,
            }
        ],
    }
    record.update(extra)
    return record


def _sample_frame_count(command):
    video_filter = command[command.index("-vf") + 1]
    fps = float(video_filter.split(",", 1)[0].removeprefix("fps="))
    duration = float(command[command.index("-t") + 1])
    return fps * duration


class DownloadFailureClassificationTests(unittest.TestCase):
    def test_age_restriction_is_classified(self) -> None:
        self.assertEqual(classify_download_failure(AGE_ERROR), "age_restricted")
        self.assertEqual(
            classify_download_failure("Sign in to confirm your age."),
            "age_restricted",
        )

    def test_operational_failures_stay_retryable(self) -> None:
        self.assertIsNone(classify_download_failure("HTTP Error 429"))
        self.assertIsNone(
            classify_download_failure("Sign in to confirm you're not a bot")
        )


class TaxonomyDownloadStallTests(unittest.TestCase):
    def test_age_restricted_video_no_longer_blocks_the_queue(self) -> None:
        state = {"videos": {"ageLimited": _completed_record()}}

        with (
            patch(
                "car_crash_pipeline.crash_taxonomy._obtain_video",
                side_effect=RuntimeError(AGE_ERROR),
            ),
            patch("car_crash_pipeline.crash_taxonomy.save_state"),
        ):
            processed = run_crash_taxonomy_stage(state, max_videos=1)

        record = state["videos"]["ageLimited"]
        segment = record["segments"][0]
        self.assertEqual(processed, 1)
        self.assertEqual(record["crash_taxonomy_status"], "age_restricted")
        self.assertEqual(segment["crash_taxonomy_status"], "age_restricted")
        self.assertEqual(segment["crash_taxonomy_version"], CRASH_TAXONOMY_VERSION)
        self.assertFalse(crash_taxonomy_pending(record))
        self.assertFalse(requires_processing(record))
        self.assertEqual(unfinished_count(state), 0)

    def test_same_video_is_not_downloaded_again_after_failure(self) -> None:
        state = {"videos": {"ageLimited": _completed_record()}}

        with (
            patch(
                "car_crash_pipeline.crash_taxonomy._obtain_video",
                side_effect=RuntimeError(AGE_ERROR),
            ) as obtain_video,
            patch("car_crash_pipeline.crash_taxonomy.save_state"),
        ):
            run_crash_taxonomy_stage(state, max_videos=1)
            self.assertEqual(run_crash_taxonomy_stage(state, max_videos=1), 0)

        obtain_video.assert_called_once()


class VisualDownloadStallTests(unittest.TestCase):
    def test_age_restricted_new_video_is_final(self) -> None:
        record = {"status": "text_accepted", "text_decision": {"include": True}}
        state = {"videos": {"newVideo": record}}

        with (
            patch("car_crash_pipeline.crash_review.CosmosCrashJudge"),
            patch(
                "car_crash_pipeline.crash_review.download_video",
                side_effect=RuntimeError(AGE_ERROR),
            ),
            patch("car_crash_pipeline.crash_review.save_state"),
            patch("car_crash_pipeline.crash_review.unload_model"),
        ):
            run_visual_stage(state)

        self.assertEqual(record["status"], DOWNLOAD_UNAVAILABLE_STATUS)
        self.assertEqual(record["download_failure"], "age_restricted")
        self.assertFalse(requires_processing(record))

        with patch("car_crash_pipeline.crash_review.CosmosCrashJudge") as judge:
            self.assertEqual(run_visual_stage(state), 0)
        judge.assert_not_called()

    def test_any_failed_download_skips_the_video(self) -> None:
        record = {"status": "text_accepted", "text_decision": {"include": True}}
        state = {"videos": {"newVideo": record}}

        with (
            patch("car_crash_pipeline.crash_review.CosmosCrashJudge"),
            patch(
                "car_crash_pipeline.crash_review.download_video",
                side_effect=RuntimeError("HTTP Error 403: Forbidden"),
            ),
            patch("car_crash_pipeline.crash_review.save_state"),
            patch("car_crash_pipeline.crash_review.unload_model"),
        ):
            run_visual_stage(state)

        self.assertEqual(record["status"], DOWNLOAD_UNAVAILABLE_STATUS)
        self.assertEqual(record["download_failure"], "download_failed")
        self.assertFalse(requires_processing(record))

    def test_processing_failure_is_skipped_after_bounded_cycles(self) -> None:
        record = {"status": "text_accepted", "text_decision": {"include": True}}
        state = {"videos": {"brokenVideo": record}}

        with (
            patch("car_crash_pipeline.crash_review.CosmosCrashJudge"),
            patch(
                "car_crash_pipeline.crash_review.analyse_video",
                side_effect=RuntimeError("Cut detection failed"),
            ),
            patch("car_crash_pipeline.crash_review.save_state"),
            patch("car_crash_pipeline.crash_review.unload_model"),
        ):
            for _ in range(settings.MAX_REVIEW_CYCLES - 1):
                run_visual_stage(state)
                self.assertEqual(record["status"], "visual_error")
            run_visual_stage(state)
            self.assertEqual(run_visual_stage(state), 0)

        self.assertEqual(record["status"], VISUAL_FAILED_STATUS)
        self.assertFalse(requires_processing(record))

    def test_location_error_without_source_is_terminalised(self) -> None:
        record = _completed_record(downloaded_path="/missing/video.mp4")
        record["segments"][0]["location_visual_review"] = {"error": "model_error"}
        state = {"videos": {"gone": record}}
        self.assertTrue(requires_processing(record))

        with patch("car_crash_pipeline.crash_review.save_state"):
            run_location_visual_stage(state)

        review = record["segments"][0]["location_visual_review"]
        self.assertIsNone(review["error"])
        self.assertTrue(review["retry_exhausted"])


class TextStageStallTests(unittest.TestCase):
    def test_metadata_failure_is_skipped_after_bounded_cycles(self) -> None:
        record = {"status": "discovered", "metadata": {"title": "crash"}}
        state = {"videos": {"v": record}}

        with (
            patch("car_crash_pipeline.metadata_filter.TextMetadataJudge") as judge,
            patch("car_crash_pipeline.metadata_filter.save_state"),
            patch("car_crash_pipeline.metadata_filter.unload_model"),
        ):
            judge.return_value.judge.side_effect = RuntimeError("CUDA error")
            for _ in range(settings.MAX_REVIEW_CYCLES):
                run_text_stage(state)
            self.assertEqual(run_text_stage(state), 0)

        self.assertEqual(record["status"], "text_failed")
        self.assertFalse(record["text_decision"]["include"])
        self.assertFalse(requires_processing(record))


class ShortSampleTests(unittest.TestCase):
    def test_very_short_clips_get_enough_frames(self) -> None:
        segment = FullSegment(start_time=10.0, end_time=10.2)
        for create in (_create_segment_sample, _create_location_sample):
            with patch("car_crash_pipeline.crash_review._run_sample_command") as run:
                create(Path("source.mp4"), segment, Path("sample.mp4"))
            self.assertGreaterEqual(
                _sample_frame_count(run.call_args.args[0]),
                MIN_SAMPLE_FRAMES - 1e-6,
            )

    def test_long_clips_keep_the_configured_rate_cap(self) -> None:
        segment = FullSegment(start_time=0.0, end_time=600.0)
        with patch("car_crash_pipeline.crash_review._run_sample_command") as run:
            _, fps = _create_segment_sample(
                Path("source.mp4"), segment, Path("sample.mp4")
            )
        self.assertLess(fps, 1.0)


class GeocodeStallTests(unittest.TestCase):
    def test_ip_address_query_is_not_location_evidence(self) -> None:
        with patch("car_crash_pipeline.location._search_nominatim") as search:
            result = geocode({"_location_query": "122.122.122.122"}, {})
        search.assert_not_called()
        self.assertEqual(result["geocode_status"], "no_evidence")

    def test_repeated_client_error_becomes_terminal(self) -> None:
        segment = {
            "segment_index": 0,
            "start_time": 0.0,
            "end_time": 1.0,
            "embedded_location_text": ["Springfield"],
        }
        state = {"videos": {"v": {"status": "complete", "segments": [segment]}}}
        error = HTTPError("u", 418, "Unknown Error", {}, BytesIO())

        with (
            patch(
                "car_crash_pipeline.location._search_nominatim",
                side_effect=error,
            ),
            patch("car_crash_pipeline.location.save_state"),
            patch("car_crash_pipeline.location.write_json_atomic"),
            patch("car_crash_pipeline.location.load_json", return_value={}),
            patch("car_crash_pipeline.location.settings.ENABLE_GEOCODING", True),
        ):
            for _ in range(MAX_GEOCODE_CLIENT_ERROR_ATTEMPTS):
                self.assertEqual(run_location_stage(state), 1)
            self.assertEqual(run_location_stage(state), 0)

        location = segment["location"]
        self.assertEqual(location["geocode_status"], "failed_terminal")
        self.assertEqual(location["location_resolution_version"], LOCATION_RESOLUTION_VERSION)
        self.assertIn("418", location["geocode_rejection_reason"])

    def test_rate_limit_stays_retryable(self) -> None:
        segment = {
            "segment_index": 0,
            "start_time": 0.0,
            "end_time": 1.0,
            "embedded_location_text": ["Springfield"],
        }
        state = {"videos": {"v": {"status": "complete", "segments": [segment]}}}
        error = HTTPError("u", 429, "Too Many Requests", {}, BytesIO())

        with (
            patch(
                "car_crash_pipeline.location._search_nominatim",
                side_effect=error,
            ),
            patch("car_crash_pipeline.location.save_state"),
            patch("car_crash_pipeline.location.write_json_atomic"),
            patch("car_crash_pipeline.location.load_json", return_value={}),
            patch("car_crash_pipeline.location.settings.ENABLE_GEOCODING", True),
        ):
            for _ in range(MAX_GEOCODE_CLIENT_ERROR_ATTEMPTS + 1):
                self.assertEqual(run_location_stage(state), 1)

        self.assertTrue(segment["location"]["geocode_status"].startswith("failed:"))


if __name__ == "__main__":
    unittest.main()
