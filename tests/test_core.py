"""Fast standard library tests that do not load models or access YouTube."""

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from car_crash_pipeline.crash_review import (
    CosmosCrashJudge,
    LocationVisualDecision,
    _apply_location_visual_decision,
    _create_location_sample,
    _create_segment_sample,
    _delete_downloaded_video_when_finished,
    _run_sample_command,
    _needs_location_visual_review,
    _restore_pre_location_visual_evidence,
    _set_visual_review_status,
    _store_location_visual_decision,
    extract_description_timestamps,
    has_visual_review_errors,
    has_location_visual_review_errors,
    is_accepted_crash_review,
    invalid_location_visual_decision,
    normalise_location_visual_decision,
    normalise_crash_decision,
    run_location_visual_stage,
    validate_location_visual_response,
    validate_crash_response,
    timestamp_labels_for_segment,
)
from car_crash_pipeline.cut_detection import (
    FullSegment,
    build_full_segments,
    merge_nearby_times,
)
from car_crash_pipeline.location import (
    LOCATION_RESOLUTION_VERSION,
    _iso3,
    _location_candidates,
    geocode,
    run_location_stage,
)
from car_crash_pipeline.output_writer import (
    MAPPING_COLUMNS,
    bracket_cell,
    iter_mapping_rows,
    iter_rows,
    write_output_csv,
)
from car_crash_pipeline.pipeline import (
    cycle_pause_seconds,
    discover_when_ready,
    requires_processing,
)
from car_crash_pipeline.shared import replace_file_with_retry
from car_crash_pipeline.youtube_discovery import YouTubeDiscovery


class CorePipelineTests(unittest.TestCase):
    def test_youtube_rate_limit_defers_discovery_without_crashing(self) -> None:
        state = {"videos": {}, "discovery": {}}
        with (
            patch(
                "car_crash_pipeline.pipeline.load_api_keys",
                return_value=["test-key"],
            ),
            patch(
                "car_crash_pipeline.pipeline.YouTubeDiscovery.discover",
                side_effect=RuntimeError("HTTP Error 429: Too Many Requests"),
            ),
            patch("car_crash_pipeline.pipeline.save_state") as save,
        ):
            discovered = discover_when_ready(state)

        self.assertEqual(discovered, 0)
        self.assertIn("429", state["discovery"]["last_error"])
        save.assert_called_once()

    def test_unfinished_location_work_uses_active_pause(self) -> None:
        with (
            patch(
                "car_crash_pipeline.pipeline.settings.ACTIVE_PAUSE_SECONDS", 5
            ),
            patch(
                "car_crash_pipeline.pipeline.settings.IDLE_PAUSE_SECONDS", 900
            ),
        ):
            self.assertEqual(cycle_pause_seconds(0, 0, 0, 1), 5)
            self.assertEqual(cycle_pause_seconds(0, 0, 0, 0), 900)

    def test_segment_prompt_does_not_include_video_description(self) -> None:
        prompt = CosmosCrashJudge._segment_prompt(
            {
                "title": "Crash compilation",
                "description": "PRIVATE_DESCRIPTION_MUST_NOT_LEAK",
            },
            12.0,
        )

        self.assertIn("Crash compilation", prompt)
        self.assertNotIn("PRIVATE_DESCRIPTION_MUST_NOT_LEAK", prompt)

    def test_persistent_visual_error_becomes_terminal_warning(self) -> None:
        record = {
            "visual_retry_cycles": 2,
            "boundary_reviews": [],
            "segment_reviews": [
                {
                    "segment_index": 75,
                    "error": "json_recovery_failed",
                }
            ],
        }

        with patch(
            "car_crash_pipeline.crash_review.settings.MAX_REVIEW_CYCLES", 3
        ):
            warning = _set_visual_review_status(
                record,
                accepted_count=223,
                boundary_error_count=0,
                segment_error_count=1,
            )

        review = record["segment_reviews"][0]
        self.assertEqual(record["status"], "complete")
        self.assertIsNone(record["error"])
        self.assertTrue(review["retry_exhausted"])
        self.assertIsNone(review["error"])
        self.assertEqual(review["terminal_error"], "json_recovery_failed")
        self.assertIn("Skipped 1", warning)
        self.assertFalse(has_visual_review_errors(record))

    def test_persistent_location_error_becomes_unknown_terminal_review(self) -> None:
        segment = {}
        decision = invalid_location_visual_decision(
            "unfinished JSON", "json_recovery_failed"
        )

        with patch(
            "car_crash_pipeline.crash_review.settings.MAX_REVIEW_CYCLES", 3
        ):
            exhausted = _store_location_visual_decision(
                segment, decision, previous_cycles=2
            )

        review = segment["location_visual_review"]
        self.assertTrue(exhausted)
        self.assertTrue(review["retry_exhausted"])
        self.assertIsNone(review["error"])
        self.assertEqual(review["terminal_error"], "json_recovery_failed")
        self.assertFalse(_needs_location_visual_review(segment))

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

    def test_low_confidence_does_not_overwrite_model_boolean(self) -> None:
        decision = normalise_crash_decision(
            {
                "is_crash": True,
                "confidence": 0.4,
                "crash_type": "near_collision",
                "location_evidence": "none",
            },
            "{}",
            5.0,
        )

        self.assertTrue(decision.is_crash)
        self.assertFalse(is_accepted_crash_review(decision.__dict__))

    def test_contradictory_near_collision_is_invalid(self) -> None:
        error = validate_crash_response(
            {
                "is_crash": False,
                "confidence": 0.9,
                "impact_time_seconds": None,
                "crash_type": "near_collision",
                "location_evidence": "none",
            }
        )

        self.assertEqual(error, "non_crash_response_must_use_unknown_crash_type")

    def test_invalid_location_enum_does_not_reject_crash_decision(self) -> None:
        data = {
            "is_crash": True,
            "confidence": 0.95,
            "impact_time_seconds": None,
            "crash_type": "near_collision",
            "location_evidence": "metadata|embedded_text|both|none",
            "locality": "Unsupported place",
        }

        self.assertIsNone(validate_crash_response(data))
        decision = normalise_crash_decision(data, json.dumps(data), 10.0)
        self.assertTrue(decision.is_crash)
        self.assertEqual(decision.location_evidence, "none")
        self.assertIsNone(decision.locality)

    def test_zero_confidence_saved_review_is_retryable(self) -> None:
        raw_response = json.dumps(
            {
                "is_crash": False,
                "confidence": 0.0,
                "impact_time_seconds": None,
                "crash_type": "near_collision",
                "location_evidence": "none",
            }
        )

        self.assertTrue(
            has_visual_review_errors(
                {
                    "segment_reviews": [
                        {
                            "segment_index": 16,
                            "confidence": 0.0,
                            "raw_response": raw_response,
                            "error": None,
                        }
                    ]
                }
            )
        )

    def test_segment_review_retries_semantic_contradiction(self) -> None:
        invalid = json.dumps(
            {
                "is_crash": False,
                "confidence": 0.0,
                "impact_time_seconds": None,
                "crash_type": "near_collision",
                "location_evidence": "metadata|embedded_text|both|none",
            }
        )
        corrected = json.dumps(
            {
                "is_crash": True,
                "confidence": 0.95,
                "impact_time_seconds": None,
                "short_description": "A vehicle swerves to avoid impact.",
                "crash_type": "near_collision",
                "camera_view": "dashcam",
                "location_evidence": "none",
            }
        )
        judge = object.__new__(CosmosCrashJudge)

        with patch.object(judge, "_generate", side_effect=[invalid, corrected]) as generate:
            decision = judge.review_segment(
                Path("sample.mp4"),
                {"title": "Crash compilation", "description": ""},
                FullSegment(start_time=0.0, end_time=10.0),
                2.4,
            )

        self.assertEqual(generate.call_count, 2)
        self.assertTrue(decision.is_crash)
        self.assertEqual(decision.confidence, 0.95)
        self.assertIsNone(decision.error)

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

    def test_mapping_output_has_requested_segment_schema(self) -> None:
        state = {
            "videos": {
                "abc": {
                    "status": "complete",
                    "segments": [
                        {
                            "segment_index": 4,
                            "start_time": 8.25,
                            "end_time": 20.5,
                            "time_of_day": "day",
                            "road_users": ["car", "truck"],
                            "location": {
                                "locality": "Toronto",
                                "state": "Ontario",
                                "country": "Canada",
                                "iso3": "CAN",
                                "continent": "North America",
                                "lat": 43.6534817,
                                "lon": -79.3839347,
                            },
                        }
                    ],
                }
            }
        }

        self.assertEqual(
            MAPPING_COLUMNS,
            [
                "id",
                "locality",
                "locality_aka",
                "state",
                "country",
                "iso3",
                "continent",
                "lat",
                "lon",
                "videos",
                "time_of_day",
                "start_time",
                "end_time",
                "vehicle_type",
            ],
        )
        row = next(iter(iter_mapping_rows(state)))
        self.assertEqual(row["id"], 1)
        self.assertEqual(row["locality"], "Toronto")
        self.assertEqual(row["videos"], "[abc]")
        self.assertEqual(row["vehicle_type"], "[car,truck]")
        self.assertEqual(row["time_of_day"], "[day]")
        self.assertEqual(row["start_time"], "[8.25]")
        self.assertEqual(row["end_time"], "[20.5]")

    def test_mapping_groups_alexandria_without_combining_segment_ranges(self) -> None:
        alexandria = {
            "locality": "Alexandria",
            "state": "Virginia",
            "country": "United States",
            "iso3": "USA",
            "continent": "North America",
            "lat": 38.8408718,
            "lon": -77.1144703,
        }
        state = {
            "videos": {
                "jJXT2zGlSc0": {
                    "status": "complete",
                    "segments": [
                        {
                            "start_time": 446.833,
                            "end_time": 455.167,
                            "time_of_day": "unknown",
                            "road_users": ["car"],
                            "location": dict(alexandria),
                        },
                        {
                            "start_time": 455.167,
                            "end_time": 463.5,
                            "time_of_day": "dawn_dusk",
                            "road_users": ["car"],
                            "location": dict(alexandria),
                        },
                    ],
                }
            }
        }

        rows = list(iter_mapping_rows(state))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["locality"], "Alexandria")
        self.assertEqual(rows[0]["videos"], "[jJXT2zGlSc0]")
        self.assertEqual(rows[0]["time_of_day"], "[unknown,dusk/dawn]")
        self.assertEqual(rows[0]["start_time"], "[446.833,455.167]")
        self.assertEqual(rows[0]["end_time"], "[455.167,463.5]")
        self.assertEqual(rows[0]["vehicle_type"], "[car,car]")

    def test_mapping_merges_state_name_and_code(self) -> None:
        common = {
            "locality": "Surfside Beach",
            "country": "United States",
            "iso3": "USA",
            "continent": "North America",
            "lat": 33.6060031,
            "lon": -78.9730887,
        }
        state = {
            "videos": {
                "jJXT2zGlSc0": {
                    "status": "complete",
                    "segments": [
                        {
                            "start_time": 1064.667,
                            "end_time": 1077.833,
                            "time_of_day": "day",
                            "road_users": ["car"],
                            "location": {**common, "state": "SC"},
                        },
                        {
                            "start_time": 1077.833,
                            "end_time": 1082.833,
                            "time_of_day": "day",
                            "road_users": ["car"],
                            "location": {**common, "state": "South Carolina"},
                        },
                    ],
                }
            }
        }

        rows = list(iter_mapping_rows(state))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["locality"], "Surfside Beach")
        self.assertEqual(rows[0]["state"], "SC")
        self.assertEqual(rows[0]["time_of_day"], "[day,day]")
        self.assertEqual(rows[0]["start_time"], "[1064.667,1077.833]")
        self.assertEqual(rows[0]["end_time"], "[1077.833,1082.833]")
        self.assertEqual(rows[0]["vehicle_type"], "[car,car]")

    def test_mapping_collapses_unknown_places_into_one_row(self) -> None:
        state = {
            "videos": {
                "abc": {
                    "status": "complete",
                    "segments": [
                        {
                            "start_time": 0.0,
                            "end_time": 5.0,
                            "time_of_day": "day",
                            "road_users": ["car"],
                            "location": {},
                        },
                        {
                            "start_time": 20.0,
                            "end_time": 30.0,
                            "time_of_day": "night",
                            "road_users": ["truck"],
                            "location": {},
                        },
                    ],
                },
                "def": {
                    "status": "complete",
                    "segments": [
                        {
                            "start_time": 4.0,
                            "end_time": 9.0,
                            "time_of_day": "unknown",
                            "road_users": ["car"],
                            "location": {},
                        }
                    ],
                },
            }
        }

        rows = list(iter_mapping_rows(state))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["locality"], "unknown")
        self.assertEqual(rows[0]["videos"], "[abc,def]")
        self.assertEqual(rows[0]["time_of_day"], "[[day,night],[unknown]]")
        self.assertEqual(rows[0]["start_time"], "[[0.0,20.0],[4.0]]")
        self.assertEqual(rows[0]["end_time"], "[[5.0,30.0],[9.0]]")
        self.assertEqual(rows[0]["vehicle_type"], "[car,truck,car]")

    def test_mapping_bracket_cells_do_not_quote_list_items(self) -> None:
        self.assertEqual(bracket_cell(["jJXT2zGlSc0"]), "[jJXT2zGlSc0]")
        self.assertEqual(bracket_cell(["car", "truck"]), "[car,truck]")

    def test_mapping_csv_is_written_with_one_row_per_locality(self) -> None:
        state = {
            "videos": {
                "abc": {
                    "status": "complete",
                    "segments": [
                        {
                            "segment_index": 2,
                            "start_time": 3.5,
                            "end_time": 9.25,
                            "time_of_day": "night",
                            "road_users": ["car"],
                            "location": {},
                        }
                    ],
                }
            }
        }
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "crash_segments.csv"
            mapping = Path(temporary) / "mapping.csv"
            with (
                patch("car_crash_pipeline.output_writer.settings.OUTPUT_CSV", output),
                patch("car_crash_pipeline.output_writer.settings.MAPPING_CSV", mapping),
            ):
                write_output_csv(state)

            with mapping.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, MAPPING_COLUMNS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "1")
        self.assertEqual(rows[0]["locality"], "unknown")
        self.assertEqual(rows[0]["videos"], "[abc]")

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

    def test_discovery_resumes_cursor_and_preserves_page_overflow(self) -> None:
        state = {"videos": {}, "discovery": {}}
        search_tokens = []

        def request(path, parameters, api_keys):
            if path == "search":
                search_tokens.append(parameters.get("pageToken"))
                if parameters.get("pageToken") == "PAGE2":
                    return {
                        "items": [{"id": {"videoId": "ccccccccccc"}}],
                        "nextPageToken": "PAGE3",
                    }
                return {
                    "items": [
                        {"id": {"videoId": "aaaaaaaaaaa"}},
                        {"id": {"videoId": "bbbbbbbbbbb"}},
                    ],
                    "nextPageToken": "PAGE2",
                }
            video_ids = parameters["id"].split(",")
            return {
                "items": [
                    {
                        "id": video_id,
                        "snippet": {"title": f"Crash {video_id}"},
                        "contentDetails": {"duration": "PT1M"},
                    }
                    for video_id in video_ids
                ]
            }

        with (
            patch("car_crash_pipeline.youtube_discovery.settings.SEED_VIDEO_IDS", []),
            patch(
                "car_crash_pipeline.youtube_discovery.settings.DISCOVERY_QUERIES",
                ["car crash"],
            ),
            patch(
                "car_crash_pipeline.youtube_discovery.settings.MAX_NEW_CANDIDATES",
                1,
            ),
            patch(
                "car_crash_pipeline.youtube_discovery.settings.MAX_PAGES_PER_QUERY",
                1,
            ),
            patch(
                "car_crash_pipeline.youtube_discovery.settings.RESULTS_PER_PAGE",
                50,
            ),
            patch("car_crash_pipeline.youtube_discovery.save_state"),
            patch(
                "car_crash_pipeline.youtube_discovery._request",
                side_effect=request,
            ),
        ):
            discovery = YouTubeDiscovery(["test-key"])
            self.assertEqual(discovery.discover(state), 1)
            self.assertEqual(discovery.discover(state), 1)
            self.assertEqual(discovery.discover(state), 1)

        self.assertEqual(
            set(state["videos"]),
            {"aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"},
        )
        self.assertEqual(search_tokens, [None, "PAGE2"])
        self.assertEqual(
            state["discovery"]["query_cursors"]["car crash"][
                "next_page_token"
            ],
            "PAGE3",
        )

    def test_location_review_error_blocks_next_discovery_batch(self) -> None:
        record = {
            "status": "complete",
            "text_decision": {"include": True},
            "visual_review_version": "cosmos3_full_clip_crash_v3",
            "segments": [
                {
                    "location_visual_review": {
                        "error": "model_error",
                    }
                }
            ],
        }

        self.assertTrue(requires_processing(record))

    def test_saved_visual_errors_are_retryable(self) -> None:
        self.assertTrue(
            has_visual_review_errors(
                {"segment_reviews": [{"segment_index": 4, "error": "sample_error"}]}
            )
        )
        self.assertFalse(
            has_visual_review_errors(
                {
                    "segment_reviews": [
                        {
                            "segment_index": 4,
                            "error": None,
                            "raw_response": json.dumps(
                                {
                                    "is_crash": False,
                                    "confidence": 0.95,
                                    "impact_time_seconds": None,
                                    "crash_type": "unknown",
                                    "location_evidence": "none",
                                }
                            ),
                        }
                    ]
                }
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

    def test_segment_sample_uses_mp4_safe_timestamps(self) -> None:
        with patch("car_crash_pipeline.crash_review._run_sample_command") as run:
            _create_segment_sample(
                Path("source.mp4"),
                FullSegment(start_time=25.333, end_time=43.666),
                Path("sample.mp4"),
            )

        command = run.call_args.args[0]
        video_filter = command[command.index("-vf") + 1]
        self.assertIn("setpts=PTS-STARTPTS", video_filter)
        self.assertEqual(command[command.index("-bf") + 1], "0")

    def test_location_sample_uses_high_resolution_and_safe_timestamps(self) -> None:
        with patch("car_crash_pipeline.crash_review._run_sample_command") as run:
            _create_location_sample(
                Path("source.mp4"),
                FullSegment(start_time=73.5, end_time=91.167),
                Path("location.mp4"),
            )

        command = run.call_args.args[0]
        video_filter = command[command.index("-vf") + 1]
        self.assertIn("scale=960:-2", video_filter)
        self.assertIn("setpts=PTS-STARTPTS", video_filter)
        self.assertEqual(command[command.index("-bf") + 1], "0")

    def test_location_review_reads_tulsa_overlay(self) -> None:
        answer = json.dumps(
            {
                "location_found": True,
                "confidence": 0.98,
                "locality": "Tulsa",
                "locality_aka": [],
                "state": "Oklahoma",
                "country": None,
                "lat": None,
                "lon": None,
                "visible_location_text": ["Tulsa, Oklahoma"],
            }
        )
        judge = object.__new__(CosmosCrashJudge)

        with patch.object(judge, "_generate", return_value=answer):
            decision = judge.review_location(Path("location.mp4"), 0.9)

        self.assertTrue(decision.location_found)
        self.assertEqual(decision.locality, "Tulsa")
        self.assertEqual(decision.state, "Oklahoma")
        self.assertIsNone(decision.error)

    def test_location_prompt_contains_no_real_place_example(self) -> None:
        prompt = CosmosCrashJudge._location_prompt()

        self.assertNotIn("Tulsa", prompt)
        self.assertNotIn("Oklahoma", prompt)
        self.assertIn('"location_found": false', prompt)

    def test_location_response_must_match_visible_text(self) -> None:
        error = validate_location_visual_response(
            {
                "location_found": True,
                "confidence": 0.98,
                "locality": "Different City",
                "state": None,
                "country": None,
                "lat": None,
                "lon": None,
                "visible_location_text": ["Readable City"],
            }
        )

        self.assertEqual(
            error, "structured_location_must_appear_in_visible_text"
        )

    def test_location_response_accepts_visible_coordinate_pair(self) -> None:
        error = validate_location_visual_response(
            {
                "location_found": True,
                "confidence": 0.98,
                "locality": None,
                "state": None,
                "country": None,
                "lat": 36.1563122,
                "lon": -95.9927516,
                "visible_location_text": ["36.1563122, -95.9927516"],
            }
        )

        self.assertIsNone(error)

    def test_location_response_applies_south_and_east_hemispheres(self) -> None:
        data = {
            "location_found": True,
            "confidence": 0.98,
            "locality": None,
            "locality_aka": [],
            "state": None,
            "country": None,
            "lat": 33.8187,
            "lon": 150.9447,
            "visible_location_text": ["S:33.8187 E:150.9447"],
        }

        self.assertIsNone(validate_location_visual_response(data))
        decision = normalise_location_visual_decision(data, "{}")

        self.assertEqual(decision.lat, -33.8187)
        self.assertEqual(decision.lon, 150.9447)

    def test_location_response_applies_compact_hemisphere_prefixes(self) -> None:
        decision = normalise_location_visual_decision(
            {
                "location_found": True,
                "confidence": 0.98,
                "locality": "E144.7032,S37.8500",
                "locality_aka": [],
                "state": None,
                "country": None,
                "lat": 37.85,
                "lon": 144.7032,
                "visible_location_text": ["E144.7032,S37.8500"],
            },
            "{}",
        )

        self.assertEqual(decision.lat, -37.85)
        self.assertEqual(decision.lon, 144.7032)

    def test_location_response_rejects_coordinates_not_in_visible_text(self) -> None:
        error = validate_location_visual_response(
            {
                "location_found": True,
                "confidence": 0.98,
                "locality": None,
                "state": None,
                "country": None,
                "lat": 36.1563122,
                "lon": -95.9927516,
                "visible_location_text": ["Coordinates unreadable"],
            }
        )

        self.assertEqual(
            error, "coordinate_pair_must_appear_in_visible_text"
        )

    def test_location_decision_updates_segment_for_geocoding(self) -> None:
        segment = {
            "location_evidence": "none",
            "embedded_location_text": [],
            "location": {"geocode_status": "not_found"},
        }
        decision = LocationVisualDecision(
            location_found=True,
            confidence=0.98,
            locality="Tulsa",
            locality_aka=[],
            state="Oklahoma",
            country=None,
            lat=None,
            lon=None,
            visible_location_text=["Tulsa, Oklahoma"],
            raw_response="{}",
        )

        _apply_location_visual_decision(segment, decision)

        self.assertEqual(segment["locality"], "Tulsa")
        self.assertEqual(segment["state"], "Oklahoma")
        self.assertEqual(segment["location_evidence"], "embedded_text")
        self.assertNotIn("location", segment)

    def test_low_confidence_location_is_not_applied(self) -> None:
        segment = {
            "location_evidence": "none",
            "embedded_location_text": [],
        }
        decision = LocationVisualDecision(
            location_found=True,
            confidence=0.80,
            locality="Uncertain City",
            locality_aka=[],
            state="Uncertain Region",
            country=None,
            lat=None,
            lon=None,
            visible_location_text=["Uncertain City, Uncertain Region"],
            raw_response="{}",
        )

        with patch(
            "car_crash_pipeline.crash_review.settings.MIN_LOCATION_CONFIDENCE",
            0.90,
        ):
            _apply_location_visual_decision(segment, decision)

        self.assertNotIn("locality", segment)
        self.assertNotIn("location", segment)

    def test_contaminated_v1_location_is_forced_to_retry(self) -> None:
        segment = {
            "locality": "Copied City",
            "location_visual_review_version": "cosmos3_location_text_v1",
            "location_visual_review": {"error": None},
            "location": {
                "locality": "Copied City",
                "geocode_status": "resolved",
            },
        }

        self.assertTrue(_needs_location_visual_review(segment))

    def test_contaminated_location_is_removed_before_retry(self) -> None:
        raw_response = json.dumps(
            {
                "is_crash": True,
                "confidence": 0.95,
                "crash_type": "near_collision",
                "location_evidence": "none",
                "embedded_location_text": [],
            }
        )
        segment = {
            "raw_response": raw_response,
            "locality": "Copied City",
            "state": "Copied Region",
            "location_evidence": "embedded_text",
            "embedded_location_text": ["Copied City, Copied Region"],
            "location": {
                "locality": "Copied City",
                "geocode_status": "resolved",
            },
        }

        _restore_pre_location_visual_evidence(segment)

        self.assertIsNone(segment["locality"])
        self.assertIsNone(segment["state"])
        self.assertEqual(segment["embedded_location_text"], [])
        self.assertEqual(segment["location_evidence"], "none")
        self.assertNotIn("location", segment)

    def test_finished_video_is_deleted_after_successful_location_review(self) -> None:
        with TemporaryDirectory() as temporary:
            video = Path(temporary) / "video.mp4"
            video.write_bytes(b"video")
            record = {
                "status": "complete",
                "downloaded_path": str(video),
                "segments": [
                    {
                        "location_visual_review": {
                            "location_found": False,
                            "confidence": 0.99,
                            "error": None,
                        }
                    }
                ],
            }

            with patch(
                "car_crash_pipeline.crash_review.settings.DELETE_VIDEO_AFTER_PROCESSING",
                True,
            ):
                deleted = _delete_downloaded_video_when_finished(record, video)

        self.assertTrue(deleted)
        self.assertFalse(video.exists())
        self.assertIsNone(record["downloaded_path"])

    def test_video_is_kept_when_location_review_needs_retry(self) -> None:
        with TemporaryDirectory() as temporary:
            video = Path(temporary) / "video.mp4"
            video.write_bytes(b"video")
            record = {
                "status": "complete",
                "downloaded_path": str(video),
                "segments": [
                    {
                        "location_visual_review": {
                            "location_found": False,
                            "confidence": 0.0,
                            "error": "model_error",
                        }
                    }
                ],
            }

            self.assertTrue(has_location_visual_review_errors(record))
            with patch(
                "car_crash_pipeline.crash_review.settings.DELETE_VIDEO_AFTER_PROCESSING",
                True,
            ):
                deleted = _delete_downloaded_video_when_finished(record, video)

            self.assertFalse(deleted)
            self.assertTrue(video.exists())
            self.assertEqual(record["downloaded_path"], str(video))

    def test_location_stage_deletes_already_finished_retained_video(self) -> None:
        with TemporaryDirectory() as temporary:
            video = Path(temporary) / "video.mp4"
            video.write_bytes(b"video")
            record = {
                "status": "complete",
                "downloaded_path": str(video),
                "segments": [
                    {
                        "location_visual_review_version":
                            "cosmos3_location_text_v2",
                        "location_visual_review": {
                            "location_found": False,
                            "confidence": 0.99,
                            "error": None,
                        },
                    }
                ],
            }
            state = {"videos": {"abc": record}}

            with (
                patch(
                    "car_crash_pipeline.crash_review.settings."
                    "DELETE_VIDEO_AFTER_PROCESSING",
                    True,
                ),
                patch("car_crash_pipeline.crash_review.save_state") as save,
                patch(
                    "car_crash_pipeline.crash_review.CosmosCrashJudge"
                ) as judge,
            ):
                reviewed = run_location_visual_stage(state)

            self.assertEqual(reviewed, 0)
            self.assertFalse(video.exists())
            self.assertIsNone(record["downloaded_path"])
            save.assert_called_once()
            judge.assert_not_called()

    def test_visible_coordinates_are_reverse_geocoded(self) -> None:
        payload = {
            "lat": "36.1563000",
            "lon": "-95.9927000",
            "address": {
                "city": "Resolved City",
                "state": "Resolved Region",
                "country": "Resolved Country",
                "country_code": "us",
                "ISO3166-2-lvl4": "US-OK",
            },
            "namedetails": {},
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with (
            patch("car_crash_pipeline.location.settings.ENABLE_GEOCODING", True),
            patch("car_crash_pipeline.location.settings.GEOCODER_DELAY_SECONDS", 0),
            patch("car_crash_pipeline.location.urlopen", return_value=Response()) as open_url,
            patch("car_crash_pipeline.location._iso3", return_value="USA"),
        ):
            result = geocode(
                {"lat": 36.1563122, "lon": -95.9927516},
                {},
            )

        self.assertIn("/reverse?", open_url.call_args.args[0].full_url)
        self.assertEqual(result["locality"], "Resolved City")
        self.assertEqual(result["state"], "OK")
        self.assertEqual(result["lat"], 36.1563122)
        self.assertEqual(result["lon"], -95.9927516)
        self.assertEqual(result["iso3"], "USA")
        self.assertEqual(result["continent"], "North America")
        self.assertEqual(result["geocode_status"], "resolved")

    def test_iso3_conversion_uses_country_code(self) -> None:
        alpha3 = {"AU": "AUS", "DE": "DEU"}
        fake_pycountry = SimpleNamespace(
            countries=SimpleNamespace(
                get=lambda *, alpha_2: SimpleNamespace(
                    alpha_3=alpha3[alpha_2]
                )
            )
        )

        with patch.dict(sys.modules, {"pycountry": fake_pycountry}):
            self.assertEqual(_iso3("au"), "AUS")
            self.assertEqual(_iso3("de"), "DEU")

    def test_location_review_rejects_unsupported_inference(self) -> None:
        error = validate_location_visual_response(
            {
                "location_found": True,
                "confidence": 0.9,
                "locality": None,
                "state": None,
                "country": None,
                "lat": None,
                "lon": None,
                "visible_location_text": [],
            }
        )

        self.assertEqual(
            error, "found_location_requires_visible_location_text"
        )

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

    def test_location_candidates_use_raw_video_text_and_timestamp_labels(self) -> None:
        segment = {
            "lat": 36.1563122,
            "lon": -95.9927516,
            "raw_response": json.dumps(
                {
                    "locality": "Sacramento",
                    "state": "California",
                    "country": "United States",
                    "embedded_location_text": ["I-80, Sacramento, California"],
                }
            ),
            "embedded_location_text": ["Santa Ana/Tustin"],
            "timestamp_labels": [
                {"timestamp_seconds": 30, "label": "Toronto, Canada"}
            ],
        }
        record = {"segments": [segment, {}]}

        candidates = _location_candidates(record, segment)

        self.assertIn(
            {
                "locality": "Sacramento",
                "state": "California",
                "country": "United States",
            },
            candidates,
        )
        self.assertIn({"_location_query": "Santa Ana/Tustin"}, candidates)
        self.assertIn({"_location_query": "Toronto, Canada"}, candidates)
        self.assertIn(
            {"lat": 36.1563122, "lon": -95.9927516}, candidates
        )

    def test_coordinate_text_is_not_used_as_a_locality(self) -> None:
        segment = {
            "locality": "S:33.8187 E:150.9447",
            "lat": 33.8187,
            "lon": 150.9447,
            "embedded_location_text": ["S:33.8187 E:150.9447"],
            "location_visual_review": {
                "visible_location_text": ["S:33.8187 E:150.9447"]
            },
        }

        candidates = _location_candidates(
            {"segments": [segment, {}]}, segment
        )

        self.assertEqual(candidates[0], {"lat": -33.8187, "lon": 150.9447})
        self.assertFalse(
            any(
                candidate.get("locality") == "S:33.8187 E:150.9447"
                or candidate.get("_location_query")
                == "S:33.8187 E:150.9447"
                for candidate in candidates
            )
        )

    def test_old_resolved_locations_are_reprocessed(self) -> None:
        segment = {
            "locality": "Melbourne",
            "state": "Victoria",
            "country": "Australia",
            "location": {
                "locality": "Melbourne",
                "state": "Victoria",
                "country": "Australia",
                "iso3": None,
                "continent": "Oceania",
                "geocode_status": "resolved",
                "location_resolution_version": "segment_evidence_location_v3",
            },
        }
        state = {
            "videos": {
                "abc": {
                    "status": "complete",
                    "segments": [segment],
                    "text_decision": {},
                }
            }
        }
        refreshed = {
            **segment["location"],
            "iso3": "AUS",
            "location_resolution_version": LOCATION_RESOLUTION_VERSION,
        }

        with (
            patch("car_crash_pipeline.location.load_json", return_value={}),
            patch("car_crash_pipeline.location.geocode", return_value=refreshed) as lookup,
            patch("car_crash_pipeline.location.write_json_atomic"),
            patch("car_crash_pipeline.location.save_state"),
        ):
            processed = run_location_stage(state)

        self.assertEqual(processed, 1)
        lookup.assert_called()
        self.assertEqual(segment["location"]["iso3"], "AUS")


if __name__ == "__main__":
    unittest.main()
