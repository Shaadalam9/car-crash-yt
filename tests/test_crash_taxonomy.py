"""Regression tests for NHTSA MMUCC crash taxonomy backfill."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from car_crash_pipeline.crash_taxonomy import (
    CRASH_TAXONOMY_STANDARD,
    CRASH_TAXONOMY_VERSION,
    crash_taxonomy_pending,
    normalise_taxonomy_response,
    run_crash_taxonomy_stage,
    validate_taxonomy_response,
)
from car_crash_pipeline.output_writer import iter_mapping_rows, iter_rows


class CrashTaxonomyTests(unittest.TestCase):
    def test_motor_vehicle_collision_uses_mmucc_c6_and_c9(self) -> None:
        data = {
            "event_kind": "collision",
            "manner_of_collision": "front_to_rear_or_rear_to_front",
            "first_harmful_event": "motor_vehicle_in_transport",
            "sequence_of_events": ["motor_vehicle_in_transport"],
        }

        self.assertIsNone(validate_taxonomy_response(data))

        result = normalise_taxonomy_response(data)

        self.assertEqual(result["event_kind"], "collision")
        self.assertEqual(
            result["manner_of_collision_code"],
            "C9_FRONT_TO_REAR_OR_REAR_TO_FRONT",
        )
        self.assertEqual(
            result["first_harmful_event_code"],
            "C7_MOTOR_VEHICLE_IN_TRANSPORT",
        )
        self.assertEqual(
            result["sequence_of_events"],
            ["motor_vehicle_in_transport"],
        )
        self.assertEqual(
            result["crash_taxonomy_standard"],
            CRASH_TAXONOMY_STANDARD,
        )
        self.assertEqual(
            result["crash_taxonomy_version"],
            CRASH_TAXONOMY_VERSION,
        )

    def test_non_motor_vehicle_first_event_requires_not_collision_manner(self) -> None:
        valid = {
            "event_kind": "collision",
            "manner_of_collision": (
                "not_collision_with_motor_vehicle_in_transport"
            ),
            "first_harmful_event": "non_motorist",
            "sequence_of_events": ["non_motorist"],
        }
        invalid = dict(valid)
        invalid["manner_of_collision"] = "angle"

        self.assertIsNone(validate_taxonomy_response(valid))
        self.assertEqual(
            validate_taxonomy_response(invalid),
            "non_motor_vehicle_first_harmful_event_requires_not_collision_manner",
        )

    def test_near_collision_has_no_mmucc_harmful_event(self) -> None:
        data = {
            "event_kind": "near_collision",
            "manner_of_collision": None,
            "first_harmful_event": None,
            "sequence_of_events": [],
        }

        self.assertIsNone(validate_taxonomy_response(data))

        result = normalise_taxonomy_response(data)

        self.assertEqual(result["event_kind"], "near_collision")
        self.assertIsNone(result["manner_of_collision"])
        self.assertIsNone(result["first_harmful_event"])
        self.assertEqual(result["sequence_of_events"], [])

    def test_sequence_keeps_non_harmful_precursor_before_first_harmful_event(
        self,
    ) -> None:
        data = {
            "event_kind": "collision",
            "manner_of_collision": (
                "not_collision_with_motor_vehicle_in_transport"
            ),
            "first_harmful_event": "rollover_overturn",
            "sequence_of_events": [
                "ran_off_roadway_right",
                "rollover_overturn",
            ],
        }

        self.assertIsNone(validate_taxonomy_response(data))

    def test_old_completed_segment_is_pending_until_backfilled(self) -> None:
        record = {
            "status": "complete",
            "segments": [{"segment_index": 0}],
        }

        self.assertTrue(crash_taxonomy_pending(record))

        record["segments"][0]["crash_taxonomy_version"] = (
            CRASH_TAXONOMY_VERSION
        )

        self.assertFalse(crash_taxonomy_pending(record))

    def test_unavailable_old_video_is_terminally_skipped(self) -> None:
        state = {
            "videos": {
                "missingVideo": {
                    "status": "complete",
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 1.0,
                            "end_time": 2.0,
                            "duration_seconds": 1.0,
                        }
                    ],
                }
            }
        }

        with (
            patch(
                "car_crash_pipeline.crash_taxonomy._obtain_video",
                side_effect=RuntimeError("Video unavailable"),
            ),
            patch(
                "car_crash_pipeline.crash_taxonomy.save_state"
            ),
        ):
            processed = run_crash_taxonomy_stage(state)

        segment = state["videos"]["missingVideo"]["segments"][0]

        self.assertEqual(processed, 1)
        self.assertEqual(
            segment["crash_taxonomy_status"],
            "video_unavailable",
        )
        self.assertEqual(
            segment["crash_taxonomy_version"],
            CRASH_TAXONOMY_VERSION,
        )
        self.assertIsNone(segment["event_kind"])
        self.assertFalse(
            crash_taxonomy_pending(state["videos"]["missingVideo"])
        )

    def test_csv_outputs_include_taxonomy(self) -> None:
        state = {
            "videos": {
                "videoA": {
                    "status": "complete",
                    "visual_review_version": "test",
                    "metadata": {},
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 10.0,
                            "end_time": 12.0,
                            "duration_seconds": 2.0,
                            "confidence": 0.99,
                            "short_description": "rear impact",
                            "crash_type": "rear_end",
                            "road_users": ["car"],
                            "time_of_day": "day",
                            "event_kind": "collision",
                            "manner_of_collision_code": (
                                "C9_FRONT_TO_REAR_OR_REAR_TO_FRONT"
                            ),
                            "manner_of_collision": (
                                "front_to_rear_or_rear_to_front"
                            ),
                            "first_harmful_event_code": (
                                "C7_MOTOR_VEHICLE_IN_TRANSPORT"
                            ),
                            "first_harmful_event": (
                                "motor_vehicle_in_transport"
                            ),
                            "sequence_of_events": [
                                "motor_vehicle_in_transport"
                            ],
                            "crash_taxonomy_standard": (
                                CRASH_TAXONOMY_STANDARD
                            ),
                            "crash_taxonomy_status": "classified",
                            "crash_taxonomy_version": (
                                CRASH_TAXONOMY_VERSION
                            ),
                            "location": {
                                "locality": "Test City",
                                "locality_aka": [],
                                "state": None,
                                "country": "Test Country",
                                "iso3": "TST",
                                "continent": "Test",
                                "lat": 1.0,
                                "lon": 2.0,
                                "osm_type": "relation",
                                "osm_id": 123,
                                "place_id": 456,
                                "geocode_status": "resolved",
                            },
                        }
                    ],
                }
            }
        }

        segment_row = list(iter_rows(state))[0]
        mapping_row = list(iter_mapping_rows(state))[0]

        self.assertEqual(segment_row["event_kind"], "collision")
        self.assertEqual(
            segment_row["first_harmful_event"],
            "motor_vehicle_in_transport",
        )
        self.assertEqual(
            segment_row["sequence_of_events"],
            '["motor_vehicle_in_transport"]',
        )

        self.assertEqual(mapping_row["event_kind"], "[collision]")
        self.assertEqual(
            mapping_row["manner_of_collision"],
            "[front_to_rear_or_rear_to_front]",
        )
        self.assertEqual(
            mapping_row["sequence_of_events"],
            "[[motor_vehicle_in_transport]]",
        )
        self.assertEqual(
            mapping_row["crash_taxonomy_status"],
            "[classified]",
        )


if __name__ == "__main__":
    unittest.main()
