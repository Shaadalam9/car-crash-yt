from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from utils.analytics.crash_results import CrashResultsAnalysis


class CrashResultsAnalysisTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        state = {
            "videos": {
                "video_a": {
                    "status": "complete",
                    "text_decision": {"include": True},
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 10.0,
                            "end_time": 20.0,
                            "duration_seconds": 10.0,
                            "time_of_day": "day",
                            "road_users": ["car", "truck"],
                            "event_kind": "collision",
                            "manner_of_collision_code": "C9_FRONT_TO_REAR_OR_REAR_TO_FRONT",
                            "manner_of_collision": "front_to_rear_or_rear_to_front",
                            "first_harmful_event_code": "C7_MOTOR_VEHICLE_IN_TRANSPORT",
                            "first_harmful_event": "motor_vehicle_in_transport",
                            "sequence_of_events": ["motor_vehicle_in_transport"],
                            "crash_taxonomy_standard": "NHTSA MMUCC 6th Edition (2024)",
                            "crash_taxonomy_status": "classified",
                            "crash_taxonomy_version": "nhtsa_mmucc6_video_v1",
                            "location": {
                                "geocode_status": "resolved",
                                "location_resolution_version": "segment_evidence_location_v7",
                                "locality": "Anthony",
                                "state": "KS",
                                "country": "United States",
                                "iso3": "USA",
                                "continent": "North America",
                                "lat": 37.1533554,
                                "lon": -98.0311728,
                                "osm_type": "relation",
                                "osm_id": 123,
                            },
                        },
                        {
                            "segment_index": 1,
                            "start_time": 25.0,
                            "end_time": 30.0,
                            "duration_seconds": 5.0,
                            "time_of_day": "night",
                            "road_users": ["car"],
                            "crash_taxonomy_status": "missing",
                            "location": {
                                "geocode_status": "not_found",
                                "location_resolution_version": "segment_evidence_location_v7",
                            },
                        },
                    ],
                },
                "video_b": {
                    "status": "text_rejected",
                    "text_decision": {"include": False},
                    "segments": [],
                },
                "video_c": {
                    "status": "visual_rejected",
                    "text_decision": {"include": True},
                    "segments": [],
                },
                "video_d": {
                    "status": "visual_error",
                    "text_decision": {"include": True},
                    "segments": [],
                },
            }
        }
        state_path = root / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")

        mapping_path = root / "mapping.csv"
        columns = [
            "id", "locality", "locality_aka", "state", "country", "iso3", "continent", "lat", "lon",
            "videos", "time_of_day", "start_time", "end_time", "vehicle_type", "event_kind",
            "manner_of_collision_code", "manner_of_collision", "first_harmful_event_code",
            "first_harmful_event", "sequence_of_events", "crash_taxonomy_standard",
            "crash_taxonomy_status", "crash_taxonomy_version",
        ]
        row = {
            "id": 1,
            "locality": "Anthony",
            "locality_aka": "[]",
            "state": "KS",
            "country": "United States",
            "iso3": "USA",
            "continent": "North America",
            "lat": 37.1533554,
            "lon": -98.0311728,
            "videos": "[video_a]",
            "time_of_day": "[day]",
            "start_time": "[10.0]",
            "end_time": "[20.0]",
            "vehicle_type": "[car,truck]",
            "event_kind": "[collision]",
            "manner_of_collision_code": "[C9_FRONT_TO_REAR_OR_REAR_TO_FRONT]",
            "manner_of_collision": "[front_to_rear_or_rear_to_front]",
            "first_harmful_event_code": "[C7_MOTOR_VEHICLE_IN_TRANSPORT]",
            "first_harmful_event": "[motor_vehicle_in_transport]",
            "sequence_of_events": "[[motor_vehicle_in_transport]]",
            "crash_taxonomy_standard": "[NHTSA MMUCC 6th Edition (2024)]",
            "crash_taxonomy_status": "[classified]",
            "crash_taxonomy_version": "[nhtsa_mmucc6_video_v1]",
        }
        with mapping_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerow(row)
        return state_path, mapping_path

    def test_summary_uses_state_for_full_dataset_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state_path, mapping_path = self._write_fixture(Path(temp))
            result = CrashResultsAnalysis(state_path, mapping_path).analyse()

            self.assertEqual(result.summary["videos"]["records"], 4)
            self.assertEqual(result.summary["videos"]["complete"], 1)
            self.assertEqual(result.summary["videos"]["text_rejected"], 1)
            self.assertEqual(result.summary["videos"]["visual_rejected"], 1)
            self.assertEqual(result.summary["videos"]["visual_error"], 1)
            self.assertEqual(result.summary["segments"]["accepted"], 2)
            self.assertEqual(result.summary["location"]["resolved"], 1)
            self.assertEqual(result.summary["location"]["statuses"]["not_found"], 1)
            self.assertEqual(result.summary["taxonomy"]["classified_segments"], 1)
            self.assertEqual(result.summary["taxonomy"]["pending_segments"], 1)

    def test_mapping_consistency_and_full_clip_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state_path, mapping_path = self._write_fixture(Path(temp))
            result = CrashResultsAnalysis(state_path, mapping_path).analyse()

            self.assertEqual(result.summary["mapping"]["expanded_segment_records"], 1)
            self.assertEqual(result.summary["mapping"]["state_resolved_segments"], 1)
            self.assertTrue(result.summary["mapping"]["segment_count_matches_state"])
            self.assertAlmostEqual(result.summary["segments"]["retained_duration_seconds"], 15.0)

    def test_canonical_locality_count_uses_osm_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state_path, mapping_path = self._write_fixture(Path(temp))
            result = CrashResultsAnalysis(state_path, mapping_path).analyse()
            self.assertEqual(result.summary["geography"]["canonical_locality_entities"], 1)

    def test_multilabel_road_users_are_segment_prevalence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state_path, mapping_path = self._write_fixture(Path(temp))
            result = CrashResultsAnalysis(state_path, mapping_path).analyse()
            road_users = result.tables["road_users"].set_index("road_user")
            self.assertEqual(int(road_users.loc["car", "segments"]), 2)
            self.assertEqual(float(road_users.loc["car", "segment_prevalence_pct"]), 100.0)
            self.assertEqual(int(road_users.loc["truck", "segments"]), 1)
            self.assertEqual(float(road_users.loc["truck", "segment_prevalence_pct"]), 50.0)


if __name__ == "__main__":
    unittest.main()
