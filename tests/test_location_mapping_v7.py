"""Regression tests for canonical locality resolution and mapping output."""

import unittest
from unittest.mock import patch

from car_crash_pipeline.location import geocode
from car_crash_pipeline.output_writer import iter_mapping_rows


class LocationMappingRegressionTests(unittest.TestCase):
    def test_jackson_city_is_selected_instead_of_jackson_county(self) -> None:
        county = {
            "place_id": 1,
            "osm_type": "relation",
            "osm_id": 111,
            "class": "boundary",
            "type": "administrative",
            "addresstype": "county",
            "name": "Jackson County",
            "lat": "34.1282559",
            "lon": "-83.5753386",
            "address": {
                "county": "Jackson County",
                "state": "Georgia",
                "country": "United States",
                "country_code": "us",
                "ISO3166-2-lvl4": "US-GA",
            },
            "namedetails": {
                "name:en": "Jackson County",
            },
        }
        city = {
            "place_id": 2,
            "osm_type": "relation",
            "osm_id": 222,
            "class": "boundary",
            "type": "administrative",
            "addresstype": "city",
            "name": "Jackson",
            "lat": "33.2946",
            "lon": "-83.9660",
            "address": {
                "city": "Jackson",
                "state": "Georgia",
                "country": "United States",
                "country_code": "us",
                "ISO3166-2-lvl4": "US-GA",
            },
            "namedetails": {
                "name:en": "Jackson",
                "official_name": "City of Jackson",
            },
        }

        with (
            patch(
                "car_crash_pipeline.location._search_nominatim",
                return_value=[county, city],
            ),
            patch(
                "car_crash_pipeline.location._iso3",
                return_value="USA",
            ),
        ):
            result = geocode(
                {"locality": "JACKSON, GEORGIA"},
                {},
            )

        self.assertEqual(result["geocode_status"], "resolved")
        self.assertEqual(result["locality"], "Jackson")
        self.assertEqual(result["state"], "GA")
        self.assertEqual(result["lat"], 33.2946)
        self.assertEqual(result["lon"], -83.966)
        self.assertEqual(result["osm_id"], 222)
        self.assertNotIn("Jackson County", result["locality_aka"])

    def test_station_name_does_not_become_containing_locality(self) -> None:
        oku_station = {
            "place_id": 3,
            "osm_type": "node",
            "osm_id": 333,
            "class": "railway",
            "type": "station",
            "addresstype": "railway",
            "name": "Oku",
            "lat": "35.7469091",
            "lon": "139.7536603",
            "address": {
                "suburb": "Kita",
                "country": "Japan",
                "country_code": "jp",
            },
            "namedetails": {
                "name:en": "Oku",
            },
        }

        with patch(
            "car_crash_pipeline.location._search_nominatim",
            return_value=[oku_station],
        ):
            result = geocode(
                {"_location_query": "Oku"},
                {},
            )

        self.assertEqual(result["geocode_status"], "not_found")
        self.assertIsNone(result["locality"])
        self.assertIsNone(result["lat"])
        self.assertIsNone(result["lon"])

    def test_visible_coordinates_are_canonicalised_to_locality(self) -> None:
        station = {
            "place_id": 3,
            "osm_type": "node",
            "osm_id": 333,
            "class": "railway",
            "type": "station",
            "addresstype": "railway",
            "name": "Oku",
            "lat": "35.7469091",
            "lon": "139.7536603",
            "address": {
                "suburb": "Kita",
                "country": "Japan",
                "country_code": "jp",
            },
            "namedetails": {
                "name:en": "Oku",
            },
        }
        kita = {
            "place_id": 4,
            "osm_type": "relation",
            "osm_id": 444,
            "class": "boundary",
            "type": "administrative",
            "addresstype": "borough",
            "name": "Kita",
            "lat": "35.7526",
            "lon": "139.7335",
            "address": {
                "borough": "Kita",
                "country": "Japan",
                "country_code": "jp",
            },
            "namedetails": {
                "name:en": "Kita",
                "official_name": "Kita City",
            },
        }

        with (
            patch(
                "car_crash_pipeline.location._reverse_nominatim",
                return_value=station,
            ),
            patch(
                "car_crash_pipeline.location._canonicalise_from_locality",
                return_value=kita,
            ),
            patch(
                "car_crash_pipeline.location._iso3",
                return_value="JPN",
            ),
        ):
            result = geocode(
                {
                    "lat": 35.7469091,
                    "lon": 139.7536603,
                },
                {},
            )

        self.assertEqual(result["geocode_status"], "resolved")
        self.assertEqual(result["locality"], "Kita")
        self.assertEqual(result["lat"], 35.7526)
        self.assertEqual(result["lon"], 139.7335)
        self.assertEqual(result["osm_id"], 444)

    def test_mapping_excludes_unresolved_locations_and_keeps_blank_state(self) -> None:
        state = {
            "videos": {
                "resolved": {
                    "status": "complete",
                    "segments": [
                        {
                            "start_time": 1.0,
                            "end_time": 2.0,
                            "time_of_day": "day",
                            "road_users": ["car"],
                            "location": {
                                "locality": "Kita",
                                "locality_aka": ["Kita City"],
                                "state": None,
                                "country": "Japan",
                                "iso3": "JPN",
                                "continent": "Asia",
                                "lat": 35.7526,
                                "lon": 139.7335,
                                "osm_type": "relation",
                                "osm_id": 444,
                                "place_id": 4,
                                "geocode_status": "resolved",
                            },
                        }
                    ],
                },
                "unresolved": {
                    "status": "complete",
                    "segments": [
                        {
                            "start_time": 3.0,
                            "end_time": 4.0,
                            "time_of_day": "day",
                            "road_users": ["car"],
                            "location": {
                                "locality": None,
                                "country": None,
                                "lat": None,
                                "lon": None,
                                "candidate_locality": "Bond44431",
                                "geocode_status": "not_found",
                            },
                        }
                    ],
                },
            }
        }

        rows = list(iter_mapping_rows(state))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["locality"], "Kita")
        self.assertEqual(rows[0]["state"], "")
        self.assertEqual(rows[0]["videos"], "[resolved]")


if __name__ == "__main__":
    unittest.main()
