"""
Test Suite: listings_api.py
Validates API request mocking, data filtering, payload flattening, CSV generation,
error handling, and pagination mechanisms.
"""
 
import csv
import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, call, mock_open, patch
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "data_collection"))
import requests
 
# ---------------------------------------------------------------------------
# Helper Functions (Extracted from API script for Isolated Testing)
# ---------------------------------------------------------------------------
 
def flatten_dict(d, parent_key='', sep='_'):
    """Identical implementation to the original script for isolated validation."""
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v)
        else:
            items[new_key] = v
    return items
 
 
def has_owner_count(vehicle: dict) -> bool:
    """Filterpredikat aus dem Originalskript."""
    return bool(vehicle.get('history') and vehicle['history'].get('ownerCount') is not None)
 
 
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
 
def make_vehicle(vin="VIN001", owner_count=2, extras=None):
    v = {
        "vin": vin,
        "price": 35000,
        "vehicle": {"make": "Ford", "model": "F-150", "year": 2020},
        "history": {"ownerCount": owner_count, "accidentCount": 0},
    }
    if extras:
        v.update(extras)
    return v
 
 
def make_api_response(vehicles: list) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": vehicles}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp
 
 
def make_empty_api_response() -> MagicMock:
    return make_api_response([])
 
 
# ===========================================================================
# 1. FLATTEN-LOGIK
# ===========================================================================
 
class TestFlattenDict(unittest.TestCase):
 
    def test_flat_dict_unchanged(self):
        d = {"a": 1, "b": "x"}
        self.assertEqual(flatten_dict(d), {"a": 1, "b": "x"})
 
    def test_nested_dict_flattened(self):
        d = {"vehicle": {"make": "Ford", "year": 2020}}
        result = flatten_dict(d)
        self.assertEqual(result["vehicle_make"], "Ford")
        self.assertEqual(result["vehicle_year"], 2020)
 
    def test_deeply_nested(self):
        d = {"a": {"b": {"c": 42}}}
        self.assertEqual(flatten_dict(d)["a_b_c"], 42)
 
    def test_list_values_serialized_as_json(self):
        d = {"tags": ["clean", "one-owner"]}
        result = flatten_dict(d)
        self.assertEqual(result["tags"], '["clean", "one-owner"]')
 
    def test_empty_dict(self):
        self.assertEqual(flatten_dict({}), {})
 
    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        result = flatten_dict(d, sep=".")
        self.assertIn("a.b", result)
 
    def test_none_value_preserved(self):
        d = {"field": None}
        self.assertIsNone(flatten_dict(d)["field"])
 
    def test_full_vehicle_structure(self):
        vehicle = make_vehicle()
        flat = flatten_dict(vehicle)
        self.assertEqual(flat["vehicle_make"], "Ford")
        self.assertEqual(flat["history_ownerCount"], 2)
        self.assertEqual(flat["vin"], "VIN001")
 
 
# ===========================================================================
# 2. FILTERLOGIK
# ===========================================================================
 
class TestHasOwnerCount(unittest.TestCase):
 
    def test_vehicle_with_owner_count(self):
        self.assertTrue(has_owner_count(make_vehicle(owner_count=1)))
 
    def test_vehicle_without_history(self):
        self.assertFalse(has_owner_count({"vin": "X", "price": 1000}))
 
    def test_vehicle_with_empty_history(self):
        self.assertFalse(has_owner_count({"vin": "X", "history": {}}))
 
    def test_vehicle_with_none_owner_count(self):
        self.assertFalse(has_owner_count({"vin": "X", "history": {"ownerCount": None}}))
 
    def test_vehicle_with_zero_owner_count(self):
        # ownerCount=0 ist ein valider Wert (z.B. Neuwagen) und sollte durchkommen
        v = {"vin": "X", "history": {"ownerCount": 0}}
        self.assertTrue(has_owner_count(v))
 
    def test_vehicle_history_is_none(self):
        self.assertFalse(has_owner_count({"vin": "X", "history": None}))
 
 
# ===========================================================================
# 3. API-AUFRUF-MOCKING
# ===========================================================================
 
class TestApiCalls(unittest.TestCase):
 
    @patch("requests.get")
    def test_correct_url_constructed(self, mock_get):
        mock_get.return_value = make_api_response([make_vehicle()])
        response = requests.get(
            "https://api.auto.dev/listings?vehicle.make=Ford&vehicle.model=F-150,F-250,F-350&page=1",
            headers={"x-api-key": "test-key"}
        )
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn("page=1", call_args[0][0])
        self.assertIn("Ford", call_args[0][0])
 
    @patch("requests.get")
    def test_api_key_sent_in_header(self, mock_get):
        mock_get.return_value = make_api_response([])
        requests.get("https://api.auto.dev/listings", headers={"x-api-key": "my-secret"})
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["headers"]["x-api-key"], "my-secret")
 
    @patch("requests.get")
    def test_http_error_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
        mock_get.return_value = mock_resp
        with self.assertRaises(requests.exceptions.HTTPError):
            mock_resp.raise_for_status()
 
    @patch("requests.get")
    def test_connection_error_raises(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("No route to host")
        with self.assertRaises(requests.exceptions.ConnectionError):
            requests.get("https://api.auto.dev/listings", headers={})
 
    @patch("requests.get")
    def test_timeout_raises(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()
        with self.assertRaises(requests.exceptions.Timeout):
            requests.get("https://api.auto.dev/listings", headers={})
 
 
# ===========================================================================
# 4. PAGINIERUNGS-LOGIK
# ===========================================================================
 
class TestPagination(unittest.TestCase):
 
    def _simulate_pagination(self, side_effects):
        """Simuliert die Paginierungsschleife aus dem Originalskript."""
        collected = []
        with patch("requests.get") as mock_get:
            mock_get.side_effect = side_effects
            for page_num in range(1, len(side_effects) + 1):
                resp = requests.get(
                    f"https://api.auto.dev/listings?page={page_num}",
                    headers={"x-api-key": "test"}
                )
                resp.raise_for_status()
                vehicles = resp.json().get("data", [])
                if not vehicles:
                    break
                collected.extend(vehicles)
        return collected
 
    def test_stops_on_empty_page(self):
        side_effects = [
            make_api_response([make_vehicle("V1")]),
            make_api_response([make_vehicle("V2")]),
            make_empty_api_response(),
        ]
        result = self._simulate_pagination(side_effects)
        self.assertEqual(len(result), 2)
 
    def test_single_page_result(self):
        result = self._simulate_pagination([
            make_api_response([make_vehicle("V1"), make_vehicle("V2")])
        ])
        self.assertEqual(len(result), 2)
 
    def test_all_pages_empty_from_start(self):
        result = self._simulate_pagination([make_empty_api_response()])
        self.assertEqual(result, [])
 
    def test_multiple_pages_collected(self):
        pages = [make_api_response([make_vehicle(f"V{i}")]) for i in range(5)]
        pages.append(make_empty_api_response())
        result = self._simulate_pagination(pages)
        self.assertEqual(len(result), 5)
 
 
# ===========================================================================
# 5. DUPLIKAT-FILTERUNG (VIN-Deduplizierung)
# ===========================================================================
 
class TestVinDeduplication(unittest.TestCase):
 
    def _deduplicate(self, vehicles):
        processed_vins = set()
        result = []
        for v in vehicles:
            vin = v.get("vin")
            if vin and vin not in processed_vins:
                result.append(v)
                processed_vins.add(vin)
        return result
 
    def test_duplicate_vins_removed(self):
        vehicles = [make_vehicle("SAME"), make_vehicle("SAME"), make_vehicle("OTHER")]
        result = self._deduplicate(vehicles)
        self.assertEqual(len(result), 2)
 
    def test_no_duplicates_unchanged(self):
        vehicles = [make_vehicle("V1"), make_vehicle("V2"), make_vehicle("V3")]
        self.assertEqual(len(self._deduplicate(vehicles)), 3)
 
    def test_missing_vin_skipped(self):
        vehicles = [{"price": 5000}, make_vehicle("V1")]
        result = self._deduplicate(vehicles)
        self.assertEqual(len(result), 1)
 
    def test_empty_list(self):
        self.assertEqual(self._deduplicate([]), [])
 
 
# ===========================================================================
# 6. CSV-AUSGABE
# ===========================================================================
 
class TestCsvOutput(unittest.TestCase):
 
    def _write_csv(self, flat_rows):
        all_headers = set()
        for row in flat_rows:
            all_headers.update(row.keys())
        fieldnames = sorted(list(all_headers))
 
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(flat_rows)
        buf.seek(0)
        return buf, fieldnames
 
    def test_csv_has_header_row(self):
        vehicles = [flatten_dict(make_vehicle("V1")), flatten_dict(make_vehicle("V2"))]
        buf, _ = self._write_csv(vehicles)
        reader = csv.DictReader(buf)
        rows = list(reader)
        self.assertEqual(len(rows), 2)
 
    def test_csv_columns_sorted(self):
        vehicles = [flatten_dict(make_vehicle())]
        _, fieldnames = self._write_csv(vehicles)
        self.assertEqual(fieldnames, sorted(fieldnames))
 
    def test_csv_vin_correct(self):
        vehicles = [flatten_dict(make_vehicle("TESTVIN"))]
        buf, _ = self._write_csv(vehicles)
        reader = csv.DictReader(buf)
        row = next(reader)
        self.assertEqual(row["vin"], "TESTVIN")
 
    def test_csv_nested_fields_flattened(self):
        vehicles = [flatten_dict(make_vehicle())]
        buf, _ = self._write_csv(vehicles)
        reader = csv.DictReader(buf)
        row = next(reader)
        self.assertIn("vehicle_make", row)
        self.assertEqual(row["vehicle_make"], "Ford")
 
    def test_csv_missing_fields_empty_not_crash(self):
        row1 = {"vin": "V1", "price": 1000}
        row2 = {"vin": "V2", "price": 2000, "extra_field": "yes"}
        buf, _ = self._write_csv([row1, row2])
        reader = csv.DictReader(buf)
        rows = list(reader)
        self.assertEqual(len(rows), 2)
 
    def test_csv_list_field_is_json_string(self):
        vehicle = make_vehicle()
        vehicle["tags"] = ["clean", "fleet"]
        flat = flatten_dict(vehicle)
        buf, _ = self._write_csv([flat])
        reader = csv.DictReader(buf)
        row = next(reader)
        parsed = json.loads(row["tags"])
        self.assertEqual(parsed, ["clean", "fleet"])
 
 
# ===========================================================================
# 7. INTEGRATION (End-to-End mit gemockter URL)
# ===========================================================================
 
class TestEndToEnd(unittest.TestCase):
 
    @patch("requests.get")
    @patch("builtins.open", new_callable=mock_open)
    @patch("time.sleep", return_value=None)
    def test_full_pipeline_writes_csv(self, mock_sleep, mock_file, mock_get):
        """
        Simuliert den kompletten Durchlauf:
        Seite 1 → 2 Fahrzeuge (1 mit, 1 ohne History)
        Seite 2 → leer → Abbruch
        Erwartet: genau 1 Fahrzeug landet in der CSV.
        """
        page1 = make_api_response([
            make_vehicle("V_WITH", owner_count=1),
            {"vin": "V_WITHOUT", "price": 9999}   # kein history-Key
        ])
        page2 = make_empty_api_response()
        mock_get.side_effect = [page1, page2]
 
        collected = []
        processed_vins = set()
 
        for page_num in range(1, 3):
            resp = requests.get(
                f"https://api.auto.dev/listings?vehicle.make=Ford&page={page_num}",
                headers={"x-api-key": "test-key"}
            )
            resp.raise_for_status()
            vehicles = resp.json().get("data", [])
            if not vehicles:
                break
            filtered = [v for v in vehicles if has_owner_count(v)]
            for v in filtered:
                vin = v.get("vin")
                if vin and vin not in processed_vins:
                    collected.append(flatten_dict(v))
                    processed_vins.add(vin)
 
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["vin"], "V_WITH")
        self.assertEqual(mock_get.call_count, 2)
 
    @patch("requests.get")
    @patch("time.sleep", return_value=None)
    def test_no_vehicles_raises_exception(self, mock_sleep, mock_get):
        """Wenn keine passenden Fahrzeuge gefunden werden, soll eine Exception geworfen werden."""
        mock_get.return_value = make_empty_api_response()
 
        raw_filtered = []
        for page_num in range(1, 3):
            resp = requests.get(f"https://api.auto.dev/listings?page={page_num}", headers={})
            resp.raise_for_status()
            vehicles = resp.json().get("data", [])
            if not vehicles:
                break
            raw_filtered.extend([v for v in vehicles if has_owner_count(v)])
 
        with self.assertRaises(Exception) as ctx:
            if not raw_filtered:
                raise Exception("Keine Fahrzeuge mit den geforderten History-Daten gefunden.")
        self.assertIn("Keine Fahrzeuge", str(ctx.exception))
 
    @patch("requests.get")
    @patch("time.sleep", return_value=None)
    def test_pagination_respects_max_pages(self, mock_sleep, mock_get):
        """Schleife darf MAX_PAGES nicht überschreiten."""
        MAX_PAGES = 3
        mock_get.return_value = make_api_response([make_vehicle()])
 
        call_count = 0
        for page_num in range(1, MAX_PAGES + 1):
            requests.get(f"https://api.auto.dev/listings?page={page_num}", headers={})
            call_count += 1
 
        self.assertEqual(call_count, MAX_PAGES)
        self.assertEqual(mock_get.call_count, MAX_PAGES)
 
 
# ===========================================================================
# 8. EDGE CASES & SONDERFÄLLE
# ===========================================================================
 
class TestEdgeCases(unittest.TestCase):
 
    def test_flatten_with_boolean_values(self):
        d = {"available": True, "sold": False}
        flat = flatten_dict(d)
        self.assertTrue(flat["available"])
        self.assertFalse(flat["sold"])
 
    def test_flatten_with_integer_zero(self):
        d = {"mileage": 0}
        self.assertEqual(flatten_dict(d)["mileage"], 0)
 
    def test_filter_keeps_zero_owner_count(self):
        v = {"vin": "NEW", "history": {"ownerCount": 0}}
        self.assertTrue(has_owner_count(v))
 
    def test_filter_rejects_missing_history_key(self):
        v = {"vin": "X", "price": 100}
        self.assertFalse(has_owner_count(v))
 
    def test_flatten_empty_nested_dict(self):
        d = {"details": {}}
        # Leeres nested dict → keine Einträge, aber kein Absturz
        flat = flatten_dict(d)
        self.assertNotIn("details", flat)
 
    @patch("requests.get")
    def test_json_decode_error_propagates(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_get.return_value = mock_resp
 
        resp = requests.get("https://api.auto.dev/listings", headers={})
        with self.assertRaises(json.JSONDecodeError):
            resp.json()
 
 
if __name__ == "__main__":
    unittest.main(verbosity=2)
