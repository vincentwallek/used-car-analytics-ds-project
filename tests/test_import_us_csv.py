import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "data_processing"))

# ---------------------------------------------------------------------------
# Patch external dependencies BEFORE importing the module under test
# ---------------------------------------------------------------------------
os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "fake-key")

with patch("supabase.create_client", return_value=MagicMock()), \
     patch("dotenv.load_dotenv"):
    import import_us_csv as us_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

US_COLUMNS = [
    "make", "model", "trim", "year", "price_usd", "mileage_mi",
    "vin", "carfax_url", "vdp_url", "primary_image_url",
    "dealer_name", "city", "state", "zip_code", "location",
    "is_used", "is_cpo", "is_online", "is_wholesale",
    "accident_count", "has_accidents", "owner_count", "one_owner",
    "personal_use", "usage_type",
    "body_style", "drivetrain", "engine", "cylinders", "doors", "seats",
    "fuel", "transmission", "exterior_color", "interior_color",
    "car_age", "price_per_mile", "listing_age_days",
]


def _make_df(n=3, missing_cols=None):
    """Return a minimal valid US CSV DataFrame."""
    data = {
        "make":              ["Ford"] * n,
        "model":             ["F-150"] * n,
        "trim":              ["XLT"] * n,
        "year":              [2020 + i for i in range(n)],
        "price_usd":         [30000.0 + i * 1000 for i in range(n)],
        "mileage_mi":        [20000.0 + i * 5000 for i in range(n)],
        "vin":               [f"1FTFW1E8{i:08d}" for i in range(n)],
        "carfax_url":        [f"https://carfax.com/report/{i}" for i in range(n)],
        "vdp_url":           [f"https://auto.dev/listing/{i}" for i in range(n)],
        "primary_image_url": [f"https://cdn.auto.dev/img/{i}.jpg" for i in range(n)],
        "dealer_name":       [f"Dealer {i}" for i in range(n)],
        "city":              ["Austin"] * n,
        "state":             ["TX"] * n,
        "zip_code":          ["78701"] * n,
        "location":          ["Austin, TX"] * n,
        "is_used":           [True] * n,
        "is_cpo":            [False] * n,
        "is_online":         [True] * n,
        "is_wholesale":      [False] * n,
        "accident_count":    [0] * n,
        "has_accidents":     [False] * n,
        "owner_count":       [1] * n,
        "one_owner":         [True] * n,
        "personal_use":      [True] * n,
        "usage_type":        ["personal"] * n,
        "body_style":        ["Pickup"] * n,
        "drivetrain":        ["4WD"] * n,
        "engine":            ["3.5L V6"] * n,
        "cylinders":         [6] * n,
        "doors":             [4] * n,
        "seats":             [5] * n,
        "fuel":              ["Gasoline"] * n,
        "transmission":      ["Automatic"] * n,
        "exterior_color":    ["White"] * n,
        "interior_color":    ["Black"] * n,
        "car_age":           [3.0 + i for i in range(n)],
        "price_per_mile":    [1.5 + i * 0.1 for i in range(n)],
        "listing_age_days":  [10 + i for i in range(n)],
    }
    df = pd.DataFrame(data)
    if missing_cols:
        df = df.drop(columns=missing_cols)
    return df


def _make_supabase_mock(n=3):
    """Return a supabase mock that yields sequential IDs on listings insert."""
    ids = iter(range(1, n + 1))

    def _insert_side_effect(rows):
        mock = MagicMock()
        mock.execute.return_value.data = [{"id": next(ids)} for _ in rows]
        return mock

    supabase = MagicMock()
    supabase.table.return_value.insert.side_effect = _insert_side_effect
    return supabase


def _run(df, supabase_mock, extra_args=None):
    argv = ["prog", "--csv", "dummy.csv", "--dataset-id", "99"] + (extra_args or [])
    with patch.object(us_mod, "supabase", supabase_mock), \
         patch("pandas.read_csv", return_value=df), \
         patch("sys.argv", argv):
        us_mod.main()


# ---------------------------------------------------------------------------
# _none_if_na
# ---------------------------------------------------------------------------

class TestNoneIfNa:
    def test_float_nan_returns_none(self):
        assert us_mod._none_if_na(float("nan")) is None

    def test_pd_na_returns_none(self):
        assert us_mod._none_if_na(pd.NA) is None

    def test_np_nan_returns_none(self):
        assert us_mod._none_if_na(np.nan) is None

    def test_string_passes_through(self):
        assert us_mod._none_if_na("hello") == "hello"

    def test_zero_passes_through(self):
        assert us_mod._none_if_na(0) == 0

    def test_false_passes_through(self):
        assert us_mod._none_if_na(False) is False

    def test_integer_passes_through(self):
        assert us_mod._none_if_na(42) == 42


# ---------------------------------------------------------------------------
# Happy-path import
# ---------------------------------------------------------------------------

class TestMainHappyPath:
    def setup_method(self):
        self.n = 3
        self.df = _make_df(self.n)
        self.supabase_mock = _make_supabase_mock(self.n)

    def test_listings_table_called(self):
        _run(self.df, self.supabase_mock)
        tables = [c.args[0] for c in self.supabase_mock.table.call_args_list]
        assert "listings" in tables

    def test_listing_us_table_called(self):
        _run(self.df, self.supabase_mock)
        tables = [c.args[0] for c in self.supabase_mock.table.call_args_list]
        assert "listing_us" in tables

    def test_dataset_id_in_listings_payload(self):
        _run(self.df, self.supabase_mock)
        insert_mock = self.supabase_mock.table.return_value
        found = any(
            r.get("dataset_id") == 99
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
        )
        assert found

    def test_default_market_is_us(self):
        _run(self.df, self.supabase_mock)
        insert_mock = self.supabase_mock.table.return_value
        markets = [
            r["market"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "market" in r
        ]
        assert all(m == "US" for m in markets)

    def test_default_currency_is_usd(self):
        _run(self.df, self.supabase_mock)
        insert_mock = self.supabase_mock.table.return_value
        currencies = [
            r["currency"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "currency" in r
        ]
        assert all(c == "USD" for c in currencies)

    def test_default_mileage_unit_is_mi(self):
        _run(self.df, self.supabase_mock)
        insert_mock = self.supabase_mock.table.return_value
        units = [
            r["mileage_unit"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "mileage_unit" in r
        ]
        assert all(u == "mi" for u in units)

    def test_title_is_none_in_listings(self):
        _run(self.df, self.supabase_mock)
        insert_mock = self.supabase_mock.table.return_value
        for c in insert_mock.insert.call_args_list:
            for r in c.args[0]:
                if "title" in r:
                    assert r["title"] is None

    def test_registration_is_none_in_listings(self):
        _run(self.df, self.supabase_mock)
        insert_mock = self.supabase_mock.table.return_value
        for c in insert_mock.insert.call_args_list:
            for r in c.args[0]:
                if "registration" in r:
                    assert r["registration"] is None


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------

class TestCliOverrides:
    def test_market_override(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock, extra_args=["--market", "CA"])
        insert_mock = mock.table.return_value
        markets = [
            r["market"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "market" in r
        ]
        assert all(m == "CA" for m in markets)

    def test_currency_override(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock, extra_args=["--currency", "CAD"])
        insert_mock = mock.table.return_value
        currencies = [
            r["currency"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "currency" in r
        ]
        assert all(c == "CAD" for c in currencies)

    def test_mileage_unit_override(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock, extra_args=["--mileage-unit", "km"])
        insert_mock = mock.table.return_value
        units = [
            r["mileage_unit"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "mileage_unit" in r
        ]
        assert all(u == "km" for u in units)

    def test_source_override(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock, extra_args=["--source", "cars.com"])
        insert_mock = mock.table.return_value
        sources = [
            r["source"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "source" in r
        ]
        assert all(s == "cars.com" for s in sources)


# ---------------------------------------------------------------------------
# Field mapping: make → brand, vdp_url → url
# ---------------------------------------------------------------------------

class TestFieldMapping:
    def test_make_mapped_to_brand_in_listings(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock)
        insert_mock = mock.table.return_value
        brands = [
            r["brand"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "brand" in r
        ]
        assert brands == ["Ford", "Ford"]

    def test_vdp_url_mapped_to_url_in_listings(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock)
        insert_mock = mock.table.return_value
        urls = [
            r["url"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "url" in r
        ]
        expected = [f"https://auto.dev/listing/{i}" for i in range(2)]
        assert urls == expected

    def test_year_in_listings_payload(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock)
        insert_mock = mock.table.return_value
        years = [
            r["year"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "year" in r
        ]
        assert years == [2020, 2021]


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNanHandling:
    def _get_us_payloads(self, mock):
        return [
            r
            for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "vin" in r  # listing_us rows always contain 'vin'
        ]

    def test_nan_price_becomes_none_in_listings(self):
        df = _make_df(1)
        df.loc[0, "price_usd"] = float("nan")
        mock = _make_supabase_mock(1)
        _run(df, mock)
        rows = [
            r for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0] if "price" in r
        ]
        assert rows[0]["price"] is None

    def test_nan_mileage_becomes_none_in_listings(self):
        df = _make_df(1)
        df.loc[0, "mileage_mi"] = float("nan")
        mock = _make_supabase_mock(1)
        _run(df, mock)
        rows = [
            r for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0] if "mileage" in r
        ]
        assert rows[0]["mileage"] is None

    def test_nan_year_becomes_none_in_listings(self):
        df = _make_df(1)
        df.loc[0, "year"] = float("nan")
        mock = _make_supabase_mock(1)
        _run(df, mock)
        rows = [
            r for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0] if "year" in r
        ]
        assert rows[0]["year"] is None

    def test_nan_vin_becomes_none_in_listing_us(self):
        df = _make_df(1)
        df.loc[0, "vin"] = float("nan")
        mock = _make_supabase_mock(1)
        _run(df, mock)
        us_rows = self._get_us_payloads(mock)
        assert us_rows[0]["vin"] is None

    def test_nan_accident_count_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "accident_count"] = float("nan")
        mock = _make_supabase_mock(1)
        _run(df, mock)
        us_rows = self._get_us_payloads(mock)
        assert us_rows[0]["accident_count"] is None

    def test_nan_owner_count_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "owner_count"] = float("nan")
        mock = _make_supabase_mock(1)
        _run(df, mock)
        us_rows = self._get_us_payloads(mock)
        assert us_rows[0]["owner_count"] is None

    def test_nan_car_age_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "car_age"] = float("nan")
        mock = _make_supabase_mock(1)
        _run(df, mock)
        us_rows = self._get_us_payloads(mock)
        assert us_rows[0]["car_age"] is None

    def test_nan_price_per_mile_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "price_per_mile"] = float("nan")
        mock = _make_supabase_mock(1)
        _run(df, mock)
        us_rows = self._get_us_payloads(mock)
        assert us_rows[0]["price_per_mile"] is None

    def test_nan_listing_age_days_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "listing_age_days"] = float("nan")
        mock = _make_supabase_mock(1)
        _run(df, mock)
        us_rows = self._get_us_payloads(mock)
        assert us_rows[0]["listing_age_days"] is None


# ---------------------------------------------------------------------------
# listing_us field completeness
# ---------------------------------------------------------------------------

class TestListingUsFields:
    EXPECTED_FIELDS = [
        "listing_id", "vin", "make", "model", "trim", "year",
        "price_usd", "mileage_mi", "carfax_url", "vdp_url",
        "primary_image_url", "dealer_name", "city", "state", "zip_code",
        "is_used", "is_cpo", "is_online", "is_wholesale",
        "accident_count", "has_accidents", "owner_count", "one_owner",
        "personal_use", "usage_type", "body_style", "drivetrain", "engine",
        "cylinders", "doors", "seats", "fuel", "transmission",
        "exterior_color", "interior_color", "car_age", "price_per_mile",
        "listing_age_days", "location",
    ]

    def test_all_expected_fields_present_in_listing_us(self):
        df = _make_df(1)
        mock = _make_supabase_mock(1)
        _run(df, mock)

        us_rows = [
            r
            for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "vin" in r
        ]
        assert len(us_rows) == 1
        for field in self.EXPECTED_FIELDS:
            assert field in us_rows[0], f"Field '{field}' missing from listing_us payload"


# ---------------------------------------------------------------------------
# listing_id propagation
# ---------------------------------------------------------------------------

class TestListingIdPropagation:
    def test_listing_ids_passed_to_listing_us(self):
        """IDs returned by listings insert must appear as listing_id in listing_us."""
        n = 3
        df = _make_df(n)

        fixed_ids = [10, 20, 30]
        call_count = {"n": 0}

        def _insert(rows):
            m = MagicMock()
            if call_count["n"] == 0:
                m.execute.return_value.data = [{"id": i} for i in fixed_ids]
            else:
                m.execute.return_value.data = []
            call_count["n"] += 1
            return m

        supabase_mock = MagicMock()
        supabase_mock.table.return_value.insert.side_effect = _insert

        with patch.object(us_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            us_mod.main()

        us_rows = [
            r
            for c in supabase_mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "vin" in r
        ]
        actual_ids = [r["listing_id"] for r in us_rows]
        assert actual_ids == fixed_ids


# ---------------------------------------------------------------------------
# year type check
# ---------------------------------------------------------------------------

class TestYearType:
    def test_year_is_int_in_listing_us(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock)

        us_rows = [
            r
            for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "vin" in r
        ]
        for row in us_rows:
            if row["year"] is not None:
                assert isinstance(row["year"], int), \
                    f"Expected int, got {type(row['year'])} for year={row['year']}"

    def test_year_is_int_in_listings(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock)

        listing_rows = [
            r
            for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "currency" in r  # listings rows always carry currency
        ]
        for row in listing_rows:
            if row.get("year") is not None:
                assert isinstance(row["year"], int), \
                    f"Expected int, got {type(row['year'])} for year={row['year']}"


# ---------------------------------------------------------------------------
# Default source
# ---------------------------------------------------------------------------

class TestDefaultSource:
    def test_default_source_is_auto_dev(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        _run(df, mock)

        sources = [
            r["source"]
            for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "source" in r
        ]
        assert all(s == "auto.dev" for s in sources)


# ---------------------------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------------------------

class TestEmptyDataFrame:
    def test_empty_csv_does_not_crash(self):
        """An empty CSV (0 rows) should complete without error."""
        df = _make_df(0)

        def _insert(rows):
            m = MagicMock()
            m.execute.return_value.data = []
            return m

        supabase_mock = MagicMock()
        supabase_mock.table.return_value.insert.side_effect = _insert

        with patch.object(us_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            us_mod.main()  # must not raise


# ---------------------------------------------------------------------------
# Mismatch guard
# ---------------------------------------------------------------------------

class TestInsertMismatch:
    def test_raises_on_id_count_mismatch(self):
        df = _make_df(3)

        supabase_mock = MagicMock()
        insert_mock = MagicMock()
        insert_mock.execute.return_value.data = [{"id": 1}]  # only 1 ID returned
        supabase_mock.table.return_value.insert.return_value = insert_mock

        with patch.object(us_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            with pytest.raises(RuntimeError, match="mismatch"):
                us_mod.main()


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_large_dataset_split_into_batches(self):
        """1200 rows → ≥3 insert calls each for listings and listing_us."""
        n = 1200
        df = _make_df(n)
        ids = iter(range(1, n + 1))

        supabase_mock = MagicMock()

        def _insert(rows):
            m = MagicMock()
            m.execute.return_value.data = [{"id": next(ids)} for _ in rows]
            return m

        supabase_mock.table.return_value.insert.side_effect = _insert

        with patch.object(us_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            us_mod.main()

        total_insert_calls = supabase_mock.table.return_value.insert.call_count
        # 3 batches for listings + 3 for listing_us = 6 minimum
        assert total_insert_calls >= 6
