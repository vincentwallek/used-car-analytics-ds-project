import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "data_processing"))

# ---------------------------------------------------------------------------
# Patch external dependencies BEFORE importing the module under test
# ---------------------------------------------------------------------------
os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "fake-key")

with patch("supabase.create_client", return_value=MagicMock()), \
     patch("dotenv.load_dotenv"):
    import import_de_csv as de_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=3, missing_cols=None):
    """Return a minimal valid DE CSV DataFrame."""
    data = {
        "title":        [f"BMW 3er {i}" for i in range(n)],
        "price_eur":    [15000.0 + i * 1000 for i in range(n)],
        "mileage_km":   [80000.0 + i * 5000 for i in range(n)],
        "power_ps":     [150.0 + i * 10 for i in range(n)],
        "transmission": ["Automatik"] * n,
        "fuel":         ["Benzin"] * n,
        "owners":       [1] * n,
        "location":     [f"Berlin {i}" for i in range(n)],
        "URL":          [f"https://mobile.de/anzeige/{i}" for i in range(n)],
        "Ausstattung":  [f"Klimaanlage, Navi {i}" for i in range(n)],
        "Beschreibung": [f"Sehr gepflegtes Fahrzeug {i}" for i in range(n)],
        "registration": [f"2020-0{i+1}" for i in range(n)],
        "brand":        ["BMW"] * n,
        "model":        ["3er"] * n,
        "car_age":      [3.0 + i for i in range(n)],
        "price_per_km": [0.19 + i * 0.01 for i in range(n)],
    }
    df = pd.DataFrame(data)
    if missing_cols:
        df = df.drop(columns=missing_cols)
    return df


def _make_supabase_mock(n=3):
    """Return a supabase mock that returns sequential IDs on insert."""
    import itertools
    ids = itertools.count(1)

    def _insert_side_effect(rows):
        mock = MagicMock()
        mock.execute.return_value.data = [{"id": next(ids)} for _ in rows]
        return mock

    supabase = MagicMock()
    supabase.table.return_value.insert.side_effect = _insert_side_effect
    return supabase


# ---------------------------------------------------------------------------
# _none_if_na
# ---------------------------------------------------------------------------

class TestNoneIfNa:
    def test_returns_none_for_float_nan(self):
        assert de_mod._none_if_na(float("nan")) is None

    def test_returns_none_for_pd_na(self):
        assert de_mod._none_if_na(pd.NA) is None

    def test_returns_none_for_np_nan(self):
        assert de_mod._none_if_na(np.nan) is None

    def test_passes_through_string(self):
        assert de_mod._none_if_na("hello") == "hello"

    def test_passes_through_zero(self):
        assert de_mod._none_if_na(0) == 0

    def test_passes_through_false(self):
        assert de_mod._none_if_na(False) is False


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

class TestColumnValidation:
    def test_raises_on_missing_required_column(self):
        df = _make_df(missing_cols=["price_eur"])
        with patch.object(de_mod, "supabase", _make_supabase_mock()), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            with pytest.raises(ValueError, match="Missing columns"):
                de_mod.main()

    def test_raises_on_multiple_missing_columns(self):
        df = _make_df(missing_cols=["mileage_km", "owners"])
        with patch.object(de_mod, "supabase", _make_supabase_mock()), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            with pytest.raises(ValueError, match="Missing columns"):
                de_mod.main()

    def test_url_column_renamed(self):
        """Capital 'URL' in CSV must be renamed to lowercase 'url'."""
        df = _make_df()
        assert "URL" in df.columns
        assert "url" not in df.columns

        supabase_mock = _make_supabase_mock(3)
        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()  # must not raise


# ---------------------------------------------------------------------------
# Happy-path import
# ---------------------------------------------------------------------------

class TestMainHappyPath:
    def setup_method(self):
        self.n = 3
        self.df = _make_df(self.n)
        self.supabase_mock = _make_supabase_mock(self.n)

    def _run(self, extra_args=None):
        argv = ["prog", "--csv", "dummy.csv", "--dataset-id", "42"] + (extra_args or [])
        with patch.object(de_mod, "supabase", self.supabase_mock), \
             patch("pandas.read_csv", return_value=self.df), \
             patch("sys.argv", argv):
            de_mod.main()

    def test_listings_table_called(self):
        self._run()
        calls = [str(c) for c in self.supabase_mock.table.call_args_list]
        assert any("listings" in c for c in calls)

    def test_listing_de_table_called(self):
        self._run()
        calls = [str(c) for c in self.supabase_mock.table.call_args_list]
        assert any("listing_de" in c for c in calls)

    def test_dataset_id_in_listings_payload(self):
        self._run()
        # Find the insert call on 'listings'
        insert_calls = []
        for table_call in self.supabase_mock.table.call_args_list:
            if table_call.args[0] == "listings":
                insert_mock = self.supabase_mock.table.return_value
                insert_calls = insert_mock.insert.call_args_list
        # At least one row should carry dataset_id=42
        found = False
        for c in insert_calls:
            rows = c.args[0]
            if any(r.get("dataset_id") == 42 for r in rows):
                found = True
        assert found

    def test_default_market_is_de(self):
        self._run()
        insert_mock = self.supabase_mock.table.return_value
        for c in insert_mock.insert.call_args_list:
            rows = c.args[0]
            for r in rows:
                if "market" in r:
                    assert r["market"] == "DE"

    def test_custom_market_override(self):
        self._run(extra_args=["--market", "AT"])
        insert_mock = self.supabase_mock.table.return_value
        markets = [
            r["market"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "market" in r
        ]
        assert all(m == "AT" for m in markets)

    def test_default_currency_is_eur(self):
        self._run()
        insert_mock = self.supabase_mock.table.return_value
        for c in insert_mock.insert.call_args_list:
            for r in c.args[0]:
                if "currency" in r:
                    assert r["currency"] == "EUR"

    def test_default_mileage_unit_is_km(self):
        self._run()
        insert_mock = self.supabase_mock.table.return_value
        units = [
            r["mileage_unit"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "mileage_unit" in r
        ]
        assert all(u == "km" for u in units)

    def test_url_value_correct_in_listings(self):
        self._run()
        insert_mock = self.supabase_mock.table.return_value
        urls = [
            r["url"]
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "url" in r
        ]
        expected = [f"https://mobile.de/anzeige/{i}" for i in range(self.n)]
        assert urls == expected

    def test_brand_lowercased(self):
        self._run()
        insert_mock = self.supabase_mock.table.return_value
        for c in insert_mock.insert.call_args_list:
            for r in c.args[0]:
                if "brand" in r and r["brand"] is not None:
                    assert r["brand"] == r["brand"].lower()

    def test_model_lowercased(self):
        self._run()
        insert_mock = self.supabase_mock.table.return_value
        for c in insert_mock.insert.call_args_list:
            for r in c.args[0]:
                if "model" in r and r["model"] is not None:
                    assert r["model"] == r["model"].lower()


# ---------------------------------------------------------------------------
# NaN handling in listings payload
# ---------------------------------------------------------------------------

class TestNanHandling:
    def test_nan_price_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "price_eur"] = float("nan")
        supabase_mock = _make_supabase_mock(1)

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        insert_mock = supabase_mock.table.return_value
        for c in insert_mock.insert.call_args_list:
            for r in c.args[0]:
                if "price" in r:
                    assert r["price"] is None

    def test_nan_mileage_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "mileage_km"] = float("nan")
        supabase_mock = _make_supabase_mock(1)

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        insert_mock = supabase_mock.table.return_value
        for c in insert_mock.insert.call_args_list:
            for r in c.args[0]:
                if "mileage" in r:
                    assert r["mileage"] is None

    def test_nan_owners_becomes_none_in_listing_de(self):
        df = _make_df(1)
        df.loc[0, "owners"] = float("nan")
        supabase_mock = _make_supabase_mock(1)

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        insert_mock = supabase_mock.table.return_value
        de_payloads = [
            r
            for c in insert_mock.insert.call_args_list
            for r in c.args[0]
            if "owners" in r
        ]
        assert de_payloads[0]["owners"] is None

    def test_owners_cast_to_int(self):
        df = _make_df(2)
        supabase_mock = _make_supabase_mock(2)

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        de_payloads = [
            r
            for c in supabase_mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "owners" in r and r["owners"] is not None
        ]
        for row in de_payloads:
            assert isinstance(row["owners"], int), \
                f"Expected int for owners, got {type(row['owners'])}"

    def test_nan_power_ps_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "power_ps"] = float("nan")
        supabase_mock = _make_supabase_mock(1)

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        de_payloads = [
            r
            for c in supabase_mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "power_ps" in r
        ]
        assert de_payloads[0]["power_ps"] is None

    def test_nan_car_age_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "car_age"] = float("nan")
        supabase_mock = _make_supabase_mock(1)

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        de_payloads = [
            r
            for c in supabase_mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "car_age" in r
        ]
        assert de_payloads[0]["car_age"] is None

    def test_nan_price_per_km_becomes_none(self):
        df = _make_df(1)
        df.loc[0, "price_per_km"] = float("nan")
        supabase_mock = _make_supabase_mock(1)

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        de_payloads = [
            r
            for c in supabase_mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "price_per_km" in r
        ]
        assert de_payloads[0]["price_per_km"] is None


# ---------------------------------------------------------------------------
# Mismatch guard
# ---------------------------------------------------------------------------

class TestInsertMismatch:
    def test_raises_on_id_count_mismatch(self):
        """If supabase returns fewer IDs than rows, RuntimeError must be raised."""
        df = _make_df(3)

        # Mock returns only 1 ID instead of 3
        supabase_mock = MagicMock()
        insert_mock = MagicMock()
        insert_mock.execute.return_value.data = [{"id": 1}]  # only 1 ID
        supabase_mock.table.return_value.insert.return_value = insert_mock

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            with pytest.raises(RuntimeError, match="mismatch"):
                de_mod.main()


# ---------------------------------------------------------------------------
# listing_de field completeness
# ---------------------------------------------------------------------------

class TestListingDeFields:
    EXPECTED_FIELDS = [
        "listing_id", "transmission", "fuel", "owners", "power_ps",
        "car_age", "price_per_km", "ausstattung", "beschreibung",
    ]

    def _get_de_rows(self, mock):
        """listing_de rows always contain 'ausstattung'."""
        return [
            r
            for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "ausstattung" in r
        ]

    def test_all_expected_fields_present(self):
        df = _make_df(1)
        mock = _make_supabase_mock(1)
        with patch.object(de_mod, "supabase", mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        de_rows = self._get_de_rows(mock)
        assert len(de_rows) == 1
        for field in self.EXPECTED_FIELDS:
            assert field in de_rows[0], f"Field '{field}' missing from listing_de payload"

    def test_ausstattung_value_correct(self):
        df = _make_df(1)
        mock = _make_supabase_mock(1)
        with patch.object(de_mod, "supabase", mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        de_rows = self._get_de_rows(mock)
        assert de_rows[0]["ausstattung"] == "Klimaanlage, Navi 0"

    def test_beschreibung_value_correct(self):
        df = _make_df(1)
        mock = _make_supabase_mock(1)
        with patch.object(de_mod, "supabase", mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        de_rows = self._get_de_rows(mock)
        assert de_rows[0]["beschreibung"] == "Sehr gepflegtes Fahrzeug 0"

    def test_power_ps_value_correct(self):
        df = _make_df(1)
        mock = _make_supabase_mock(1)
        with patch.object(de_mod, "supabase", mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        de_rows = self._get_de_rows(mock)
        assert de_rows[0]["power_ps"] == 150.0


# ---------------------------------------------------------------------------
# listing_id propagation
# ---------------------------------------------------------------------------

class TestListingIdPropagation:
    def test_listing_ids_passed_to_listing_de(self):
        """IDs returned by listings insert must appear as listing_id in listing_de."""
        n = 3
        df = _make_df(n)

        # listings insert returns IDs 10, 20, 30
        fixed_ids = [10, 20, 30]
        call_count = {"n": 0}

        def _insert(rows):
            m = MagicMock()
            if call_count["n"] == 0:
                # first call = listings batch
                m.execute.return_value.data = [{"id": i} for i in fixed_ids]
            else:
                m.execute.return_value.data = []
            call_count["n"] += 1
            return m

        supabase_mock = MagicMock()
        supabase_mock.table.return_value.insert.side_effect = _insert

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        de_rows = [
            r
            for c in supabase_mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "ausstattung" in r
        ]
        actual_ids = [r["listing_id"] for r in de_rows]
        assert actual_ids == fixed_ids


# ---------------------------------------------------------------------------
# Default source
# ---------------------------------------------------------------------------

class TestDefaultSource:
    def test_default_source_is_mobile(self):
        df = _make_df(2)
        mock = _make_supabase_mock(2)
        with patch.object(de_mod, "supabase", mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        sources = [
            r["source"]
            for c in mock.table.return_value.insert.call_args_list
            for r in c.args[0]
            if "source" in r
        ]
        assert all(s == "mobile" for s in sources)


# ---------------------------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------------------------

class TestEmptyDataFrame:
    def test_empty_csv_does_not_crash(self):
        """An empty CSV (0 rows) should complete without error."""
        df = _make_df(0)
        mock = _make_supabase_mock(0)

        # Override side_effect: every insert returns empty data
        def _insert(rows):
            m = MagicMock()
            m.execute.return_value.data = []
            return m

        mock.table.return_value.insert.side_effect = _insert

        with patch.object(de_mod, "supabase", mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()  # must not raise


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_large_dataset_is_split_into_batches(self):
        """1200 rows → at least 3 insert calls on 'listings' (batch_size=500)."""
        n = 1200
        df = _make_df(n)
        ids = iter(range(1, n + 1))

        supabase_mock = MagicMock()

        def _insert(rows):
            m = MagicMock()
            m.execute.return_value.data = [{"id": next(ids)} for _ in rows]
            return m

        supabase_mock.table.return_value.insert.side_effect = _insert

        with patch.object(de_mod, "supabase", supabase_mock), \
             patch("pandas.read_csv", return_value=df), \
             patch("sys.argv", ["prog", "--csv", "x.csv", "--dataset-id", "1"]):
            de_mod.main()

        total_insert_calls = supabase_mock.table.return_value.insert.call_count
        # 3 batches for listings + 3 batches for listing_de = 6
        assert total_insert_calls >= 6
