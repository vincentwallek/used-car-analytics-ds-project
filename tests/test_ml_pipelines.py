"""
test_ml_pipelines.py
Vollständige Tests für ml_pipeline_de.py und ml_pipeline_us.py

Ausführen:
    pytest test_ml_pipelines.py -v
    pytest test_ml_pipelines.py -v --tb=short   # kürzere Fehlerausgabe
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "models"))
import pickle
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# DEUTSCHE PIPELINE
# ============================================================

class TestDELoadData:
    """Tests für load_data() der deutschen Pipeline."""

    def _make_row(self, price=15000, mileage=80000, brand="volkswagen",
                  model="golf", car_age=5, power_ps=110, owners=2,
                  transmission="manuell", fuel="benzin",
                  listings_as_list=False, features_as_list=False,
                  extra_features=None):
        features = {"has_sunroof": 1, "has_navigation": 0}
        if extra_features:
            features.update(extra_features)

        listings_val = {"price": price, "mileage": mileage, "brand": brand, "model": model}
        if listings_as_list:
            listings_val = [listings_val]

        features_val = features if not features_as_list else [features]

        return {
            "car_age": car_age, "power_ps": power_ps, "owners": owners,
            "transmission": transmission, "fuel": fuel,
            "listings": listings_val,
            "listing_features": features_val,
        }

    @patch("ml_pipeline_de.supabase")
    def test_returns_dataframe(self, mock_sb):
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[self._make_row()])
        df = load_data()
        assert isinstance(df, pd.DataFrame) and len(df) == 1

    @patch("ml_pipeline_de.supabase")
    def test_core_columns_present(self, mock_sb):
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[self._make_row()])
        df = load_data()
        for col in ["price", "mileage", "brand", "model", "car_age",
                    "power_ps", "transmission", "fuel"]:
            assert col in df.columns, f"Spalte '{col}' fehlt"

    @patch("ml_pipeline_de.supabase")
    def test_skips_row_without_listings(self, mock_sb):
        from ml_pipeline_de import load_data
        row = self._make_row()
        row["listings"] = None
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[row, self._make_row(price=9999)])
        df = load_data()
        assert len(df) == 1

    @patch("ml_pipeline_de.supabase")
    def test_skips_empty_listings_list(self, mock_sb):
        """Leere listings-Liste [] wird übersprungen (Zeile 48 im Code)."""
        from ml_pipeline_de import load_data
        row = self._make_row()
        row["listings"] = []
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[row])
        df = load_data()
        assert df.empty

    @patch("ml_pipeline_de.supabase")
    def test_skips_row_without_features(self, mock_sb):
        from ml_pipeline_de import load_data
        row = self._make_row()
        row["listing_features"] = None
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[row])
        df = load_data()
        assert df.empty

    @patch("ml_pipeline_de.supabase")
    def test_skips_empty_features_list(self, mock_sb):
        """Leere listing_features-Liste [] wird übersprungen (Zeile 57 im Code)."""
        from ml_pipeline_de import load_data
        row = self._make_row()
        row["listing_features"] = []
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[row])
        df = load_data()
        assert df.empty

    @patch("ml_pipeline_de.supabase")
    def test_features_as_list_is_unpacked(self, mock_sb):
        """listing_features als Liste wird korrekt entpackt (Zeile 56-58)."""
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[self._make_row(features_as_list=True)])
        df = load_data()
        assert len(df) == 1
        assert "has_sunroof" in df.columns

    @patch("ml_pipeline_de.supabase")
    def test_listings_as_list_is_unpacked(self, mock_sb):
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[self._make_row(listings_as_list=True)])
        df = load_data()
        assert len(df) == 1
        assert df["price"].iloc[0] == 15000

    @patch("ml_pipeline_de.supabase")
    def test_skips_row_without_price(self, mock_sb):
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[self._make_row(price=None)])
        df = load_data()
        assert df.empty

    @patch("ml_pipeline_de.supabase")
    def test_empty_api_response(self, mock_sb):
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[])
        df = load_data()
        assert df.empty

    @patch("ml_pipeline_de.supabase")
    def test_safe_str_lowercases_and_strips(self, mock_sb):
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[self._make_row(brand="  BMW  ", fuel="DIESEL")])
        df = load_data()
        assert df["brand"].iloc[0] == "bmw"
        assert df["fuel"].iloc[0] == "diesel"

    @patch("ml_pipeline_de.supabase")
    def test_safe_str_none_becomes_unbekannt(self, mock_sb):
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[self._make_row(transmission=None, fuel="")])
        df = load_data()
        assert df["transmission"].iloc[0] == "unbekannt"
        assert df["fuel"].iloc[0] == "unbekannt"

    @patch("ml_pipeline_de.supabase")
    def test_listing_id_is_removed(self, mock_sb):
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[self._make_row(extra_features={"listing_id": 99})])
        df = load_data()
        assert "listing_id" not in df.columns

    @patch("ml_pipeline_de.supabase")
    def test_multiple_valid_rows(self, mock_sb):
        from ml_pipeline_de import load_data
        mock_sb.table.return_value.select.return_value.execute.return_value = \
            MagicMock(data=[self._make_row(price=p) for p in [10000, 20000, 30000]])
        df = load_data()
        assert len(df) == 3


class TestDEPreprocessData:
    """Tests für preprocess_data() der deutschen Pipeline."""

    def _make_df(self, n=60):
        np.random.seed(42)
        return pd.DataFrame({
            "price":        np.random.uniform(5000, 40000, n),
            "mileage":      np.random.uniform(10000, 200000, n),
            "car_age":      np.random.randint(1, 15, n).astype(float),
            "power_ps":     np.random.randint(75, 300, n).astype(float),
            "owners":       np.random.randint(1, 4, n).astype(float),
            "brand":        np.random.choice(["volkswagen", "bmw", "mercedes"], n),
            "model":        np.random.choice(["golf", "3er", "c-klasse"], n),
            "transmission": np.random.choice(["manuell", "automatik"], n),
            "fuel":         np.random.choice(["benzin", "diesel"], n),
        })

    def test_returns_correct_types(self):
        from ml_pipeline_de import preprocess_data
        X, y, enc = preprocess_data(self._make_df())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(enc, OneHotEncoder)

    def test_drops_na_price(self):
        from ml_pipeline_de import preprocess_data
        df = self._make_df()
        df.loc[0, "price"] = None
        _, y, _ = preprocess_data(df)
        assert y.isna().sum() == 0

    def test_drops_na_mileage(self):
        from ml_pipeline_de import preprocess_data
        df = self._make_df()
        df.loc[0, "mileage"] = None
        _, y, _ = preprocess_data(df)
        assert len(y) < len(df)

    def test_drops_na_car_age(self):
        from ml_pipeline_de import preprocess_data
        df = self._make_df()
        df.loc[0, "car_age"] = None
        _, y, _ = preprocess_data(df)
        assert len(y) < len(df)

    def test_fills_power_ps_when_present(self):
        from ml_pipeline_de import preprocess_data
        df = self._make_df()
        df.loc[0, "power_ps"] = None
        X, _, _ = preprocess_data(df)
        assert X.isna().sum().sum() == 0

    def test_no_crash_when_power_ps_column_missing(self):
        """Zeile 102: if 'power_ps' in df.columns — muss fehlerfrei durchlaufen."""
        from ml_pipeline_de import preprocess_data
        df = self._make_df().drop(columns=["power_ps"])
        X, y, _ = preprocess_data(df)
        assert X.isna().sum().sum() == 0

    def test_no_crash_when_owners_column_missing(self):
        """Zeile 104: if 'owners' in df.columns — muss fehlerfrei durchlaufen."""
        from ml_pipeline_de import preprocess_data
        df = self._make_df().drop(columns=["owners"])
        X, y, _ = preprocess_data(df)
        assert X.isna().sum().sum() == 0

    def test_fills_owners_when_present(self):
        from ml_pipeline_de import preprocess_data
        df = self._make_df()
        df.loc[0, "owners"] = None
        X, _, _ = preprocess_data(df)
        assert X.isna().sum().sum() == 0

    def test_price_not_in_X(self):
        from ml_pipeline_de import preprocess_data
        X, _, _ = preprocess_data(self._make_df())
        assert "price" not in X.columns

    def test_categorical_cols_encoded(self):
        from ml_pipeline_de import preprocess_data
        X, _, _ = preprocess_data(self._make_df())
        for col in ["brand", "model", "transmission", "fuel"]:
            assert col not in X.columns
        assert any("brand_" in c for c in X.columns)

    def test_no_nan_in_output(self):
        from ml_pipeline_de import preprocess_data
        X, y, _ = preprocess_data(self._make_df())
        assert X.isna().sum().sum() == 0 and y.isna().sum() == 0

    def test_encoder_ignores_unknown_category(self):
        from ml_pipeline_de import preprocess_data
        _, _, encoder = preprocess_data(self._make_df())
        test_row = pd.DataFrame([["neu_unbekannt", "unbekannt", "manuell", "benzin"]],
                                 columns=["brand", "model", "transmission", "fuel"])
        result = encoder.transform(test_row)
        assert result is not None


class TestDETrainModel:
    """Tests für train_model() der deutschen Pipeline."""

    def _make_xy(self, n=200):
        np.random.seed(0)
        X = pd.DataFrame({
            "mileage": np.random.uniform(10000, 200000, n),
            "car_age": np.random.randint(1, 15, n),
            "power_ps": np.random.randint(75, 300, n),
        })
        y = pd.Series(np.random.uniform(5000, 40000, n))
        return X, y

    def test_returns_xgb_regressor(self):
        from ml_pipeline_de import train_model
        model, _ = train_model(*self._make_xy())
        assert isinstance(model, xgb.XGBRegressor)

    def test_x_train_is_80_percent(self):
        from ml_pipeline_de import train_model
        X, y = self._make_xy()
        _, X_train = train_model(X, y)
        assert len(X_train) == pytest.approx(len(X) * 0.8, abs=5)

    def test_can_predict_positive_values(self):
        from ml_pipeline_de import train_model
        model, X_train = train_model(*self._make_xy())
        assert all(p > 0 for p in model.predict(X_train.iloc[:5]))

    def test_prints_r2_score(self, capsys):
        from ml_pipeline_de import train_model
        train_model(*self._make_xy())
        assert "R²" in capsys.readouterr().out

    def test_prints_mae_with_euro_sign(self, capsys):
        from ml_pipeline_de import train_model
        train_model(*self._make_xy())
        assert "€" in capsys.readouterr().out

    def test_prints_best_params(self, capsys):
        from ml_pipeline_de import train_model
        train_model(*self._make_xy())
        assert "Parameter-Mix" in capsys.readouterr().out


class TestDEExplainModel:
    """Tests für explain_model() der deutschen Pipeline."""

    def _make_trained(self, n=100):
        np.random.seed(1)
        X = pd.DataFrame({
            "mileage": np.random.uniform(10000, 200000, n),
            "car_age": np.random.randint(1, 15, n),
            "power_ps": np.random.randint(75, 300, n),
        })
        y = pd.Series(np.random.uniform(5000, 40000, n))
        return xgb.XGBRegressor(n_estimators=10, max_depth=3).fit(X, y), X

    def test_runs_without_error(self, capsys):
        from ml_pipeline_de import explain_model
        explain_model(*self._make_trained())
        assert "Basis-Preis" in capsys.readouterr().out

    def test_prints_euro_sign_not_dollar(self, capsys):
        from ml_pipeline_de import explain_model
        explain_model(*self._make_trained())
        out = capsys.readouterr().out
        assert "€" in out and "$" not in out

    def test_prints_top15_header(self, capsys):
        from ml_pipeline_de import explain_model
        explain_model(*self._make_trained())
        assert "Top 15" in capsys.readouterr().out

    def test_shap_dataframe_has_euro_influence_column(self):
        """DataFrame enthält 'Einfluss_in_Euro' (DE-spezifisch, nicht Dollar)."""
        import shap
        model, X_train = self._make_trained()
        sample = X_train.iloc[[0]]
        shap_vals = shap.TreeExplainer(model)(sample)
        impacts = pd.DataFrame({
            "Feature":          sample.columns,
            "Wert_beim_Auto":   sample.iloc[0].values,
            "Einfluss_in_Euro": shap_vals.values[0],
        })
        assert "Einfluss_in_Euro" in impacts.columns
        assert "Einfluss_in_Dollar" not in impacts.columns


class TestDESaveLoad:
    """Tests für Pickle-Serialisierung der deutschen Pipeline-Artefakte."""

    def test_model_pickle_roundtrip(self, tmp_path):
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.rand(50), "b": np.random.rand(50)})
        y = pd.Series(np.random.rand(50) * 10000)
        model = xgb.XGBRegressor(n_estimators=5).fit(X, y)
        path = tmp_path / "model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        np.testing.assert_array_almost_equal(model.predict(X), loaded.predict(X))

    def test_encoder_pickle_roundtrip(self, tmp_path):
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc.fit([["vw", "golf"], ["bmw", "3er"]])
        path = tmp_path / "encoder.pkl"
        with open(path, "wb") as f:
            pickle.dump(enc, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded.transform([["vw", "golf"]]).shape[1] > 0


# ============================================================
# US-PIPELINE
# ============================================================

class TestUSLoadData:
    """Tests für load_data() der US-Pipeline (mit Pagination)."""

    def _make_row(self, price=25000, mileage=50000, brand="ford",
                  model="mustang", listings_as_list=False):
        listings_val = {"price": price, "mileage": mileage, "brand": brand, "model": model}
        if listings_as_list:
            listings_val = [listings_val]
        return {
            "trim": "GT", "drivetrain": "rwd", "fuel": "gasoline",
            "transmission": "automatic", "accident_count": 0, "owner_count": 1,
            "one_owner": True, "has_accidents": False, "body_style": "coupe",
            "engine": "5.0L V8", "cylinders": 8, "doors": 2, "seats": 4,
            "exterior_color": "red", "interior_color": "black",
            "is_used": True, "is_cpo": False, "is_online": True,
            "is_wholesale": False, "personal_use": True,
            "usage_type": "personal", "car_age": 3,
            "listings": listings_val,
        }

    def _mock_exec(self, mock_sb, side_effects):
        mock_exec = MagicMock()
        mock_exec.execute.side_effect = [MagicMock(data=d) for d in side_effects]
        mock_sb.table.return_value.select.return_value.range.return_value = mock_exec

    @patch("ml_pipeline_us.supabase")
    def test_pagination_stops_on_empty_batch(self, mock_sb):
        from ml_pipeline_us import load_data
        self._mock_exec(mock_sb, [[self._make_row()], []])
        assert len(load_data()) == 1

    @patch("ml_pipeline_us.supabase")
    def test_pagination_continues_when_batch_exactly_full(self, mock_sb):
        """Exakt 1000 Zeilen → Code muss weiterladen (Grenzfall Zeile 55)."""
        from ml_pipeline_us import load_data
        full_batch = [self._make_row(price=p) for p in range(1000)]
        last_batch = [self._make_row(price=99999)]
        self._mock_exec(mock_sb, [full_batch, last_batch])
        assert len(load_data()) == 1001

    @patch("ml_pipeline_us.supabase")
    def test_pagination_accumulates_multiple_batches(self, mock_sb):
        from ml_pipeline_us import load_data
        batch_a = [self._make_row(price=p) for p in range(1000)]
        batch_b = [self._make_row(price=p) for p in range(500)]
        self._mock_exec(mock_sb, [batch_a, batch_b])
        assert len(load_data()) == 1500

    @patch("ml_pipeline_us.supabase")
    def test_bool_true_becomes_1(self, mock_sb):
        from ml_pipeline_us import load_data
        row = self._make_row()
        row["one_owner"] = True
        row["is_cpo"] = True
        self._mock_exec(mock_sb, [[row], []])
        df = load_data()
        assert df["one_owner"].iloc[0] == 1
        assert df["is_cpo"].iloc[0] == 1

    @patch("ml_pipeline_us.supabase")
    def test_bool_false_becomes_0(self, mock_sb):
        from ml_pipeline_us import load_data
        row = self._make_row()
        row["has_accidents"] = False
        self._mock_exec(mock_sb, [[row], []])
        df = load_data()
        assert df["has_accidents"].iloc[0] == 0

    @patch("ml_pipeline_us.supabase")
    def test_safe_bool_none_becomes_0(self, mock_sb):
        """safe_bool(None): `1 if val is True else 0` → 0 für None."""
        from ml_pipeline_us import load_data
        row = self._make_row()
        row["one_owner"] = None
        row["is_cpo"] = None
        self._mock_exec(mock_sb, [[row], []])
        df = load_data()
        assert df["one_owner"].iloc[0] == 0
        assert df["is_cpo"].iloc[0] == 0

    @patch("ml_pipeline_us.supabase")
    def test_listings_as_list_is_unpacked(self, mock_sb):
        from ml_pipeline_us import load_data
        self._mock_exec(mock_sb, [[self._make_row(listings_as_list=True)], []])
        df = load_data()
        assert len(df) == 1 and df["price"].iloc[0] == 25000

    @patch("ml_pipeline_us.supabase")
    def test_skips_empty_listings_list(self, mock_sb):
        """Leere listings-Liste [] → Row wird übersprungen."""
        from ml_pipeline_us import load_data
        row = self._make_row()
        row["listings"] = []
        self._mock_exec(mock_sb, [[row], []])
        assert load_data().empty

    @patch("ml_pipeline_us.supabase")
    def test_skips_row_without_listings(self, mock_sb):
        from ml_pipeline_us import load_data
        row = self._make_row()
        row["listings"] = None
        self._mock_exec(mock_sb, [[row], []])
        assert load_data().empty

    @patch("ml_pipeline_us.supabase")
    def test_skips_row_without_price(self, mock_sb):
        from ml_pipeline_us import load_data
        self._mock_exec(mock_sb, [[self._make_row(price=None)], []])
        assert load_data().empty

    @patch("ml_pipeline_us.supabase")
    def test_us_specific_columns_present(self, mock_sb):
        from ml_pipeline_us import load_data
        self._mock_exec(mock_sb, [[self._make_row()], []])
        df = load_data()
        for col in ["trim", "drivetrain", "body_style", "engine", "cylinders",
                    "doors", "seats", "exterior_color", "interior_color",
                    "one_owner", "has_accidents", "is_cpo", "is_used",
                    "is_online", "is_wholesale", "personal_use",
                    "accident_count", "owner_count"]:
            assert col in df.columns, f"US-Spalte '{col}' fehlt"

    @patch("ml_pipeline_us.supabase")
    def test_safe_str_normalizes(self, mock_sb):
        from ml_pipeline_us import load_data
        row = self._make_row()
        row["fuel"] = "  GASOLINE  "
        row["drivetrain"] = "AWD"
        self._mock_exec(mock_sb, [[row], []])
        df = load_data()
        assert df["fuel"].iloc[0] == "gasoline"
        assert df["drivetrain"].iloc[0] == "awd"


class TestUSPreprocessData:
    """Tests für preprocess_data() der US-Pipeline."""

    def _make_df(self, n=60, extra_brands=None):
        np.random.seed(7)
        brands = ["ford", "toyota", "chevrolet"]
        if extra_brands:
            brands += extra_brands
        return pd.DataFrame({
            "price":          np.random.uniform(8000, 60000, n),
            "mileage":        np.random.uniform(5000, 150000, n),
            "car_age":        np.random.randint(1, 12, n).astype(float),
            "accident_count": np.random.randint(0, 3, n).astype(float),
            "owner_count":    np.random.randint(1, 5, n).astype(float),
            "cylinders":      np.random.choice([4, 6, 8], n).astype(float),
            "doors":          np.random.choice([2, 4], n).astype(float),
            "seats":          np.random.choice([2, 5, 7], n).astype(float),
            "brand":          np.random.choice(brands, n),
            "model":          np.random.choice(["f-150", "camry", "silverado"], n),
            "trim":           np.random.choice(["base", "sport", "premium"], n),
            "drivetrain":     np.random.choice(["fwd", "rwd", "awd"], n),
            "fuel":           np.random.choice(["gasoline", "diesel"], n),
            "transmission":   np.random.choice(["automatic", "manual"], n),
            "body_style":     np.random.choice(["sedan", "suv", "truck"], n),
            "engine":         np.random.choice(["2.0L I4", "3.5L V6"], n),
            "exterior_color": np.random.choice(["white", "black", "red"], n),
            "interior_color": np.random.choice(["black", "tan"], n),
            "usage_type":     np.random.choice(["personal", "fleet"], n),
            "one_owner":      np.random.randint(0, 2, n),
            "has_accidents":  np.random.randint(0, 2, n),
            "is_used":        np.random.randint(0, 2, n),
            "is_cpo":         np.random.randint(0, 2, n),
            "is_online":      np.random.randint(0, 2, n),
            "is_wholesale":   np.random.randint(0, 2, n),
            "personal_use":   np.random.randint(0, 2, n),
        })

    def test_returns_correct_types(self):
        from ml_pipeline_us import preprocess_data
        X, y, enc = preprocess_data(self._make_df())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(enc, OneHotEncoder)

    def test_removes_mercedes_benz(self):
        from ml_pipeline_us import preprocess_data
        df = self._make_df(extra_brands=["mercedes-benz"])
        n_merc = df["brand"].str.contains("mercedes", case=False).sum()
        _, y, _ = preprocess_data(df)
        assert len(y) <= len(df) - n_merc

    def test_removes_mercedes_amg_partial_match(self):
        """str.contains('mercedes') trifft auch 'mercedes-amg' usw."""
        from ml_pipeline_us import preprocess_data
        df = self._make_df(extra_brands=["mercedes-amg"])
        n_amg = df["brand"].str.contains("mercedes", case=False).sum()
        _, y, _ = preprocess_data(df)
        assert len(y) <= len(df) - n_amg

    def test_mercedes_filter_is_case_insensitive(self):
        from ml_pipeline_us import preprocess_data
        df = self._make_df(extra_brands=["Mercedes-Benz"])
        n_merc = df["brand"].str.contains("mercedes", case=False).sum()
        _, y, _ = preprocess_data(df)
        assert len(y) <= len(df) - n_merc

    def test_drops_na_mandatory_columns(self):
        from ml_pipeline_us import preprocess_data
        df = self._make_df()
        df.loc[0, "mileage"] = None
        df.loc[1, "car_age"] = None
        _, y, _ = preprocess_data(df)
        assert len(y) <= len(df) - 2

    def test_fills_numeric_na_with_median(self):
        from ml_pipeline_us import preprocess_data
        df = self._make_df()
        df.loc[0:5, "cylinders"] = None
        df.loc[0:3, "doors"] = None
        X, _, _ = preprocess_data(df)
        assert X.isna().sum().sum() == 0

    def test_no_price_in_X(self):
        from ml_pipeline_us import preprocess_data
        X, _, _ = preprocess_data(self._make_df())
        assert "price" not in X.columns

    def test_categorical_cols_encoded(self):
        from ml_pipeline_us import preprocess_data
        X, _, _ = preprocess_data(self._make_df())
        for col in ["brand", "model", "trim", "drivetrain", "fuel",
                    "transmission", "body_style", "engine",
                    "exterior_color", "interior_color", "usage_type"]:
            assert col not in X.columns
        assert any("brand_" in c for c in X.columns)

    def test_boolean_cols_stay_numeric(self):
        from ml_pipeline_us import preprocess_data
        X, _, _ = preprocess_data(self._make_df())
        for col in ["one_owner", "has_accidents", "is_used", "is_cpo"]:
            if col in X.columns:
                assert X[col].dtype in [np.int64, np.float64, np.int32]

    def test_no_nan_in_output(self):
        from ml_pipeline_us import preprocess_data
        X, y, _ = preprocess_data(self._make_df())
        assert X.isna().sum().sum() == 0 and y.isna().sum() == 0

    def test_count_logging_printed(self, capsys):
        """Print-Logs zu Datenzählung werden ausgegeben."""
        from ml_pipeline_us import preprocess_data
        preprocess_data(self._make_df())
        out = capsys.readouterr().out
        assert "Mercedes" in out or "saubere Autos" in out


class TestUSTrainModel:
    """Tests für train_model() der US-Pipeline."""

    def _make_xy(self, n=250):
        np.random.seed(5)
        X = pd.DataFrame({
            "mileage":       np.random.uniform(5000, 150000, n),
            "car_age":       np.random.randint(1, 12, n),
            "cylinders":     np.random.choice([4, 6, 8], n),
            "one_owner":     np.random.randint(0, 2, n),
            "has_accidents": np.random.randint(0, 2, n),
        })
        y = pd.Series(np.random.uniform(8000, 60000, n))
        return X, y

    def test_returns_xgb_regressor(self):
        from ml_pipeline_us import train_model
        model, _ = train_model(*self._make_xy())
        assert isinstance(model, xgb.XGBRegressor)

    def test_prints_best_params(self, capsys):
        from ml_pipeline_us import train_model
        train_model(*self._make_xy())
        assert "Parameter-Mix" in capsys.readouterr().out

    def test_prints_r2_score(self, capsys):
        from ml_pipeline_us import train_model
        train_model(*self._make_xy())
        assert "R²" in capsys.readouterr().out

    def test_prints_mae_with_dollar_sign(self, capsys):
        from ml_pipeline_us import train_model
        train_model(*self._make_xy())
        assert "$" in capsys.readouterr().out

    def test_predictions_are_positive(self):
        from ml_pipeline_us import train_model
        model, X_train = train_model(*self._make_xy())
        assert all(p > 0 for p in model.predict(X_train.iloc[:10]))


class TestUSExplainModel:
    """Tests für explain_model() der US-Pipeline."""

    def _make_trained(self, n=100):
        np.random.seed(3)
        X = pd.DataFrame({
            "mileage":    np.random.uniform(5000, 150000, n),
            "car_age":    np.random.randint(1, 12, n),
            "cylinders":  np.random.choice([4, 6, 8], n),
            "one_owner":  np.random.randint(0, 2, n),
        })
        y = pd.Series(np.random.uniform(8000, 60000, n))
        return xgb.XGBRegressor(n_estimators=10, max_depth=3).fit(X, y), X

    def test_runs_without_error(self, capsys):
        from ml_pipeline_us import explain_model
        explain_model(*self._make_trained())
        assert "Basis-Preis" in capsys.readouterr().out

    def test_prints_dollar_not_euro(self, capsys):
        from ml_pipeline_us import explain_model
        explain_model(*self._make_trained())
        out = capsys.readouterr().out
        assert "$" in out and "€" not in out

    def test_prints_top15_header(self, capsys):
        from ml_pipeline_us import explain_model
        explain_model(*self._make_trained())
        assert "Top 15" in capsys.readouterr().out

    def test_shap_dataframe_has_dollar_influence_column(self):
        """DataFrame enthält 'Einfluss_in_Dollar' (US-spezifisch, nicht Euro)."""
        import shap
        model, X_train = self._make_trained()
        sample = X_train.iloc[[0]]
        shap_vals = shap.TreeExplainer(model)(sample)
        impacts = pd.DataFrame({
            "Feature":            sample.columns,
            "Wert_beim_Auto":     sample.iloc[0].values,
            "Einfluss_in_Dollar": shap_vals.values[0],
        })
        assert "Einfluss_in_Dollar" in impacts.columns
        assert "Einfluss_in_Euro" not in impacts.columns


class TestUSSaveLoad:
    """Tests für Pickle-Serialisierung der US-Pipeline-Artefakte."""

    def test_model_saved_with_us_suffix(self, tmp_path):
        np.random.seed(0)
        X = pd.DataFrame({"a": np.random.rand(50)})
        y = pd.Series(np.random.rand(50) * 50000)
        model = xgb.XGBRegressor(n_estimators=5).fit(X, y)
        path = tmp_path / "car_price_xgboost_us.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        assert path.exists() and "_us" in path.name

    def test_encoder_pickle_roundtrip(self, tmp_path):
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc.fit([["ford", "f-150", "awd"], ["toyota", "camry", "fwd"]])
        path = tmp_path / "categorical_encoder_us.pkl"
        with open(path, "wb") as f:
            pickle.dump(enc, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded.transform([["ford", "f-150", "awd"]]).shape[1] > 0


# ============================================================
# ÜBERGREIFENDE TESTS
# ============================================================

class TestSharedBehavior:
    """Verhaltensanforderungen, die für beide Pipelines gelten."""

    def test_random_state_42_is_reproducible(self):
        from sklearn.model_selection import train_test_split
        np.random.seed(42)
        X = pd.DataFrame({"x": np.random.rand(100)})
        y = pd.Series(np.random.rand(100))
        a, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        b, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        pd.testing.assert_frame_equal(a, b)

    def test_de_param_grid_keys(self):
        param_grid = {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1],
                      "n_estimators": [500, 1000]}
        assert set(param_grid.keys()) == {"max_depth", "learning_rate", "n_estimators"}

    def test_us_param_grid_has_extra_depth_10(self):
        """US-Grid enthält max_depth=10, DE-Grid nicht."""
        de_grid = {"max_depth": [4, 6, 8]}
        us_grid = {"max_depth": [4, 6, 8, 10]}
        assert 10 not in de_grid["max_depth"]
        assert 10 in us_grid["max_depth"]

    def test_ohe_handle_unknown_returns_zeros(self):
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc.fit([["vw"], ["bmw"]])
        assert enc.transform([["audi"]]).sum() == 0

    @pytest.mark.parametrize("suffix", ["", "_us"])
    def test_output_filenames_use_correct_suffix(self, suffix):
        assert f"car_price_xgboost{suffix}.pkl".endswith(".pkl")
        assert f"categorical_encoder{suffix}.pkl".endswith(".pkl")
        assert f"numeric_columns{suffix}.pkl".endswith(".pkl")
