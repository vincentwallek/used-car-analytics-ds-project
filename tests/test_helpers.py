import base64
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = PROJECT_ROOT.parent / "app" / "helpers.py"


class AttrDict(dict):
    """Dict with attribute access, compatible with st.session_state."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _fake_cache_data(ttl=None):
    def decorator(func):
        return func
    return decorator


@pytest.fixture
def helpers_module(monkeypatch):
    """
    Lädt helpers.py isoliert.
    Streamlit und SHAP werden gemockt, damit keine App und kein echter SHAP-Explainer nötig sind.
    """
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.session_state = AttrDict()
    fake_streamlit.cache_data = _fake_cache_data
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    fake_shap = types.ModuleType("shap")
    fake_shap.TreeExplainer = lambda model: (lambda X: "fake_shap_values")
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    spec = importlib.util.spec_from_file_location("helpers_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------
# Kleine Utility-Funktionen
# ------------------------------------------------------------

def test_fmt_h_formats_words_and_short_abbreviations(helpers_module):
    assert helpers_module._fmt_h("bmw-x5") == "BMW-X5"
    assert helpers_module._fmt_h("mercedes-benz") == "Mercedes-Benz"
    assert helpers_module._fmt_h("") == ""


def test_fmt_formats_brand_model_for_display(helpers_module):
    assert helpers_module._fmt("bmw") == "BMW"
    assert helpers_module._fmt("mercedes-benz") == "Mercedes-Benz"
    assert helpers_module._fmt(None) is None


def test_img_to_base64_reads_file(helpers_module, tmp_path):
    file_path = tmp_path / "logo.bin"
    file_path.write_bytes(b"abc123")

    result = helpers_module.img_to_base64(file_path)

    assert result == base64.b64encode(b"abc123").decode()


def test_translate_shap_direct_feature_name(helpers_module):
    assert helpers_module._translate_shap("mileage") == "Kilometerstand"
    assert helpers_module._translate_shap("car_age") == "Fahrzeugalter"


def test_translate_shap_encoded_feature_name(helpers_module):
    assert helpers_module._translate_shap("brand_bmw") == "Marke: BMW"
    assert helpers_module._translate_shap("model_mercedes-benz") == "Modell: Mercedes-Benz"


def test_translate_shap_unknown_feature_name(helpers_module):
    assert helpers_module._translate_shap("unknown_feature") == "Unknown Feature"


# ------------------------------------------------------------
# Fake Encoder / Fake Model
# ------------------------------------------------------------

class FakeEncoder:
    def __init__(self):
        self.feature_names_in_ = np.array(["brand", "model"])
        self.categories_ = [
            np.array(["bmw", "audi"]),
            np.array(["x5", "a4"]),
        ]
        self.last_input = None

    def transform(self, df):
        self.last_input = df.copy()
        return np.ones((len(df), len(df.columns)))

    def get_feature_names_out(self, cat_cols):
        return [f"{col}_encoded" for col in cat_cols]


class FakeModel:
    def __init__(self, prediction=12345.67):
        self.prediction = prediction
        self.last_X = None

    def predict(self, X):
        self.last_X = X.copy()
        return np.array([self.prediction] * len(X))


# ------------------------------------------------------------
# get_encoder_categories()
# ------------------------------------------------------------

def test_get_encoder_categories_returns_empty_dict_without_models(helpers_module):
    assert helpers_module.get_encoder_categories(None, "DE") == {}


def test_get_encoder_categories_returns_sorted_categories_de(helpers_module):
    encoder = FakeEncoder()
    trained_models = {"de_encoder": encoder}

    result = helpers_module.get_encoder_categories(trained_models, "DE")

    assert result == {
        "brand": ["audi", "bmw"],
        "model": ["a4", "x5"],
    }


# ------------------------------------------------------------
# predict_price_fast()
# ------------------------------------------------------------

def test_predict_price_fast_returns_zero_without_models(helpers_module):
    result = helpers_module.predict_price_fast(None, "DE", {})

    assert result == 0.0


def test_predict_price_fast_de_builds_model_input_and_returns_float(helpers_module):
    encoder = FakeEncoder()
    model = FakeModel(prediction=25000.0)
    trained_models = {
        "de_model": model,
        "de_encoder": encoder,
        "de_num_cols": ["mileage", "car_age", "power_ps", "owners"],
    }
    input_data = {
        "brand": " BMW ",
        "model": " X5 ",
        "transmission": " Automatic ",
        "fuel": " Diesel ",
        "mileage": 50000,
        "car_age": 4,
    }

    result = helpers_module.predict_price_fast(trained_models, "DE", input_data)

    assert result == 25000.0
    assert encoder.last_input.iloc[0]["brand"] == "bmw"
    assert encoder.last_input.iloc[0]["model"] == "x5"
    assert "owners" in model.last_X.columns
    assert model.last_X.iloc[0]["owners"] == 0.0


# ------------------------------------------------------------
# generate_recommendations()
# ------------------------------------------------------------

def test_generate_recommendations_returns_empty_without_models(helpers_module):
    result = helpers_module.generate_recommendations(
        trained_models=None,
        market="DE",
        input_data={},
        base_price=20000,
        db_data=pd.DataFrame(),
        currency_symbol="€",
    )

    assert result == []


def test_generate_recommendations_de_buyer_creates_saving_mileage_and_alternative(helpers_module, monkeypatch):
    def fake_predict_price_fast(trained_models, market, input_data):
        if input_data.get("tuv_neu") == 0.0:
            return 19000
        if input_data.get("mileage") == 50000:
            return 23000
        if input_data.get("model") == "x3":
            return 18000
        return 20000

    monkeypatch.setattr(helpers_module, "predict_price_fast", fake_predict_price_fast)

    input_data = {
        "brand": "bmw",
        "model": "x5",
        "mileage": 60000,
        "tuv_neu": 1.0,
    }
    db_data = pd.DataFrame({"brand": ["bmw", "bmw"], "model": ["x5", "x3"]})

    recs = helpers_module.generate_recommendations(
        trained_models={"dummy": True},
        market="DE",
        input_data=input_data,
        base_price=20000,
        db_data=db_data,
        currency_symbol="€",
        role="buyer",
    )

    rec_types = {r["type"] for r in recs}
    assert "saving" in rec_types
    assert "mileage_tip" in rec_types
    assert "alternative" in rec_types
    assert recs == sorted(recs, key=lambda r: r["saving"], reverse=True)


def test_generate_recommendations_de_seller_only_uses_equipment_features(helpers_module, monkeypatch):
    def fake_predict_price_fast(trained_models, market, input_data):
        if input_data.get("ausstattung_pano") == 1.0:
            return 21500
        if input_data.get("unfallfrei") == 1.0:
            return 50000
        return 20000

    monkeypatch.setattr(helpers_module, "predict_price_fast", fake_predict_price_fast)

    input_data = {
        "brand": "mercedes",
        "model": "c-class",
        "mileage": 30000,
        "ausstattung_pano": 0.0,
        "unfallfrei": 0.0,
    }

    recs = helpers_module.generate_recommendations(
        trained_models={"dummy": True},
        market="DE",
        input_data=input_data,
        base_price=20000,
        db_data=pd.DataFrame(),
        currency_symbol="€",
        role="seller",
    )

    assert len(recs) == 1
    assert recs[0]["type"] == "upgrade"
    assert "Panoramadach" in recs[0]["text"]
    assert "Unfallfrei" not in recs[0]["text"]


def test_generate_recommendations_us_buyer_creates_alternative_mileage_and_cylinder_saving(helpers_module, monkeypatch):
    def fake_predict_price_fast(trained_models, market, input_data):
        if input_data.get("model") == "focus":
            return 18000
        if input_data.get("mileage") == 50000:
            return 23000
        if input_data.get("cylinders") == 4.0:
            return 19000
        return 20000

    monkeypatch.setattr(helpers_module, "predict_price_fast", fake_predict_price_fast)

    input_data = {
        "brand": "ford",
        "model": "f-150",
        "mileage": 60000,
        "cylinders": 6.0,
    }
    db_data = pd.DataFrame({"brand": ["ford", "ford"], "model": ["f-150", "focus"]})

    recs = helpers_module.generate_recommendations(
        trained_models={"dummy": True},
        market="US",
        input_data=input_data,
        base_price=20000,
        db_data=db_data,
        currency_symbol="$",
        role="buyer",
    )

    rec_types = {r["type"] for r in recs}
    assert "alternative" in rec_types
    assert "mileage_tip" in rec_types
    assert "saving" in rec_types


# ------------------------------------------------------------
# get_market_data()
# ------------------------------------------------------------

class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeTable:
    def __init__(self, data):
        self.data = data
        self.selected = None
        self.eq_filter = None

    def select(self, columns):
        self.selected = columns
        return self

    def eq(self, column, value):
        self.eq_filter = (column, value)
        return self

    def execute(self):
        return FakeResponse(self.data)


class FakeSupabase:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        if name not in self.tables:
            raise KeyError(name)
        return FakeTable(self.tables[name])


class FailingSupabase:
    def table(self, name):
        raise RuntimeError("database unavailable")


def test_get_market_data_returns_empty_dataframe_on_error(helpers_module):
    result = helpers_module.get_market_data("DE", FailingSupabase())

    assert list(result.columns) == ["brand", "model", "title", "price", "mileage", "location", "url"]
    assert result.empty


def test_get_market_data_de_joins_listing_de_and_features(helpers_module):
    supabase = FakeSupabase(
        {
            "listings": [
                {"id": 1, "brand": "Mercedes-Benz", "model": "C-Class", "price": 30000, "mileage": 40000, "location": "Berlin", "url": "u"}
            ],
            "listing_de": [
                {"listing_id": 1, "title": "Mercedes C-Class", "fuel": "diesel"}
            ],
            "listing_features": [
                {"listing_id": 1, "ausstattung_burmester_standart": 1}
            ],
        }
    )

    result = helpers_module.get_market_data("DE", supabase)

    assert len(result) == 1
    assert result.iloc[0]["brand"] == "mercedes-benz"
    assert result.iloc[0]["model"] == "c-class"
    assert "ausstattung_burmester_standard" in result.columns


def test_get_market_data_us_joins_listing_us_and_builds_title(helpers_module):
    supabase = FakeSupabase(
        {
            "listings": [
                {"id": 1, "brand": "BMW", "model": "X5", "price": 50000, "mileage": 20000, "location": "Miami", "url": "u"}
            ],
            "listing_us": [
                {"listing_id": 1, "trim": "M Sport", "engine": "3.0L"}
            ],
        }
    )

    result = helpers_module.get_market_data("US", supabase)

    assert len(result) == 1
    assert result.iloc[0]["brand"] == "bmw"
    assert result.iloc[0]["model"] == "x5"
    assert result.iloc[0]["title"] == "BMW X5 M Sport"


# ------------------------------------------------------------
# predict_price() und run_ml_prediction()
# ------------------------------------------------------------

def test_predict_price_returns_zero_and_none_without_models(helpers_module):
    price, shap_values = helpers_module.predict_price(None, "DE", {})

    assert price == 0.0
    assert shap_values is None


def test_predict_price_de_returns_prediction_and_shap_values(helpers_module, monkeypatch):
    encoder = FakeEncoder()
    model = FakeModel(prediction=30000.0)
    trained_models = {
        "de_model": model,
        "de_encoder": encoder,
        "de_num_cols": ["mileage", "car_age", "power_ps", "owners"],
    }
    input_data = {
        "brand": "BMW",
        "model": "X5",
        "transmission": "Automatic",
        "fuel": "Diesel",
        "mileage": 50000,
        "car_age": 4,
    }

    monkeypatch.setattr(helpers_module.shap, "TreeExplainer", lambda model: (lambda X: "shap-result"))

    price, shap_values = helpers_module.predict_price(trained_models, "DE", input_data)

    assert price == 30000.0
    assert shap_values == "shap-result"


def test_run_ml_prediction_returns_error_when_model_missing(helpers_module):
    result = helpers_module.run_ml_prediction(
        trained_models=None,
        market="DE",
        brand="BMW",
        model_name="X5",
        car_age=4,
        mileage=50000,
    )

    parsed = json.loads(result)
    assert parsed == {"error": "Modell konnte nicht geladen werden."}


def test_run_ml_prediction_de_returns_json_with_price_and_impacts(helpers_module, monkeypatch):
    class FakeShapRow:
        feature_names = ["mileage", "brand_bmw"]
        values = [-1200.123, 2500.567]

    fake_shap_values = [FakeShapRow()]

    monkeypatch.setattr(
        helpers_module,
        "predict_price",
        lambda trained_models, market, input_vals: (29999.99, fake_shap_values),
    )

    result = helpers_module.run_ml_prediction(
        trained_models={"dummy": True},
        market="DE",
        brand="BMW",
        model_name="X5",
        car_age=4,
        mileage=50000,
    )

    parsed = json.loads(result)

    assert parsed["berechneter_preis"] == 29999.99
    assert parsed["waehrung"] == "€"
    assert parsed["alle_preis_einflussfaktoren"] == {
        "mileage": -1200.12,
        "brand_bmw": 2500.57,
    }


def test_run_ml_prediction_us_uses_dollar_currency(helpers_module, monkeypatch):
    class FakeShapRow:
        feature_names = ["cylinders"]
        values = [900.0]

    monkeypatch.setattr(
        helpers_module,
        "predict_price",
        lambda trained_models, market, input_vals: (45000.0, [FakeShapRow()]),
    )

    result = helpers_module.run_ml_prediction(
        trained_models={"dummy": True},
        market="US",
        brand="Ford",
        model_name="F-150",
        car_age=3,
        mileage=30000,
        cylinders=6,
    )

    parsed = json.loads(result)

    assert parsed["berechneter_preis"] == 45000.0
    assert parsed["waehrung"] == "$"
    assert parsed["alle_preis_einflussfaktoren"] == {"cylinders": 900.0}


# ------------------------------------------------------------
# Streamlit Session State Helper
# ------------------------------------------------------------

def test_b_reads_checkbox_from_session_state(helpers_module):
    helpers_module.st.session_state["feature_enabled"] = True
    helpers_module.st.session_state["feature_disabled"] = False

    assert helpers_module._b("feature_enabled") == 1.0
    assert helpers_module._b("feature_disabled") == 0.0
    assert helpers_module._b("missing_key") == 0.0


def test_collect_inputs_de_reads_session_state(helpers_module):
    s = helpers_module.st.session_state
    s.de_mileage = 50000
    s.de_age = 4
    s.de_power = 190
    s.de_owners = 1
    s.de_trans = "automatic"
    s.de_fuel = "diesel"
    s.de_garantie = 12
    s.de_tuv = True
    s.de_unfall = True
    s.de_mangel = False
    s.de_scheckh = True
    s.de_pano = True
    s.de_amg = False
    s.de_distronic = True
    s.de_multibeam = False
    s.de_klima4 = True
    s.de_klima2 = False
    s.de_burm3d = True
    s.de_burmstd = False
    s.de_reif8 = True
    s.de_reifall = False

    result = helpers_module._collect_inputs("DE", "bmw", "x5")

    assert result["brand"] == "bmw"
    assert result["model"] == "x5"
    assert result["mileage"] == 50000.0
    assert result["tuv_neu"] == 1.0
    assert result["mangel_vorhanden"] == 0.0
    assert result["ausstattung_pano"] == 1.0


def test_collect_inputs_us_reads_session_state_and_derives_binary_features(helpers_module):
    s = helpers_module.st.session_state
    s.us_mileage = 30000
    s.us_age = 3
    s.us_accidents = 2
    s.us_owners = 1
    s.us_cyl = 6
    s.us_doors = 4
    s.us_seats = 5
    s.us_trim = "limited"
    s.us_drive = "awd"
    s.us_fuel = "gasoline"
    s.us_trans = "automatic"
    s.us_body = "truck"
    s.us_engine = "3.5l"
    s.us_ext_color = "black"
    s.us_int_color = "gray"
    s.us_usage = "Personal Use"

    result = helpers_module._collect_inputs("US", "ford", "f-150")

    assert result["brand"] == "ford"
    assert result["model"] == "f-150"
    assert result["mileage"] == 30000.0
    assert result["one_owner"] == 1.0
    assert result["has_accidents"] == 1.0
    assert result["personal_use"] == 1.0
    assert result["is_used"] == 1.0