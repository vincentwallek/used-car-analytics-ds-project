import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = PROJECT_ROOT.parent / "src" / "feature_engineering" / "process_llm_features.py"


@pytest.fixture
def feature_module(monkeypatch):
    """
    Lädt feature_extraction.py isoliert.
    Supabase und dotenv werden gemockt, damit keine echten externen Abhängigkeiten nötig sind.
    """

    fake_supabase = types.ModuleType("supabase")

    def fake_create_client(url, key):
        return {"url": url, "key": key}

    fake_supabase.create_client = fake_create_client
    monkeypatch.setitem(sys.modules, "supabase", fake_supabase)

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    spec = importlib.util.spec_from_file_location("feature_extraction_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ------------------------------------------------------------
# robust_parse()
# ------------------------------------------------------------

def test_robust_parse_valid_json(feature_module):
    raw = '{"tuv": "neu", "garantie": 24}'

    result = feature_module.robust_parse(raw)

    assert result == {"tuv": "neu", "garantie": 24}


def test_robust_parse_dict_returns_dict(feature_module):
    raw = {"tuv": "neu"}

    result = feature_module.robust_parse(raw)

    assert result == {"tuv": "neu"}


def test_robust_parse_list_returns_list(feature_module):
    raw = [{"a": 1}]

    result = feature_module.robust_parse(raw)

    assert result == [{"a": 1}]


def test_robust_parse_empty_string_returns_empty_dict(feature_module):
    result = feature_module.robust_parse("")

    assert result == {}


def test_robust_parse_none_returns_empty_dict(feature_module):
    result = feature_module.robust_parse(None)

    assert result == {}


def test_robust_parse_invalid_string_returns_empty_dict(feature_module):
    result = feature_module.robust_parse("not valid json")

    assert result == {}


def test_robust_parse_single_quotes_and_boolean_values(feature_module):
    raw = "{'tuv': true, 'mangel': false, 'garantie': null}"

    result = feature_module.robust_parse(raw)

    assert result == {
        "tuv": True,
        "mangel": False,
        "garantie": None,
    }


# ------------------------------------------------------------
# process_ai_features()
# ------------------------------------------------------------

def test_process_ai_features_extracts_positive_history_features(feature_module):
    raw = {
        "historie": {
            "TÜV": "HU/AU neu gemacht",
            "Scheckheft": "lückenlos Scheckheft gepflegt",
            "Bereifung": "8-fach Winter und Sommer",
            "Allwetter": "true",
            "Garantie": "24 Monate",
            "Unfallfrei": "ja",
            "Mängel": "Keine",
        }
    }

    result = feature_module.process_ai_features(raw)

    assert result["tuv_neu"] == 1
    assert result["scheckheft_gepflegt"] == 1
    assert result["bereifung_8_fach"] == 1
    assert result["bereifung_allwetter"] == 1
    assert result["garantie_monate"] == 24
    assert result["unfallfrei"] == 1
    assert result["mangel_vorhanden"] == 0


def test_process_ai_features_extracts_negative_history_features(feature_module):
    raw = {
        "historie": {
            "TÜV": "abgelaufen",
            "Scheckheft": "nicht vorhanden",
            "Bereifung": "Sommerreifen",
            "Garantie": "keine Garantie",
            "Unfall": "Nachlackierung wegen Parkrempler",
            "Mängel": "Delle und Steinschlag",
        }
    }

    result = feature_module.process_ai_features(raw)

    assert result["tuv_neu"] == 0
    assert result["scheckheft_gepflegt"] == 0
    assert result["bereifung_8_fach"] == 0
    assert result["bereifung_allwetter"] == 0
    assert result["garantie_monate"] == 0
    assert result["unfallfrei"] == 0
    assert result["mangel_vorhanden"] == 1


def test_process_ai_features_extracts_equipment_features(feature_module):
    raw = {
        "ausstattung": {
            "assistenz": "DISTRONIC ACC",
            "licht": "MULTIBEAM LED Matrix",
            "klima": "4-Zonen THERMOTRONIC",
            "sound": "Burmester 3D High-End",
            "paket": "AMG Line",
            "dach": "Panorama-Schiebedach",
        }
    }

    result = feature_module.process_ai_features(raw)

    assert result["ausstattung_distronic"] == 1
    assert result["ausstattung_multibeam"] == 1
    assert result["ausstattung_klima_4_zonen"] == 1
    assert result["ausstattung_burmester_3d"] == 1
    assert result["ausstattung_amg_line"] == 1
    assert result["ausstattung_pano"] == 1


def test_process_ai_features_false_equipment_values_stay_zero(feature_module):
    raw = {
        "ausstattung": {
            "distronic": "false",
            "multibeam": "nein",
            "amg line": "0",
            "pano": "none",
        }
    }

    result = feature_module.process_ai_features(raw)

    assert result["ausstattung_distronic"] == 0
    assert result["ausstattung_multibeam"] == 0
    assert result["ausstattung_amg_line"] == 0
    assert result["ausstattung_pano"] == 0


def test_process_ai_features_returns_empty_dict_for_list_input(feature_module):
    raw = [{"tuv": "neu"}]

    result = feature_module.process_ai_features(raw)

    assert result == {}


def test_process_ai_features_handles_json_string(feature_module):
    raw = '{"TÜV": "neu", "Garantie": "12 Monate", "Unfallfrei": "true"}'

    result = feature_module.process_ai_features(raw)

    assert result["tuv_neu"] == 1
    assert result["garantie_monate"] == 12
    assert result["unfallfrei"] == 1


def test_process_ai_features_detects_two_zone_climate(feature_module):
    raw = {
        "ausstattung": {
            "klima": "2-Zonen THERMATIC"
        }
    }

    result = feature_module.process_ai_features(raw)

    assert result["ausstattung_klima_2_zonen"] == 1


def test_process_ai_features_detects_standard_burmester(feature_module):
    raw = {
        "ausstattung": {
            "sound": "Burmester Surround Soundsystem"
        }
    }

    result = feature_module.process_ai_features(raw)

    assert result["ausstattung_burmester_standard"] == 1


# ------------------------------------------------------------
# fetch_data()
# ------------------------------------------------------------

def test_fetch_data_returns_dataframe(feature_module):
    class FakeResponse:
        data = [
            {"listing_id": 1, "ai_features": '{"TÜV": "neu"}'},
            {"listing_id": 2, "ai_features": '{"TÜV": "alt"}'},
        ]

    class FakeTable:
        def __init__(self):
            self.selected = None

        def select(self, columns):
            self.selected = columns
            return self

        def execute(self):
            return FakeResponse()

    class FakeSupabase:
        def __init__(self):
            self.table_name = None
            self.fake_table = FakeTable()

        def table(self, name):
            self.table_name = name
            return self.fake_table

    fake_supabase = FakeSupabase()

    result = feature_module.fetch_data(fake_supabase)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert fake_supabase.table_name == "listing_de"
    assert fake_supabase.fake_table.selected == "listing_id, ai_features"


# ------------------------------------------------------------
# upload_to_supabase()
# ------------------------------------------------------------

def test_upload_to_supabase_uploads_batches(feature_module):
    batches = []

    class FakeQuery:
        def upsert(self, batch):
            batches.append(batch)
            return self

        def execute(self):
            return {"ok": True}

    class FakeSupabase:
        def table(self, name):
            assert name == "listing_features"
            return FakeQuery()

    df = pd.DataFrame(
        {
            "listing_id": range(505),
            "tuv_neu": [1] * 505,
        }
    )

    feature_module.upload_to_supabase(FakeSupabase(), df)

    assert len(batches) == 2
    assert len(batches[0]) == 500
    assert len(batches[1]) == 5


def test_upload_to_supabase_handles_empty_dataframe(feature_module):
    batches = []

    class FakeQuery:
        def upsert(self, batch):
            batches.append(batch)
            return self

        def execute(self):
            return {"ok": True}

    class FakeSupabase:
        def table(self, name):
            return FakeQuery()

    df = pd.DataFrame(columns=["listing_id", "tuv_neu"])

    feature_module.upload_to_supabase(FakeSupabase(), df)

    assert batches == []


# ------------------------------------------------------------
# save_local()
# ------------------------------------------------------------

def test_save_local_creates_csv_file(feature_module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = pd.DataFrame(
        {
            "listing_id": [1],
            "tuv_neu": [1],
        }
    )

    feature_module.save_local(df)

    files = list((tmp_path / "data" / "processed").glob("listing_features_*.csv"))

    assert len(files) == 1

    saved = pd.read_csv(files[0])
    assert saved.iloc[0]["listing_id"] == 1
    assert saved.iloc[0]["tuv_neu"] == 1


# ------------------------------------------------------------
# run_pipeline()
# ------------------------------------------------------------

def test_run_pipeline_exits_when_no_data_found(feature_module, monkeypatch, capsys):
    monkeypatch.setattr(feature_module, "get_supabase_client", lambda: "fake-client")
    monkeypatch.setattr(feature_module, "fetch_data", lambda supabase: pd.DataFrame())

    feature_module.run_pipeline()

    captured = capsys.readouterr()

    assert "No data found in Supabase." in captured.out


def test_run_pipeline_processes_features_and_calls_save_and_upload(feature_module, monkeypatch):
    calls = {
        "saved": False,
        "uploaded": False,
        "uploaded_df": None,
    }

    raw_df = pd.DataFrame(
        {
            "listing_id": [1, 2],
            "ai_features": [
                '{"TÜV": "neu", "Garantie": "12 Monate"}',
                '{"Unfallfrei": "ja", "Mängel": "Keine"}',
            ],
        }
    )

    def fake_save_local(df):
        calls["saved"] = True

    def fake_upload_to_supabase(supabase, df):
        calls["uploaded"] = True
        calls["uploaded_df"] = df.copy()

    monkeypatch.setattr(feature_module, "get_supabase_client", lambda: "fake-client")
    monkeypatch.setattr(feature_module, "fetch_data", lambda supabase: raw_df)
    monkeypatch.setattr(feature_module, "save_local", fake_save_local)
    monkeypatch.setattr(feature_module, "upload_to_supabase", fake_upload_to_supabase)
    monkeypatch.setattr(feature_module, "SAVE_LOCAL", True)
    monkeypatch.setattr(feature_module, "UPLOAD_TO_SUPABASE", True)

    feature_module.run_pipeline()

    assert calls["saved"] is True
    assert calls["uploaded"] is True
    assert "listing_id" in calls["uploaded_df"].columns
    assert "tuv_neu" in calls["uploaded_df"].columns
    assert "garantie_monate" in calls["uploaded_df"].columns