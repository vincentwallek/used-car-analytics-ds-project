import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = PROJECT_ROOT.parent / "src" / "feature_engineering" / "llm_extraction.py"


def import_llm_module(monkeypatch, tmp_path, fake_df=None, fake_post=None):
    """
    Loads llm_extraction.py in an isolated environment.
    Note: The module executes database and LLM queries upon import.
    Therefore, dependencies like sqlalchemy, pandas.read_sql, and requests.post 
    must be mocked prior to importing.
    """

    monkeypatch.chdir(tmp_path)

    if fake_df is None:
        fake_df = pd.DataFrame(columns=["id", "beschreibung"])

    captured = {
        "engine_url": None,
        "queries": [],
        "post_calls": [],
    }

    fake_sqlalchemy = types.ModuleType("sqlalchemy")

    def fake_create_engine(url):
        captured["engine_url"] = url
        return "fake-engine"

    fake_sqlalchemy.create_engine = fake_create_engine
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake_sqlalchemy)

    def fake_read_sql(query, engine):
        captured["queries"].append(query)
        assert engine == "fake-engine"
        return fake_df

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    fake_requests = types.ModuleType("requests")

    if fake_post is None:
        def fake_post(url, json):
            captured["post_calls"].append(
                {
                    "url": url,
                    "json": json,
                }
            )

            class FakeResponse:
                def json(self):
                    return {
                        "response": '{"tuv": true, "garantie": 12}'
                    }

            return FakeResponse()

    fake_requests.post = fake_post
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    spec = importlib.util.spec_from_file_location("llm_extraction_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module, captured


# ------------------------------------------------------------
# Import / Top-Level Verhalten
# ------------------------------------------------------------

def test_import_does_not_use_real_database(monkeypatch, tmp_path):
    module, captured = import_llm_module(monkeypatch, tmp_path)

    assert captured["engine_url"].startswith("postgresql://")
    assert len(captured["queries"]) == 1
    assert "SELECT" in captured["queries"][0]
    assert "listing_de" in captured["queries"][0]
    assert "BETWEEN 100 AND 817" in captured["queries"][0]


def test_import_creates_empty_results_csv_when_no_rows(monkeypatch, tmp_path):
    module, captured = import_llm_module(monkeypatch, tmp_path)

    output_file = tmp_path / "data" / "processed" / "llm_results2.csv"

    assert output_file.exists()


def test_top_level_processing_writes_results_csv(monkeypatch, tmp_path):
    fake_df = pd.DataFrame(
        {
            "id": [101, 102],
            "beschreibung": [
                "Mercedes mit TÜV neu und Garantie",
                "BMW mit AMG Line und Pano",
            ],
        }
    )

    module, captured = import_llm_module(
        monkeypatch,
        tmp_path,
        fake_df=fake_df,
    )

    output_file = tmp_path / "data" / "processed" / "llm_results2.csv"

    assert output_file.exists()

    result_df = pd.read_csv(output_file)

    assert len(result_df) == 2
    assert list(result_df.columns) == ["listing_id", "raw_output"]
    assert result_df.iloc[0]["listing_id"] == 101
    assert result_df.iloc[1]["listing_id"] == 102
    assert len(captured["post_calls"]) == 2


def test_top_level_processing_continues_when_llm_request_fails(monkeypatch, tmp_path, capsys):
    fake_df = pd.DataFrame(
        {
            "id": [101],
            "beschreibung": ["Text"],
        }
    )

    def failing_post(url, json):
        raise RuntimeError("LLM not available")

    module, captured = import_llm_module(
        monkeypatch,
        tmp_path,
        fake_df=fake_df,
        fake_post=failing_post,
    )

    captured_output = capsys.readouterr()

    assert "Error:" in captured_output.out

    output_file = tmp_path / "data" / "processed" / "llm_results2.csv"
    assert output_file.exists()


# ------------------------------------------------------------
# extract_features()
# ------------------------------------------------------------

def test_extract_features_posts_to_local_ollama(monkeypatch, tmp_path):
    module, captured = import_llm_module(monkeypatch, tmp_path)

    post_calls = []

    def fake_post(url, json):
        post_calls.append(
            {
                "url": url,
                "json": json,
            }
        )

        class FakeResponse:
            def json(self):
                return {
                    "response": '{"unfallfrei": true}'
                }

        return FakeResponse()

    module.requests.post = fake_post

    result = module.extract_features("Das Fahrzeug ist unfallfrei und hat TÜV neu.")

    assert result == '{"unfallfrei": true}'
    assert len(post_calls) == 1
    assert post_calls[0]["url"] == "http://localhost:11434/api/generate"
    assert post_calls[0]["json"]["model"] == "mistral"
    assert post_calls[0]["json"]["stream"] is False
    assert "Das Fahrzeug ist unfallfrei und hat TÜV neu." in post_calls[0]["json"]["prompt"]


def test_extract_features_prompt_contains_required_sections(monkeypatch, tmp_path):
    module, captured = import_llm_module(monkeypatch, tmp_path)

    post_calls = []

    def fake_post(url, json):
        post_calls.append(json)

        class FakeResponse:
            def json(self):
                return {
                    "response": "{}"
                }

        return FakeResponse()

    module.requests.post = fake_post

    module.extract_features("Beispieltext")

    prompt = post_calls[0]["prompt"]

    assert "TEIL 1: HISTORIE & ZUSTAND" in prompt
    assert "TEIL 2: AUSSTATTUNG" in prompt
    assert "Antworte strikt im JSON-Format" in prompt
    assert "Beispieltext" in prompt


def test_extract_features_raises_keyerror_if_response_key_missing(monkeypatch, tmp_path):
    module, captured = import_llm_module(monkeypatch, tmp_path)

    def fake_post(url, json):
        class FakeResponse:
            def json(self):
                return {
                    "not_response": "{}"
                }

        return FakeResponse()

    module.requests.post = fake_post

    with pytest.raises(KeyError):
        module.extract_features("Text")


# ------------------------------------------------------------
# parse_output()
# ------------------------------------------------------------

def test_parse_output_extracts_json_from_text(monkeypatch, tmp_path):
    module, captured = import_llm_module(monkeypatch, tmp_path)

    text = 'Here is the result: {"tuv": true, "garantie": 12} Thanks.'

    result = module.parse_output(text)

    assert result == {
        "tuv": True,
        "garantie": 12,
    }


def test_parse_output_returns_none_for_invalid_json(monkeypatch, tmp_path):
    module, captured = import_llm_module(monkeypatch, tmp_path)

    result = module.parse_output("this is not json")

    assert result is None


def test_parse_output_returns_none_for_broken_json(monkeypatch, tmp_path):
    module, captured = import_llm_module(monkeypatch, tmp_path)

    result = module.parse_output('{"tuv": true')

    assert result is None


def test_parse_output_handles_nested_json(monkeypatch, tmp_path):
    module, captured = import_llm_module(monkeypatch, tmp_path)

    text = 'Output: {"history": {"tuv": true}, "equipment": {"pano": false}}'

    result = module.parse_output(text)

    assert result == {
        "history": {
            "tuv": True,
        },
        "equipment": {
            "pano": False,
        },
    }