import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


# ------------------------------------------------------------
# data-pipeline.py importieren
# Wichtig: Weil der Dateiname einen Bindestrich hat, kann man
# nicht einfach "import data-pipeline" schreiben.
# Deshalb wird die Datei über importlib geladen.
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = PROJECT_ROOT.parent / "src" / "data_processing" / "data_pipeline.py"

spec = importlib.util.spec_from_file_location("data_pipeline", PIPELINE_PATH)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


# ------------------------------------------------------------
# Tests für _run()
# ------------------------------------------------------------

def test_run_calls_subprocess_with_correct_command(monkeypatch):
    called = {}

    class FakeCompletedProcess:
        returncode = 0

    def fake_subprocess_run(cmd, text):
        called["cmd"] = cmd
        called["text"] = text
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    pipeline._run(["python", "script.py"])

    assert called["cmd"] == ["python", "script.py"]
    assert called["text"] is True


def test_run_raises_system_exit_when_subprocess_fails(monkeypatch):
    class FakeCompletedProcess:
        returncode = 1

    def fake_subprocess_run(cmd, text):
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        pipeline._run(["python", "broken_script.py"])

    assert exc.value.code == 1


def test_run_raises_system_exit_with_correct_error_code(monkeypatch):
    class FakeCompletedProcess:
        returncode = 127

    def fake_subprocess_run(cmd, text):
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        pipeline._run(["python", "missing_script.py"])

    assert exc.value.code == 127


# ------------------------------------------------------------
# Tests für _default_cleaned_path()
# ------------------------------------------------------------

def test_default_cleaned_path_adds_cleaned_prefix():
    result = pipeline._default_cleaned_path("raw.csv", "US")

    assert result == "cleaned_raw.csv"


def test_default_cleaned_path_keeps_directory():
    result = pipeline._default_cleaned_path("data/raw.csv", "DE")

    expected = str(Path("data") / "cleaned_raw.csv")
    assert result == expected


# ------------------------------------------------------------
# Tests für cmd_clean()
# ------------------------------------------------------------

def test_cmd_clean_us_with_input(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)

    args = argparse.Namespace(
        market="US",
        input="raw_us.csv",
        output=None
    )

    pipeline.cmd_clean(args)

    assert called["cmd"] == [
        "python",
        "data-preperation_us.py",
        "raw_us.csv"
    ]


def test_cmd_clean_us_without_input(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)

    args = argparse.Namespace(
        market="US",
        input=None,
        output=None
    )

    pipeline.cmd_clean(args)

    assert called["cmd"] == [
        "python",
        "data-preperation_us.py"
    ]


def test_cmd_clean_de_with_input(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)

    args = argparse.Namespace(
        market="DE",
        input="raw_de.csv",
        output=None
    )

    pipeline.cmd_clean(args)

    assert called["cmd"] == [
        "python",
        "data-preperation_de.py",
        "raw_de.csv"
    ]


def test_cmd_clean_de_without_input(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)

    args = argparse.Namespace(
        market="DE",
        input=None,
        output=None
    )

    pipeline.cmd_clean(args)

    assert called["cmd"] == [
        "python",
        "data-preperation_de.py"
    ]


def test_cmd_clean_market_is_case_insensitive(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)

    args = argparse.Namespace(
        market="us",
        input="raw.csv",
        output=None
    )

    pipeline.cmd_clean(args)

    assert called["cmd"] == [
        "python",
        "data-preperation_us.py",
        "raw.csv"
    ]


def test_cmd_clean_invalid_market_raises_system_exit():
    args = argparse.Namespace(
        market="FR",
        input="raw.csv",
        output=None
    )

    with pytest.raises(SystemExit) as exc:
        pipeline.cmd_clean(args)

    assert str(exc.value) == "market must be US or DE"


# ------------------------------------------------------------
# Tests für cmd_import()
# ------------------------------------------------------------

def test_cmd_import_us(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)

    args = argparse.Namespace(
        market="US",
        cleaned="cleaned_us.csv",
        dataset_id=8
    )

    pipeline.cmd_import(args)

    assert called["cmd"] == [
        "python",
        "import_us_csv.py",
        "--csv",
        "cleaned_us.csv",
        "--dataset-id",
        "8"
    ]


def test_cmd_import_de(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)

    args = argparse.Namespace(
        market="DE",
        cleaned="cleaned_de.csv",
        dataset_id=12
    )

    pipeline.cmd_import(args)

    assert called["cmd"] == [
        "python",
        "import_de_csv.py",
        "--csv",
        "cleaned_de.csv",
        "--dataset-id",
        "12"
    ]


def test_cmd_import_market_is_case_insensitive(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)

    args = argparse.Namespace(
        market="de",
        cleaned="cleaned.csv",
        dataset_id=5
    )

    pipeline.cmd_import(args)

    assert called["cmd"] == [
        "python",
        "import_de_csv.py",
        "--csv",
        "cleaned.csv",
        "--dataset-id",
        "5"
    ]


def test_cmd_import_invalid_market_raises_system_exit():
    args = argparse.Namespace(
        market="FR",
        cleaned="cleaned.csv",
        dataset_id=1
    )

    with pytest.raises(SystemExit) as exc:
        pipeline.cmd_import(args)

    assert str(exc.value) == "market must be US or DE"


# ------------------------------------------------------------
# Tests für cmd_run()
# ------------------------------------------------------------

def test_cmd_run_executes_clean_then_import(monkeypatch):
    calls = []

    def fake_clean(args):
        calls.append(("clean", args.market, args.input, args.output))

    def fake_import(args):
        calls.append(("import", args.market, args.cleaned, args.dataset_id))

    monkeypatch.setattr(pipeline, "cmd_clean", fake_clean)
    monkeypatch.setattr(pipeline, "cmd_import", fake_import)

    args = argparse.Namespace(
        market="US",
        input="raw.csv",
        output=None,
        cleaned="cleaned_us.csv",
        dataset_id=8
    )

    pipeline.cmd_run(args)

    assert calls == [
        ("clean", "US", "raw.csv", None),
        ("import", "US", "cleaned_us.csv", 8)
    ]


def test_cmd_run_de_executes_clean_then_import(monkeypatch):
    calls = []

    def fake_clean(args):
        calls.append(("clean", args.market, args.input, args.output))

    def fake_import(args):
        calls.append(("import", args.market, args.cleaned, args.dataset_id))

    monkeypatch.setattr(pipeline, "cmd_clean", fake_clean)
    monkeypatch.setattr(pipeline, "cmd_import", fake_import)

    args = argparse.Namespace(
        market="DE",
        input="raw_de.csv",
        output="cleaned_de.csv",
        cleaned="cleaned_de.csv",
        dataset_id=12
    )

    pipeline.cmd_run(args)

    assert calls == [
        ("clean", "DE", "raw_de.csv", "cleaned_de.csv"),
        ("import", "DE", "cleaned_de.csv", 12)
    ]


def test_cmd_run_missing_cleaned_raises_system_exit(monkeypatch):
    calls = []

    def fake_clean(args):
        calls.append(("clean", args.market, args.input, args.output))

    def fake_import(args):
        calls.append(("import", args.market, args.cleaned, args.dataset_id))

    monkeypatch.setattr(pipeline, "cmd_clean", fake_clean)
    monkeypatch.setattr(pipeline, "cmd_import", fake_import)

    args = argparse.Namespace(
        market="US",
        input="raw.csv",
        output=None,
        cleaned=None,
        dataset_id=8
    )

    with pytest.raises(SystemExit) as exc:
        pipeline.cmd_run(args)

    assert "Missing --cleaned" in str(exc.value)

    # Cleaning wird noch ausgeführt,
    # Import darf aber nicht mehr ausgeführt werden.
    assert calls == [
        ("clean", "US", "raw.csv", None)
    ]


# ------------------------------------------------------------
# Tests für main() und CLI-Verhalten
# ------------------------------------------------------------

def test_main_clean_command(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-pipeline.py",
            "clean",
            "--market",
            "US",
            "--input",
            "raw.csv"
        ]
    )

    pipeline.main()

    assert called["cmd"] == [
        "python",
        "data-preperation_us.py",
        "raw.csv"
    ]


def test_main_import_command(monkeypatch):
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(pipeline, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-pipeline.py",
            "import",
            "--market",
            "DE",
            "--cleaned",
            "cleaned_de.csv",
            "--dataset-id",
            "12"
        ]
    )

    pipeline.main()

    assert called["cmd"] == [
        "python",
        "import_de_csv.py",
        "--csv",
        "cleaned_de.csv",
        "--dataset-id",
        "12"
    ]


def test_main_run_command(monkeypatch):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)

    monkeypatch.setattr(pipeline, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-pipeline.py",
            "run",
            "--market",
            "US",
            "--input",
            "raw_us.csv",
            "--cleaned",
            "cleaned_us.csv",
            "--dataset-id",
            "8"
        ]
    )

    pipeline.main()

    assert calls == [
        [
            "python",
            "data-preperation_us.py",
            "raw_us.csv"
        ],
        [
            "python",
            "import_us_csv.py",
            "--csv",
            "cleaned_us.csv",
            "--dataset-id",
            "8"
        ]
    ]


def test_main_without_command_raises_system_exit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-pipeline.py"
        ]
    )

    with pytest.raises(SystemExit):
        pipeline.main()


def test_main_clean_invalid_market_raises_system_exit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-pipeline.py",
            "clean",
            "--market",
            "FR"
        ]
    )

    with pytest.raises(SystemExit):
        pipeline.main()


def test_main_import_missing_required_cleaned_raises_system_exit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-pipeline.py",
            "import",
            "--market",
            "US",
            "--dataset-id",
            "8"
        ]
    )

    with pytest.raises(SystemExit):
        pipeline.main()


def test_main_import_missing_required_dataset_id_raises_system_exit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-pipeline.py",
            "import",
            "--market",
            "US",
            "--cleaned",
            "cleaned_us.csv"
        ]
    )

    with pytest.raises(SystemExit):
        pipeline.main()


def test_main_run_missing_required_cleaned_raises_system_exit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-pipeline.py",
            "run",
            "--market",
            "US",
            "--input",
            "raw.csv",
            "--dataset-id",
            "8"
        ]
    )

    with pytest.raises(SystemExit):
        pipeline.main()


def test_main_run_missing_required_dataset_id_raises_system_exit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-pipeline.py",
            "run",
            "--market",
            "US",
            "--input",
            "raw.csv",
            "--cleaned",
            "cleaned_us.csv"
        ]
    )

    with pytest.raises(SystemExit):
        pipeline.main()