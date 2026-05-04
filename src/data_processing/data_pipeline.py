import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _default_cleaned_path(raw_csv: str, market: str) -> str:
    """
    Your existing cleaning scripts already output files like:
      cleaned_source_brand_model.csv

    So for run-mode we DON'T try to predict the cleaned filename.
    This function is only used if you want to pass --cleaned explicitly.
    """
    raw = Path(raw_csv)
    return str(raw.with_name(f"cleaned_{raw.name}"))


def cmd_clean(args: argparse.Namespace) -> None:
    market = args.market.upper()

    if market == "US":
        # US cleaner expects: python data_preperation_us.py <input_csv>
        cmd = ["python", "data-preperation_us.py"]
        if args.input:
            cmd += [args.input]
        _run(cmd)
        return

    if market == "DE":
        # DE cleaner expects: python data_preperation_de.py <input_csv>
        cmd = ["python", "data-preperation_de.py"]
        if args.input:
            cmd += [args.input]
        _run(cmd)
        return

    raise SystemExit("market must be US or DE")


def cmd_import(args: argparse.Namespace) -> None:
    market = args.market.upper()

    if market == "US":
        cmd = [
            "python",
            "import_us_csv.py",
            "--csv",
            args.cleaned,
            "--dataset-id",
            str(args.dataset_id),
        ]
        _run(cmd)
        return

    if market == "DE":
        cmd = [
            "python",
            "import_de_csv.py",
            "--csv",
            args.cleaned,
            "--dataset-id",
            str(args.dataset_id),
        ]
        _run(cmd)
        return

    raise SystemExit("market must be US or DE")


def cmd_run(args: argparse.Namespace) -> None:
    # 1) clean (runs existing preprocessing script)
    clean_args = argparse.Namespace(market=args.market, input=args.input, output=args.output)
    cmd_clean(clean_args)

    # 2) import (you pass the cleaned csv explicitly)
    # Because your cleaned naming is cleaned_source_brand_model.csv,
    # the pipeline cannot reliably guess it unless you tell it.
    if not args.cleaned:
        raise SystemExit(
            "Missing --cleaned. Your cleaning output name is not predictable.\n"
            "Run with e.g.:\n"
            "  python pipeline.py run --market US --input raw.csv --cleaned cleaned_auto.dev_ford_f-series.csv --dataset-id 8"
        )

    import_args = argparse.Namespace(market=args.market, cleaned=args.cleaned, dataset_id=args.dataset_id)
    cmd_import(import_args)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Clean + import pipeline wrapper for used-car-analytics-ds-project."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_clean = sub.add_parser("clean", help="Run the existing cleaning script for a market.")
    p_clean.add_argument("--market", required=True, choices=["DE", "US"])
    p_clean.add_argument("--input", required=False, help="Optional: raw CSV path (if your cleaner supports it).")
    p_clean.add_argument("--output", required=False, help="Optional: output CSV path (if your cleaner supports it).")
    p_clean.set_defaults(func=cmd_clean)

    p_import = sub.add_parser("import", help="Import a cleaned CSV into Supabase.")
    p_import.add_argument("--market", required=True, choices=["DE", "US"])
    p_import.add_argument("--cleaned", required=True, help="Path to cleaned_*.csv")
    p_import.add_argument("--dataset-id", required=True, type=int)
    p_import.set_defaults(func=cmd_import)

    p_run = sub.add_parser("run", help="Run clean, then import.")
    p_run.add_argument("--market", required=True, choices=["DE", "US"])
    p_run.add_argument("--input", required=False, help="Optional: raw CSV path (if your cleaner supports it).")
    p_run.add_argument("--output", required=False, help="Optional: cleaned output (if your cleaner supports it).")
    p_run.add_argument("--cleaned", required=True, help="The cleaned csv path produced by cleaning.")
    p_run.add_argument("--dataset-id", required=True, type=int)
    p_run.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()