import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


# ------------------------------------------------------------
# Module Import Setup
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = PROJECT_ROOT.parent / "src" / "data_processing" / "data_preparation_de.py"

spec = importlib.util.spec_from_file_location("data_preparation_de", SCRIPT_PATH)
prep_de = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep_de)


def make_valid_de_df():
    return pd.DataFrame(
        {
            "Preis": ["12.345 €"],
            "Kilometerstand": ["45.000 km"],
            "Erstzulassung": ["EZ 03/2020"],
            "PS": ["150 PS"],
            "Getriebe": ["Automatik"],
            "Kraftstoff": ["Benzin"],
            "Fahrzeughalter": ["2 Fahrzeughalter"],
            "Standort": ["Berlin"],
            "Titel": ["Volkswagen Golf"],
        }
    )


# ------------------------------------------------------------
# Grundlegende Cleaning-Tests
# ------------------------------------------------------------

def test_clean_de_data_returns_dataframe():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_clean_de_data_renames_columns():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    expected_columns = {
        "price_eur",
        "mileage_km",
        "registration",
        "power_ps",
        "transmission",
        "fuel",
        "owners",
        "location",
        "title",
        "brand",
        "model",
        "car_age",
        "price_per_km",
        "market",
        "source",
    }

    assert expected_columns.issubset(set(result.columns))


def test_clean_de_data_removes_original_year_columns():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    assert "year" not in result.columns
    assert "year_num" not in result.columns


def test_clean_de_data_does_not_modify_original_dataframe():
    df = make_valid_de_df()
    original_columns = df.columns.tolist()

    prep_de.clean_de_data(df)

    assert df.columns.tolist() == original_columns
    assert "Preis" in df.columns


# ------------------------------------------------------------
# Format-Konvertierungen
# ------------------------------------------------------------

def test_price_is_cleaned_and_converted_to_number():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["price_eur"] == 12345


def test_price_with_comma_decimal_is_converted_correctly():
    df = make_valid_de_df()
    df.loc[0, "Preis"] = "19.999,50 €"

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["price_eur"] == 19999.50


def test_price_with_non_breaking_space_is_converted_correctly():
    df = make_valid_de_df()
    df.loc[0, "Preis"] = "12\u00a0345 €"

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["price_eur"] == 12345


def test_mileage_is_cleaned_and_converted_to_number():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["mileage_km"] == 45000


def test_power_is_cleaned_and_converted_to_number():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["power_ps"] == 150


def test_owners_are_extracted_as_number():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["owners"] == 2


def test_registration_is_cleaned_and_formatted_as_month_year():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["registration"] == "03/2020"


def test_registration_without_ez_prefix_still_works():
    df = make_valid_de_df()
    df.loc[0, "Erstzulassung"] = "05/2021"

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["registration"] == "05/2021"


# ------------------------------------------------------------
# Brand / Model Parsing
# ------------------------------------------------------------

def test_title_is_split_into_brand_and_model():
    df = make_valid_de_df()
    df.loc[0, "Titel"] = "BMW M3 Competition"

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["brand"] == "bmw"
    assert result.iloc[0]["model"] == "m3"


def test_title_values_are_lowercase():
    df = make_valid_de_df()
    df.loc[0, "Titel"] = "Mercedes C63"

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["brand"] == "mercedes"
    assert result.iloc[0]["model"] == "c63"


# ------------------------------------------------------------
# Mapping-Tests für Getriebe und Kraftstoff
# ------------------------------------------------------------

def test_transmission_automatic_is_mapped_correctly():
    df = make_valid_de_df()
    df.loc[0, "Getriebe"] = "Automatik"

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["transmission"] == "automatic"


def test_transmission_manual_is_mapped_correctly():
    df = make_valid_de_df()
    df.loc[0, "Getriebe"] = "Schaltgetriebe"

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["transmission"] == "manual"


@pytest.mark.parametrize(
    "raw_fuel, expected_fuel",
    [
        ("Benzin", "petrol"),
        ("Diesel", "diesel"),
        ("Hybrid", "hybrid"),
        ("Elektro", "electric"),
    ],
)
def test_fuel_mapping(raw_fuel, expected_fuel):
    df = make_valid_de_df()
    df.loc[0, "Kraftstoff"] = raw_fuel

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["fuel"] == expected_fuel


# ------------------------------------------------------------
# Filter- und Drop-Tests
# ------------------------------------------------------------

def test_invalid_price_below_minimum_is_removed():
    df = make_valid_de_df()
    df.loc[0, "Preis"] = "400 €"

    result = prep_de.clean_de_data(df)

    assert len(result) == 0


def test_invalid_price_above_maximum_is_removed():
    df = make_valid_de_df()
    df.loc[0, "Preis"] = "250.000 €"

    result = prep_de.clean_de_data(df)

    assert len(result) == 0


def test_invalid_mileage_above_limit_is_removed():
    df = make_valid_de_df()
    df.loc[0, "Kilometerstand"] = "350.000 km"

    result = prep_de.clean_de_data(df)

    assert len(result) == 0


def test_negative_mileage_is_removed():
    df = make_valid_de_df()
    df.loc[0, "Kilometerstand"] = "-10 km"

    result = prep_de.clean_de_data(df)

    assert len(result) == 0


def test_invalid_registration_format_is_removed():
    df = make_valid_de_df()
    df.loc[0, "Erstzulassung"] = "not a date"

    result = prep_de.clean_de_data(df)

    assert len(result) == 0


def test_registration_year_before_1991_is_removed():
    df = make_valid_de_df()
    df.loc[0, "Erstzulassung"] = "01/1990"

    result = prep_de.clean_de_data(df)

    assert len(result) == 0


def test_registration_year_after_2026_is_removed():
    df = make_valid_de_df()
    df.loc[0, "Erstzulassung"] = "01/2027"

    result = prep_de.clean_de_data(df)

    assert len(result) == 0


def test_valid_registration_year_2026_is_kept():
    df = make_valid_de_df()
    df.loc[0, "Erstzulassung"] = "01/2026"

    result = prep_de.clean_de_data(df)

    assert len(result) == 1
    assert result.iloc[0]["registration"] == "01/2026"


def test_invalid_price_text_is_removed():
    df = make_valid_de_df()
    df.loc[0, "Preis"] = "auf Anfrage"

    result = prep_de.clean_de_data(df)

    assert len(result) == 0


def test_invalid_mileage_text_is_removed():
    df = make_valid_de_df()
    df.loc[0, "Kilometerstand"] = "unknown"

    result = prep_de.clean_de_data(df)

    assert len(result) == 0


# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------

def test_market_and_source_are_added():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    assert result.iloc[0]["market"] == "DE"
    assert result.iloc[0]["source"] == "mobile"


def test_price_per_km_is_calculated_correctly():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    expected = 12345 / 45000

    assert result.iloc[0]["price_per_km"] == pytest.approx(expected)


def test_price_per_km_is_na_when_mileage_is_zero():
    df = make_valid_de_df()
    df.loc[0, "Kilometerstand"] = "0 km"

    result = prep_de.clean_de_data(df)

    assert pd.isna(result.iloc[0]["price_per_km"])


def test_car_age_is_calculated():
    df = make_valid_de_df()

    result = prep_de.clean_de_data(df)

    assert "car_age" in result.columns
    assert result.iloc[0]["car_age"] > 0


# ------------------------------------------------------------
# Duplikate
# ------------------------------------------------------------

def test_duplicate_rows_are_removed():
    df = make_valid_de_df()
    df = pd.concat([df, df], ignore_index=True)

    result = prep_de.clean_de_data(df)

    assert len(result) == 1


# ------------------------------------------------------------
# Mehrere Zeilen zusammen
# ------------------------------------------------------------

def test_clean_de_data_keeps_only_valid_rows_from_mixed_dataframe():
    valid = make_valid_de_df()

    invalid_price = make_valid_de_df()
    invalid_price.loc[0, "Preis"] = "100 €"

    invalid_mileage = make_valid_de_df()
    invalid_mileage.loc[0, "Kilometerstand"] = "500.000 km"

    invalid_date = make_valid_de_df()
    invalid_date.loc[0, "Erstzulassung"] = "01/1989"

    df = pd.concat(
        [valid, invalid_price, invalid_mileage, invalid_date],
        ignore_index=True
    )

    result = prep_de.clean_de_data(df)

    assert len(result) == 1
    assert result.iloc[0]["price_eur"] == 12345


# ------------------------------------------------------------
# Tests für main()
# ------------------------------------------------------------

def test_main_without_input_argument_exits_with_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["data_preparation_de.py"])

    with pytest.raises(SystemExit) as exc:
        prep_de.main()

    captured = capsys.readouterr()

    assert exc.value.code == 1
    assert "Usage: python data_preparation.py <input_csv>" in captured.out


def test_main_with_missing_file_exits_with_error(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["data_preparation_de.py", "missing_file.csv"]
    )

    with pytest.raises(SystemExit) as exc:
        prep_de.main()

    captured = capsys.readouterr()

    assert exc.value.code == 1
    assert "File not found: missing_file.csv" in captured.out


def test_main_reads_csv_cleans_data_and_writes_output(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "raw_de.csv"

    df = make_valid_de_df()
    df.to_csv(input_file, index=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["data_preparation_de.py", str(input_file)]
    )

    prep_de.main()

    output_file = tmp_path / "cleaned_raw_de.csv"

    assert output_file.exists()

    cleaned = pd.read_csv(output_file)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["price_eur"] == 12345
    assert cleaned.iloc[0]["mileage_km"] == 45000
    assert cleaned.iloc[0]["registration"] == "03/2020"
    assert cleaned.iloc[0]["market"] == "DE"
    assert cleaned.iloc[0]["source"] == "mobile"

    captured = capsys.readouterr()

    assert "Loading" in captured.out
    assert "Rows before cleaning: 1" in captured.out
    assert "Rows after cleaning:  1" in captured.out
    assert "Saved cleaned data to: cleaned_raw_de.csv" in captured.out