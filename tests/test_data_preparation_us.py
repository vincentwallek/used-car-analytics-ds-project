import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


# ------------------------------------------------------------
# Module Import Setup
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = PROJECT_ROOT.parent / "src" / "data_processing" / "data_preparation_us.py"

spec = importlib.util.spec_from_file_location("data_preparation_us", SCRIPT_PATH)
prep_us = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep_us)


def make_valid_us_df():
    return pd.DataFrame(
        {
            "@id": ["car-1"],
            "createdAt": ["2025-01-01T00:00:00Z"],
            "location": ["Los Angeles, CA"],
            "online": ["true"],
            "vin": ["1HGCM82633A004352"],

            "history_accidentCount": ["0"],
            "history_accidents": ["false"],
            "history_oneOwner": ["yes"],
            "history_ownerCount": ["1"],
            "history_personalUse": ["true"],
            "history_usageType": ["Personal"],

            "retailListing_carfaxUrl": ["https://example.com/carfax"],
            "retailListing_city": ["Los Angeles"],
            "retailListing_cpo": ["false"],
            "retailListing_dealer": ["Test Dealer"],
            "retailListing_miles": ["45,000 mi"],
            "retailListing_photoCount": ["12"],
            "retailListing_price": ["$24,999"],
            "retailListing_primaryImage": ["https://example.com/image.jpg"],
            "retailListing_state": ["CA"],
            "retailListing_used": ["true"],
            "retailListing_vdp": ["https://example.com/vdp"],
            "retailListing_zip": ["90001"],

            "vehicle_baseInvoice": ["23000"],
            "vehicle_baseMsrp": ["30000"],
            "vehicle_bodyStyle": ["Sedan"],
            "vehicle_confidence": ["0.95"],
            "vehicle_cylinders": ["4"],
            "vehicle_doors": ["4"],
            "vehicle_drivetrain": ["FWD"],
            "vehicle_engine": ["2.0L I4"],
            "vehicle_exteriorColor": ["Black"],
            "vehicle_fuel": ["Gasoline"],
            "vehicle_interiorColor": ["Black"],
            "vehicle_make": ["Honda"],
            "vehicle_model": ["Civic"],
            "vehicle_seats": ["5"],
            "vehicle_series": ["EX"],
            "vehicle_squishVin": ["1HGCM82633A"],
            "vehicle_style": ["EX Sedan"],
            "vehicle_transmission": ["Automatic"],
            "vehicle_trim": ["EX"],
            "vehicle_type": ["Car"],
            "vehicle_vin": ["1HGCM82633A004352"],
            "vehicle_year": ["2020"],
            "wholesaleListing": ["false"],
        }
    )


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def test_to_bool_maps_true_values():
    s = pd.Series(["true", "1", "yes", "y", "t", " TRUE "])

    result = prep_us._to_bool(s)

    assert result.tolist() == [True, True, True, True, True, True]


def test_to_bool_maps_false_values():
    s = pd.Series(["false", "0", "no", "n", "f", " FALSE "])

    result = prep_us._to_bool(s)

    assert result.tolist() == [False, False, False, False, False, False]


def test_to_bool_maps_missing_values_to_na():
    s = pd.Series(["nan", "none", "null", "", None])

    result = prep_us._to_bool(s)

    assert result.isna().sum() == 5


def test_to_numeric_clean_removes_currency_and_text():
    s = pd.Series(["$24,999", "45,000 mi", "€12,500.50", "N/A"])

    result = prep_us._to_numeric_clean(s)

    assert result.iloc[0] == 24999
    assert result.iloc[1] == 45000
    assert result.iloc[2] == 12500.50
    assert pd.isna(result.iloc[3])


def test_to_numeric_clean_handles_negative_values():
    s = pd.Series(["-10", "-500 mi"])

    result = prep_us._to_numeric_clean(s)

    assert result.iloc[0] == -10
    assert result.iloc[1] == -500


# ------------------------------------------------------------
# Grundlegende Cleaning-Tests
# ------------------------------------------------------------

def test_clean_us_data_returns_dataframe():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_clean_us_data_does_not_modify_original_dataframe():
    df = make_valid_us_df()
    original_columns = df.columns.tolist()

    prep_us.clean_us_data(df)

    assert df.columns.tolist() == original_columns
    assert "retailListing_price" in df.columns


def test_clean_us_data_contains_expected_columns():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    expected_columns = {
        "id",
        "created_at",
        "market",
        "source",
        "vin",
        "make",
        "model",
        "year",
        "price_usd",
        "mileage_mi",
        "car_age",
        "price_per_mile",
        "listing_age_days",
        "is_used",
        "is_cpo",
        "is_online",
        "is_wholesale",
        "accident_count",
        "has_accidents",
        "owner_count",
        "one_owner",
        "personal_use",
    }

    assert expected_columns.issubset(set(result.columns))


def test_clean_us_data_removes_redundant_vin_columns():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert "vehicle_vin" not in result.columns
    assert "vin_raw" not in result.columns


def test_clean_us_data_removes_unknown_columns_by_whitelist():
    df = make_valid_us_df()
    df["random_unused_column"] = ["should be removed"]

    result = prep_us.clean_us_data(df)

    assert "random_unused_column" not in result.columns


# ------------------------------------------------------------
# Datentypen und Konvertierungen
# ------------------------------------------------------------

def test_price_is_converted_to_numeric():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert result.iloc[0]["price_usd"] == 24999


def test_mileage_is_converted_to_numeric():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert result.iloc[0]["mileage_mi"] == 45000


def test_year_is_converted_to_numeric():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert result.iloc[0]["year"] == 2020


def test_owner_and_accident_count_are_numeric():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert result.iloc[0]["owner_count"] == 1
    assert result.iloc[0]["accident_count"] == 0


def test_created_at_is_parsed_as_datetime_utc():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert str(result["created_at"].dtype) == "datetime64[ns, UTC]"


def test_boolean_columns_are_converted():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    row = result.iloc[0]

    assert bool(row["is_online"]) is True
    assert bool(row["has_accidents"]) is False
    assert bool(row["one_owner"]) is True
    assert bool(row["personal_use"]) is True
    assert bool(row["is_cpo"]) is False
    assert bool(row["is_used"]) is True
    assert bool(row["is_wholesale"]) is False


def test_string_columns_are_stripped_and_lowercased():
    df = make_valid_us_df()
    df.loc[0, "vehicle_make"] = "  HONDA  "
    df.loc[0, "vehicle_model"] = "  CIVIC  "
    df.loc[0, "vehicle_fuel"] = "  GASOLINE  "
    df.loc[0, "vehicle_transmission"] = "  AUTOMATIC  "
    df.loc[0, "vehicle_drivetrain"] = "  FWD  "
    df.loc[0, "vehicle_bodyStyle"] = "  SEDAN  "
    df.loc[0, "retailListing_state"] = "  CA  "

    result = prep_us.clean_us_data(df)

    row = result.iloc[0]

    assert row["make"] == "honda"
    assert row["model"] == "civic"
    assert row["fuel"] == "gasoline"
    assert row["transmission"] == "automatic"
    assert row["drivetrain"] == "fwd"
    assert row["body_style"] == "sedan"
    assert row["state"] == "ca"


# ------------------------------------------------------------
# VIN-Tests
# ------------------------------------------------------------

def test_vehicle_vin_is_used_as_main_vin():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert result.iloc[0]["vin"] == "1HGCM82633A004352"


def test_vin_raw_is_used_when_vehicle_vin_is_missing():
    df = make_valid_us_df()
    df.loc[0, "vehicle_vin"] = None
    df.loc[0, "vin"] = "2HGCM82633A004353"

    result = prep_us.clean_us_data(df)

    assert str(result.iloc[0]["vin"]) == "2HGCM82633A004353"


def test_invalid_vin_is_set_to_na_but_row_is_kept():
    df = make_valid_us_df()
    df.loc[0, "vehicle_vin"] = "INVALIDVIN!!!"
    df.loc[0, "vin"] = "INVALIDVIN!!!"

    result = prep_us.clean_us_data(df)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["vin"])


def test_vin_is_uppercased_and_stripped():
    df = make_valid_us_df()
    df.loc[0, "vehicle_vin"] = "  1hgcm82633a004352  "

    result = prep_us.clean_us_data(df)

    assert result.iloc[0]["vin"] == "1HGCM82633A004352"


# ------------------------------------------------------------
# Required Fields / Drop NA
# ------------------------------------------------------------

def test_missing_price_removes_row():
    df = make_valid_us_df()
    df.loc[0, "retailListing_price"] = "N/A"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_missing_mileage_removes_row():
    df = make_valid_us_df()
    df.loc[0, "retailListing_miles"] = "N/A"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_missing_year_removes_row():
    df = make_valid_us_df()
    df.loc[0, "vehicle_year"] = "N/A"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_missing_make_removes_row():
    df = make_valid_us_df()
    df.loc[0, "vehicle_make"] = ""

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_missing_model_removes_row():
    df = make_valid_us_df()
    df.loc[0, "vehicle_model"] = ""

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


# ------------------------------------------------------------
# Outlier-Tests
# ------------------------------------------------------------

def test_price_below_minimum_is_removed():
    df = make_valid_us_df()
    df.loc[0, "retailListing_price"] = "$499"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_price_minimum_boundary_is_kept():
    df = make_valid_us_df()
    df.loc[0, "retailListing_price"] = "$500"

    result = prep_us.clean_us_data(df)

    assert len(result) == 1


def test_price_above_maximum_is_removed():
    df = make_valid_us_df()
    df.loc[0, "retailListing_price"] = "$300,001"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_price_maximum_boundary_is_kept():
    df = make_valid_us_df()
    df.loc[0, "retailListing_price"] = "$300,000"

    result = prep_us.clean_us_data(df)

    assert len(result) == 1


def test_negative_mileage_is_removed():
    df = make_valid_us_df()
    df.loc[0, "retailListing_miles"] = "-1"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_mileage_above_maximum_is_removed():
    df = make_valid_us_df()
    df.loc[0, "retailListing_miles"] = "400,001"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_mileage_maximum_boundary_is_kept():
    df = make_valid_us_df()
    df.loc[0, "retailListing_miles"] = "400,000"

    result = prep_us.clean_us_data(df)

    assert len(result) == 1


def test_year_below_minimum_is_removed():
    df = make_valid_us_df()
    df.loc[0, "vehicle_year"] = "1989"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_year_above_maximum_is_removed():
    df = make_valid_us_df()
    df.loc[0, "vehicle_year"] = "2027"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_year_boundaries_are_kept():
    df = make_valid_us_df()

    df_1990 = df.copy()
    df_1990.loc[0, "vehicle_year"] = "1990"

    df_2026 = df.copy()
    df_2026.loc[0, "vehicle_year"] = "2026"
    df_2026.loc[0, "vehicle_vin"] = "2HGCM82633A004353"
    df_2026.loc[0, "vin"] = "2HGCM82633A004353"

    mixed = pd.concat([df_1990, df_2026], ignore_index=True)

    result = prep_us.clean_us_data(mixed)

    assert len(result) == 2
    assert set(result["year"].tolist()) == {1990, 2026}


def test_owner_count_above_limit_is_removed():
    df = make_valid_us_df()
    df.loc[0, "history_ownerCount"] = "16"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_owner_count_na_is_allowed():
    df = make_valid_us_df()
    df.loc[0, "history_ownerCount"] = "unknown"

    result = prep_us.clean_us_data(df)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["owner_count"])


def test_accident_count_above_limit_is_removed():
    df = make_valid_us_df()
    df.loc[0, "history_accidentCount"] = "21"

    result = prep_us.clean_us_data(df)

    assert len(result) == 0


def test_accident_count_na_is_allowed():
    df = make_valid_us_df()
    df.loc[0, "history_accidentCount"] = "unknown"

    result = prep_us.clean_us_data(df)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["accident_count"])


# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------

def test_market_and_source_are_added():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert result.iloc[0]["market"] == "US"
    assert result.iloc[0]["source"] == "auto.dev"


def test_car_age_is_calculated_from_current_year():
    df = make_valid_us_df()
    df.loc[0, "vehicle_year"] = "2020"

    result = prep_us.clean_us_data(df)

    expected_age = pd.Timestamp.now(tz="UTC").year - 2020

    assert result.iloc[0]["car_age"] == expected_age


def test_price_per_mile_is_calculated_correctly():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    expected = 24999 / 45000

    assert result.iloc[0]["price_per_mile"] == pytest.approx(expected)


def test_price_per_mile_is_na_when_mileage_is_zero():
    df = make_valid_us_df()
    df.loc[0, "retailListing_miles"] = "0"

    result = prep_us.clean_us_data(df)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["price_per_mile"])


def test_listing_age_days_is_created_when_created_at_exists():
    df = make_valid_us_df()

    result = prep_us.clean_us_data(df)

    assert "listing_age_days" in result.columns
    assert result.iloc[0]["listing_age_days"] >= 0


def test_future_listing_age_days_is_set_to_zero():
    df = make_valid_us_df()

    future_date = (
        pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    df.loc[0, "createdAt"] = future_date

    result = prep_us.clean_us_data(df)

    assert result.iloc[0]["listing_age_days"] == 0


def test_listing_age_days_not_created_without_created_at_column():
    df = make_valid_us_df()
    df = df.drop(columns=["createdAt"])

    result = prep_us.clean_us_data(df)

    assert "listing_age_days" not in result.columns


# ------------------------------------------------------------
# Duplikate
# ------------------------------------------------------------

def test_duplicate_rows_are_removed_by_dedup_keys():
    df = make_valid_us_df()
    df = pd.concat([df, df], ignore_index=True)

    result = prep_us.clean_us_data(df)

    assert len(result) == 1


def test_rows_with_different_vin_are_not_removed_as_duplicates():
    df1 = make_valid_us_df()

    df2 = make_valid_us_df()
    df2.loc[0, "vehicle_vin"] = "2HGCM82633A004353"
    df2.loc[0, "vin"] = "2HGCM82633A004353"

    mixed = pd.concat([df1, df2], ignore_index=True)

    result = prep_us.clean_us_data(mixed)

    assert len(result) == 2


# ------------------------------------------------------------
# Mixed DataFrame
# ------------------------------------------------------------

def test_clean_us_data_keeps_only_valid_rows_from_mixed_dataframe():
    valid = make_valid_us_df()

    invalid_price = make_valid_us_df()
    invalid_price.loc[0, "retailListing_price"] = "$100"
    invalid_price.loc[0, "vehicle_vin"] = "2HGCM82633A004353"
    invalid_price.loc[0, "vin"] = "2HGCM82633A004353"

    invalid_mileage = make_valid_us_df()
    invalid_mileage.loc[0, "retailListing_miles"] = "999,999"
    invalid_mileage.loc[0, "vehicle_vin"] = "3HGCM82633A004354"
    invalid_mileage.loc[0, "vin"] = "3HGCM82633A004354"

    invalid_year = make_valid_us_df()
    invalid_year.loc[0, "vehicle_year"] = "1980"
    invalid_year.loc[0, "vehicle_vin"] = "4HGCM82633A004355"
    invalid_year.loc[0, "vin"] = "4HGCM82633A004355"

    df = pd.concat(
        [valid, invalid_price, invalid_mileage, invalid_year],
        ignore_index=True
    )

    result = prep_us.clean_us_data(df)

    assert len(result) == 1
    assert result.iloc[0]["price_usd"] == 24999
    assert result.iloc[0]["mileage_mi"] == 45000
    assert result.iloc[0]["year"] == 2020


# ------------------------------------------------------------
# Tests für main()
# ------------------------------------------------------------

def test_main_without_input_argument_exits_with_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["data-preperation-us.py"])

    with pytest.raises(SystemExit) as exc:
        prep_us.main()

    captured = capsys.readouterr()

    assert exc.value.code == 1
    assert "Usage: python data_preparation_us.py <input_csv>" in captured.out


def test_main_with_missing_file_exits_with_error(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["data-preperation-us.py", "missing_file.csv"]
    )

    with pytest.raises(SystemExit) as exc:
        prep_us.main()

    captured = capsys.readouterr()

    assert exc.value.code == 1
    assert "File not found: missing_file.csv" in captured.out


def test_main_reads_csv_cleans_data_and_writes_output(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "raw_us.csv"

    df = make_valid_us_df()
    df.to_csv(input_file, index=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["data-preperation-us.py", str(input_file)]
    )

    prep_us.main()

    output_file = tmp_path / "cleaned_raw_us.csv"

    assert output_file.exists()

    cleaned = pd.read_csv(output_file)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["price_usd"] == 24999
    assert cleaned.iloc[0]["mileage_mi"] == 45000
    assert cleaned.iloc[0]["year"] == 2020
    assert cleaned.iloc[0]["market"] == "US"
    assert cleaned.iloc[0]["source"] == "auto.dev"

    captured = capsys.readouterr()

    assert "Loading" in captured.out
    assert "Rows before cleaning: 1" in captured.out
    assert "Rows after cleaning:  1" in captured.out
    assert "Saved cleaned data to: cleaned_raw_us.csv" in captured.out