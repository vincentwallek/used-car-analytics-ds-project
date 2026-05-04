import pandas as pd
import sys
import os


def clean_de_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rename columns
    df = df.rename(columns={
        "Preis": "price_eur",
        "Kilometerstand": "mileage_km",
        "Erstzulassung": "year",
        "PS": "power_ps",
        "Getriebe": "transmission",
        "Kraftstoff": "fuel",
        "Fahrzeughalter": "owners",
        "Standort": "location",
        "Titel": "title"
    })

    # -------- PRICE --------
    df["price_eur"] = (
        df["price_eur"]
        .astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["price_eur"] = pd.to_numeric(df["price_eur"], errors="coerce")

    # -------- MILEAGE --------
    df["mileage_km"] = (
        df["mileage_km"]
        .astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace("km", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.strip()
    )
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")

    # -------- REGISTRATION (month+year only) --------
    df["registration"] = (
        df["year"]
        .astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.replace("Erstzulassung", "", regex=False)
        .str.replace("EZ", "", regex=False)
        .str.strip()
    )

    # Parse strictly as MM/YYYY
    parsed_reg = pd.to_datetime(df["registration"], format="%m/%Y", errors="coerce")

    # Keep as string "MM/YYYY" in output (no day column shown)
    df["registration"] = parsed_reg.dt.strftime("%m/%Y")

    # Temporary numeric year only for filtering
    df["year_num"] = parsed_reg.dt.year

    # -------- POWER --------
    df["power_ps"] = (
        df["power_ps"]
        .astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace("PS", "", regex=False)
        .str.strip()
    )
    df["power_ps"] = pd.to_numeric(df["power_ps"], errors="coerce")

    # -------- OWNERS --------
    df["owners"] = df["owners"].astype(str).str.extract(r"(\d+)", expand=False)
    df["owners"] = pd.to_numeric(df["owners"], errors="coerce")

    # -------- TITLE → BRAND / MODEL --------
    df["title"] = df["title"].astype(str).str.strip()
    title_parts = df["title"].str.split()
    df["brand"] = title_parts.str[0].str.lower()
    df["model"] = title_parts.str[1].str.lower()

    # -------- TRANSMISSION --------
    trans_map = {
        "Automatik": "automatic",
        "Schaltgetriebe": "manual"
    }
    df["transmission"] = df["transmission"].astype(str).str.strip().map(trans_map)

    # -------- FUEL --------
    fuel_map = {
        "Benzin": "petrol",
        "Diesel": "diesel",
        "Hybrid": "hybrid",
        "Elektro": "electric"
    }
    df["fuel"] = df["fuel"].astype(str).str.strip().map(fuel_map)

    # -------- DROP NA --------
    df = df.dropna(subset=["price_eur", "mileage_km", "registration"])

    # -------- OUTLIERS --------
    df = df[(df["price_eur"] > 500) & (df["price_eur"] < 200000)]
    df = df[(df["mileage_km"] >= 0) & (df["mileage_km"] < 300000)]
    df = df[(df["year_num"] > 1990) & (df["year_num"] <= 2026)]

    # -------- FEATURES --------
    current_date = pd.to_datetime("2026-01-01")
    df["car_age"] = (current_date - parsed_reg.loc[df.index]).dt.days / 365
    df["price_per_km"] = df["price_eur"] / df["mileage_km"]
    df.loc[df["mileage_km"] == 0, "price_per_km"] = pd.NA

    # -------- META --------
    df["market"] = "DE"
    df["source"] = "mobile"

    # -------- DROP DUPLICATE YEAR COLS --------
    # Drop original renamed year and temporary numeric year
    df = df.drop(columns=["year", "year_num"], errors="ignore")

    # -------- DUPLICATES --------
    df = df.drop_duplicates()

    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python data_preparation.py <input_csv>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    print(f"Loading {input_path}...")

    df = pd.read_csv(input_path, keep_default_na=True)

    print(f"Rows before cleaning: {len(df)}")
    df_clean = clean_de_data(df)
    print(f"Rows after cleaning:  {len(df_clean)}")

    output_path = f"cleaned_{os.path.basename(input_path)}"
    df_clean.to_csv(output_path, index=False)

    print(f"Saved cleaned data to: {output_path}")


if __name__ == "__main__":
    main()