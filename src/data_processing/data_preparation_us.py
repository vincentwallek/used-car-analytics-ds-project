import pandas as pd
import numpy as np
import sys
import os


def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
        "y": True, "n": False,
        "t": True, "f": False
    }
    s = series.astype(str).str.strip().str.lower()
    s = s.replace({"nan": pd.NA, "none": pd.NA, "null": pd.NA, "": pd.NA})
    return s.map(mapping).astype("boolean")


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(r"[,$€£]", "", regex=True)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .str.strip()
    )
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "N/A": np.nan})
    return pd.to_numeric(s, errors="coerce")


def clean_us_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "@id": "id",
        "createdAt": "created_at",
        "location": "location",
        "online": "is_online",
        "vin": "vin_raw",

        "history_accidentCount": "accident_count",
        "history_accidents": "has_accidents",
        "history_oneOwner": "one_owner",
        "history_ownerCount": "owner_count",
        "history_personalUse": "personal_use",
        "history_usageType": "usage_type",

        "retailListing_carfaxUrl": "carfax_url",
        "retailListing_city": "city",
        "retailListing_cpo": "is_cpo",
        "retailListing_dealer": "dealer_name",
        "retailListing_miles": "mileage_mi",
        "retailListing_photoCount": "photo_count",
        "retailListing_price": "price_usd",
        "retailListing_primaryImage": "primary_image_url",
        "retailListing_state": "state",
        "retailListing_used": "is_used",
        "retailListing_vdp": "vdp_url",
        "retailListing_zip": "zip_code",

        "vehicle_baseInvoice": "base_invoice_usd",
        "vehicle_baseMsrp": "base_msrp_usd",
        "vehicle_bodyStyle": "body_style",
        "vehicle_confidence": "vehicle_confidence",
        "vehicle_cylinders": "cylinders",
        "vehicle_doors": "doors",
        "vehicle_drivetrain": "drivetrain",
        "vehicle_engine": "engine",
        "vehicle_exteriorColor": "exterior_color",
        "vehicle_fuel": "fuel",
        "vehicle_interiorColor": "interior_color",
        "vehicle_make": "make",
        "vehicle_model": "model",
        "vehicle_seats": "seats",
        "vehicle_series": "series",
        "vehicle_squishVin": "squish_vin",
        "vehicle_style": "style",
        "vehicle_transmission": "transmission",
        "vehicle_trim": "trim",
        "vehicle_type": "vehicle_type",
        "vehicle_vin": "vehicle_vin",
        "vehicle_year": "year",
        "wholesaleListing": "is_wholesale",
    }
    df = df.rename(columns=rename_map)

    # Datetime
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).astype("datetime64[ns, UTC]")

    # Numeric
    numeric_cols = [
        "accident_count", "owner_count", "mileage_mi", "photo_count", "price_usd",
        "base_invoice_usd", "base_msrp_usd", "vehicle_confidence", "cylinders",
        "doors", "seats", "year"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_numeric_clean(df[col])

    # Bool
    bool_cols = [
        "has_accidents", "one_owner", "personal_use", "is_online", "is_cpo",
        "is_used", "is_wholesale"
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = _to_bool(df[col])

    # Strings
    str_cols = [
        "usage_type", "city", "state", "zip_code", "dealer_name", "make", "model",
        "trim", "series", "body_style", "style", "drivetrain", "engine",
        "fuel", "transmission", "vehicle_type", "exterior_color", "interior_color",
        "vehicle_vin", "vin_raw", "squish_vin", "carfax_url", "vdp_url", "primary_image_url",
        "location"
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    for col in ["make", "model", "fuel", "transmission", "drivetrain", "body_style", "state"]:
        if col in df.columns:
            df[col] = df[col].str.lower()

    # VIN harmonization
    if "vehicle_vin" in df.columns and "vin_raw" in df.columns:
        df["vin"] = df["vehicle_vin"].fillna(df["vin_raw"])
    elif "vehicle_vin" in df.columns:
        df["vin"] = df["vehicle_vin"]
    elif "vin_raw" in df.columns:
        df["vin"] = df["vin_raw"]
    else:
        df["vin"] = pd.NA

    df["vin"] = df["vin"].astype(str).str.upper().str.strip()
    df.loc[~df["vin"].str.match(r"^[A-HJ-NPR-Z0-9]{11,17}$", na=False), "vin"] = pd.NA

    # Required fields
    required = [c for c in ["price_usd", "mileage_mi", "year", "make", "model"] if c in df.columns]
    if required:
        df = df.dropna(subset=required)

    # Outliers
    df = df[(df["price_usd"] >= 500) & (df["price_usd"] <= 300000)]
    df = df[(df["mileage_mi"] >= 0) & (df["mileage_mi"] <= 400000)]
    df = df[(df["year"] >= 1990) & (df["year"] <= 2026)]

    if "owner_count" in df.columns:
        df = df[(df["owner_count"].isna()) | ((df["owner_count"] >= 0) & (df["owner_count"] <= 15))]
    if "accident_count" in df.columns:
        df = df[(df["accident_count"].isna()) | ((df["accident_count"] >= 0) & (df["accident_count"] <= 20))]

    # Features (nur sinnvolle)
    now_utc = pd.Timestamp.now(tz="UTC")

    df["car_age"] = now_utc.year - df["year"]

    df["price_per_mile"] = df["price_usd"] / df["mileage_mi"]
    df.loc[df["mileage_mi"] == 0, "price_per_mile"] = pd.NA

    # listing_age_days optional und nicht-negativ
    if "created_at" in df.columns:
        df["listing_age_days"] = (now_utc - df["created_at"]).dt.days
        df.loc[df["listing_age_days"] < 0, "listing_age_days"] = 0  # oder pd.NA, wenn du lieber NA willst

    # metadata
    df["market"] = "US"
    df["source"] = "auto.dev"

    # Drop redundant
    df = df.drop(columns=[c for c in ["vehicle_vin", "vin_raw"] if c in df.columns], errors="ignore")

    # Dedup
    dedup_keys = [c for c in ["vin", "price_usd", "mileage_mi", "year"] if c in df.columns]
    if len(dedup_keys) >= 2:
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
    else:
        df = df.drop_duplicates()

    # FINAL: nur sinnvolle Spalten behalten (Whitelist)
    keep_cols = [
        # IDs / provenance
        "id", "created_at", "market", "source",

        # vehicle
        "vin", "make", "model", "trim", "year",
        "body_style", "drivetrain", "engine", "cylinders", "doors", "seats",
        "fuel", "transmission", "exterior_color", "interior_color",

        # listing
        "price_usd", "mileage_mi", "carfax_url", "vdp_url", "primary_image_url",
        "dealer_name", "city", "state", "zip_code",
        "is_used", "is_cpo", "is_online", "is_wholesale",

        # history
        "accident_count", "has_accidents", "owner_count", "one_owner", "personal_use", "usage_type",

        # derived (optional)
        "car_age", "price_per_mile", "listing_age_days",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python data_preparation_us.py <input_csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, keep_default_na=True, low_memory=False)

    print(f"Rows before cleaning: {len(df)}")
    df_clean = clean_us_data(df)
    print(f"Rows after cleaning:  {len(df_clean)}")

    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"cleaned_{os.path.basename(input_path)}")
    df_clean.to_csv(output_file, index=False)
    print(f"Saved cleaned data to: {output_file}")


if __name__ == "__main__":
    main()