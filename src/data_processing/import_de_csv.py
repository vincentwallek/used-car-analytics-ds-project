import os
import argparse
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # Local/Backend use only!

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Defaults (can be overridden via CLI)
DEFAULT_MARKET = "DE"
DEFAULT_SOURCE = "mobile"
DEFAULT_CURRENCY = "EUR"
DEFAULT_MILEAGE_UNIT = "km"


def _none_if_na(v):
    return None if pd.isna(v) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to cleaned CSV")
    ap.add_argument("--dataset-id", required=True, type=int, help="datasets.id")
    ap.add_argument("--market", default=DEFAULT_MARKET)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--currency", default=DEFAULT_CURRENCY)
    ap.add_argument("--mileage-unit", default=DEFAULT_MILEAGE_UNIT)
    args = ap.parse_args()

    csv_path = args.csv
    dataset_id = args.dataset_id
    market = args.market
    source = args.source
    currency = args.currency
    mileage_unit = args.mileage_unit

    df = pd.read_csv(csv_path)

    # Normalize CSV: Match URL column to DB schema
    if "URL" in df.columns and "url" not in df.columns:
        df = df.rename(columns={"URL": "url"})

    # (Optional) sanity check
    required_cols = [
        "title", "price_eur", "mileage_km", "power_ps", "transmission", "fuel",
        "owners", "location", "url", "Ausstattung", "Beschreibung",
        "registration", "brand", "model", "car_age", "price_per_km"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # 1) listings (Base data)
    listings_rows = []
    for _, r in df.iterrows():
        listings_rows.append({
            "dataset_id": dataset_id,
            "market": market,
            "source": source,

            "brand": (str(r["brand"]).lower().strip() if pd.notna(r["brand"]) else None),
            "model": (str(r["model"]).lower().strip() if pd.notna(r["model"]) else None),
            "title": _none_if_na(r.get("title")),

            "price": (float(r["price_eur"]) if pd.notna(r["price_eur"]) else None),
            "currency": currency,

            "mileage": (float(r["mileage_km"]) if pd.notna(r["mileage_km"]) else None),
            "mileage_unit": mileage_unit,

            "registration": _none_if_na(r.get("registration")),
            "location": _none_if_na(r.get("location")),
            "url": _none_if_na(r.get("url")),
        })

    batch_size = 500
    inserted_listing_ids = []

    for i in range(0, len(listings_rows), batch_size):
        batch = listings_rows[i:i + batch_size]
        resp = supabase.table("listings").insert(batch).execute()
        data = resp.data or []
        inserted_listing_ids.extend([row["id"] for row in data])

    if len(inserted_listing_ids) != len(df):
        raise RuntimeError(
            f"Inserted listings mismatch: inserted={len(inserted_listing_ids)} csv_rows={len(df)}. "
            "Wenn du unique constraints hast oder Errors auftreten, kann das passieren."
        )

    # 2) listing_de (DE-specific details incl. equipment/description)
    de_rows = []
    for listing_id, (_, r) in zip(inserted_listing_ids, df.iterrows()):
        de_rows.append({
            "listing_id": listing_id,
            "transmission": _none_if_na(r.get("transmission")),
            "fuel": _none_if_na(r.get("fuel")),
            "owners": (int(r["owners"]) if pd.notna(r["owners"]) else None),
            "power_ps": (float(r["power_ps"]) if pd.notna(r["power_ps"]) else None),
            "car_age": (float(r["car_age"]) if pd.notna(r["car_age"]) else None),
            "price_per_km": (float(r["price_per_km"]) if pd.notna(r["price_per_km"]) else None),

            # NEW: save all
            "ausstattung": _none_if_na(r.get("Ausstattung")),
            "beschreibung": _none_if_na(r.get("Beschreibung")),
        })

    for i in range(0, len(de_rows), batch_size):
        batch = de_rows[i:i + batch_size]
        supabase.table("listing_de").insert(batch).execute()

    print(f"Import done: {len(df)} rows")
    print(f"dataset_id: {dataset_id}")


if __name__ == "__main__":
    main()