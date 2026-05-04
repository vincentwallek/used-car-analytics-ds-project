import os
import argparse
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # nur lokal/Backend!

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Defaults (können jetzt per CLI überschrieben werden)
DEFAULT_MARKET = "US"
DEFAULT_SOURCE = "auto.dev"
DEFAULT_CURRENCY = "USD"
DEFAULT_MILEAGE_UNIT = "mi"


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

    # created_at wird NICHT importiert -> kein Parsing nötig

    # 1) Insert into listings (Basis)
    listings_rows = []
    for _, r in df.iterrows():
        listings_rows.append({
            "dataset_id": dataset_id,
            "market": market,
            "source": source,
            "brand": _none_if_na(r.get("make")),  # wir nutzen brand=make im US-Fall
            "model": _none_if_na(r.get("model")),
            "title": None,
            "price": _none_if_na(r.get("price_usd")),
            "currency": currency,
            "mileage": _none_if_na(r.get("mileage_mi")),
            "mileage_unit": mileage_unit,
            "year": (int(r["year"]) if pd.notna(r.get("year")) else None),
            "registration": None,
            "location": _none_if_na(r.get("location")),
            "url": _none_if_na(r.get("vdp_url")),  # als "url" der Listing-Detailseite
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
            f"Inserted listings mismatch: inserted={len(inserted_listing_ids)} csv_rows={len(df)}"
        )

    # 2) Insert into listing_us (Details)
    us_rows = []
    for listing_id, (_, r) in zip(inserted_listing_ids, df.iterrows()):
        us_rows.append({
            "listing_id": listing_id,
            "vin": _none_if_na(r.get("vin")),

            "make": _none_if_na(r.get("make")),
            "model": _none_if_na(r.get("model")),
            "trim": _none_if_na(r.get("trim")),
            "year": (int(r["year"]) if pd.notna(r.get("year")) else None),

            "price_usd": _none_if_na(r.get("price_usd")),
            "mileage_mi": _none_if_na(r.get("mileage_mi")),

            "carfax_url": _none_if_na(r.get("carfax_url")),
            "vdp_url": _none_if_na(r.get("vdp_url")),
            "primary_image_url": _none_if_na(r.get("primary_image_url")),

            "dealer_name": _none_if_na(r.get("dealer_name")),
            "city": _none_if_na(r.get("city")),
            "state": _none_if_na(r.get("state")),
            "zip_code": _none_if_na(r.get("zip_code")),

            "is_used": _none_if_na(r.get("is_used")),
            "is_cpo": _none_if_na(r.get("is_cpo")),
            "is_online": _none_if_na(r.get("is_online")),
            "is_wholesale": _none_if_na(r.get("is_wholesale")),

            "accident_count": _none_if_na(r.get("accident_count")),
            "has_accidents": _none_if_na(r.get("has_accidents")),
            "owner_count": _none_if_na(r.get("owner_count")),
            "one_owner": _none_if_na(r.get("one_owner")),
            "personal_use": _none_if_na(r.get("personal_use")),
            "usage_type": _none_if_na(r.get("usage_type")),

            "body_style": _none_if_na(r.get("body_style")),
            "drivetrain": _none_if_na(r.get("drivetrain")),
            "engine": _none_if_na(r.get("engine")),
            "cylinders": _none_if_na(r.get("cylinders")),
            "doors": _none_if_na(r.get("doors")),
            "seats": _none_if_na(r.get("seats")),
            "fuel": _none_if_na(r.get("fuel")),
            "transmission": _none_if_na(r.get("transmission")),
            "exterior_color": _none_if_na(r.get("exterior_color")),
            "interior_color": _none_if_na(r.get("interior_color")),

            "car_age": _none_if_na(r.get("car_age")),
            "price_per_mile": _none_if_na(r.get("price_per_mile")),
            "listing_age_days": _none_if_na(r.get("listing_age_days")),

            "location": _none_if_na(r.get("location")),
        })

    for i in range(0, len(us_rows), batch_size):
        batch = us_rows[i:i + batch_size]
        supabase.table("listing_us").insert(batch).execute()

    print(f"US import done: {len(df)} rows")
    print(f"dataset_id: {dataset_id}")


if __name__ == "__main__":
    main()