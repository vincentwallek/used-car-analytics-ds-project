import requests
import csv
import time
import json

# ============================ KONFIGURATION ===============================
API_KEY = "sk_ad_ZF_y6274NrUEe4T97RjrYTaX"

# Seiten-Parameter
MAX_PAGES = 200
# ==========================================================================

BASE_URL = "https://api.auto.dev/listings"
OUTPUT_FILE = 'ford_f_series_mit_ownerCount.csv'

headers = {
    'x-api-key': API_KEY
}

all_vehicles_flat = []
processed_vins = set()
all_headers = set()

try:
    if not API_KEY or API_KEY == "IHR_API_SCHLÜSSEL":
        raise ValueError("API-Schlüssel wurde nicht in der Konfiguration festgelegt.")

    print("--- Sammle und filtere Fahrzeug-Listings (nur mit 'history'-Daten) ---")

    raw_filtered_listings = []
    for page_num in range(1, MAX_PAGES + 1):
        request_url = f"{BASE_URL}?vehicle.make=Ford&vehicle.model=F-150,F-250,F-350&page={page_num}"
        print(f"Sende Anfrage für Seite {page_num}...")
        response = requests.get(request_url, headers=headers)
        response.raise_for_status()

        data = response.json()
        vehicle_listings = data.get('data', [])

        if not vehicle_listings:
            print("Keine weiteren Fahrzeuge gefunden. Paginierung beendet.")
            break

        # --- HIER FINDET DIE FILTERUNG STATT ---
        initial_count = len(vehicle_listings)
        filtered_page_listings = [
            vehicle for vehicle in vehicle_listings if
            vehicle.get('history') and vehicle['history'].get('ownerCount') is not None
        ]
        # -----------------------------------------

        if filtered_page_listings:
            raw_filtered_listings.extend(filtered_page_listings)
            print(
                f"{len(filtered_page_listings)} von {initial_count} Fahrzeugen auf dieser Seite hatten die gewünschten History-Daten.")
        else:
            print(f"Keines der {initial_count} Fahrzeuge auf dieser Seite hatte die gewünschten History-Daten.")

        time.sleep(0.5)

    if not raw_filtered_listings:
        raise Exception(
            "Es wurden keine Fahrzeuge für 'Mercedes-Benz S-Class' mit den geforderten History-Daten gefunden.")

    print(f"\n--- Verarbeite {len(raw_filtered_listings)} gefilterte Fahrzeuge für die CSV-Datei ---")

    for vehicle_data in raw_filtered_listings:
        vin = vehicle_data.get('vin')
        if vin and vin not in processed_vins:

            def flatten_dict(d, parent_key='', sep='_'):
                items = {}
                for k, v in d.items():
                    new_key = parent_key + sep + k if parent_key else k
                    if isinstance(v, dict):
                        items.update(flatten_dict(v, new_key, sep=sep))
                    elif isinstance(v, list):
                        items[new_key] = json.dumps(v)
                    else:
                        items[new_key] = v
                return items


            flat_row = flatten_dict(vehicle_data)
            all_vehicles_flat.append(flat_row)
            processed_vins.add(vin)
            all_headers.update(flat_row.keys())

    print(f"\n--- Schreibe {len(all_vehicles_flat)} Fahrzeuge mit {len(all_headers)} Spalten in die CSV-Datei ---")

    fieldnames = sorted(list(all_headers))

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_vehicles_flat)

    print(f"Daten erfolgreich in '{OUTPUT_FILE}' gespeichert!")

except Exception as e:
    print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
