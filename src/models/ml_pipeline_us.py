"""
03_ml_pipeline_us.py
Training des XGBoost Preisvorhersage-Modells für den US Markt
(Mit Pagination für unlimitierte Datensätze und US-spezifischen Variablen)
"""

import os
import pandas as pd
import xgboost as xgb
from supabase import create_client
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import pickle

# =========================
# 1. DATEN LADEN
# =========================
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_data():
    print("Lade US-Daten aus Supabase (in Batches, um API-Limits zu umgehen)...")

    all_rows = []
    limit = 1000  # Wir laden in 1000er Schritten
    offset = 0

    while True:
        response = (
            supabase.table("listing_us")
            .select("""
                trim, drivetrain, fuel, transmission, accident_count, owner_count, 
                one_owner, has_accidents, body_style, engine, cylinders, doors, 
                seats, exterior_color, interior_color, is_used, is_cpo, is_online, 
                is_wholesale, personal_use, usage_type, car_age, 
                listings(price, mileage, brand, model)
            """)
            .range(offset, offset + limit - 1)  # Pagination (z.B. 0-999, 1000-1999)
            .execute()
        )

        data = response.data
        if not data:
            break  # Keine Daten mehr gefunden -> Schleife beenden

        all_rows.extend(data)

        # Wenn weniger Daten zurückkamen als das Limit, sind wir am Ende
        if len(data) < limit:
            break

        offset += limit

    print(f"-> {len(all_rows)} Roh-Datensätze komplett heruntergeladen. Verarbeite...")

    rows = []
    for row in all_rows:
        # Basisdaten entpacken
        listings_data = row.get("listings")
        if not listings_data:
            continue

        if isinstance(listings_data, list):
            if len(listings_data) == 0: continue
            listings_data = listings_data[0]

        # Helfer für saubere Strings
        def safe_str(val):
            return str(val).lower().strip() if pd.notnull(val) and val != "" else "unbekannt"

        # Helfer für Booleans (Wandelt True/False in 1/0 um)
        def safe_bool(val):
            return 1 if val is True else 0

        combined = {
            # Basisdaten
            "price": listings_data.get("price"),
            "mileage": listings_data.get("mileage"),
            "brand": safe_str(listings_data.get("brand")),
            "model": safe_str(listings_data.get("model")),

            # US-Spezifische numerische Daten
            "car_age": row.get("car_age"),
            "accident_count": row.get("accident_count"),
            "owner_count": row.get("owner_count"),
            "cylinders": row.get("cylinders"),
            "doors": row.get("doors"),
            "seats": row.get("seats"),

            # US-Spezifische kategoriale Daten (Text)
            "trim": safe_str(row.get("trim")),
            "drivetrain": safe_str(row.get("drivetrain")),
            "fuel": safe_str(row.get("fuel")),
            "transmission": safe_str(row.get("transmission")),
            "body_style": safe_str(row.get("body_style")),
            "engine": safe_str(row.get("engine")),
            "exterior_color": safe_str(row.get("exterior_color")),
            "interior_color": safe_str(row.get("interior_color")),
            "usage_type": safe_str(row.get("usage_type")),

            # US-Spezifische Boolesche Daten (Wahr/Falsch zu 1/0)
            "one_owner": safe_bool(row.get("one_owner")),
            "has_accidents": safe_bool(row.get("has_accidents")),
            "is_used": safe_bool(row.get("is_used")),
            "is_cpo": safe_bool(row.get("is_cpo")),
            "is_online": safe_bool(row.get("is_online")),
            "is_wholesale": safe_bool(row.get("is_wholesale")),
            "personal_use": safe_bool(row.get("personal_use"))
        }

        # Nur aufnehmen, wenn ein Preis vorhanden ist
        if pd.notna(combined["price"]):
            rows.append(combined)

    df = pd.DataFrame(rows)
    print(f"Erfolgreich {len(df)} US-Datensätze mit Preis für das Training vorbereitet.")
    return df

# =========================
# 2. PREPROCESSING
# =========================
def preprocess_data(df):
    print("\nBereite US-Daten vor...")
    count_start = len(df)

    # =========================================================================
    # ⚠️ TEMPORÄRER FILTER: START (Diese Zeilen später einfach löschen)
    # =========================================================================
    # Wirf alle Autos raus, die "mercedes" im Markennamen enthalten.
    df = df[~df["brand"].str.contains("mercedes", case=False, na=False)]

    count_no_mercedes = len(df)
    print(f"ℹ️ Info: {count_start - count_no_mercedes} Mercedes-Fahrzeuge wurden temporär aussortiert.")
    # =========================================================================
    # ⚠️ TEMPORÄRER FILTER: ENDE
    # =========================================================================

    # Harte Ausreißer ohne Alter oder Kilometer entfernen
    df = df.dropna(subset=["price", "mileage", "car_age"])

    count_after_na = len(df)
    print(f"ℹ️ Info: {count_no_mercedes - count_after_na} Autos wurden gelöscht, weil Preis, Kilometer oder Alter fehlen.")
    print(f"-> Es verbleiben {count_after_na} saubere Autos für das Training.\n")

    # Numerische Lücken mit dem Median füllen
    numeric_fills = ["accident_count", "owner_count", "cylinders", "doors", "seats"]
    for col in numeric_fills:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # X und y trennen
    X = df.drop(columns=["price"])
    y = df["price"]

    # Kategorien encodieren
    categorical_cols = [
        "brand", "model", "trim", "drivetrain", "fuel", "transmission",
        "body_style", "engine", "exterior_color", "interior_color", "usage_type"
    ]

    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(X[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=X.index)

    X_final = pd.concat([X[numeric_cols], df_encoded], axis=1)

    return X_final, y, encoder

# =========================
# 3. MODELL TRAINING (MIT GRID SEARCH)
# =========================
def train_model(X, y):
    print("Suche automatisch nach der besten Baumtiefe und Parameter-Kombination...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        'max_depth': [4, 6, 8, 10],           # Mehr Auswahl, da wir mehr US-Features haben
        'learning_rate': [0.05, 0.1],
        'n_estimators': [500, 1000]
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"\n--- Bester gefundener Parameter-Mix (US) ---")
    print(grid_search.best_params_)

    # Evaluation
    predictions = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- Modellauswertung US (Bestes Modell) ---")
    print(f"R² Score: {r2:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} $\n")

    return best_model, X_train

# =========================
# 4. FEATURE IMPACT (SHAP)
# =========================
def explain_model(model, X_train):
    print("Berechne beispielhaft Feature-Impact mit SHAP...")
    explainer = shap.TreeExplainer(model)

    sample_car = X_train.iloc[[0]]
    shap_values = explainer(sample_car)

    basis_preis = shap_values.base_values[0]
    print(f"Basis-Preis: {basis_preis:.2f} $")
    print("Top 15 Einflussfaktoren für dieses spezifische US-Auto:")

    impacts = pd.DataFrame({
        'Feature': sample_car.columns,
        'Wert_beim_Auto': sample_car.iloc[0].values,
        'Einfluss_in_Dollar': shap_values.values[0]
    }).sort_values(by="Einfluss_in_Dollar", key=abs, ascending=False)

    print(impacts.head(15).to_string(index=False))
    print("\n")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df = load_data()

    if not df.empty:
        X, y, encoder = preprocess_data(df)
        model, X_train = train_model(X, y)

        explain_model(model, X_train)

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Modelle mit "_us" Suffix speichern, um deutsche Dateien nicht zu überschreiben
        with open(os.path.join(BASE_DIR, "models", "car_price_xgboost_us.pkl"), "wb") as f:
            pickle.dump(model, f)

        with open(os.path.join(BASE_DIR, "models", "categorical_encoder_us.pkl"), "wb") as f:
            pickle.dump(encoder, f)

        categorical_cols = [
            "brand", "model", "trim", "drivetrain", "fuel", "transmission",
            "body_style", "engine", "exterior_color", "interior_color", "usage_type"
        ]
        numeric_cols = [c for c in df.columns if c not in categorical_cols and c != "price"]

        with open(os.path.join(BASE_DIR, "models", "numeric_columns_us.pkl"), "wb") as f:
            pickle.dump(numeric_cols, f)

        print("✅ US-Pipeline erfolgreich beendet. Das BESTE Modell wurde als '_us.pkl' gespeichert!")
    else:
        print("Keine US-Daten für das Training gefunden.")