"""
02_ml_pipeline_de.py
Training des XGBoost Preisvorhersage-Modells
(Final: Korrektes Daten-Schema + Automatisches Hyperparameter-Tuning)
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
    print("Lade Daten aus Supabase...")

    # PERFEKTER JOIN NACH DEINEM SCHEMA:
    # car_age, power_ps, owners, transmission, fuel aus listing_de
    # price, mileage, brand, model aus listings
    # Alle Features aus listing_features
    response = (
        supabase.table("listing_de")
        .select("car_age, power_ps, owners, transmission, fuel, listings(price, mileage, brand, model), listing_features(*)")
        .execute()
    )

    rows = []
    for row in response.data:
        # --- Basisdaten aus der verknüpften 'listings' Tabelle entpacken ---
        listings_data = row.get("listings")
        if not listings_data:
            continue

        if isinstance(listings_data, list):
            if len(listings_data) == 0: continue
            listings_data = listings_data[0]

        # --- ML-Features aus 'listing_features' entpacken ---
        features = row.get("listing_features")
        if not features:
            continue

        if isinstance(features, list):
            if len(features) == 0: continue
            features = features[0]

        # --- Alles zusammenbauen ---
        def safe_str(val):
            return str(val).lower().strip() if pd.notnull(val) and val != "" else "unbekannt"

        combined = {
            # Aus Tabelle 'listings'
            "price": listings_data.get("price"),
            "mileage": listings_data.get("mileage"),
            "brand": safe_str(listings_data.get("brand")),
            "model": safe_str(listings_data.get("model")),

            # Aus Tabelle 'listing_de' (row)
            "car_age": row.get("car_age"),
            "power_ps": row.get("power_ps"),
            "owners": row.get("owners"),
            "transmission": safe_str(row.get("transmission")),
            "fuel": safe_str(row.get("fuel"))
        }

        # KI-Features (1 und 0 Spalten) hinzufügen
        combined.update(features)

        # Unnötige IDs entfernen
        combined.pop("listing_id", None)

        # Nur aufnehmen, wenn ein Preis vorhanden ist
        if pd.notna(combined["price"]):
            rows.append(combined)

    df = pd.DataFrame(rows)
    print(f"Erfolgreich {len(df)} Datensätze geladen.")
    return df

# =========================
# 2. PREPROCESSING
# =========================
def preprocess_data(df):
    print("Bereite Daten vor...")
    # Ausreißer ohne Preis, Alter oder Kilometer entfernen
    df = df.dropna(subset=["price", "mileage", "car_age"])

    # --- Numerische Werte reparieren ---
    if "power_ps" in df.columns:
        df["power_ps"] = df["power_ps"].fillna(df["power_ps"].median())
    if "owners" in df.columns:
        df["owners"] = df["owners"].fillna(df["owners"].median())

    # X und y trennen
    X = df.drop(columns=["price"])
    y = df["price"]

    # --- Kategorien encodieren ---
    categorical_cols = ["brand", "model", "transmission", "fuel"]
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

    # Basis-Modell definieren
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    # Hier testet der Computer verschiedene Parameter (Rumspielen automatisiert)
    param_grid = {
        'max_depth': [4, 6, 8],               # Testet verschiedene Baum-Tiefen
        'learning_rate': [0.05, 0.1],         # Testet Lernraten
        'n_estimators': [500, 1000]           # Testet Anzahl der Bäume
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=1 # Zeigt den Fortschritt in der Konsole an
    )

    # Suche starten
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"\n--- Bester gefundener Parameter-Mix ---")
    print(grid_search.best_params_)

    # Evaluation des Gewinner-Modells
    predictions = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- Modellauswertung (Bestes Modell) ---")
    print(f"R² Score: {r2:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} €\n")

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
    print(f"Basis-Preis: {basis_preis:.2f} €")
    print("Top 15 Einflussfaktoren für dieses spezifische Auto:")

    impacts = pd.DataFrame({
        'Feature': sample_car.columns,
        'Wert_beim_Auto': sample_car.iloc[0].values,
        'Einfluss_in_Euro': shap_values.values[0]
    }).sort_values(by="Einfluss_in_Euro", key=abs, ascending=False)

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
        models_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Modelle speichern
        with open(os.path.join(models_dir, "car_price_xgboost.pkl"), "wb") as f:
            pickle.dump(model, f)

        with open(os.path.join(models_dir, "categorical_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)

        # Spalten speichern
        categorical_cols = ["brand", "model", "transmission", "fuel"]
        numeric_cols = [c for c in df.columns if c not in categorical_cols and c != "price"]

        with open(os.path.join(models_dir, "numeric_columns.pkl"), "wb") as f:
            pickle.dump(numeric_cols, f)

        print("✅ Pipeline erfolgreich beendet. Das BESTE Modell wurde gespeichert!")
    else:
        print("Keine Daten für das Training gefunden.")