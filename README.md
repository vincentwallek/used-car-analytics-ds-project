# Gebrauchtwagen Preisanalyse und Vorhersage

Ein Data-Science-Projekt zur Sammlung, Bereinigung und Analyse von Fahrzeugdaten aus dem deutschen (DE) und US-amerikanischen Markt. Das System trainiert marktspezifische XGBoost-Modelle zur Preisvorhersage und nutzt ein lokales LLM (Mistral 7B via Ollama), um strukturierte Ausstattungsmerkmale aus Freitextbeschreibungen zu extrahieren.

**Live-Demo:** [autovalue.streamlit.app](https://autovalue.streamlit.app)

[![CI - Test Suite](https://github.com/vincentwallek/used-car-analytics-ds-project/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/vincentwallek/used-car-analytics-ds-project/actions/workflows/ci.yml)

---

## Features

- **Datensammlung** — Webscraping (mobile.de) und REST-API-Anbindung (auto.dev) für DE- und US-Markt
- **Datenbereinigungs-Pipeline** — Marktspezifische Vorverarbeitung mit Ausreißererkennung, Imputation fehlender Werte und Typnormalisierung
- **LLM-Feature-Extraktion** — Lokale Mistral 7B Inferenz via Ollama zur Extraktion von 15 strukturierten Ausstattungs- und Zustandsmerkmalen aus deutschen Inseratsbeschreibungen
- **XGBoost-Preisvorhersage** — Automatisiertes Hyperparameter-Tuning via GridSearchCV mit SHAP-basierter Modellerklärbarkeit
- **Interaktives Dashboard** — Streamlit-Webanwendung zur Echtzeit-Preisschätzung, bereitgestellt über Streamlit Community Cloud

---

## Projektstruktur

```
used-car-analytics-ds-project/
├── app/                            # Streamlit-Frontend (deployed via Community Cloud)
│   ├── app.py
│   └── helpers.py
├── src/
│   ├── data_collection/            # Scraper (mobile.de) und API-Client (auto.dev)
│   │   ├── scraper.py
│   │   └── listings_api.py
│   ├── data_processing/            # Bereinigung, Import und Pipeline-Orchestrierung
│   │   ├── data_preparation_de.py
│   │   ├── data_preparation_us.py
│   │   ├── import_de_csv.py
│   │   ├── import_us_csv.py
│   │   └── data_pipeline.py
│   ├── feature_engineering/        # LLM-Extraktion und Feature-Verarbeitung
│   │   ├── llm_extraction.py
│   │   └── process_llm_features.py
│   └── models/                     # XGBoost-Training mit GridSearch und SHAP
│       ├── ml_pipeline_de.py
│       └── ml_pipeline_us.py
├── tests/                          # Vollständig gemockte Test-Suite (11 Module)
├── data/                           # Roh- und bereinigte CSVs
├── models/                         # Serialisierte Modelle und Encoder (.pkl)
├── .github/workflows/ci.yml        # GitHub Actions CI-Pipeline
└── requirements.txt
```

---

## Tech Stack

| Kategorie | Technologien |
|---|---|
| ML und Datenverarbeitung | pandas, numpy, scikit-learn, xgboost, shap |
| Datensammlung | seleniumbase, beautifulsoup4, requests |
| Datenbank | Supabase (PostgreSQL), SQLAlchemy |
| LLM | Mistral 7B (lokal via Ollama) |
| Frontend | Streamlit |
| Testing und CI | pytest, pytest-cov, GitHub Actions |

---

## Installation

### 1. Repository klonen und Abhängigkeiten installieren

```bash
git clone https://github.com/vincentwallek/used-car-analytics-ds-project.git
cd used-car-analytics-ds-project

python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Umgebungsvariablen konfigurieren

Eine `.env`-Datei im Projektverzeichnis erstellen:

```env
SUPABASE_URL=deine_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=dein_supabase_service_key
SUPABASE_DB_URL=deine_supabase_db_verbindung
```

### 3. Ollama installieren (für LLM-Feature-Extraktion)

Die LLM-Extraktionspipeline setzt eine lokale Ollama-Instanz mit Mistral 7B voraus:

```bash
# Ollama installieren von https://ollama.com
ollama pull mistral
ollama serve
```

---

## Nutzung

### 1. Webscraping (deutscher Markt)

Der Scraper sammelt Fahrzeugdaten von mobile.de. Vor der Ausführung müssen zwei Stellen in `src/data_collection/scraper.py` angepasst werden:

1. **Such-URL anpassen** (Zeile 18): Die `base_search_url` auf die gewünschte mobile.de-Suche setzen. Dazu die gewünschte Suche auf mobile.de konfigurieren und die URL aus der Browserleiste kopieren.
2. **Dateiname anpassen** (Zeile 27): Den `filename` auf den gewünschten Ausgabenamen setzen, z. B. `mobile_de_mercedes_sklasse.csv`.

```bash
# Scraper ausführen (Standard: 7 Übersichtsseiten, max. 200 Inserate)
python src/data_collection/scraper.py
```

Die Ausgabe wird in `data/raw/` gespeichert. Die Anzahl der durchsuchten Seiten kann über den Parameter `max_pages` in der letzten Zeile der Datei angepasst werden.

### 2. API-Abfrage (US-Markt)

Das Skript `listings_api.py` ruft Fahrzeugdaten über die auto.dev REST-API ab. Vor der Ausführung anpassen:

1. **API-Key** (Zeile 8): Eigenen auto.dev API-Schlüssel eintragen.
2. **Fahrzeugmodell** (Zeile 36): Die `request_url` auf die gewünschte Marke und das Modell anpassen, z. B. `vehicle.make=Ford&vehicle.model=F-150`.
3. **Dateiname** (Zeile 18): Den `OUTPUT_FILE` auf den gewünschten Dateinamen setzen.

```bash
python src/data_collection/listings_api.py
```

Die Ausgabe wird in `data/raw/` gespeichert.

### 3. Datenbereinigung

Rohe CSV-Daten für einen bestimmten Markt bereinigen. Die Ausgabe wird in `data/processed/` gespeichert.

```bash
# Deutscher Markt
python src/data_processing/data_preparation_de.py data/raw/mobile_de_erweitert_Sklasse.csv

# US-Markt
python src/data_processing/data_preparation_us.py data/raw/ford_f_series.csv
```

### 4. Datenbank-Import

Bereinigte CSVs in Supabase importieren:

```bash
python src/data_processing/import_de_csv.py --csv data/processed/cleaned_mobile_de_erweitert_Sklasse.csv --dataset-id 1
python src/data_processing/import_us_csv.py --csv data/processed/cleaned_ford_f_series.csv --dataset-id 2
```

### 5. Gesamte Pipeline (Bereinigung + Import)

```bash
python src/data_processing/data_pipeline.py run --market DE --input data/raw/raw_de.csv --cleaned data/processed/cleaned_raw_de.csv --dataset-id 1
python src/data_processing/data_pipeline.py run --market US --input data/raw/raw_us.csv --cleaned data/processed/cleaned_raw_us.csv --dataset-id 2
```

### 6. LLM-Feature-Extraktion (nur DE)

Setzt eine lokal laufende Ollama-Instanz mit dem Mistral-Modell voraus. Extrahiert Ausstattungsmerkmale aus den Freitextbeschreibungen und speichert sie in der Supabase-Datenbank.

```bash
# Schritt 1: Rohe LLM-Ausgaben generieren
python src/feature_engineering/llm_extraction.py

# Schritt 2: Strukturierte Features extrahieren und in Supabase hochladen
python src/feature_engineering/process_llm_features.py
```

### 7. Modelltraining

Trainiert XGBoost-Modelle mit automatisiertem GridSearchCV-Tuning. Die serialisierten Modelle werden in `models/` gespeichert.

```bash
python src/models/ml_pipeline_de.py
python src/models/ml_pipeline_us.py
```

---

## Testing

Die Test-Suite läuft vollständig isoliert mit gemockten externen Abhängigkeiten (Supabase, Selenium, Ollama). Keine API-Schlüssel oder Datenbankverbindungen erforderlich.

```bash
# Gesamte Test-Suite ausführen
pytest tests/ -v

# Mit Coverage-Report ausführen
pytest --cov=src --cov=app tests/
```

CI läuft automatisch bei jedem Push auf `main`, `dev` und `feature/**` Branches via GitHub Actions.

---

## Branch-Strategie

| Branch | Zweck |
|---|---|
| `main` | Produktionsreife Releases |
| `dev` | Aktive Entwicklung und Integration |
| `feature/*` | Feature-Branches, werden via PR in `dev` gemergt |