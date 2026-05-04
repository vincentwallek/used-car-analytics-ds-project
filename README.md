# Data Science Projekt: Gebrauchtwagen Preisanalyse & Vorhersage

Dieses Repository beinhaltet den Code für ein umfassendes Data Science Projekt zur Sammlung, Bereinigung und Analyse von Fahrzeugdaten. Das Hauptziel des Projekts ist die präzise Preisvorhersage für Gebrauchtwagen auf dem europäischen (DE) und amerikanischen (US) Markt unter Verwendung moderner Machine-Learning-Techniken. Zudem kommt ein Large Language Model (LLM) zum Einsatz, um unstrukturierte Beschreibungen in strukturierte Features umzuwandeln.

---

## Projektziele & Features

- Automatisierte Datensammlung: Scraping von detaillierten Fahrzeuginformationen (mobile.de) sowie Anbindung an externe Fahrzeug-APIs für den US-Markt.
- Data Engineering Pipeline: Robuste Skripte zur Bereinigung von Rohdaten (Outlier-Detection, Imputation von Missing Values, Encoding von kategorialen Variablen).
- LLM Feature Extraktion: Nutzung der Groq API (Llama-3), um aus Freitext-Fahrzeugbeschreibungen boolesche Ausstattungsmerkmale (z. B. "Navigationssystem vorhanden") zu extrahieren.
- Machine Learning (XGBoost): Marktspezifisches Modell-Training mit Hyperparameter-Tuning (GridSearch) für hochpräzise Preisvorhersagen.
- Explainable AI (SHAP): Analyse des Feature-Impacts, um vorhergesagte Preise transparent und nachvollziehbar zu machen.
- Interaktives Dashboard: Eine intuitive Streamlit-Webanwendung, über die Nutzer Fahrzeugdaten eingeben und sofortige Preiseinschätzungen sowie datengetriebene Empfehlungen erhalten können.

---

## Tech Stack

Das Projekt setzt auf einen modernen, robusten Python-Stack:

- Machine Learning & Datenverarbeitung: pandas, numpy, scikit-learn, xgboost, shap
- Data Collection: seleniumbase, beautifulsoup4, requests
- Datenbank & Persistenz: supabase (PostgreSQL), SQLAlchemy
- Generative AI: groq (Llama-3 8b/70b)
- Frontend: streamlit
- Qualitätssicherung: pytest, pytest-cov, unittest.mock

---

## Architektur & Struktur

Das Projekt folgt einer strengen Trennung von Frontend-Logik und Backend-Pipelines:

```text
used-car-analytics-ds-project/
│
├── app/                        # Streamlit Frontend
│   ├── app.py                  # Interaktives UI-Dashboard
│   └── helpers.py              # UI-Logik, Formatting & Business Rules
│
├── src/                        # Core Backend & Pipelines
│   ├── data_collection/        # Web-Scraper und API-Routinen
│   ├── data_processing/        # Data Cleaning & Pipeline-Orchestrierung
│   ├── feature_engineering/    # LLM-gestützte Daten-Extraktion
│   └── models/                 # XGBoost Modellierung & SHAP-Analysen
│
├── tests/                      # Automatisierte Unit-Tests
├── data/                       # Lokaler Datenspeicher (Roh- & bereinigte CSVs)
├── models/                     # Serialisierte XGBoost Modelle (.pkl)
└── requirements.txt            # Abhängigkeiten
```

---

## Setup & Installation

### 1. Repository klonen und Environment aufsetzen
```bash
git clone https://github.com/vincentwallek/used-car-analytics-ds-project.git
cd used-car-analytics-ds-project

# Virtuelles Environment erstellen und aktivieren
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 2. Umgebungsvariablen (.env) konfigurieren
Erstelle im Hauptverzeichnis eine Datei namens `.env` und hinterlege deine API-Schlüssel:
```env
SUPABASE_URL=deine_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=dein_supabase_service_key
GROQ_API_KEY=dein_groq_api_key
```

---

## Nutzung & Ausführung

### 1. Daten-Pipeline starten
Die Bereinigung und der Datenbank-Upload können zentral gesteuert werden:

```bash
# Pipeline für den deutschen Markt
python src/data_processing/data_pipeline.py run DE

# Pipeline für den US-Markt
python src/data_processing/data_pipeline.py run US
```

### 2. Modell trainieren
Sobald die Daten in Supabase liegen, können die Vorhersagemodelle neu trainiert werden. Das Skript übernimmt automatisch das GridSearch-Tuning und speichert das beste Modell.
```bash
python src/models/ml_pipeline_de.py
python src/models/ml_pipeline_us.py
```

### 3. Dashboard starten
Starte das Streamlit UI, um interaktiv mit dem trainierten Modell zu arbeiten:
```bash
streamlit run app/app.py
```

---

## Testing

Das Projekt legt großen Wert auf Softwarequalität. Die gesamte Test-Suite (`tests/`) kann isoliert und ohne aktive Datenbankverbindung (vollständig gemockt) ausgeführt werden.

```bash
# Gesamte Test-Suite ausführen
pytest tests/ -v

# Tests mit Coverage-Report ausführen
pytest --cov=src --cov=app tests/
```