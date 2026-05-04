"""
ML Data Pipeline for cleaning unstructured JSON from Supabase.
Extracts hard features and uploads them to the 'listing_features' table.

Requirements:
pip install supabase pandas python-dotenv
"""

import os
import json
import ast
import re
import pandas as pd
from supabase import create_client
from datetime import datetime
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================

load_dotenv()

# =========================
# CONFIG
# =========================

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

SOURCE_TABLE = "listing_de"
TARGET_TABLE = "listing_features"  # Die NEUE Tabelle für ML-Features

SAVE_LOCAL = True
UPLOAD_TO_SUPABASE = True  # Auf True setzen für den automatischen Upload


# =========================
# SUPABASE CONNECTION
# =========================

def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================
# DATA FETCH
# =========================

def fetch_data(supabase):
    """
    Holt nur die listing_id und die unstrukturierten KI-Features.
    Wir brauchen hier keine Preis- oder Modelldaten, das spart Traffic!
    """
    response = (
        supabase
        .table(SOURCE_TABLE)
        .select("listing_id, ai_features")
        .execute()
    )

    data = response.data
    return pd.DataFrame(data)


# =========================
# JSON PARSING & EXTRACTION
# =========================

def robust_parse(raw):
    """Sicheres Parsen von fehlerhaftem JSON vom LLM."""
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return {}

    raw = raw.strip()
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Wenn JSON-Parse fehlschlägt, probiere ast.literal_eval (repariert z.B. single quotes)
    fixed = re.sub(r'\btrue\b', 'True', raw)
    fixed = re.sub(r'\bfalse\b', 'False', fixed)
    fixed = re.sub(r'\bnull\b', 'None', fixed)

    try:
        parsed = ast.literal_eval(fixed)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    return {}


def process_ai_features(raw_data):
    """
    Extrahieren der exakten Features aus dem Prompt.
    """
    data = robust_parse(raw_data)
    if not isinstance(data, dict):
        return {}

    # Alles rekursiv in string-basiertes flaches Dictionary schreiben
    flat_data = {}

    def _traverse(node, prefix=''):
        if isinstance(node, dict):
            for k, v in node.items():
                _traverse(v, str(k).lower().strip())
        elif isinstance(node, list):
            pass
        else:
            flat_data[prefix] = str(node).lower().strip()

    _traverse(data)

    features = {}

    # --- HELPER FUNKTION ---
    def get_val_for_keys(keys_list):
        for k, v in flat_data.items():
            if any(key in k for key in keys_list):
                return v
        return ''

    # --- TEIL 1: HISTORIE & ZUSTAND ---

    # TÜV
    tuv_val = get_val_for_keys(['tüv', 'tuv', 'hu/au', 'hu'])
    features['tuv_neu'] = 1 if any(w in tuv_val for w in ['neu', 'gültig', 'gemacht', 'true', '1', 'ja']) else 0

    # Scheckheft
    scheckheft_val = get_val_for_keys(['scheckheft', 'service'])
    features['scheckheft_gepflegt'] = 1 if any(
        w in scheckheft_val for w in ['lückenlos', 'scheckheft', 'service a neu', 'true', '1']) else 0

    # Bereifung
    bereifung_val = get_val_for_keys(['bereifung', 'reifen'])
    allwetter_val = get_val_for_keys(['allwetter', 'ganzjahres'])
    features['bereifung_8_fach'] = 1 if any(
        w in bereifung_val for w in ['8-fach', '8 fach', 'winter und sommer', 'winter- und sommer']) else 0
    features['bereifung_allwetter'] = 1 if any(
        w in bereifung_val + allwetter_val for w in ['allwetter', 'ganzjahres', 'true']) else 0

    # Garantie
    garantie_val = get_val_for_keys(['garantie'])
    match = re.search(r'(\d+)', garantie_val)  # Sucht nach der Zahl (Monate)
    features['garantie_monate'] = int(match.group(1)) if match else 0

    # Unfallfrei
    unfall_val = get_val_for_keys(['unfallfrei', 'unfall'])
    if any(w in unfall_val for w in ['false', '0', 'nein', 'nachlackierung', 'parkrempler', 'unfallwagen']):
        features['unfallfrei'] = 0
    elif any(w in unfall_val for w in ['true', '1', 'ja', 'yes', 'unfallfrei']):
        features['unfallfrei'] = 1
    else:
        features['unfallfrei'] = 0

    # Mängel
    mangel_val = get_val_for_keys(['mängel', 'mangel'])
    features['mangel_vorhanden'] = 0 if (
                'keine' in mangel_val or not mangel_val or mangel_val in ['none', 'false', '0']) else 1

    # --- TEIL 2: AUSSTATTUNG ---

    equip_keys = {
        'ausstattung_distronic': ['distronic', 'acc', 'abstandsregeltempomat'],
        'ausstattung_multibeam': ['multibeam', 'ils', 'matrix'],
        'ausstattung_klima_4_zonen': ['4-zonen', '4_zonen', 'thermotronic', 'klima hinten'],
        'ausstattung_klima_2_zonen': ['2-zonen', '2_zonen', 'thermatic'],
        'ausstattung_burmester_3d': ['burmester 3d', 'high-end', 'highend', '3d-soundsystem'],
        'ausstattung_burmester_standard': ['burmester (standard)', 'burmester standard', 'surround'],
        'ausstattung_amg_line': ['amg line', 'amg_line', 'amg-line'],
        'ausstattung_pano': ['pano', 'panorama', 'schiebedach']
    }

    for feat_name, keywords in equip_keys.items():
        feat_val = 0
        for k, v in flat_data.items():
            if any(kw in k for kw in keywords) or any(kw in v for kw in keywords):
                if v not in ['false', '0', 'nein', 'no', 'none', 'null', '']:
                    feat_val = 1
                    break
        features[feat_name] = feat_val

    return features


# =========================
# SAVE / UPLOAD RESULTS
# =========================

def save_local(df):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"listing_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved locally: {filename}")


def upload_to_supabase(supabase, df):
    # NaN Werte in None (für die Datenbank als Null-Werte) umwandeln
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")

    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        # upsert macht ein Insert oder Update (falls die ID schon existiert)
        supabase.table(TARGET_TABLE).upsert(batch).execute()

    print(f"Uploaded {len(records)} rows to {TARGET_TABLE} successfully!")


# =========================
# MAIN PIPELINE
# =========================

def run_pipeline():
    print("Connecting to Supabase...")
    supabase = get_supabase_client()

    print("Fetching raw AI features from listing_de...")
    df = fetch_data(supabase)

    if df.empty:
        print("No data found in Supabase.")
        return

    print("Extracting structured features from JSON...")
    # Extrahiere die Features
    features_df = df["ai_features"].apply(process_ai_features).apply(pd.Series)

    print("Combining listing_ids with features...")
    # Verbinde listing_id aus den Rohdaten mit den extrahierten Spalten
    # Wir lassen ai_features bewusst weg, um es nicht mit in die neue DB hochzuladen
    df_final = pd.concat([df[["listing_id"]], features_df], axis=1)

    # Entferne Zeilen, in denen keine Features generiert werden konnten
    # (falls z.B. alle Spalten NaN sind außer listing_id)
    df_final = df_final.dropna(subset=features_df.columns, how='all')

    if SAVE_LOCAL:
        save_local(df_final)

    if UPLOAD_TO_SUPABASE:
        print("Uploading cleanly structured features to Supabase...")
        upload_to_supabase(supabase, df_final)

    print("Feature Extraction Pipeline finished.")


# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    run_pipeline()