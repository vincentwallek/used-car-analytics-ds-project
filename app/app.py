import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from supabase import create_client
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from groq import Groq
import json
import os
from helpers import (
    img_to_base64, get_encoder_categories, generate_recommendations,
    DE_BINARY_LABELS, DE_CONDITION_KEYS, DE_EQUIP_KEYS,
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AutoValue | Fahrzeug-Intelligenz",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. THEME ENGINE
# ==========================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# --- Theme Palette ---
if st.session_state.theme == "dark":
    T = {
        "bg": "#0f172a",
        "bg_secondary": "#1e293b",
        "card_bg": "#1e293b",
        "card_border": "#334155",
        "text_primary": "#f1f5f9",
        "text_secondary": "#94a3b8",
        "text_heading": "#f8fafc",
        "accent": "#3b82f6",
        "accent_hover": "#60a5fa",
        "btn_bg": "#3b82f6",
        "btn_text": "#ffffff",
        "btn_hover_bg": "#2563eb",
        "divider": "#334155",
        "input_bg": "#1e293b",
        "input_border": "#475569",
        "input_text": "#f1f5f9",
        "shadow": "rgba(0,0,0,0.4)",
        "tab_active": "#3b82f6",
        "tab_inactive": "#94a3b8",
        "metric_bg": "#1e293b",
        "sidebar_bg": "#0f172a",
    }
else:
    T = {
        "bg": "#f8fafc",
        "bg_secondary": "#ffffff",
        "card_bg": "#ffffff",
        "card_border": "#e2e8f0",
        "text_primary": "#1e293b",
        "text_secondary": "#64748b",
        "text_heading": "#0f172a",
        "accent": "#2563eb",
        "accent_hover": "#1d4ed8",
        "btn_bg": "#2563eb",
        "btn_text": "#ffffff",
        "btn_hover_bg": "#1d4ed8",
        "divider": "#e2e8f0",
        "input_bg": "#f1f5f9",
        "input_border": "#cbd5e1",
        "input_text": "#1e293b",
        "shadow": "rgba(0,0,0,0.08)",
        "tab_active": "#2563eb",
        "tab_inactive": "#64748b",
        "metric_bg": "#f1f5f9",
        "sidebar_bg": "#ffffff",
    }

# ==========================================
# 3. GLOBAL STYLESHEET
# ==========================================
st.markdown(f"""
    <style>
        /* --- Google Font --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* --- Root Reset --- */
        html, body, .stApp {{
            background-color: {T['bg']} !important;
            font-family: 'Inter', sans-serif !important;
        }}

        /* --- Hide sidebar completely --- */
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}

        /* --- Theme toggle icon button --- */
        .theme-toggle-col .stButton > button {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            font-size: 1.5rem;
            padding: 0.25rem 0.5rem;
            min-width: auto;
            width: auto;
        }}
        .theme-toggle-col .stButton > button:hover {{
            background: transparent !important;
            transform: scale(1.15);
        }}

        /* --- Typography --- */
        .stMarkdown, p, span, label, li {{
            color: {T['text_primary']} !important;
            font-family: 'Inter', sans-serif !important;
        }}
        h1, h2, h3, h4 {{
            color: {T['text_heading']} !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
        }}

        /* --- Cards --- */
        .av-card {{
            background-color: {T['card_bg']};
            padding: 2rem;
            border-radius: 14px;
            box-shadow: 0 4px 24px {T['shadow']};
            border: 1px solid {T['card_border']};
            margin-bottom: 1.5rem;
            min-height: 240px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .av-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 32px {T['shadow']};
        }}
        .av-card h3 {{
            color: {T['text_heading']} !important;
            margin-bottom: 0.5rem;
            font-size: 1.25rem;
        }}
        .av-card p {{
            color: {T['text_secondary']} !important;
            font-size: 0.9rem;
            line-height: 1.6;
        }}
        .av-card .card-tag {{
            display: inline-block;
            background-color: {T['accent']};
            color: #ffffff;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            margin-bottom: 0.75rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        /* --- Buttons --- */
        .stButton > button {{
            width: 100%;
            background-color: {T['btn_bg']} !important;
            color: {T['btn_text']} !important;
            border-radius: 10px;
            padding: 0.75rem 1.25rem;
            border: none !important;
            font-weight: 600;
            font-size: 0.9rem;
            font-family: 'Inter', sans-serif !important;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.15s ease;
            letter-spacing: 0.2px;
        }}
        .stButton > button:hover {{
            background-color: {T['btn_hover_bg']} !important;
            color: {T['btn_text']} !important;
            transform: translateY(-1px);
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}
        /* Fix: force button text color on all states */
        .stButton > button p,
        .stButton > button span,
        .stButton > button div {{
            color: {T['btn_text']} !important;
        }}

        /* --- Form submit button --- */
        .stFormSubmitButton > button {{
            background-color: {T['accent']} !important;
            color: #ffffff !important;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.75rem;
            border: none !important;
            transition: background-color 0.2s ease;
        }}
        .stFormSubmitButton > button:hover {{
            background-color: {T['accent_hover']} !important;
            color: #ffffff !important;
        }}
        .stFormSubmitButton > button p,
        .stFormSubmitButton > button span {{
            color: #ffffff !important;
        }}

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            border-bottom: 2px solid {T['divider']};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 48px;
            padding: 0 1.5rem;
            background-color: transparent !important;
            color: {T['tab_inactive']} !important;
            border: none;
            font-weight: 500;
            font-family: 'Inter', sans-serif !important;
            transition: color 0.2s ease;
        }}
        .stTabs [aria-selected="true"] {{
            color: {T['tab_active']} !important;
            border-bottom: 2px solid {T['tab_active']} !important;
            font-weight: 600;
        }}

        /* --- Input fields --- */
        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {{
            background-color: {T['input_bg']} !important;
            color: {T['input_text']} !important;
            border: 1px solid {T['input_border']} !important;
            border-radius: 8px !important;
        }}
        .stSelectbox label,
        .stNumberInput label,
        .stTextInput label,
        .stCheckbox label {{
            color: {T['text_secondary']} !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }}
        .stSelectbox svg {{
            fill: {T['text_secondary']} !important;
        }}

        /* --- Checkbox --- */
        .stCheckbox span {{
            color: {T['text_primary']} !important;
        }}

        /* --- Metrics --- */
        [data-testid="stMetricValue"] {{
            color: {T['accent']} !important;
            font-weight: 700 !important;
            font-size: 2rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {T['text_secondary']} !important;
        }}
        [data-testid="metric-container"] {{
            background-color: {T['metric_bg']};
            border: 1px solid {T['card_border']};
            padding: 1rem 1.5rem;
            border-radius: 12px;
        }}

        /* --- Divider --- */
        hr {{
            border-color: {T['divider']} !important;
        }}

        /* --- Dataframe --- */
        .stDataFrame {{
            border-radius: 10px;
            overflow: hidden;
        }}

        /* --- HTML Table (Passende Angebote) --- */
        table {{
            width: 100% !important;
            border-collapse: collapse;
            border-radius: 10px;
            overflow: hidden;
            font-size: 0.9rem;
        }}
        table th {{
            background-color: {T['card_bg']};
            color: {T['text_secondary']};
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid {T['divider']};
        }}
        table td {{
            padding: 0.6rem 1rem;
            border-bottom: 1px solid {T['divider']};
            color: {T['text_primary']};
        }}
        table tr:hover {{
            background-color: {T['card_bg']};
        }}
        table a {{
            color: {T['accent']} !important;
            text-decoration: none;
        }}
        table a:hover {{
            text-decoration: underline;
        }}

        /* --- Chat --- */
        .stChatMessage {{
            background-color: {T['card_bg']} !important;
            border: 1px solid {T['card_border']} !important;
            border-radius: 12px !important;
        }}

        /* --- Spinner --- */
        .stSpinner > div {{
            border-top-color: {T['accent']} !important;
        }}

        /* --- Hide Streamlit branding --- */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* --- Logo container --- */
        .av-logo-wrap {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .av-logo-wrap img {{
            border-radius: 10px;
            background-color: transparent;
        }}
        .av-logo-text {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {T['text_heading']};
            letter-spacing: -0.5px;
        }}
        .av-logo-sub {{
            font-size: 0.78rem;
            color: {T['text_secondary']};
            margin-top: -2px;
        }}

        /* --- Header bar --- */
        .av-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
        }}

        /* --- Hero Section --- */
        .av-hero {{
            text-align: center;
            padding: 2rem 0 1rem 0;
        }}
        .av-hero h2 {{
            font-size: 2rem;
            font-weight: 700;
            color: {T['text_heading']} !important;
            margin-bottom: 0.5rem;
        }}
        .av-hero p {{
            color: {T['text_secondary']} !important;
            font-size: 1.05rem;
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.7;
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. STATE MANAGEMENT & ROUTING
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'role' not in st.session_state:
    st.session_state.role = None
if 'market' not in st.session_state:
    st.session_state.market = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def nav(page, role=None, market=None):
    """Set navigation state."""
    st.session_state.page = page
    st.session_state.role = role
    st.session_state.market = market


# ==========================================
# 5. INITIALIZATION & DATA ENGINE
# ==========================================
@st.cache_resource
def init_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


supabase = init_supabase()
geolocator = Nominatim(user_agent="autovalue_v3")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


@st.cache_resource
def load_models():
    models = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base_dir, "car_price_xgboost.pkl"), "rb") as f:
            models["de_model"] = pickle.load(f)
        with open(os.path.join(base_dir, "categorical_encoder.pkl"), "rb") as f:
            models["de_encoder"] = pickle.load(f)
        with open(os.path.join(base_dir, "numeric_columns.pkl"), "rb") as f:
            models["de_num_cols"] = pickle.load(f)
        with open(os.path.join(base_dir, "car_price_xgboost_us.pkl"), "rb") as f:
            models["us_model"] = pickle.load(f)
        with open(os.path.join(base_dir, "categorical_encoder_us.pkl"), "rb") as f:
            models["us_encoder"] = pickle.load(f)
        with open(os.path.join(base_dir, "numeric_columns_us.pkl"), "rb") as f:
            models["us_num_cols"] = pickle.load(f)
        return models
    except:
        return None


trained_models = load_models()


@st.cache_data(ttl=600)
def get_market_data(market_code):
    """Load listings from Supabase with proper table joins.
    DE: listings + listing_de (title) + listing_features (equipment).
    US: listings + listing_us (trim, engine, etc.).
    All linked via listings.id = *.listing_id.
    """
    empty = pd.DataFrame(columns=["brand", "model", "title", "price", "mileage", "location", "url"])
    try:
        # 1. Load base listings filtered by market
        res = supabase.table("listings").select("*").eq("market", market_code.upper()).execute()
        df = pd.DataFrame(res.data)
        if df.empty:
            return empty
        df.columns = [c.lower() for c in df.columns]
        df["brand"] = df["brand"].astype(str).str.lower().str.strip()
        df["model"] = df["model"].astype(str).str.lower().str.strip()

        if market_code.upper() == "DE":
            # 2a. Join listing_de (title, transmission, fuel, etc.)
            try:
                res_de = supabase.table("listing_de").select("*").execute()
                df_de = pd.DataFrame(res_de.data)
                if not df_de.empty:
                    df_de.columns = [c.lower() for c in df_de.columns]
                    df = df.merge(df_de, left_on="id", right_on="listing_id",
                                  how="left", suffixes=("", "_de"))
            except:
                pass
            # 2b. Join listing_features (equipment booleans)
            try:
                res_feat = supabase.table("listing_features").select("*").execute()
                df_feat = pd.DataFrame(res_feat.data)
                if not df_feat.empty:
                    df_feat.columns = [c.lower() for c in df_feat.columns]
                    # Fix DB typo: standart -> standard
                    if "ausstattung_burmester_standart" in df_feat.columns:
                        df_feat = df_feat.rename(columns={"ausstattung_burmester_standart": "ausstattung_burmester_standard"})
                    df = df.merge(df_feat, left_on="id", right_on="listing_id",
                                  how="left", suffixes=("", "_feat"))
            except:
                pass
            # Ensure title column
            if "title" not in df.columns:
                df["title"] = df["brand"].apply(_fmt) + " " + df["model"].apply(_fmt)
        else:
            # 2. Join listing_us (trim, engine, colors, etc.)
            try:
                res_us = supabase.table("listing_us").select("*").execute()
                df_us = pd.DataFrame(res_us.data)
                if not df_us.empty:
                    df_us.columns = [c.lower() for c in df_us.columns]
                    df = df.merge(df_us, left_on="id", right_on="listing_id",
                                  how="left", suffixes=("", "_us"))
            except:
                pass
            # Compose title: Brand Model Trim
            trim_col = df["trim"].astype(str).str.strip() if "trim" in df.columns else pd.Series([""] * len(df))
            df["title"] = (df["brand"].apply(_fmt) + " " + df["model"].apply(_fmt)
                           + (" " + trim_col).where(trim_col.ne("") & trim_col.ne("nan"), ""))

        # Ensure required columns exist
        for col in ["brand", "model", "title", "price", "mileage", "location", "url"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return empty


@st.cache_data
def get_coords(loc_string):
    if not loc_string or loc_string == "unbekannt":
        return None, None
    try:
        loc = geocode(loc_string)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None


def predict_price(market, input_data):
    if not trained_models:
        return 0.0, None
    m_code = "de" if market == "DE" else "us"
    model = trained_models[f"{m_code}_model"]
    encoder = trained_models[f"{m_code}_encoder"]
    num_cols = trained_models[f"{m_code}_num_cols"]
    cat_cols = (
        ["brand", "model", "transmission", "fuel"]
        if market == "DE"
        else [
            "brand", "model", "trim", "drivetrain", "fuel",
            "transmission", "body_style", "engine",
            "exterior_color", "interior_color", "usage_type",
        ]
    )
    df_input = pd.DataFrame([input_data])
    for col in num_cols:
        if col not in df_input.columns:
            df_input[col] = 0.0
            
    for col in cat_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).str.lower().str.strip()
            
    encoded_cats = encoder.transform(df_input[cat_cols])
    df_encoded = pd.DataFrame(
        encoded_cats,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df_input.index,
    )
    X_final = pd.concat([df_input[num_cols], df_encoded], axis=1)
    prediction = model.predict(X_final)[0]
    shap_values = shap.TreeExplainer(model)(X_final)
    return prediction, shap_values

def run_ml_prediction(market, brand, model_name, car_age, mileage, transmission="automatic", fuel="benzin", power_ps=150, cylinders=6):
    """Dieses Tool wird vom LLM aufgerufen, um echte ML-Preise zu berechnen."""
    if market == "DE":
        input_vals = {"brand": brand.lower(), "model": model_name.lower(), "car_age": float(car_age), "mileage": float(mileage),
                      "transmission": transmission, "fuel": fuel, "power_ps": float(power_ps), "owners": 1.0,
                      "ausstattung_pano": 0.0, "ausstattung_amg_line": 0.0}
    else:
        input_vals = {"brand": brand.lower(), "model": model_name.lower(), "car_age": float(car_age), "mileage": float(mileage),
                      "transmission": transmission, "fuel": fuel, "cylinders": float(cylinders), "engine": "unbekannt",
                      "has_accidents": 0.0, "is_cpo": 0.0, "doors": 4.0, "seats": 5.0, "trim": "unbekannt", 
                      "drivetrain": "unbekannt", "body_style": "unbekannt", "exterior_color": "unbekannt", 
                      "interior_color": "unbekannt", "usage_type": "unbekannt"}

    price, s_vals = predict_price(market, input_vals)
    
    if s_vals is None:
        return json.dumps({"error": "Modell konnte nicht geladen werden."})
        
    # Wir geben ALLE Faktoren an die KI, damit sie gezielt filtern kann
    impact_dict = {feat: round(float(val), 2) for feat, val in zip(s_vals[0].feature_names, s_vals[0].values)}
    
    result = {
        "berechneter_preis": round(float(price), 2),
        "waehrung": "€" if market == "DE" else "$",
        "alle_preis_einflussfaktoren": impact_dict
    }
    return json.dumps(result)


# ==========================================
# 6. DISPLAY HELPERS
# ==========================================

def _fmt(s):
    """Capitalize brand/model for display. E.g. 'mercedes-benz' -> 'Mercedes-Benz', 'nx' -> 'NX'."""
    if not s:
        return s
    parts = s.split("-")
    return "-".join(p.upper() if (p.isalpha() and len(p) <= 3) else p.capitalize() for p in parts)


# --- SHAP Label Translation ---
_SHAP_DIRECT = {
    "mileage": "Kilometerstand", "car_age": "Fahrzeugalter",
    "power_ps": "Leistung (PS)", "owners": "Vorbesitzer",
    "garantie_monate": "Garantie (Monate)", "cylinders": "Zylinder",
    "doors": "T\u00fcren", "seats": "Sitze", "accident_count": "Unf\u00e4lle",
    "owner_count": "Vorbesitzer", "one_owner": "Erstbesitzer",
    "has_accidents": "Hat Unf\u00e4lle", "is_used": "Gebraucht",
    "is_cpo": "Zertifiziert", "is_online": "Online-Kauf",
    "is_wholesale": "Gro\u00dfhandel", "personal_use": "Privatnutzung",
    "tuv_neu": "T\u00dcV neu", "unfallfrei": "Unfallfrei",
    "mangel_vorhanden": "M\u00e4ngel", "scheckheft_gepflegt": "Scheckheft",
    "ausstattung_pano": "Panoramadach", "ausstattung_amg_line": "AMG Line",
    "ausstattung_distronic": "Distronic", "ausstattung_multibeam": "Multibeam LED",
    "ausstattung_klima_4_zonen": "4-Zonen Klima", "ausstattung_klima_2_zonen": "2-Zonen Klima",
    "ausstattung_burmester_3d": "Burmester 3D", "ausstattung_burmester_standard": "Burmester Std.",
    "bereifung_8_fach": "8-fach Bereifung", "bereifung_allwetter": "Allwetterreifen",
    "price_per_km": "Preis/km", "price_per_mile": "Preis/Meile",
    "listing_age_days": "Inseratsalter (Tage)",
}
_SHAP_PREFIX = {
    "brand_": "Marke", "model_": "Modell", "fuel_": "Kraftstoff",
    "transmission_": "Getriebe", "drivetrain_": "Antrieb",
    "body_style_": "Karosserie", "engine_": "Motor",
    "exterior_color_": "Au\u00dfenfarbe", "interior_color_": "Innenfarbe",
    "usage_type_": "Nutzungsart", "trim_": "Ausstattungslinie",
}


def _translate_shap(name):
    """Translate encoded SHAP feature name to readable German."""
    if name in _SHAP_DIRECT:
        return _SHAP_DIRECT[name]
    for prefix, label in _SHAP_PREFIX.items():
        if name.startswith(prefix):
            value = name[len(prefix):]
            return f"{label}: {_fmt(value)}"
    return name.replace("_", " ").title()


def _render_shap_display(s_vals, csym, input_vals):
    """Render SHAP values as clean text-based impact cards (excluding other model impacts)."""
    sv = s_vals[0]
    features = sv.feature_names
    values = sv.values
    base = sv.base_values

    # Filter out model_ features (other models are not relevant for the analysis)
    impacts = [(f, v) for f, v in zip(features, values) if not f.startswith("model_")]
    impacts = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)[:10]

    st.caption(f"Basiswert (Durchschnitt): {csym}{base:,.0f}")
    
    # US Binary Mappings
    US_BINARY_LABELS = {
        "has_accidents": "Unfälle",
        "one_owner": "Erster Hand",
        "is_online": "Online-Kauf",
        "is_used": "Gebrauchtwagen",
        "is_cpo": "CPO-Fahrzeug",
        "is_wholesale": "Wholesale-Angebot",
        "personal_use": "Private Nutzung",
    }
    
    # Grammatical gender mapping for "Kein" vs "Keine"
    GENDER_MAP = {
        "Motor": "Kein",
        "Karosserie": "Keine",
        "Außenfarbe": "Keine",
        "Innenfarbe": "Keine",
        "Marke": "Keine",
        "Kraftstoff": "Kein",
        "Antrieb": "Kein",
        "Getriebe": "Kein",
        "Ausstattungslinie": "Keine",
        "Modell": "Kein",
        "Nutzungsart": "Keine",
    }

    for feat, val in impacts:
        label = ""
        
        if feat in DE_BINARY_LABELS:
            is_active = input_vals.get(feat, 0.0) == 1.0
            prefix = "Mit" if is_active else "Ohne"
            label = f"{prefix} {DE_BINARY_LABELS[feat]}"
            
        elif feat in US_BINARY_LABELS:
            is_active = input_vals.get(feat, 0.0) == 1.0
            if feat == "has_accidents":
                label = "Mit Unfällen" if is_active else "Unfallfrei"
            elif feat == "one_owner":
                label = "Aus 1. Hand" if is_active else "Mehrere Vorbesitzer"
            else:
                prefix = "Mit" if is_active else "Ohne"
                label = f"{prefix} {US_BINARY_LABELS[feat]}"
                
        else:
            is_prefix_feat = False
            for p_key, p_label in _SHAP_PREFIX.items():
                if feat.startswith(p_key):
                    is_prefix_feat = True
                    actual_value = feat[len(p_key):]
                    input_key = p_key[:-1]
                    user_value = input_vals.get(input_key, "")
                    
                    is_active = (str(user_value).lower() == str(actual_value).lower())
                    gender = GENDER_MAP.get(p_label, "Kein")
                    
                    if is_active:
                        label = f"{p_label}: {_fmt(actual_value)}"
                    else:
                        label = f"{gender} {p_label}: {_fmt(actual_value)}"
                    break
                    
            if not is_prefix_feat:
                if feat in input_vals:
                    v = input_vals[feat]
                    base_label = _translate_shap(feat)
                    if feat == "mileage":
                        unit = "km" if csym == "€" else "Meilen"
                        label = f"{base_label} von {v:,.0f} {unit}"
                    elif feat == "car_age":
                        label = f"{base_label} von {int(v)} Jahren"
                    elif feat == "power_ps":
                        label = f"{base_label} von {int(v)} PS"
                    elif feat in ["owners", "owner_count"]:
                        label = f"{int(v)} Vorbesitzer"
                    elif feat == "accident_count":
                        label = f"{int(v)} gemeldete Unfälle"
                    elif feat == "cylinders":
                        label = f"{int(v)}-Zylinder-Motor"
                    elif feat == "doors":
                        label = f"{int(v)} Türen"
                    elif feat == "seats":
                        label = f"{int(v)} Sitze"
                    elif feat == "garantie_monate":
                        label = f"{int(v)} Monate Garantie"
                    else:
                        label = f"{base_label}: {v}"
                else:
                    label = _translate_shap(feat)
            
        direction = "\u2191" if val > 0 else "\u2193"
        color = "#22c55e" if val > 0 else "#ef4444"
        bg = "rgba(34,197,94,0.1)" if val > 0 else "rgba(239,68,68,0.1)"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:0.6rem 1rem;margin:0.3rem 0;border-radius:8px;"
            f"background:{bg};border-left:3px solid {color};'>"
            f"<span style='color:{T['text_primary']};font-size:0.9rem;'>{label}</span>"
            f"<span style='color:{color};font-weight:600;font-size:0.9rem;'>"
            f"{direction} {val:+,.0f} {csym}</span></div>",
            unsafe_allow_html=True,
        )


# ==========================================
# 7. UI VIEWS
# ==========================================

def view_header():
    """Render the top header bar with logo, theme toggle, and back-navigation."""
    if st.session_state.page == "home":
        # Dashboard layout: Only space and the theme toggle on the right
        col_spacer, col_theme = st.columns([9, 1])
        with col_theme:
            st.markdown('<div class="theme-toggle-col" style="position: relative; z-index: 9999;">', unsafe_allow_html=True)
            icon = "☀️" if st.session_state.theme == "dark" else "🌙"
            if st.button(icon, key="theme_toggle", help="Design wechseln"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Large centered logo ABOVE the divider line
        base_path = os.path.dirname(os.path.abspath(__file__))
        if st.session_state.theme == "dark":
            logo_path = os.path.join(base_path, "dark_logo.png")
        else:
            logo_path = os.path.join(base_path, "light_logo.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(base_path, "logo.png")
            
        if os.path.exists(logo_path):
            st.markdown(
                f'<div style="display: flex; justify-content: center; margin-top: -6rem; margin-bottom: 2rem; pointer-events: none;">'
                f'<img src="data:image/png;base64,{img_to_base64(logo_path)}" width="300" />'
                f'</div>',
                unsafe_allow_html=True,
            )
            
    else:
        # Standard sub-pages layout: Logo on the left, nav, theme toggle
        col_logo, col_spacer, col_nav, col_theme = st.columns([2.5, 4.5, 2, 1])
        with col_logo:
            base_path = os.path.dirname(os.path.abspath(__file__))
            if st.session_state.theme == "dark":
                logo_path = os.path.join(base_path, "dark_logo.png")
            else:
                logo_path = os.path.join(base_path, "light_logo.png")
            if not os.path.exists(logo_path):
                logo_path = os.path.join(base_path, "logo.png")
                
            if os.path.exists(logo_path):
                st.markdown(
                    f'<div class="av-logo-wrap">'
                    f'<img src="data:image/png;base64,{img_to_base64(logo_path)}" width="110" height="110" />'
                    f'<div><div class="av-logo-text">AutoValue</div>'
                    f'<div class="av-logo-sub">Fahrzeug-Intelligenz</div></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="av-logo-wrap"><div>'
                    '<div class="av-logo-text">AutoValue</div>'
                    '<div class="av-logo-sub">Fahrzeug-Intelligenz</div></div></div>',
                    unsafe_allow_html=True,
                )
        with col_nav:
            st.write("")
            if st.button("Zurück zum Dashboard", key="btn_back"):
                nav("home")
                st.rerun()
                
        with col_theme:
            st.markdown('<div class="theme-toggle-col">', unsafe_allow_html=True)
            icon = "☀️" if st.session_state.theme == "dark" else "🌙"
            if st.button(icon, key="theme_toggle", help="Design wechseln"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"<hr style='border:none;border-top:1px solid {T['divider']};margin:0.5rem 0 1.5rem 0;'>",
                unsafe_allow_html=True)



def view_home():
    """Render the home / dashboard page."""
    st.markdown(
        f"""
        <div class="av-hero">
            <h2>Professionelle Fahrzeugbewertung</h2>
            <p>Nutzen Sie Machine-Learning-Modelle, trainiert auf echten Marktdaten,
            für präzise Fahrzeugbewertungen auf dem deutschen und US-Markt.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            f"""
            <div class="av-card">
                <span class="card-tag">Verkäufer</span>
                <h3>Verkäufer-Intelligenz</h3>
                <p>Geben Sie Ihre Fahrzeugdaten ein und erhalten Sie eine
                datenbasierte Marktbewertung. Verstehen Sie, welche Merkmale
                den Preis beeinflussen.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1a, c1b = st.columns(2)
        with c1a:
            if st.button("Bewertung starten -- DE", key="btn_s_de"):
                nav("app", "seller", "DE")
                st.rerun()
        with c1b:
            if st.button("Bewertung starten -- US", key="btn_s_us"):
                nav("app", "seller", "US")
                st.rerun()

    with col2:
        st.markdown(
            f"""
            <div class="av-card">
                <span class="card-tag">Käufer</span>
                <h3>Käufer-Intelligenz</h3>
                <p>Durchsuchen Sie echte Inserate, vergleichen Sie Preise mit
                dem vorhergesagten Marktwert und finden Sie die besten
                Angebote auf dem Markt.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c2a, c2b = st.columns(2)
        with c2a:
            if st.button("Inventar suchen -- DE", key="btn_b_de"):
                nav("app", "buyer", "DE")
                st.rerun()
        with c2b:
            if st.button("Inventar suchen -- US", key="btn_b_us"):
                nav("app", "buyer", "US")
                st.rerun()


def view_app():
    """Render the analysis / chat application page."""
    market = st.session_state.market
    role = st.session_state.role
    db_data = get_market_data(market)
    currency = "EUR" if market == "DE" else "USD"
    csym = "\u20ac" if market == "DE" else "$"
    enc_cats = get_encoder_categories(trained_models, market)

    role_de = "Verkäufer" if role == "seller" else "Käufer"
    st.subheader(f"{role_de}-Analyse  |  Markt: {market}")
    tab_engine, tab_chat = st.tabs(["Analyse", "AutoValue Assistent"])

    with tab_engine:
        # Brand→Model fallback when db_data is unavailable
        BRAND_MODELS = {
            # DE
            "mercedes-benz": ["c-klasse", "e-klasse", "s-klasse", "a-klasse", "b-klasse",
                              "cla", "cls", "glc", "gle", "gla", "glb", "gls",
                              "eqe", "eqs", "eqa", "eqb", "eqc", "v-klasse", "g-klasse"],
            # US
            "ford": ["f-150", "f-250", "f-350"],
            "lexus": ["nx"],
        }
        if market == "DE":
            bm1, bm2, bm3 = st.columns(3)
        else:
            bm1, bm2 = st.columns(2)
            
        with bm1:
            brand_options = enc_cats.get("brand", sorted(db_data['brand'].unique()) if not db_data.empty else ["mercedes-benz"])
            brand = st.selectbox("Marke", brand_options, key="sel_brand", format_func=_fmt)
        with bm2:
            model_options = BRAND_MODELS.get(brand, ["unknown"])
            if not db_data.empty:
                filtered = sorted(db_data[db_data['brand'] == brand]['model'].unique())
                if filtered:
                    model_options = filtered
            model_name = st.selectbox("Modell", model_options, key="sel_model", format_func=_fmt)
            
        if market == "DE":
            with bm3:
                motor_options = ["Alle"]
                if not db_data.empty:
                    subset = db_data[(db_data["brand"] == brand) & (db_data["model"] == model_name)]
                    if "title" in subset.columns:
                        import re
                        brand_pattern = re.compile(rf'\b{re.escape(str(brand))}\b', re.IGNORECASE)
                        extracted = []
                        for t in subset["title"].dropna().unique():
                            t_clean = brand_pattern.sub('', str(t)).strip()
                            t_clean = re.sub(r'\s+', ' ', t_clean)
                            if t_clean:
                                extracted.append(t_clean)
                        if extracted:
                            motor_options = ["Alle"] + sorted(list(set(extracted)))
                st.selectbox("Motorleistung / Trim", motor_options, key="sel_motorleistung")

        # Advanced options toggle (outside form to avoid _arrow_right bug)
        show_advanced = True
        if role == "buyer":
            show_advanced = st.toggle("Erweiterte Optionen anzeigen", value=False, key="show_advanced")

        with st.form("valuation_form"):
            if market == "DE":
                _render_de_form_fields(enc_cats, role, show_advanced)
            else:
                _render_us_form_fields(enc_cats, db_data, brand, model_name, role, show_advanced)
            submitted = st.form_submit_button("Analyse starten")

        if submitted:
            with st.spinner("Marktwert wird berechnet ..."):
                input_vals = _collect_inputs(market, brand, model_name)
                price, s_vals = predict_price(market, input_vals)
                # Store results in session_state so they persist across reruns
                st.session_state.last_result = {
                    "price": price, "s_vals": s_vals, "input_vals": input_vals,
                    "brand": brand, "model_name": model_name,
                    "market": market, "role": role, "csym": csym,
                }
                # Reset show-all states on new submission
                st.session_state.show_all_matches = False
                st.session_state.show_all_recs = False

        # ── Display results (persisted in session_state) ──
        result = st.session_state.get("last_result")
        if result and result.get("market") == market and result.get("role") == role:
            price = result["price"]
            s_vals = result["s_vals"]
            input_vals = result["input_vals"]
            res_brand = result["brand"]
            res_model = result["model_name"]

            st.markdown(f"<hr style='border:none;border-top:1px solid {T['divider']};margin:1.5rem 0;'>",
                        unsafe_allow_html=True)
            st.metric("Gesch\u00e4tzter Marktwert", f"{csym}{price:,.2f}")

            if role == "seller" and s_vals:
                st.markdown("### Einflussfaktoren auf den Preis")
                _render_shap_display(s_vals, csym, input_vals)

            if role == "buyer" and not db_data.empty:
                st.markdown("### Passende Angebote")
                matches = db_data[(db_data['brand'] == res_brand) & (db_data['model'] == res_model)].copy()

                if market == "DE" and "sel_motorleistung" in st.session_state:
                    sel_trim = st.session_state.sel_motorleistung
                    if sel_trim != "Alle":
                        import re
                        brand_pattern = re.compile(rf'\b{re.escape(str(res_brand))}\b', re.IGNORECASE)
                        def matches_trim(row):
                            title_clean = brand_pattern.sub('', str(row.get("title", ""))).strip()
                            title_clean = re.sub(r'\s+', ' ', title_clean)
                            return title_clean.lower() == sel_trim.lower()
                        matches = matches[matches.apply(matches_trim, axis=1)]

                # Smart filtering: apply input criteria as bounds
                user_mileage = input_vals.get("mileage", 0)
                if user_mileage > 0 and "mileage" in matches.columns:
                    matches["mileage"] = pd.to_numeric(matches["mileage"], errors="coerce")
                    matches = matches[matches["mileage"].fillna(0) <= user_mileage]
                user_age = input_vals.get("car_age", 0)
                if user_age > 0 and "car_age" in matches.columns:
                    matches["car_age"] = pd.to_numeric(matches["car_age"], errors="coerce")
                    matches = matches[matches["car_age"].fillna(0) <= user_age]
                user_owners = input_vals.get("owners", 0) if market == "DE" else input_vals.get("owner_count", 0)
                if user_owners > 0 and "owners" in matches.columns:
                    matches["owners"] = pd.to_numeric(matches["owners"], errors="coerce")
                    matches = matches[matches["owners"].fillna(99) <= user_owners]
                user_ps = input_vals.get("power_ps", 0)
                if user_ps > 0 and "power_ps" in matches.columns:
                    matches["power_ps"] = pd.to_numeric(matches["power_ps"], errors="coerce")
                    matches = matches[matches["power_ps"].fillna(0) >= user_ps]

                if market == "US":
                    user_fuel = input_vals.get("fuel", "unknown")
                    if user_fuel != "unknown" and "fuel" in matches.columns:
                        matches = matches[matches["fuel"].astype(str).str.lower() == str(user_fuel).lower()]
                    
                    user_cyl = input_vals.get("cylinders", 0)
                    if user_cyl > 0 and "cylinders" in matches.columns:
                        matches = matches[pd.to_numeric(matches["cylinders"], errors="coerce") == user_cyl]
                        
                    user_trim = input_vals.get("trim", "unknown")
                    if user_trim != "unknown" and "trim" in matches.columns:
                        matches = matches[matches["trim"].astype(str).str.lower() == str(user_trim).lower()]
                        
                    user_engine = input_vals.get("engine", "unknown")
                    if user_engine != "unknown" and "engine" in matches.columns:
                        matches = matches[matches["engine"].astype(str).str.lower() == str(user_engine).lower()]
                        
                elif market == "DE":
                    user_fuel = input_vals.get("fuel")
                    if user_fuel and "fuel" in matches.columns:
                        matches = matches[matches["fuel"].astype(str).str.lower() == str(user_fuel).lower()]
                    
                    user_trans = input_vals.get("transmission")
                    if user_trans and "transmission" in matches.columns:
                        matches = matches[matches["transmission"].astype(str).str.lower() == str(user_trans).lower()]

                if not matches.empty:
                    # Location filter for map interaction
                    filter_col, sort_col = st.columns([2, 2])
                    with sort_col:
                        sort_option = st.selectbox("Sortieren nach", [
                            "Preis aufsteigend", "Preis absteigend",
                            "Kilometerstand aufsteigend", "Kilometerstand absteigend"
                        ], key="sort_matches")
                    locations = []
                    if "location" in matches.columns:
                        locations = sorted(matches["location"].dropna().astype(str).unique())
                        locations = [l for l in locations if l and l != "nan"]
                    with filter_col:
                        loc_options = ["Alle Standorte"] + locations
                        loc_filter = st.selectbox("Standort filtern", loc_options, key="loc_filter")
                    if loc_filter != "Alle Standorte" and "location" in matches.columns:
                        matches = matches[matches["location"] == loc_filter]

                    sort_map = {
                        "Preis aufsteigend": ("price", True),
                        "Preis absteigend": ("price", False),
                        "Kilometerstand aufsteigend": ("mileage", True),
                        "Kilometerstand absteigend": ("mileage", False),
                    }
                    s_c, s_a = sort_map[sort_option]
                    if s_c in matches.columns:
                        matches = matches.sort_values(s_c, ascending=s_a, na_position="last")

                    # Show count control
                    show_all = st.session_state.get("show_all_matches", False)
                    display_matches = matches if show_all else matches.head(20)

                    # Build display dataframe — full width
                    if market == "DE":
                        disp = display_matches[[c for c in ["title", "price", "mileage", "location"] if c in display_matches.columns]].copy()
                        if "url" in display_matches.columns:
                            disp["Link"] = display_matches["url"].apply(
                                lambda u: f'<a href="{u}" target="_blank">Zum Inserat</a>' if pd.notna(u) and str(u).startswith("http") else ""
                            )
                        st.markdown(
                            "<div style='width:100%;overflow-x:auto;'>" + disp.to_html(escape=False, index=False) + "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        disp_cols = [c for c in ["title", "price", "mileage", "location"] if c in display_matches.columns]
                        st.dataframe(display_matches[disp_cols], use_container_width=True, hide_index=True)

                    if len(matches) > 20 and not show_all:
                        if st.button(f"Alle {len(matches)} Angebote anzeigen", key="btn_show_all_matches"):
                            st.session_state.show_all_matches = True
                            st.rerun()
                    elif show_all and len(matches) > 20:
                        st.caption(f"{len(matches)} Angebote angezeigt.")

                    # Map with location data
                    if "location" in display_matches.columns:
                        map_data = []
                        for _, row in display_matches.head(30).iterrows():
                            lt, ln = get_coords(row.get("location", ""))
                            if lt:
                                map_data.append({"lat": lt, "lon": ln, "title": str(row.get("title", "")), "location": str(row.get("location", ""))})
                        if map_data:
                            st.caption("Klicken Sie auf einen Standort oben, um die Tabelle zu filtern.")
                            st.map(pd.DataFrame(map_data))
                else:
                    st.caption("Keine passenden Angebote f\u00fcr diese Konfiguration gefunden.")

            st.markdown("### Empfehlungen")
            recs = generate_recommendations(trained_models, market, input_vals, price, db_data, csym, role)
            if recs:
                show_all_recs = st.session_state.get("show_all_recs", False)
                show_count = len(recs) if show_all_recs else min(5, len(recs))
                for r in recs[:show_count]:
                    st.info(r["text"])
                if len(recs) > 5 and not show_all_recs:
                    if st.button(f"Alle {len(recs)} Empfehlungen anzeigen", key="btn_show_all_recs"):
                        st.session_state.show_all_recs = True
                        st.rerun()
            else:
                st.caption("Keine relevanten Empfehlungen f\u00fcr diese Konfiguration gefunden.")

    # ── Chat Tab ──
    with tab_chat:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            chat_key = f"chat_{market}_{role}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = []
                
            rolle_text = "Verkäufer" if role == "seller" else "Käufer"
            verfuegbare_marken = ", ".join(sorted(db_data['brand'].dropna().unique())) if not db_data.empty else "unserer Datenbank"
            
            # 1. Basis-Anweisung & Erklärungs-Kontext für das Modell
            system_anweisung = (
                f"Du bist der exklusive KI-Assistent von AutoValue für den {market}-Automarkt. Du berätst einen {rolle_text}.\n\n"
                f"TRANSPARENZ & METHODIK (Wie du rechnest):\n"
                f"Erkläre dem Nutzer bei Preisberechnungen gerne kurz, wie du auf die Werte kommst: Du nutzt ein fortschrittliches Machine-Learning-Modell (XGBoost), das auf tausenden echten Marktdaten trainiert wurde. Um Preiseinflüsse exakt zu beziffern, wertest du sogenannte SHAP-Werte aus.\n\n"
                f"ALLGEMEINE REGELN:\n"
                f"1. FEHLENDE DATEN (WICHTIG!): Das Tool 'run_ml_prediction' benötigt ZWINGEND Marke, Modell, Alter und Kilometerstand. Wenn der Nutzer einen dieser Werte vergisst (z.B. das Alter nicht nennt), ERFINDE NIEMALS WERTE! Rufe das Tool in diesem Fall NICHT auf, sondern frage den Nutzer freundlich nach der fehlenden Information (z.B. 'Wie alt ist das Fahrzeug?').\n"
                f"2. WERTVERLUST: Wenn nach dem 'Wertverlust' über x Jahre gefragt wird, rufe das Tool 'run_ml_prediction' ZWEIMAL auf (Neuwagen vs. gefragtes Alter/KM) und bilde die Differenz aus den berechneten Preisen.\n"
                f"3. PREIS-FAKTOREN: Nutze die 'alle_preis_einflussfaktoren' aus dem Tool, um exakt zu begründen, warum ein Auto diesen Preis hat.\n"
                f"4. KEINE EXTERNEN QUELLEN: Nutze niemals KBB, Schwacke etc. Verlasse dich nur auf dein Tool.\n"
            )

            # 2. Spezifische Einschränkungen je nach Rolle
            if role == "buyer":
                system_anweisung += (
                    f"5. EINSCHRÄNKUNG FÜR KÄUFER: Weise den Nutzer freundlich darauf hin, dass du KEINE allgemeinen Fahrzeuginformationen, allgemeinen Kaufberatungen oder pauschale Budget-Empfehlungen (z.B. 'Welches Auto für 30.000€?') geben kannst. Deine Aufgabe ist AUSSCHLIESSLICH die datenbasierte Preisermittlung. Der Nutzer muss dir zwingend konkrete Daten (Marke, Modell, Alter, Kilometerstand) nennen, damit du das ML-Modell aufrufen kannst.\n"
                )
            else:
                system_anweisung += (
                    f"5. FOKUS FÜR VERKÄUFER: Hilf dem Nutzer, den optimalen Marktwert für sein spezifisches Fahrzeug zu ermitteln und zeige ihm auf, welche Ausstattungen den Preis treiben oder mindern.\n"
                )

            # 3. Zwingender Disclaimer
            system_anweisung += (
                f"\n6. HAFTUNGSAUSSCHLUSS: Beende JEDE deiner Antworten zwingend mit folgendem Satz in kursiver Schrift: "
                f"'*Hinweis: Diese Angaben sind ohne Gewähr und wurden von einer Künstlichen Intelligenz auf Basis eines Machine-Learning-Modells ermittelt.*'\n"
                f"Antworte auf Deutsch, präzise und professionell."
            )
            
            sys_prompt = {"role": "system", "content": system_anweisung}

            # Definition des Werkzeugs für Groq
            tools = [{
                "type": "function",
                "function": {
                    "name": "run_ml_prediction",
                    "description": "Berechnet den exakten Marktwert und die Preis-Einflussfaktoren über unser XGBoost ML-Modell.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "market": {"type": "string", "enum": ["DE", "US"], "description": "Markt (DE oder US)"},
                            "brand": {"type": "string", "description": "Automarke"},
                            "model_name": {"type": "string", "description": "Modellname"},
                            "car_age": {"type": "number", "description": "Alter in Jahren"},
                            "mileage": {"type": "number", "description": "Kilometerstand"},
                        },
                        "required": ["market", "brand", "model_name", "car_age", "mileage"]
                    }
                }
            }]

            for m in st.session_state[chat_key]:
                avatar_icon = "👤" if m["role"] == "user" else "🤖"
                with st.chat_message(m["role"], avatar=avatar_icon): 
                    st.markdown(m["content"])

            if p := st.chat_input("Konkrete Fahrzeuge bewerten (Marke, Modell, Alter, KM) ..."):
                with st.chat_message("user", avatar="👤"): st.markdown(p)
                st.session_state[chat_key].append({"role": "user", "content": p})

                with st.spinner("AutoValue Assistent analysiert Daten..."):
                    try:
                        history_for_api = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state[chat_key][:-1]]
                        messages = [sys_prompt] + history_for_api + [{"role": "user", "content": p}]
                        
                        # 1. Anfrage an Groq (darf Tools nutzen)
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=messages,
                            tools=tools,
                            tool_choice="auto"
                        )
                        
                        response_msg = response.choices[0].message
                        
                        # 2. Prüfen, ob Groq das Tool aufgerufen hat
                        if response_msg.tool_calls:
                            messages.append(response_msg)
                            
                            for tool_call in response_msg.tool_calls:
                                if tool_call.function.name == "run_ml_prediction":
                                    args = json.loads(tool_call.function.arguments)
                                    tool_result = run_ml_prediction(**args)
                                    
                                    messages.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": "run_ml_prediction",
                                        "content": tool_result
                                    })
                            
                            # 3. Zweiter Aufruf für finale Textantwort
                            final_response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=messages
                            )
                            final_text = final_response.choices[0].message.content
                        else:
                            final_text = response_msg.content

                        with st.chat_message("assistant", avatar="🤖"): st.markdown(final_text)
                        st.session_state[chat_key].append({"role": "assistant", "content": final_text})
                        
                    except Exception as e:
                        st.error(f"Fehler bei der KI-Anfrage: {e}")
        else:
            st.warning("Chat Assistant nicht konfiguriert (GROQ_API_KEY fehlt).")


def _render_de_form_fields(enc_cats, role, show_advanced):
    """Deutsche Markt-Eingabefelder."""
    brand = st.session_state.get("sel_brand", "")
    model_name = st.session_state.get("sel_model", "")
    sel_trim = st.session_state.get("sel_motorleistung", "Alle")
    
    default_ps = 150
    if sel_trim != "Alle" and brand and model_name:
        db_data = get_market_data("DE")
        if not db_data.empty and "power_ps" in db_data.columns and "title" in db_data.columns:
            subset = db_data[(db_data["brand"] == brand) & (db_data["model"] == model_name)]
            import re
            brand_pattern = re.compile(rf'\b{re.escape(str(brand))}\b', re.IGNORECASE)
            
            matching_ps = []
            for _, row in subset.iterrows():
                t = str(row["title"])
                ps = row["power_ps"]
                if pd.isna(ps) or float(ps) <= 0:
                    continue
                t_clean = brand_pattern.sub('', t).strip()
                t_clean = re.sub(r'\s+', ' ', t_clean)
                if t_clean.lower() == sel_trim.lower():
                    matching_ps.append(float(ps))
            
            if matching_ps:
                import numpy as np
                default_ps = int(np.median(matching_ps))

    st.markdown("#### Fahrzeugdaten")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Kilometerstand", 0, 500000, 50000, 5000, key="de_mileage")
        st.number_input("Alter (Jahre)", 0, 40, 3, key="de_age")
        # Check if user already entered a different PS value to avoid overwriting typed input
        current_ps = st.session_state.get("de_power", float(default_ps))
        st.number_input("Leistung (PS)", 30, 1000, int(current_ps), key="de_power")
    with c2:
        st.number_input("Vorbesitzer", 1, 10, 1, key="de_owners")
        trans_opts = enc_cats.get("transmission", ["automatic", "manual"])
        st.selectbox("Getriebe", trans_opts, key="de_trans", format_func=_fmt)
        fuel_opts = enc_cats.get("fuel", ["benzin", "diesel", "elektro", "hybrid"])
        st.selectbox("Kraftstoff", fuel_opts, key="de_fuel", format_func=_fmt)
    with c3:
        st.number_input("Garantie (Monate)", 0, 60, 0, key="de_garantie")

    st.markdown("#### Zustand")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.checkbox("TÜV neu", key="de_tuv")
    with d2:
        st.checkbox("Unfallfrei", key="de_unfall")
    with d3:
        st.checkbox("Mängel vorhanden", key="de_mangel")
    with d4:
        st.checkbox("Scheckheft gepflegt", key="de_scheckh")

    if show_advanced:
        st.markdown("#### Ausstattung")
        _render_de_equipment()


def _render_de_equipment():
    """DE Ausstattungs-Checkboxen."""
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.checkbox("Panoramadach", key="de_pano")
        st.checkbox("AMG Line", key="de_amg")
    with e2:
        st.checkbox("Distronic", key="de_distronic")
        st.checkbox("Multibeam LED", key="de_multibeam")
    with e3:
        st.checkbox("4-Zonen Klima", key="de_klima4")
        st.checkbox("2-Zonen Klima", key="de_klima2")
    with e4:
        st.checkbox("Burmester 3D", key="de_burm3d")
        st.checkbox("Burmester Std.", key="de_burmstd")
    with e5:
        st.checkbox("8-fach Bereifung", key="de_reif8")
        st.checkbox("Allwetterreifen", key="de_reifall")


def _render_us_form_fields(enc_cats, db_data, brand, model_name, role, show_advanced):
    """US-Markt Eingabefelder."""
    st.markdown("#### Fahrzeugdaten")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Kilometerstand", 0, 500000, 30000, 5000, key="us_mileage")
        st.number_input("Alter (Jahre)", 0, 40, 3, key="us_age")
        st.number_input("Unfälle", 0, 10, 0, key="us_accidents")
    with c2:
        st.number_input("Vorbesitzer", 1, 10, 1, key="us_owners")
        st.number_input("Zylinder", 0, 16, 6, key="us_cyl")
        st.number_input("Türen", 2, 6, 4, key="us_doors")
    with c3:
        st.number_input("Sitze", 1, 12, 5, key="us_seats")
        dt_opts = enc_cats.get("drivetrain", ["unknown"])
        st.selectbox("Antrieb", dt_opts, key="us_drive", format_func=_fmt)
        fuel_opts = enc_cats.get("fuel", ["gasoline", "diesel", "electric", "hybrid"])
        st.selectbox("Kraftstoff", fuel_opts, key="us_fuel", format_func=_fmt)

    st.markdown("#### Klassifizierung")
    g1, g2 = st.columns(2)
    with g1:
        trans_opts = enc_cats.get("transmission", ["automatic", "manual"])
        st.selectbox("Getriebe", trans_opts, key="us_trans", format_func=_fmt)
        body_opts = enc_cats.get("body_style", ["sedan", "suv", "truck", "coupe"])
        st.selectbox("Karosserie", body_opts, key="us_body", format_func=_fmt)
    with g2:
        use_opts = enc_cats.get("usage_type", ["personal", "fleet"])
        st.selectbox("Nutzungsart", use_opts, key="us_usage", format_func=_fmt)

    if show_advanced:
        st.markdown("#### Erweiterte Optionen")
        _render_us_advanced(enc_cats, db_data, model_name)


def _render_us_advanced(enc_cats, db_data=None, model_name=None):
    """US erweiterte Klassifizierungsfelder — gefiltert nach Modell."""
    # Filter options by selected model from db_data
    model_data = pd.DataFrame()
    if db_data is not None and not db_data.empty and model_name:
        model_data = db_data[db_data['model'] == model_name]
        
        fuel = st.session_state.get("us_fuel")
        if fuel and "fuel" in model_data.columns:
            model_data = model_data[model_data["fuel"].astype(str).str.lower() == str(fuel).lower()]
            
        cyl = st.session_state.get("us_cyl")
        if cyl is not None and "cylinders" in model_data.columns:
            model_data = model_data[pd.to_numeric(model_data["cylinders"], errors="coerce") == cyl]

    f1, f2, f3 = st.columns(3)
    with f1:
        if not model_data.empty and 'trim' in model_data.columns:
            trim_opts = sorted(model_data['trim'].dropna().astype(str).unique())
            trim_opts = [t for t in trim_opts if t and t != 'nan']
        else:
            trim_opts = enc_cats.get("trim", ["unknown"])
        if not trim_opts:
            trim_opts = ["unknown"]
        st.selectbox("Ausstattungslinie", trim_opts, key="us_trim_adv", format_func=_fmt)
    with f2:
        if not model_data.empty and 'engine' in model_data.columns:
            eng_opts = sorted(model_data['engine'].dropna().astype(str).unique())
            eng_opts = [e for e in eng_opts if e and e != 'nan']
        else:
            eng_opts = enc_cats.get("engine", ["unknown"])
        if not eng_opts:
            eng_opts = ["unknown"]
        st.selectbox("Motor", eng_opts, key="us_engine", format_func=_fmt)
    with f3:
        if not model_data.empty and 'exterior_color' in model_data.columns:
            ext_opts = sorted(model_data['exterior_color'].dropna().astype(str).unique())
            ext_opts = [e for e in ext_opts if e and e != 'nan']
        else:
            ext_opts = enc_cats.get("exterior_color", ["unknown"])
        if not ext_opts:
            ext_opts = ["unknown"]
        st.selectbox("Au\u00dfenfarbe", ext_opts, key="us_ext_color", format_func=_fmt)
    f4, f5 = st.columns(2)
    with f4:
        if not model_data.empty and 'interior_color' in model_data.columns:
            int_opts = sorted(model_data['interior_color'].dropna().astype(str).unique())
            int_opts = [i for i in int_opts if i and i != 'nan']
        else:
            int_opts = enc_cats.get("interior_color", ["unknown"])
        if not int_opts:
            int_opts = ["unknown"]
        st.selectbox("Innenfarbe", int_opts, key="us_int_color", format_func=_fmt)
    with f5:
        pass


def _b(key):
    """Read a checkbox boolean from session_state and return 1.0/0.0."""
    return 1.0 if st.session_state.get(key, False) else 0.0


def _collect_inputs(market, brand, model_name):
    """Collect all form widget values from session state into a model-input dict."""
    if market == "DE":
        return {
            "brand": brand, "model": model_name,
            "mileage": float(st.session_state.de_mileage),
            "car_age": float(st.session_state.de_age),
            "power_ps": float(st.session_state.de_power),
            "owners": float(st.session_state.de_owners),
            "transmission": st.session_state.de_trans,
            "fuel": st.session_state.de_fuel,
            "garantie_monate": float(st.session_state.de_garantie),
            "tuv_neu": _b("de_tuv"),
            "unfallfrei": _b("de_unfall"),
            "mangel_vorhanden": _b("de_mangel"),
            "scheckheft_gepflegt": _b("de_scheckh"),
            "ausstattung_pano": _b("de_pano"),
            "ausstattung_amg_line": _b("de_amg"),
            "ausstattung_distronic": _b("de_distronic"),
            "ausstattung_multibeam": _b("de_multibeam"),
            "ausstattung_klima_4_zonen": _b("de_klima4"),
            "ausstattung_klima_2_zonen": _b("de_klima2"),
            "ausstattung_burmester_3d": _b("de_burm3d"),
            "ausstattung_burmester_standard": _b("de_burmstd"),
            "bereifung_8_fach": _b("de_reif8"),
            "bereifung_allwetter": _b("de_reifall"),
        }
    else:
        acc = int(st.session_state.us_accidents)
        own = int(st.session_state.us_owners)
        usage = st.session_state.us_usage
        return {
            "brand": brand, "model": model_name,
            "mileage": float(st.session_state.us_mileage),
            "car_age": float(st.session_state.us_age),
            "accident_count": float(acc),
            "owner_count": float(own),
            "cylinders": float(st.session_state.us_cyl),
            "doors": float(st.session_state.us_doors),
            "seats": float(st.session_state.us_seats),
            "trim": st.session_state.get("us_trim_adv", st.session_state.get("us_trim", "unknown")),
            "drivetrain": st.session_state.us_drive,
            "fuel": st.session_state.us_fuel,
            "transmission": st.session_state.us_trans,
            "body_style": st.session_state.us_body,
            "engine": st.session_state.get("us_engine", "unknown"),
            "exterior_color": st.session_state.get("us_ext_color", "unknown"),
            "interior_color": st.session_state.get("us_int_color", "unknown"),
            "usage_type": usage,
            # Derived binary features
            "one_owner": 1.0 if own == 1 else 0.0,
            "has_accidents": 1.0 if acc > 0 else 0.0,
            "is_used": 1.0,
            "is_cpo": 0.0,
            "is_online": 0.0,
            "is_wholesale": 0.0,
            "personal_use": 1.0 if "personal" in str(usage).lower() else 0.0,
        }


# ==========================================
# 8. MAIN ROUTER
# ==========================================
view_header()
if st.session_state.page == "home":
    view_home()
else:
    view_app()