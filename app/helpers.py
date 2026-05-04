"""AutoValue helper functions: labels, recommendations, utilities."""
import base64
import pandas as pd
import numpy as np
import streamlit as st
import shap
import json

def _fmt_h(s):
    """Capitalize for display in recommendation text."""
    if not s:
        return s
    parts = s.split("-")
    return "-".join(p.upper() if (p.isalpha() and len(p) <= 3) else p.capitalize() for p in parts)

# ── Feature label mappings ─────────────────────────────────────────
DE_BINARY_LABELS = {
    "tuv_neu": "TÜV neu",
    "scheckheft_gepflegt": "Scheckheft gepflegt",
    "bereifung_8_fach": "8-fach Bereifung",
    "bereifung_allwetter": "Allwetterreifen",
    "unfallfrei": "Unfallfrei",
    "mangel_vorhanden": "Mängel vorhanden",
    "ausstattung_distronic": "Distronic",
    "ausstattung_multibeam": "Multibeam LED",
    "ausstattung_klima_4_zonen": "4-Zonen Klima",
    "ausstattung_klima_2_zonen": "2-Zonen Klima",
    "ausstattung_burmester_3d": "Burmester 3D",
    "ausstattung_burmester_standard": "Burmester Standard",
    "ausstattung_amg_line": "AMG Line",
    "ausstattung_pano": "Panoramadach",
}

DE_CONDITION_KEYS = ["tuv_neu", "unfallfrei", "mangel_vorhanden", "scheckheft_gepflegt"]
DE_EQUIP_KEYS = [
    "bereifung_8_fach", "bereifung_allwetter",
    "ausstattung_distronic", "ausstattung_multibeam",
    "ausstattung_klima_4_zonen", "ausstattung_klima_2_zonen",
    "ausstattung_burmester_3d", "ausstattung_burmester_standard",
    "ausstattung_amg_line", "ausstattung_pano",
]


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_encoder_categories(trained_models, market):
    """Extract valid categories for each categorical feature."""
    if not trained_models:
        return {}
    m_code = "de" if market == "DE" else "us"
    encoder = trained_models[f"{m_code}_encoder"]
    return {
        name: sorted(cats.tolist())
        for name, cats in zip(encoder.feature_names_in_, encoder.categories_)
    }


def predict_price_fast(trained_models, market, input_data):
    """Predict price without SHAP (fast, for recommendations)."""
    if not trained_models:
        return 0.0
    m_code = "de" if market == "DE" else "us"
    model = trained_models[f"{m_code}_model"]
    encoder = trained_models[f"{m_code}_encoder"]
    num_cols = trained_models[f"{m_code}_num_cols"]
    cat_cols = (
        ["brand", "model", "transmission", "fuel"] if market == "DE"
        else ["brand", "model", "trim", "drivetrain", "fuel", "transmission",
              "body_style", "engine", "exterior_color", "interior_color", "usage_type"]
    )
    df = pd.DataFrame([input_data])
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            
    enc = encoder.transform(df[cat_cols])
    df_enc = pd.DataFrame(enc, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    X = pd.concat([df[num_cols], df_enc], axis=1)
    return float(model.predict(X)[0])


def generate_recommendations(trained_models, market, input_data, base_price, db_data, currency_symbol, role="buyer"):
    """Generate role-aware recommendations.
    Buyer: savings (remove features, lower mileage, alternatives).
    Seller: upgrades (add equipment only — no condition features).
    """
    recs = []
    if not trained_models:
        return recs

    if market == "DE":
        # Seller: only equipment keys (adding 'Unfallfrei' makes no sense)
        # Buyer: all keys (condition + equipment)
        if role == "seller":
            toggle_keys = DE_EQUIP_KEYS
        else:
            toggle_keys = DE_CONDITION_KEYS + DE_EQUIP_KEYS

        for key in toggle_keys:
            if key not in input_data:
                continue
            alt = dict(input_data)
            current_val = alt[key]
            alt[key] = 0.0 if current_val == 1.0 else 1.0
            try:
                alt_price = predict_price_fast(trained_models, market, alt)
            except:
                continue
            label = DE_BINARY_LABELS.get(key, key)

            if role == "buyer" and current_val == 1.0 and base_price > alt_price + 100:
                saving = base_price - alt_price
                recs.append({
                    "text": f"Wenn du auf '{label}' verzichtest, sparst du ca. {saving:,.0f} {currency_symbol}.",
                    "saving": saving, "type": "saving"
                })
            elif role == "seller" and current_val == 0.0 and alt_price > base_price + 100:
                gain = alt_price - base_price
                recs.append({
                    "text": f"Wenn du '{label}' anbietest, nimmt der Preis um ca. {gain:,.0f} {currency_symbol} zu.",
                    "saving": gain, "type": "upgrade"
                })

        # Mileage-based recommendations for buyers
        if role == "buyer":
            current_mileage = input_data.get("mileage", 0)
            if current_mileage > 20000:
                for km_less in [10000, 20000, 50000]:
                    if current_mileage - km_less < 5000:
                        continue
                    alt = dict(input_data)
                    alt["mileage"] = current_mileage - km_less
                    try:
                        alt_price = predict_price_fast(trained_models, market, alt)
                        diff = alt_price - base_price
                        if diff > 200:
                            recs.append({
                                "text": f"Fahrzeuge mit {km_less:,.0f} km weniger Laufleistung kosten ca. {diff:,.0f} {currency_symbol} mehr — achte auf Angebote mit weniger Kilometern!",
                                "saving": diff, "type": "mileage_tip"
                            })
                            break  # Only show best mileage suggestion
                    except:
                        continue

        # Alternative models from same brand (buyer only)
        if role == "buyer" and not db_data.empty:
            brand = input_data.get("brand", "")
            current_model = input_data.get("model", "")
            alt_models = [m for m in db_data[db_data["brand"] == brand]["model"].unique()
                          if m != current_model][:5]
            for alt_m in alt_models:
                alt = dict(input_data)
                alt["model"] = alt_m
                try:
                    alt_price = predict_price_fast(trained_models, market, alt)
                    diff = base_price - alt_price
                    if diff > 500:
                        recs.append({
                            "text": f"Ein '{_fmt_h(alt_m)}' mit gleicher Ausstattung würde ca. {diff:,.0f} {currency_symbol} weniger kosten.",
                            "saving": diff, "type": "alternative"
                        })
                except:
                    pass

    else:  # US
        # Alternative models (buyer only)
        if role == "buyer" and not db_data.empty:
            brand = input_data.get("brand", "")
            current_model = input_data.get("model", "")
            alt_models = [m for m in db_data[db_data["brand"] == brand]["model"].unique()
                          if m != current_model][:5]
            for alt_m in alt_models:
                alt = dict(input_data)
                alt["model"] = alt_m
                try:
                    alt_price = predict_price_fast(trained_models, market, alt)
                    diff = base_price - alt_price
                    if diff > 500:
                        recs.append({
                            "text": f"Ein '{_fmt_h(alt_m)}' mit gleicher Ausstattung würde ca. ${diff:,.0f} weniger kosten.",
                            "saving": diff, "type": "alternative"
                        })
                except:
                    pass

        # Mileage-based recommendations for buyers
        if role == "buyer":
            current_mileage = input_data.get("mileage", 0)
            if current_mileage > 20000:
                for miles_less in [10000, 20000, 50000]:
                    if current_mileage - miles_less < 5000:
                        continue
                    alt = dict(input_data)
                    alt["mileage"] = current_mileage - miles_less
                    try:
                        alt_price = predict_price_fast(trained_models, market, alt)
                        diff = alt_price - base_price
                        if diff > 200:
                            recs.append({
                                "text": f"Fahrzeuge mit {miles_less:,.0f} Meilen weniger kosten ca. ${diff:,.0f} mehr — achte auf niedrigere Laufleistung!",
                                "saving": diff, "type": "mileage_tip"
                            })
                            break
                    except:
                        continue

        # Cylinder alternatives
        if input_data.get("cylinders", 0) > 4:
            alt = dict(input_data)
            alt["cylinders"] = max(4.0, alt["cylinders"] - 2.0)
            try:
                alt_price = predict_price_fast(trained_models, market, alt)
                diff = base_price - alt_price
                if role == "buyer" and diff > 300:
                    recs.append({
                        "text": f"Ein {int(alt['cylinders'])}-Zylinder-Motor würde ca. ${diff:,.0f} sparen.",
                        "saving": diff, "type": "saving"
                    })
                elif role == "seller" and alt_price > base_price + 300:
                    gain = alt_price - base_price
                    recs.append({
                        "text": f"Mit einem {int(input_data['cylinders'] + 2)}-Zylinder-Motor könnte der Preis um ca. ${gain:,.0f} steigen.",
                        "saving": gain, "type": "upgrade"
                    })
            except:
                pass

    recs.sort(key=lambda r: r["saving"], reverse=True)
    return recs  # Return ALL — UI handles display limit


@st.cache_data(ttl=600)
def get_market_data(market_code, _supabase):
    """Load listings from Supabase with proper table joins.
    DE: listings + listing_de (title) + listing_features (equipment).
    US: listings + listing_us (trim, engine, etc.).
    All linked via listings.id = *.listing_id.
    """
    empty = pd.DataFrame(columns=["brand", "model", "title", "price", "mileage", "location", "url"])
    try:
        # 1. Load base listings filtered by market
        res = _supabase.table("listings").select("*").eq("market", market_code.upper()).execute()
        df = pd.DataFrame(res.data)
        if df.empty:
            return empty
        df.columns = [c.lower() for c in df.columns]
        df["brand"] = df["brand"].astype(str).str.lower().str.strip()
        df["model"] = df["model"].astype(str).str.lower().str.strip()

        if market_code.upper() == "DE":
            # 2a. Join listing_de (title, transmission, fuel, etc.)
            try:
                res_de = _supabase.table("listing_de").select("*").execute()
                df_de = pd.DataFrame(res_de.data)
                if not df_de.empty:
                    df_de.columns = [c.lower() for c in df_de.columns]
                    df = df.merge(df_de, left_on="id", right_on="listing_id",
                                  how="left", suffixes=("", "_de"))
            except:
                pass
            # 2b. Join listing_features (equipment booleans)
            try:
                res_feat = _supabase.table("listing_features").select("*").execute()
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
                res_us = _supabase.table("listing_us").select("*").execute()
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


def predict_price(trained_models, market, input_data):
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


def run_ml_prediction(trained_models, market, brand, model_name, car_age, mileage, transmission="automatic", fuel="benzin", power_ps=150, cylinders=6):
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

    price, s_vals = predict_price(trained_models, market, input_vals)
    
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


def _fmt(s):
    """Capitalize brand/model for display. E.g. 'mercedes-benz' -> 'Mercedes-Benz', 'nx' -> 'NX'."""
    if not s:
        return s
    parts = s.split("-")
    return "-".join(p.upper() if (p.isalpha() and len(p) <= 3) else p.capitalize() for p in parts)


_SHAP_DIRECT = {
    "mileage": "Kilometerstand", "car_age": "Fahrzeugalter",
    "power_ps": "Leistung (PS)", "owners": "Vorbesitzer",
    "garantie_monate": "Garantie (Monate)", "cylinders": "Zylinder",
    "doors": "Türen", "seats": "Sitze", "accident_count": "Unfälle",
    "owner_count": "Vorbesitzer", "one_owner": "Erstbesitzer",
    "has_accidents": "Hat Unfälle", "is_used": "Gebraucht",
    "is_cpo": "Zertifiziert", "is_online": "Online-Kauf",
    "is_wholesale": "Großhandel", "personal_use": "Privatnutzung",
    "tuv_neu": "TÜV neu", "unfallfrei": "Unfallfrei",
    "mangel_vorhanden": "Mängel", "scheckheft_gepflegt": "Scheckheft",
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
    "exterior_color_": "Außenfarbe", "interior_color_": "Innenfarbe",
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

