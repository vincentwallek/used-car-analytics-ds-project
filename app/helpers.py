"""AutoValue helper functions: labels, recommendations, utilities."""
import base64
import pandas as pd
import numpy as np


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


