import pandas as pd
import requests
from sqlalchemy import create_engine


# =========================
# DB CONNECTION
# =========================
engine = create_engine(
    "postgresql://postgres.tnwwljxazuaefezooesq:DzIKYJfQ1h0RHuzm@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
)

# =========================
# LOAD DATA (nur Text!)
# =========================
query = """
SELECT 
    l.id,
    d.beschreibung
FROM listings l
JOIN listing_de d ON l.id = d.listing_id
WHERE d.beschreibung IS NOT NULL
    AND l.id BETWEEN 100 AND 817
"""

df = pd.read_sql(query, engine)
print(f"Geladene Zeilen: {len(df)}")

# =========================
# LLM FUNCTION
# =========================
def extract_features(text):

    prompt = f""""Du bist ein Senior Data Engineer für Fahrzeugdaten.\n\n"
            "TEIL 1: HISTORIE & ZUSTAND (Sehr wichtig!)\n"
            "Analysiere den Text nach harten Fakten:\n"
            "- TÜV: 'HU/AU neu', 'TÜV wird bei Übergabe neu gemacht', 'TÜV bis 10/25' (wenn nah dran) -> true\n"
            "- Scheckheft: 'lückenlos bei Mercedes', 'Scheckheft', 'Service A neu' -> true\n"
            "- Bereifung: '8-fach bereift', 'Winter und Sommer' -> '8-fach'. 'Allwetter', 'Ganzjahresreifen' -> 'Allwetterreifen'.\n"
            "- Garantie: 'Junge Sterne 24 Monate', 'Garantie 1 Jahr' -> Extrahiere nur die Anzahl der Monate (24 oder 12).\n"
            "- Unfallfrei: Achte auf versteckte Hinweise! 'Unfallfrei' -> true. 'Nachlackierung an der Tür wegen Parkrempler' -> false! 'Unfallwagen' -> false.\n"
            "- Mängel: Sammle alles Negative ('Steinschlag Frontscheibe', 'Felge zerkratzt', 'Delle'). Sei präzise. Wenn nichts steht: 'Keine'.\n\n"
            "TEIL 2: AUSSTATTUNG (Mappe strikt auf die Enum-Werte)\n"
            "- DISTRONIC, ACC -> Abstandsregeltempomat (DISTRONIC)\n"
            "- MULTIBEAM, ILS -> Matrix-LED / MULTIBEAM\n"
            "- 4-Zonen, THERMOTRONIC, Klima hinten -> Klimaautomatik (4-Zonen / THERMOTRONIC)\n"
            "- 2-Zonen, THERMATIC -> Klimaautomatik (2-Zonen / THERMATIC)\n"
            "- Burmester 3D -> Burmester High-End 3D-Soundsystem\n"
            "- Burmester (Standard) -> Burmester Surround-Soundsystem\n"
            "- AMG Line -> AMG Line (Achte darauf, ob auch Night-Paket erwähnt wird!)\n"
            "- Pano -> Panorama-Schiebedach\n\n"
            "Antworte strikt im JSON-Format gemäß Schema, ohne Erklärungen, ohne Markdown-Backticks.
Text:
{text}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


# =========================
# APPLY (erstmal klein!)
# =========================
results = []

for i, row in df.iterrows():   # ❗ erst testen!
    print(f"Processing {i}")

    try:
        output = extract_features(row["beschreibung"])

        results.append({
            "listing_id": row["id"],
            "raw_output": output
        })

    except Exception as e:
        print("Error:", e)


result_df = pd.DataFrame(results)
result_df.to_csv("llm_results2.csv", index=False)

import json

def parse_output(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except:
        return None

print("Done!")
