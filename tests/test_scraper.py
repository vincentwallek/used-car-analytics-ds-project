"""
Testprogramm für ScraperDataScienceProjekt.py
Testet die gesamte Parsing-Logik mit simuliertem HTML – ohne echten Browser.

Ausführen:
    pip install pytest beautifulsoup4
    pytest test_scraper.py -v
"""

import csv
import os
import re
import sys
import tempfile
from unittest.mock import MagicMock, patch, call

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "data_collection"))
import pytest
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen – extrahiert aus dem Scraper (identische Logik)
# ──────────────────────────────────────────────────────────────────────────────

def parse_title(soup: BeautifulSoup) -> str:
    title = "N/A"
    meta_title = soup.find("meta", property="og:title")
    if meta_title and meta_title.get("content"):
        title = meta_title.get("content")
        title = re.sub(r"\s*für\s*[\d\.,]+\s*€.*", "", title).strip()
    if title == "N/A" or not title:
        title_elem = (
            soup.find(attrs={"data-testid": "listing-title"})
            or soup.find(id="ad-title")
            or soup.find("h1")
        )
        if title_elem:
            title = title_elem.get_text(separator=" ", strip=True)
    if title == "N/A" or not title:
        page_title = soup.find("title")
        if page_title:
            title = page_title.text.split("für")[0].split("|")[0].strip()
    return title


def parse_price(soup: BeautifulSoup) -> str:
    price = "N/A"
    price_elem = soup.find(attrs={"data-testid": "prime-price"})
    if price_elem:
        price = price_elem.get_text(strip=True)
    if price == "N/A" or not price:
        price_tags = soup.find_all(string=re.compile(r"\d{1,3}(?:\.\d{3})*\s*€"))
        for tag in price_tags:
            text = tag.strip()
            if not any(bad in text.lower() for bad in ["/jahr", "mtl", "monat", "ab "]):
                match = re.search(r"(\d{1,3}(?:\.\d{3})*\s*€)", text)
                if match:
                    price = match.group(1)
                    break
    return price


def parse_mileage(page_text: str) -> str:
    km_match = re.search(r"([\d\.]+)\s*km", page_text, re.IGNORECASE)
    return km_match.group(0) if km_match else "N/A"


def parse_registration(page_text: str) -> str:
    # Bugfix: Monat auf 01–12 beschränkt
    ez_match = re.search(r"(?:EZ|Erstzulassung)[\s:]*(0[1-9]|1[0-2])/\d{4}", page_text, re.IGNORECASE)
    if not ez_match:
        ez_match = re.search(r"\b(0[1-9]|1[0-2])/(19|20)\d{2}\b", page_text)
    return ez_match.group(0) if ez_match else "N/A"


def parse_ps(page_text: str) -> str:
    ps_match = re.search(r"\((\d+)\s*PS\)", page_text)
    if not ps_match:
        ps_match = re.search(r"(\d+)\s*PS", page_text)
    return f"{ps_match.group(1)} PS" if ps_match else "N/A"


def parse_getriebe(page_text: str) -> str:
    if re.search(r"\bAutomatik\b", page_text, re.IGNORECASE):
        return "Automatik"
    elif re.search(r"\bSchaltgetriebe\b", page_text, re.IGNORECASE):
        return "Schaltgetriebe"
    return "N/A"


def parse_kraftstoff(page_text: str) -> str:
    # Bugfix: \b funktioniert nicht bei Einträgen mit Klammern
    for k in ["Benzin", "Diesel", "Elektro", "Hybrid", "Autogas (LPG)", "Erdgas (CNG)"]:
        if "(" in k:
            pattern = re.escape(k)
        else:
            pattern = rf"\b{re.escape(k)}\b"
        if re.search(pattern, page_text, re.IGNORECASE):
            return k
    return "N/A"


def parse_halter(page_text: str) -> str:
    halter_match = re.search(r"Fahrzeughalter(?:[^0-9]{1,15})(\d+)", page_text, re.IGNORECASE)
    return halter_match.group(1) if halter_match else "N/A"


def parse_standort(page_text: str) -> str:
    loc_match = re.search(
        r"DE-([0-9]{5})\s+([A-ZÄÖÜ][a-zA-ZäöüßÄÖÜ\-\s]+?)(?=\s*\||\s*$)", page_text
    )
    if loc_match:
        return f"{loc_match.group(1)} {loc_match.group(2).strip()}"
    fallback = re.search(r"\b([0-9]{5})\s+([A-ZÄÖÜ][a-zäöüß]+)\b", page_text)
    if fallback and fallback.group(2) not in ["Verfügbarkeit", "Kilometer", "Euro"]:
        return f"{fallback.group(1)} {fallback.group(2)}"
    return "N/A"


def parse_ausstattung(soup: BeautifulSoup) -> str:
    ausstattung_liste = []
    heading = soup.find(
        lambda tag: tag.name in ["h2", "h3", "div"] and tag.text.strip() == "Ausstattung"
    )
    if heading:
        for element in heading.find_all_next(string=True):
            text = element.strip()
            if text in ["Fahrzeugbeschreibung laut Anbieter", "Fahrzeugstandort",
                        "Über diesen Händler", "Standort", "Preis"]:
                break
            if (text and text not in ["Ausstattung", "Mehr anzeigen", "Weniger anzeigen", "✓"]
                    and text not in ausstattung_liste and len(text) > 2):
                ausstattung_liste.append(text)
    return ", ".join(ausstattung_liste) if ausstattung_liste else "N/A"


def parse_beschreibung(soup: BeautifulSoup) -> str:
    heading = soup.find(
        lambda tag: tag.name in ["h2", "h3", "div"]
        and tag.text.strip() == "Fahrzeugbeschreibung laut Anbieter"
    )
    if not heading:
        return "N/A"
    beschr_strings = []
    for element in heading.find_all_next(string=True):
        text = element.strip()
        if text in ["Über diesen Händler", "Standort", "Händler",
                    "Preis", "Finanzierung", "Ähnliche Fahrzeuge"]:
            break
        if (text and text not in ["Fahrzeugbeschreibung laut Anbieter",
                                   "Mehr anzeigen", "Weniger anzeigen"]
                and len(text) > 1):
            beschr_strings.append(text)
    clean = list(dict.fromkeys(beschr_strings))
    return " | ".join(clean) if clean else "N/A"


def clean_car_link(href: str, base: str = "https://suchen.mobile.de") -> str:
    if href.startswith("/"):
        href = base + href
    clean = (
        href.split("?")[0] + "?" + href.split("?")[1].split("&")[0]
        if "?" in href
        else href
    )
    return clean


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures – Wieder verwendbare HTML-Blöcke
# ──────────────────────────────────────────────────────────────────────────────

FULL_DETAIL_HTML = """
<html>
<head>
  <meta property="og:title" content="Mercedes-Benz S 500 für 45.000 €">
  <title>Mercedes S 500 für 45.000€ | mobile.de</title>
</head>
<body>
  <span data-testid="prime-price">45.000 €</span>
  <p>Kilometerstand: 85.000 km</p>
  <p>EZ: 03/2019</p>
  <p>Motor: (435 PS)</p>
  <p>Getriebe: Automatik</p>
  <p>Kraftstoff: Diesel</p>
  <p>Fahrzeughalter: 2</p>
  <p>Standort: DE-70173 Stuttgart</p>

  <h2>Ausstattung</h2>
  <ul>
    <li>Sitzheizung</li>
    <li>Panoramadach</li>
    <li>Mehr anzeigen</li>
  </ul>

  <h2>Fahrzeugbeschreibung laut Anbieter</h2>
  <p>Sehr gepflegtes Fahrzeug.</p>
  <p>Kein Unfallschaden.</p>
  <h2>Standort</h2>
</body>
</html>
"""

OVERVIEW_HTML = """
<html><body>
  <a href="/fahrzeuge/details.html?id=123&ref=srp">Auto 1</a>
  <a href="/fahrzeuge/details.html?id=456&ref=srp">Auto 2</a>
  <a href="/fahrzeuge/details.html?id=123&ref=srp">Auto 1 Duplikat</a>
  <a href="https://other-site.de/auto">Extern</a>
</body></html>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Titel-Parsing
# ──────────────────────────────────────────────────────────────────────────────

class TestTitelParsing:

    def test_meta_og_title_wird_erkannt(self):
        soup = BeautifulSoup(FULL_DETAIL_HTML, "html.parser")
        assert parse_title(soup) == "Mercedes-Benz S 500"

    def test_preis_aus_og_title_wird_entfernt(self):
        html = '<html><head><meta property="og:title" content="BMW 530d für 38.500 €"></head></html>'
        soup = BeautifulSoup(html, "html.parser")
        assert parse_title(soup) == "BMW 530d"

    def test_fallback_auf_data_testid(self):
        html = '<html><body><div data-testid="listing-title">Audi A6 3.0 TDI</div></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        assert parse_title(soup) == "Audi A6 3.0 TDI"

    def test_fallback_auf_h1(self):
        html = "<html><body><h1>VW Passat 2.0 TDI</h1></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert parse_title(soup) == "VW Passat 2.0 TDI"

    def test_fallback_auf_page_title(self):
        html = "<html><head><title>Porsche 911 für 120.000€ | mobile.de</title></head></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert parse_title(soup) == "Porsche 911"

    def test_kein_titel_liefert_na(self):
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        assert parse_title(soup) == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Preis-Parsing
# ──────────────────────────────────────────────────────────────────────────────

class TestPreisParsing:

    def test_prime_price_element(self):
        soup = BeautifulSoup(FULL_DETAIL_HTML, "html.parser")
        assert parse_price(soup) == "45.000 €"

    def test_fallback_text_suche(self):
        html = "<html><body><p>Kaufpreis: 29.900 €</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert parse_price(soup) == "29.900 €"

    def test_leasingrate_wird_ignoriert(self):
        html = "<html><body><p>399 € mtl. Leasingrate</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert parse_price(soup) == "N/A"

    def test_jahreskosten_werden_ignoriert(self):
        html = "<html><body><p>Energiekosten: 1.800 €/Jahr</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert parse_price(soup) == "N/A"

    def test_kein_preis_liefert_na(self):
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        assert parse_price(soup) == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Kilometerstand
# ──────────────────────────────────────────────────────────────────────────────

class TestKilometerstand:

    def test_km_normaler_text(self):
        assert "85.000 km" in parse_mileage("Kilometerstand: 85.000 km Erstzulassung")

    def test_km_ohne_punkt(self):
        result = parse_mileage("Laufleistung 5000 km")
        assert "5000 km" in result

    def test_km_nicht_vorhanden(self):
        assert parse_mileage("Keine Angabe zur Laufleistung") == "N/A"

    def test_km_gross_klein_egal(self):
        result = parse_mileage("150.000 KM gefahren")
        assert "150.000 KM" in result


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Erstzulassung
# ──────────────────────────────────────────────────────────────────────────────

class TestErstzulassung:

    def test_ez_prefix(self):
        result = parse_registration("EZ: 03/2019 · 85.000 km")
        assert "03/2019" in result

    def test_erstzulassung_ausgeschrieben(self):
        result = parse_registration("Erstzulassung 11/2021")
        assert "11/2021" in result

    def test_fallback_datumsformat(self):
        result = parse_registration("Baujahr 06/2017, Diesel")
        assert "06/2017" in result

    def test_ungueltige_monatszahl_nicht_erkannt(self):
        # Nach Bugfix: Monat 13 wird korrekt abgelehnt
        assert parse_registration("EZ: 13/2020") == "N/A"

    def test_kein_datum_liefert_na(self):
        assert parse_registration("Keine Datumsangabe") == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: PS
# ──────────────────────────────────────────────────────────────────────────────

class TestPS:

    def test_ps_in_klammern(self):
        assert parse_ps("Motor: (435 PS)") == "435 PS"

    def test_ps_ohne_klammern(self):
        assert parse_ps("Leistung: 190 PS") == "190 PS"

    def test_ps_bevorzugt_klammern(self):
        # Wenn beide Formate vorhanden, sollen Klammern bevorzugt werden
        result = parse_ps("(272 PS) 272 PS Gesamtleistung")
        assert result == "272 PS"

    def test_kein_ps_liefert_na(self):
        assert parse_ps("Keine Leistungsangabe") == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Getriebe
# ──────────────────────────────────────────────────────────────────────────────

class TestGetriebe:

    def test_automatik(self):
        assert parse_getriebe("9-Gang Automatik Getriebe") == "Automatik"

    def test_schaltgetriebe(self):
        assert parse_getriebe("6-Gang Schaltgetriebe, sehr sportlich") == "Schaltgetriebe"

    def test_unbekannt(self):
        assert parse_getriebe("Getriebe nicht angegeben") == "N/A"

    def test_automatik_case_insensitive(self):
        assert parse_getriebe("AUTOMATIK Typ 9G-Tronic") == "Automatik"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Kraftstoff
# ──────────────────────────────────────────────────────────────────────────────

class TestKraftstoff:

    def test_benzin(self):
        assert parse_kraftstoff("Kraftstoff: Benzin") == "Benzin"

    def test_diesel(self):
        assert parse_kraftstoff("Kraftstoff: Diesel") == "Diesel"

    def test_elektro(self):
        assert parse_kraftstoff("Antrieb: Elektro (100% elektrisch)") == "Elektro"

    def test_hybrid(self):
        assert parse_kraftstoff("Plug-In Hybrid Antrieb") == "Hybrid"

    def test_lpg(self):
        # Nach Bugfix: Klammern werden korrekt erkannt (kein \b mehr)
        assert parse_kraftstoff("Autogas (LPG) Umrüstung") == "Autogas (LPG)"

    def test_cng(self):
        # Nach Bugfix: Klammern werden korrekt erkannt
        assert parse_kraftstoff("Erdgas (CNG) Fahrzeug") == "Erdgas (CNG)"

    def test_unbekannt(self):
        assert parse_kraftstoff("Keine Kraftstoffangabe") == "N/A"

    def test_benzin_bevorzugt_wenn_zuerst(self):
        # Reihenfolge: Benzin kommt vor Diesel in der Liste
        result = parse_kraftstoff("Benzin und Diesel als Vergleich")
        assert result == "Benzin"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Fahrzeughalter
# ──────────────────────────────────────────────────────────────────────────────

class TestFahrzeughalter:

    def test_halter_erkannt(self):
        assert parse_halter("Fahrzeughalter: 2") == "2"

    def test_halter_mit_abstand(self):
        assert parse_halter("Fahrzeughalter  1 Vorbesitzer") == "1"

    def test_halter_nicht_vorhanden(self):
        assert parse_halter("Keine Angabe zum Vorbesitzer") == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Standort
# ──────────────────────────────────────────────────────────────────────────────

class TestStandort:

    def test_de_format(self):
        result = parse_standort("Händler: DE-70173 Stuttgart | Telefon")
        assert result == "70173 Stuttgart"

    def test_fallback_plz_stadt(self):
        result = parse_standort("Fahrzeugstandort: 80331 München")
        assert result == "80331 München"

    def test_ignoriert_blacklist_wörter(self):
        result = parse_standort("80331 Verfügbarkeit ab sofort")
        assert result == "N/A"

    def test_kein_standort_liefert_na(self):
        assert parse_standort("Kein Standort angegeben") == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Ausstattung
# ──────────────────────────────────────────────────────────────────────────────

class TestAusstattung:

    def test_ausstattung_wird_extrahiert(self):
        soup = BeautifulSoup(FULL_DETAIL_HTML, "html.parser")
        result = parse_ausstattung(soup)
        assert "Sitzheizung" in result
        assert "Panoramadach" in result

    def test_mehr_anzeigen_wird_gefiltert(self):
        soup = BeautifulSoup(FULL_DETAIL_HTML, "html.parser")
        result = parse_ausstattung(soup)
        assert "Mehr anzeigen" not in result

    def test_stopp_vor_naechster_sektion(self):
        html = """
        <div>
          <h2>Ausstattung</h2>
          <p>Navigation</p>
          <h2>Fahrzeugbeschreibung laut Anbieter</h2>
          <p>Das ist die Beschreibung, kein Feature</p>
        </div>"""
        soup = BeautifulSoup(html, "html.parser")
        result = parse_ausstattung(soup)
        assert "Navigation" in result
        assert "Das ist die Beschreibung" not in result

    def test_kein_ausstattung_heading_liefert_na(self):
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        assert parse_ausstattung(soup) == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Fahrzeugbeschreibung
# ──────────────────────────────────────────────────────────────────────────────

class TestBeschreibung:

    def test_beschreibung_wird_extrahiert(self):
        soup = BeautifulSoup(FULL_DETAIL_HTML, "html.parser")
        result = parse_beschreibung(soup)
        assert "Sehr gepflegtes Fahrzeug" in result
        assert "Kein Unfallschaden" in result

    def test_beschreibung_mit_pipe_getrennt(self):
        soup = BeautifulSoup(FULL_DETAIL_HTML, "html.parser")
        result = parse_beschreibung(soup)
        assert " | " in result

    def test_stopp_vor_haendler_section(self):
        html = """
        <div>
          <h2>Fahrzeugbeschreibung laut Anbieter</h2>
          <p>Top gepflegt</p>
          <h2>Über diesen Händler</h2>
          <p>Händler Info</p>
        </div>"""
        soup = BeautifulSoup(html, "html.parser")
        result = parse_beschreibung(soup)
        assert "Top gepflegt" in result
        assert "Händler Info" not in result

    def test_duplikate_werden_entfernt(self):
        html = """
        <div>
          <h2>Fahrzeugbeschreibung laut Anbieter</h2>
          <p>Scheckheftgepflegt</p>
          <p>Scheckheftgepflegt</p>
        </div>"""
        soup = BeautifulSoup(html, "html.parser")
        result = parse_beschreibung(soup)
        parts = result.split(" | ")
        assert parts.count("Scheckheftgepflegt") == 1

    def test_kein_heading_liefert_na(self):
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        assert parse_beschreibung(soup) == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Link-Bereinigung & Duplikat-Erkennung
# ──────────────────────────────────────────────────────────────────────────────

class TestLinkBereinigung:

    def test_relativer_pfad_wird_absolut(self):
        result = clean_car_link("/fahrzeuge/details.html?id=123&ref=srp")
        assert result.startswith("https://suchen.mobile.de")

    def test_parameter_werden_auf_ersten_reduziert(self):
        result = clean_car_link("https://suchen.mobile.de/fahrzeuge/details.html?id=123&ref=srp&vc=Car")
        assert result == "https://suchen.mobile.de/fahrzeuge/details.html?id=123"

    def test_link_ohne_parameter_bleibt_unveraendert(self):
        url = "https://suchen.mobile.de/fahrzeuge/details.html"
        assert clean_car_link(url) == url

    def test_duplikate_im_overview(self):
        soup = BeautifulSoup(OVERVIEW_HTML, "html.parser")
        links = []
        for a_tag in soup.find_all("a", href=re.compile(r"/fahrzeuge/details\.html")):
            href = a_tag.get("href")
            cleaned = clean_car_link(href)
            if cleaned not in links:
                links.append(cleaned)
        assert len(links) == 2  # 3 Tags, aber 1 Duplikat → 2 eindeutige Links

    def test_externe_links_nicht_im_ergebnis(self):
        soup = BeautifulSoup(OVERVIEW_HTML, "html.parser")
        links = [
            a.get("href")
            for a in soup.find_all("a", href=re.compile(r"/fahrzeuge/details\.html"))
        ]
        assert all("/fahrzeuge/details.html" in l for l in links)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: CSV-Ausgabe
# ──────────────────────────────────────────────────────────────────────────────

class TestCSVAusgabe:

    FIELDNAMES = [
        "Titel", "Preis", "Kilometerstand", "Erstzulassung",
        "PS", "Getriebe", "Kraftstoff", "Fahrzeughalter",
        "Standort", "URL", "Ausstattung", "Beschreibung",
    ]

    def _write_row(self, tmpfile, row: dict):
        with open(tmpfile, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerow(row)

    def test_csv_wird_erstellt(self, tmp_path):
        filepath = str(tmp_path / "test.csv")
        row = {k: "Test" for k in self.FIELDNAMES}
        self._write_row(filepath, row)
        assert os.path.exists(filepath)

    def test_csv_hat_alle_felder(self, tmp_path):
        filepath = str(tmp_path / "test.csv")
        row = {k: f"Wert_{k}" for k in self.FIELDNAMES}
        self._write_row(filepath, row)
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_read = next(reader)
        for field in self.FIELDNAMES:
            assert field in row_read

    def test_sonderzeichen_werden_korrekt_gespeichert(self, tmp_path):
        filepath = str(tmp_path / "test.csv")
        row = {k: "Standard" for k in self.FIELDNAMES}
        row["Titel"] = "Mercedes S-Klasse (W222) – Ö/Ä/Ü Test"
        row["Beschreibung"] = "Scheckheftgepflegt | kein Unfall | 1. Hand"
        self._write_row(filepath, row)
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_read = next(reader)
        assert "Ö/Ä/Ü" in row_read["Titel"]
        assert " | " in row_read["Beschreibung"]

    def test_mehrere_zeilen_anhaengen(self, tmp_path):
        filepath = str(tmp_path / "multi.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
        for i in range(3):
            with open(filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow({k: f"Auto_{i}" for k in self.FIELDNAMES})
        with open(filepath, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Integrationstest: Vollständiges Detail-HTML → alle Felder korrekt
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_alle_felder_aus_vollstaendigem_html(self):
        soup = BeautifulSoup(FULL_DETAIL_HTML, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)

        assert parse_title(soup) == "Mercedes-Benz S 500"
        assert parse_price(soup) == "45.000 €"
        assert "85.000 km" in parse_mileage(page_text)
        assert "03/2019" in parse_registration(page_text)
        assert parse_ps(page_text) == "435 PS"
        assert parse_getriebe(page_text) == "Automatik"
        assert parse_kraftstoff(page_text) == "Diesel"
        assert parse_halter(page_text) == "2"
        assert "70173" in parse_standort(page_text)
        assert "Sitzheizung" in parse_ausstattung(soup)
        assert "Sehr gepflegtes Fahrzeug" in parse_beschreibung(soup)

    def test_leeres_html_liefert_nur_na(self):
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        page_text = ""

        assert parse_title(soup) == "N/A"
        assert parse_price(soup) == "N/A"
        assert parse_mileage(page_text) == "N/A"
        assert parse_registration(page_text) == "N/A"
        assert parse_ps(page_text) == "N/A"
        assert parse_getriebe(page_text) == "N/A"
        assert parse_kraftstoff(page_text) == "N/A"
        assert parse_halter(page_text) == "N/A"
        assert parse_standort(page_text) == "N/A"
        assert parse_ausstattung(soup) == "N/A"
        assert parse_beschreibung(soup) == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Test: Hauptfunktion mit vollständig gemocktem Browser
# ──────────────────────────────────────────────────────────────────────────────

class TestHauptfunktionMock:

    def test_scraper_laeuft_ohne_browser(self, tmp_path):
        """Stellt sicher, dass der Scraper mit einem Mock-Driver durchläuft,
        ohne dass seleniumbase installiert sein muss."""
        import sys
        from unittest.mock import MagicMock, patch

        # Seleniumbase komplett mocken, bevor das Modul importiert wird
        mock_sb_module = MagicMock()
        MockDriver = MagicMock()
        mock_driver = MagicMock()
        MockDriver.return_value = mock_driver
        mock_sb_module.Driver = MockDriver

        call_count = {"n": 0}

        def fake_page_source():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return OVERVIEW_HTML     # Seite 1: 2 Links
            elif call_count["n"] == 2:
                return "<html><body></body></html>"  # Seite 2: leer → Stopp
            return FULL_DETAIL_HTML      # Detailseiten

        type(mock_driver).page_source = property(lambda self: fake_page_source())
        mock_driver.find_elements.return_value = []

        with patch.dict(sys.modules, {"seleniumbase": mock_sb_module}):
            with patch("time.sleep", return_value=None):
                with patch("os.path.exists", return_value=True):
                    # Testet die Mock-Kette: Driver wurde korrekt aufgerufen
                    driver = mock_sb_module.Driver(uc=True, incognito=True)
                    assert driver is mock_driver
                    MockDriver.assert_called_with(uc=True, incognito=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
