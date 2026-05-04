# Install seleniumbase first (if not present):
# pip install seleniumbase

from seleniumbase import Driver
from bs4 import BeautifulSoup
import time
import random
import csv
import re
import os


def scrape_mobile_de_details(max_pages=5):
    print("Starte getarnten Browser...")
    driver = Driver(uc=True, incognito=True)

    # Hier URL anpassen, welche Seite gescrapt werden soll.
    base_search_url = (
        "https://suchen.mobile.de/fahrzeuge/search.html?cn=DE&dam=false"
        "&isSearchRequest=true&ms=17200%3B%3B16%3B&od=up&ref=srp"
        "&refId=c0d9ecaa-2e9d-52e5-6f6c-f7ea6ddbf3b2&s=Car&sb=rel&st=DEALER&vc=Car"
    )
    # Adjust the filename here.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(BASE_DIR, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "mobile_de_erweitert_Sklasse.csv")
    fieldnames = [
        "Titel", "Preis", "Kilometerstand", "Erstzulassung",
        "PS", "Getriebe", "Kraftstoff", "Fahrzeughalter",
        "Standort", "URL", "Ausstattung", "Beschreibung",
    ]

    # CSV initialisieren (Header schreiben, falls Datei neu ist)
    if not os.path.exists(filename):
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    all_car_links = []

    try:
        # --- 1. LINKS ÜBER MEHRERE SEITEN SAMMELN ---
        for page in range(1, max_pages + 1):
            current_search_url = f"{base_search_url}&pageNumber={page}"
            print(f"\n--- Lade Übersichtsseite {page} ---")

            driver.uc_open_with_reconnect(current_search_url, reconnect_time=5)

            if page == 1:
                print("Warte 15 Sekunden auf der ersten Seite (evtl. Captcha/Cookies)...")
                time.sleep(15)
            else:
                time.sleep(random.uniform(5, 8))

            driver.execute_script("window.scrollBy(0, 1500);")
            time.sleep(3)

            soup_overview = BeautifulSoup(driver.page_source, "html.parser")
            page_links = []

            for a_tag in soup_overview.find_all("a", href=re.compile(r"/fahrzeuge/details\.html")):
                href = a_tag.get("href")
                if href.startswith("/"):
                    href = "https://suchen.mobile.de" + href
                # URL bereinigen, um Duplikate zu vermeiden
                clean_href = (
                    href.split("-)[0] + "- + href.split("-)[1].split("&")[0]
                    if "- in href
                    else href
                )
                if clean_href not in all_car_links:
                    page_links.append(clean_href)

            if not page_links:
                print("Keine weiteren Links gefunden. Beende das Sammeln der Links.")
                break

            all_car_links.extend(page_links)
            print(
                f"Es wurden {len(page_links)} Auto-Links auf Seite {page} gefunden. "
                f"(Gesamt: {len(all_car_links)})"
            )

        # LIMIT FÜR DEN TESTLAUF (entfernen, wenn alle Links gescrapt werden sollen)
        all_car_links = all_car_links[:200]
        print(f"\nBesuche nun {len(all_car_links)} Autos im Detail...")

        # --- 2. DETAILSEITEN BESUCHEN & DIREKT SPEICHERN ---
        for index, link in enumerate(all_car_links, 1):
            try:
                print(f"[{index}/{len(all_car_links)}] Öffne Auto...")
                driver.uc_open_with_reconnect(link, reconnect_time=4)
                time.sleep(random.uniform(4.0, 6.0))
                driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(1.0)

                # Click all "Show more" buttons via JavaScript
                try:
                    mehr_anzeigen_buttons = driver.find_elements(
                        "xpath", "//*[contains(text(), 'Mehr anzeigen')]"
                    )
                    for btn in mehr_anzeigen_buttons:
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(0.5)
                except Exception:
                    pass

                soup = BeautifulSoup(driver.page_source, "html.parser")
                page_text = soup.get_text(separator=" ", strip=True)

                # -- TITLE (3-Stage-Method) --
                title = "N/A"

                # Stufe 1: Meta-Tag
                meta_title = soup.find("meta", property="og:title")
                if meta_title and meta_title.get("content"):
                    title = meta_title.get("content")
                    title = re.sub(r"\s*für\s*[\d\.,]+\s*€.*", "", title).strip()

                # Stufe 2: Interne IDs oder H1
                if title == "N/A" or not title:
                    title_elem = (
                        soup.find(attrs={"data-testid": "listing-title"})
                        or soup.find(id="ad-title")
                        or soup.find("h1")
                    )
                    if title_elem:
                        title = title_elem.get_text(separator=" ", strip=True)

                # Stage 3: Website Tab Title
                if title == "N/A" or not title:
                    page_title = soup.find("title")
                    if page_title:
                        title = page_title.text.split("für")[0].split("|")[0].strip()

                # -- KAUFPREIS --
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

                # -- KILOMETERSTAND --
                km_match = re.search(r"([\d\.]+)\s*km", page_text, re.IGNORECASE)
                mileage = km_match.group(0) if km_match else "N/A"

                # -- INITIAL REGISTRATION --
                # Bugfix: Monat auf 01–12 beschränkt (vorher \d{2} akzeptierte z.B. 13)
                ez_match = re.search(
                    r"(?:EZ|Erstzulassung)[\s:]*(0[1-9]|1[0-2])/\d{4}", page_text, re.IGNORECASE
                )
                if not ez_match:
                    ez_match = re.search(r"\b(0[1-9]|1[0-2])/(19|20)\d{2}\b", page_text)
                registration = ez_match.group(0) if ez_match else "N/A"

                # -- PS --
                ps_match = re.search(r"\((\d+)\s*PS\)", page_text)
                if not ps_match:
                    ps_match = re.search(r"(\d+)\s*PS", page_text)
                ps = f"{ps_match.group(1)} PS" if ps_match else "N/A"

                # -- GETRIEBE --
                if re.search(r"\bAutomatik\b", page_text, re.IGNORECASE):
                    getriebe = "Automatik"
                elif re.search(r"\bSchaltgetriebe\b", page_text, re.IGNORECASE):
                    getriebe = "Schaltgetriebe"
                else:
                    getriebe = "N/A"

                # -- KRAFTSTOFF --
                # Bugfix:  does not work for fuels with parentheses like "Autogas (LPG)".
                # Lösung: Wörter ohne Klammern nutzen \b, Einträge mit Klammern re.escape ohne \b.
                kraftstoff = "N/A"
                for k in ["Benzin", "Diesel", "Elektro", "Hybrid", "Autogas (LPG)", "Erdgas (CNG)"]:
                    if "(" in k:
                        pattern = re.escape(k)
                    else:
                        pattern = rf"\b{re.escape(k)}\b"
                    if re.search(pattern, page_text, re.IGNORECASE):
                        kraftstoff = k
                        break

                # -- FAHRZEUGHALTER --
                halter_match = re.search(
                    r"Fahrzeughalter(?:[^0-9]{1,15})(\d+)", page_text, re.IGNORECASE
                )
                halter = halter_match.group(1) if halter_match else "N/A"

                # -- STANDORT --
                standort = "N/A"
                loc_match = re.search(
                    r"DE-([0-9]{5})\s+([A-ZÄÖÜ][a-zA-ZäöüßÄÖÜ\-\s]+?)(?=\s*\||\s*$)",
                    page_text,
                )
                if loc_match:
                    standort = f"{loc_match.group(1)} {loc_match.group(2).strip()}"
                else:
                    fallback = re.search(r"\b([0-9]{5})\s+([A-ZÄÖÜ][a-zäöüß]+)\b", page_text)
                    if fallback and fallback.group(2) not in ["Verfügbarkeit", "Kilometer", "Euro"]:
                        standort = f"{fallback.group(1)} {fallback.group(2)}"

                # -- EQUIPMENT --
                ausstattung_liste = []
                ausstattung_heading = soup.find(
                    lambda tag: tag.name in ["h2", "h3", "div"]
                    and tag.text.strip() == "Ausstattung"
                )
                if ausstattung_heading:
                    for element in ausstattung_heading.find_all_next(string=True):
                        text = element.strip()
                        if text in [
                            "Fahrzeugbeschreibung laut Anbieter", "Fahrzeugstandort",
                            "Über diesen Händler", "Standort", "Preis",
                        ]:
                            break
                        if (
                            text
                            and text not in ["Ausstattung", "Mehr anzeigen", "Weniger anzeigen", "✓"]
                            and text not in ausstattung_liste
                            and len(text) > 2
                        ):
                            ausstattung_liste.append(text)

                ausstattung_str = ", ".join(ausstattung_liste) if ausstattung_liste else "N/A"

                # -- FAHRZEUGBESCHREIBUNG --
                beschreibung_text = "N/A"
                beschr_heading = soup.find(
                    lambda tag: tag.name in ["h2", "h3", "div"]
                    and tag.text.strip() == "Fahrzeugbeschreibung laut Anbieter"
                )
                if beschr_heading:
                    beschr_strings = []
                    for element in beschr_heading.find_all_next(string=True):
                        text = element.strip()
                        if text in [
                            "Über diesen Händler", "Standort", "Händler",
                            "Preis", "Finanzierung", "Ähnliche Fahrzeuge",
                        ]:
                            break
                        if (
                            text
                            and text not in ["Fahrzeugbeschreibung laut Anbieter", "Mehr anzeigen", "Weniger anzeigen"]
                            and len(text) > 1
                        ):
                            beschr_strings.append(text)

                    clean_strings = list(dict.fromkeys(beschr_strings))  # Remove duplicates
                    beschreibung_text = " | ".join(clean_strings) if clean_strings else "N/A"

                # --- DATEN DIREKT IN CSV SPEICHERN ---
                car_data = {
                    "Titel": title,
                    "Preis": price,
                    "Kilometerstand": mileage,
                    "Erstzulassung": registration,
                    "PS": ps,
                    "Getriebe": getriebe,
                    "Kraftstoff": kraftstoff,
                    "Fahrzeughalter": halter,
                    "Standort": standort,
                    "URL": link,
                    "Ausstattung": ausstattung_str,
                    "Beschreibung": beschreibung_text,
                }

                with open(filename, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(car_data)

                print(f"   -> Erfolg & Gespeichert: {title[:40]}... | {price} | Standort: {standort}")

            except Exception as e:
                print(f"   -> Fehler bei diesem Auto übersprungen: {e}")
                continue

    except Exception as e:
        print(f"Ein genereller Fehler ist aufgetreten: {e}")
    finally:
        driver.quit()
        print(f"\nScraping beendet! Alle gesammelten Daten sind in '{filename}'.")


# --- Execution ---
# Du kannst die Anzahl der durchsuchten Seiten hier anpassen (z.B. max_pages=10)
if __name__ == "__main__":
    scrape_mobile_de_details(max_pages=7)
