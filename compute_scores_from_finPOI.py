import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
#  MAPOWANIE KATEGORII OSM → KATEGORIE LOGICZNE UŻYWANE W W_POI
# python compute_scores_from_finPOI.py C:\Users\antek\PycharmProjects\PythonProjectMiasto\data\apartments_out.json
# -------------------------------------------------------------------------

OSM_TO_LOGIC = {
    # SKLEPY
    "supermarket": "sklep",
    "convenience": "sklep",
    "bakery": "sklep",

    # TRANSPORT
    "bus_stop": "przystanek_autobus",
    "tram_stop": "przystanek_tramwaj",

    # ZDROWIE
    "clinic_hospital": "szpital_przychodnia",
    "pharmacy": "apteka",

    # EDU + DZIECI
    "playground": "plac_zabaw",
    "kinder_childcare": "szkola_przedszkole",
    "school": "szkola_przedszkole",
    "university": "uczelnia",

    # REKREACJA
    "park": "park",
    "library": "biblioteka",

    # ROZRYWKA
    "nightclub": "klub",
    "pub": "pub",

    # ZWIERZĘTA
    "pet_shop": "sklep_zoologiczny",
    "veterinary": "weterynarz",
}

# -------------------------------------------------------------------------
#  FUZZY (tylko czas)
# -------------------------------------------------------------------------

def time_fuzzy(t: float) -> float:
    """
    Fuzzy usefulness only based on time_min.
    0-5 min: Near → 1.0
    5-10 min: Mid → 1.0 → 0.6
    10-15 min: Far → 0.6 → 0.3
    """
    if t <= 5:
        return 1.0
    if t <= 10:
        return 1.0 - 0.4 * ((t - 5) / 5)
    if t <= 15:
        return 0.6 - 0.3 * ((t - 10) / 5)
    return 0.0

# -------------------------------------------------------------------------
#  GŁÓWNY PROGRAM
# -------------------------------------------------------------------------

def main(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    rows = []

    # wszystkie kategorie logiczne z W_POI
    all_logic_cats = set(OSM_TO_LOGIC.values())

    for apt in data:
        apt_id = apt["source_id"]
        poi_list = apt.get("pois", [])

        # mapowanie OSM → LOGIC
        mapped = {}
        for poi in poi_list:
            osm_cat = poi["category"]
            if osm_cat not in OSM_TO_LOGIC:
                continue  # ignorujemy POI nieużywane w fuzzy logic

            logic_cat = OSM_TO_LOGIC[osm_cat]
            t = float(poi["time_min"])

            # może być wiele OSM-kategorii mapowanych do jednej logicznej
            # bierzemy NAJLEPSZY (maksymalny poi_feature)
            feature = time_fuzzy(t)

            if logic_cat not in mapped:
                mapped[logic_cat] = (t, feature)
            else:
                # wybieramy POI z lepszym fuzzy
                if feature > mapped[logic_cat][1]:
                    mapped[logic_cat] = (t, feature)

        # tworzymy pełny zestaw kategorii logicznych
        for logic_cat in all_logic_cats:
            if logic_cat in mapped:
                t, feature = mapped[logic_cat]
                rows.append({
                    "apt_id": apt_id,
                    "kategoria": logic_cat,
                    "best_time_min": t,
                    "poi_feature": feature
                })
            else:
                # brak POI → 0
                rows.append({
                    "apt_id": apt_id,
                    "kategoria": logic_cat,
                    "best_time_min": np.nan,
                    "poi_feature": 0.0
                })

    df = pd.DataFrame(rows)

    out = json_path.parent / "apartments_poi_scores.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print("✔ Zapisano:", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python compute_scores_from_finPOI.py path\\apartments_out.json")
        sys.exit(1)
    main(Path(sys.argv[1]))
