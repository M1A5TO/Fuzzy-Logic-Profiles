# compute_attractiveness_profiles.py
# Użycie:
#   python compute_attractiveness_profiles.py PATH_TO_apartments_poi_scores.csv PATH_TO_apartments_simulated.csv
#
# Wejście:
#   - apartments_poi_scores.csv  (z poprzedniego skryptu: apt_id, kategoria, poi_feature, ...)
#   - apartments_simulated.csv   (ma kolumny: apt_id, lat, lon; jeśli nie ma realnych atrybutów, skrypt je zasymuluje)
#
# Wyjście:
#   - attractiveness_by_profile.csv   (POI, CENA, M2, ZDJ, Atrakcyjność dla 5 profili)
#   - ranking_<profil>.csv            (posortowana atrakcyjność + uzasadnienie TOP± wkładów)
#
# Noty:
#   - Obsługuje wagi ujemne (kary) w macierzy wag POI.
#   - Gdy brak realnych: symuluje price/m2, size, photos (łatwo podmienić na realne dane).
#   - Normalizacje: robust p10/p95 dla ceny; docelowe s_target per profil dla m2.

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ====== KONFIG: wagi POI per profil ======
# Skala: wysokie ~ 1.0, średnie ~ 0.65, niskie ~ 0.35, zero = 0.0, kary < 0
# Dodaj/usuń kategorie wg swoich danych (muszą zgadzać się z "kategoria" z CSV)
HIGH, MID, LOW, ZERO = 1.0, 0.65, 0.35, 0.0

W_POI: Dict[str, Dict[str, float]] = {
    # 1) Rodzinny
    "rodzinny": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "szkola_przedszkole": HIGH, "plac_zabaw": MID, "park": HIGH,
        "biblioteka": MID,
        "kawiarnia_restauracja": LOW,
        "galeria": LOW,
        "bank_atm": MID, "fryzjer": MID,
        "silownia": LOW,
        "weterynarz": MID, "sklep_zoologiczny": MID,
        # kary:
        "klub": -0.5, "pub": -0.3,
    },

    # 2) Studencki
    "studencki": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "uczelnia": HIGH, "biblioteka": HIGH, "silownia": HIGH,
        "kawiarnia_restauracja": MID,
        "klub": HIGH, "pub": HIGH,
        "park": MID, "bank_atm": MID, "fryzjer": LOW, "galeria": LOW,
        # ignorowane rodzinne:
        "szkola_przedszkole": ZERO, "plac_zabaw": ZERO,
    },

    # 3) Singiel
    "singiel": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "kawiarnia_restauracja": HIGH, "silownia": HIGH, "klub": HIGH, "pub": HIGH,
        "park": MID, "bank_atm": MID, "fryzjer": MID, "galeria": LOW,
        "biblioteka": LOW,
        "szkola_przedszkole": ZERO, "plac_zabaw": ZERO,
    },

    # 4) Właściciel psa
    "wlasciciel_psa": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "park": HIGH, "weterynarz": HIGH, "sklep_zoologiczny": HIGH,
        "kawiarnia_restauracja": MID, "silownia": LOW, "bank_atm": MID, "fryzjer": LOW, "galeria": LOW,
        "klub": -0.2, "pub": -0.1,
        "szkola_przedszkole": LOW, "plac_zabaw": LOW, "biblioteka": LOW,
    },

    # 5) Uniwersalny
    "uniwersalny": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "park": MID, "biblioteka": MID, "bank_atm": MID, "fryzjer": MID,
        "galeria": LOW, "silownia": MID, "kawiarnia_restauracja": MID,
        "szkola_przedszkole": MID, "plac_zabaw": MID,
        "klub": LOW, "pub": LOW,
        "weterynarz": MID, "sklep_zoologiczny": MID,
    },
}

# ====== KONFIG: wagi bloków (POI/CENA/M2/ZDJ) per profil ======
WB: Dict[str, Dict[str, float]] = {
    "rodzinny":      {"POI": 0.35, "CENA": 0.20, "M2": 0.40, "ZDJ": 0.05},
    "studencki":     {"POI": 0.40, "CENA": 0.35, "M2": 0.20, "ZDJ": 0.05},
    "singiel":       {"POI": 0.25, "CENA": 0.35, "M2": 0.35, "ZDJ": 0.05},
    "wlasciciel_psa":{"POI": 0.45, "CENA": 0.20, "M2": 0.30, "ZDJ": 0.05},
    "uniwersalny":   {"POI": 0.30, "CENA": 0.25, "M2": 0.30, "ZDJ": 0.15},
}

# ====== KONFIG: docelowe metraże per profil (dla normalizacji M2) ======
SIZE_TARGET: Dict[str, Tuple[float, float]] = {
    "rodzinny":       (25, 65),
    "studencki":      (20, 45),
    "singiel":        (20, 45),
    "wlasciciel_psa": (25, 55),
    "uniwersalny":    (22, 55),
}

# ====== FUNKCJE NORMALIZACJI ======
def robust_price_score(price, p10, p95):
    if p95 <= p10:
        return 0.5
    return float(np.clip((p95 - price) / (p95 - p10), 0, 1))

def size_score(size_m2, s_min, s_target):
    if s_target <= s_min:
        return 0.5
    return float(np.clip((size_m2 - s_min) / (s_target - s_min), 0, 1))

def photos_score(count):
    return float(1 - np.exp(-count/5.0))

def weighted_mean_signed(values, weights):
    v = np.array(values, float)
    w = np.array(weights, float)
    if np.all(w == 0):
        return 0.0
    # normalizujemy przez sumę |wag|, aby skala pozostawała w [0,1] przy dodatnich i lekkich ujemnych
    return float((v * w).sum() / np.abs(w).sum())

# ====== PIPELINE ======
def load_inputs(scores_path: Path, apartments_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(scores_path)
    required = {"apt_id", "kategoria", "poi_feature"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn w apartments_poi_scores.csv: {missing}")
    apts = pd.read_csv(apartments_path)
    if "apt_id" not in apts.columns:
        raise ValueError("Brakuje kolumny 'apt_id' w apartments_simulated.csv")
    return scores, apts

def simulate_attributes_if_missing(apts: pd.DataFrame, seed=123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = apts.copy()

    if "price_pln_m2" not in out.columns:
        # ceny w Gdyni/Gdańsku ~ 9k–18k/m2 (bardzo z grubsza); losowo z kilkoma ogonkami
        base = rng.normal(12500, 1800, size=len(out))
        base = np.clip(base, 8000, 20000)
        out["price_pln_m2"] = base.round(0)

    if "size_m2" not in out.columns:
        # metraże 18–85 m2
        size = rng.normal(48, 14, size=len(out))
        size = np.clip(size, 18, 85)
        out["size_m2"] = size.round(1)

    if "photos_count" not in out.columns:
        # zdjęcia 0–15, rozkład Poissona
        out["photos_count"] = rng.poisson(6, size=len(out)).clip(0, 20)

    return out

def compute_blocks_and_final(scores: pd.DataFrame, apts: pd.DataFrame) -> pd.DataFrame:
    # p10/p95 dla ceny (robust scaling)
    p10 = float(np.percentile(apts["price_pln_m2"], 10))
    p95 = float(np.percentile(apts["price_pln_m2"], 95))

    # zagnieżdżona struktura: apt_id -> {kategoria: poi_feature}
    piv = scores.pivot_table(index="apt_id", columns="kategoria", values="poi_feature", aggfunc="mean").fillna(0.0)
    piv.columns.name = None
    # dopnij brakujące kategorie (na wypadek gdy w obszarze nie wystąpiły)
    all_cats = sorted(set().union(*[set(d.keys()) for d in W_POI.values()]))
    for cat in all_cats:
        if cat not in piv.columns:
            piv[cat] = 0.0

    rows = []
    for apt_id, row in piv.iterrows():
        # dane mieszkania
        apt_row = apts.loc[apts["apt_id"] == apt_id]
        if apt_row.empty:
            # jeżeli w apts tego id nie ma (nie powinno), pomijamy
            continue
        price = float(apt_row["price_pln_m2"].iloc[0])
        size  = float(apt_row["size_m2"].iloc[0])
        photos = int(apt_row["photos_count"].iloc[0])

        for prof, weights in W_POI.items():
            # 1) POI: średnia ważona z wagami (obsługa wag ujemnych)
            cats = list(weights.keys())
            vals = [float(row.get(c, 0.0)) for c in cats]
            wts  = [float(weights[c]) for c in cats]
            poi_score = weighted_mean_signed(vals, wts)

            # 2) Cena, metraż, zdjęcia
            price_s = robust_price_score(price, p10, p95)
            smin, starget = SIZE_TARGET[prof]
            m2_s = size_score(size, smin, starget)
            photo_s = photos_score(photos)

            # 3) Sugeno (bloki)
            wblk = WB[prof]
            num = wblk["POI"]*poi_score + wblk["CENA"]*price_s + wblk["M2"]*m2_s + wblk["ZDJ"]*photo_s
            den = wblk["POI"] + wblk["CENA"] + wblk["M2"] + wblk["ZDJ"]
            final = float(num/den if den > 0 else 0.0)

            # 4) Uzasadnienie (wkłady per kategoria; dodatnie i ujemne top3)
            contrib = pd.Series({c: weights[c]*row.get(c, 0.0) for c in cats})
            top_plus = ", ".join([f"{c}(+{contrib[c]:.2f})" for c in contrib.sort_values(ascending=False).head(3).index])
            top_minus = ", ".join([f"{c}({contrib[c]:.2f})" for c in contrib.sort_values().head(3).index])

            rows.append({
                "apt_id": apt_id,
                "profile": prof,
                "POI": round(poi_score, 4),
                "CENA": round(price_s, 4),
                "M2": round(m2_s, 4),
                "ZDJ": round(photo_s, 4),
                "ATRAKCYJNOSC": round(final, 4),
                "price_pln_m2": round(price, 0),
                "size_m2": round(size, 1),
                "photos_count": photos,
                "TOP_PLUS": top_plus,
                "TOP_MINUS": top_minus,
            })
    return pd.DataFrame(rows)

def save_rankings(df: pd.DataFrame, out_dir: Path, topn=30):
    # jeden plik zbiorczy
    out_all = out_dir / "attractiveness_by_profile.csv"
    df.to_csv(out_all, index=False, encoding="utf-8")

    # rankingi i uzasadnienia per profil
    for prof in df["profile"].unique():
        sub = df[df["profile"] == prof].sort_values("ATRAKCYJNOSC", ascending=False)
        sub.head(topn).to_csv(out_dir / f"ranking_{prof}.csv", index=False, encoding="utf-8")

def main():
    if len(sys.argv) < 3:
        print("Użycie: python compute_attractiveness_profiles.py PATH_TO_apartments_poi_scores.csv PATH_TO_apartments_simulated.csv")
        sys.exit(1)

    scores_path = Path(sys.argv[1])
    apts_path = Path(sys.argv[2])
    out_dir = scores_path.parent

    scores, apts = load_inputs(scores_path, apts_path)
    apts = simulate_attributes_if_missing(apts, seed=123)
    out = compute_blocks_and_final(scores, apts)
    save_rankings(out, out_dir)

    print("[OK] Zapisano:")
    print(f"- {out_dir/'attractiveness_by_profile.csv'}")
    for prof in W_POI.keys():
        print(f"- {out_dir/f'ranking_{prof}.csv'} (TOP mieszkania dla profilu '{prof}')")

if __name__ == "__main__":
    main()
