# compute_attractiveness_profiles.py — z blokiem ROOMS
# Użycie:
#   python compute_attractiveness_profiles.py PATH_TO_apartments_poi_scores.csv PATH_TO_offers.csv
#
# Wejście:
#   - apartments_poi_scores.csv  (apt_id, kategoria, poi_feature ~ [0,1])
#   - offers.csv                 (offer_id, area_m2, price_per_m2 / price_amount+area_m2+price_currency, rooms, ...)
#
# Wyjście:
#   - attractiveness_by_profile.csv
#   - ranking_<profil>.csv

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# ====== KONFIG: wagi POI per profil (warstwa 1) ======
HIGH, MID, LOW, ZERO = 1.0, 0.65, 0.35, 0.0

W_POI: Dict[str, Dict[str, float]] = {
    "rodzinny": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "szkola_przedszkole": HIGH, "plac_zabaw": MID, "park": HIGH,
        "biblioteka": MID, "kawiarnia_restauracja": LOW, "galeria": LOW,
        "bank_atm": MID, "fryzjer": MID, "silownia": LOW,
        "weterynarz": MID, "sklep_zoologiczny": MID,
        "klub": -0.5, "pub": -0.3,
    },
    "studencki": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "uczelnia": HIGH, "biblioteka": HIGH, "silownia": HIGH,
        "kawiarnia_restauracja": MID, "klub": HIGH, "pub": HIGH,
        "park": MID, "bank_atm": MID, "fryzjer": LOW, "galeria": LOW,
        "szkola_przedszkole": ZERO, "plac_zabaw": ZERO,
    },
    "singiel": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "kawiarnia_restauracja": HIGH, "silownia": HIGH, "klub": HIGH, "pub": HIGH,
        "park": MID, "bank_atm": MID, "fryzjer": MID, "galeria": LOW, "biblioteka": LOW,
        "szkola_przedszkole": ZERO, "plac_zabaw": ZERO,
    },
    "wlasciciel_psa": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "park": HIGH, "weterynarz": HIGH, "sklep_zoologiczny": HIGH,
        "kawiarnia_restauracja": MID, "silownia": LOW, "bank_atm": MID, "fryzjer": LOW, "galeria": LOW,
        "klub": -0.2, "pub": -0.1,
        "szkola_przedszkole": LOW, "plac_zabaw": LOW, "biblioteka": LOW,
    },
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

# ====== Wagi bloków do fallbacku (dodano ROOMS) ======
WB: Dict[str, Dict[str, float]] = {
    "rodzinny":      {"POI": 0.30, "CENA": 0.18, "M2": 0.32, "ROOMS": 0.15, "ZDJ": 0.05},
    "studencki":     {"POI": 0.38, "CENA": 0.34, "M2": 0.12, "ROOMS": 0.11, "ZDJ": 0.05},
    "singiel":       {"POI": 0.25, "CENA": 0.35, "M2": 0.20, "ROOMS": 0.15, "ZDJ": 0.05},
    "wlasciciel_psa":{"POI": 0.42, "CENA": 0.18, "M2": 0.22, "ROOMS": 0.13, "ZDJ": 0.05},
    "uniwersalny":   {"POI": 0.28, "CENA": 0.24, "M2": 0.24, "ROOMS": 0.09, "ZDJ": 0.15},
}

# ====== Docelowe metraże per profil (dla M2) ======
SIZE_TARGET: Dict[str, Tuple[float, float]] = {
    "rodzinny":       (25, 65),
    "studencki":      (20, 45),
    "singiel":        (20, 45),
    "wlasciciel_psa": (25, 55),
    "uniwersalny":    (22, 55),
}

# ====== Docelowa liczba pokoi per profil (dla ROOMS) ======
# Format: (min_ok, target_low, target_high, max_ok)
ROOMS_TARGET_RANGES: Dict[str, Tuple[float, float, float, float]] = {
    "rodzinny":       (2.0, 3.0, 4.0, 5.0),  # 3–4 idealne
    "studencki":      (1.0, 1.0, 2.0, 3.0),  # 1–2 idealne
    "singiel":        (1.0, 1.0, 1.0, 2.0),  # 1 idealnie
    "wlasciciel_psa": (2.0, 2.0, 3.0, 4.0),  # 2–3 idealne
    "uniwersalny":    (2.0, 2.0, 3.0, 4.0),  # 2–3 idealne
}

# ====== Konsekwencje (Sugeno 0-order) dla 1D bloków ======
PRICE_RULES = {
    "rodzinny":      {"Cheap": 1.00, "Mid": 0.60, "Expensive": 0.20},
    "studencki":     {"Cheap": 1.00, "Mid": 0.55, "Expensive": 0.10},
    "singiel":       {"Cheap": 0.90, "Mid": 0.65, "Expensive": 0.25},
    "wlasciciel_psa":{"Cheap": 0.95, "Mid": 0.60, "Expensive": 0.20},
    "uniwersalny":   {"Cheap": 0.95, "Mid": 0.60, "Expensive": 0.25},
}
SIZE_RULES = {
    "rodzinny":      {"Small": 0.20, "Target": 1.00, "Large": 0.70},
    "studencki":     {"Small": 0.30, "Target": 0.95, "Large": 0.60},
    "singiel":       {"Small": 0.35, "Target": 0.90, "Large": 0.65},
    "wlasciciel_psa":{"Small": 0.25, "Target": 1.00, "Large": 0.75},
    "uniwersalny":   {"Small": 0.30, "Target": 0.95, "Large": 0.70},
}
PHOTOS_RULES = {
    "rodzinny":      {"Few": 0.20, "Some": 0.60, "Many": 1.00},
    "studencki":     {"Few": 0.20, "Some": 0.55, "Many": 0.95},
    "singiel":       {"Few": 0.20, "Some": 0.60, "Many": 0.95},
    "wlasciciel_psa":{"Few": 0.20, "Some": 0.55, "Many": 0.90},
    "uniwersalny":   {"Few": 0.20, "Some": 0.60, "Many": 1.00},
}
ROOMS_RULES = {
    # Zasada: Target -> wysoka stała; TooFew/TooMany -> obniżenie (nieco mniej surowe dla "studencki")
    "rodzinny":      {"TooFew": 0.30, "Target": 1.00, "TooMany": 0.60},
    "studencki":     {"TooFew": 0.55, "Target": 0.95, "TooMany": 0.50},
    "singiel":       {"TooFew": 0.60, "Target": 1.00, "TooMany": 0.50},
    "wlasciciel_psa":{"TooFew": 0.40, "Target": 1.00, "TooMany": 0.60},
    "uniwersalny":   {"TooFew": 0.45, "Target": 0.95, "TooMany": 0.60},
}

# ====== Helpery ======
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def weighted_mean_signed(values, weights):
    v = np.array(values, float)
    w = np.array(weights, float)
    if np.all(w == 0): return 0.0
    return float((v * w).sum() / np.abs(w).sum())

# ====== Fuzzification 1D ======
def price_memberships(price: float, p10: float, p50: float, p95: float):
    if price <= p10: cheap = 1.0
    elif price < p50: cheap = (p50 - price) / (p50 - p10)
    else: cheap = 0.0
    if price <= p50: expensive = 0.0
    elif price < p95: expensive = (price - p50) / (p95 - p50)
    else: expensive = 1.0
    if price <= p10 or price >= p95: mid = 0.0
    elif price == p50: mid = 1.0
    elif price < p50: mid = (price - p10) / (p50 - p10)
    else: mid = (p95 - price) / (p95 - p50)
    return {"Cheap": _clamp01(cheap), "Mid": _clamp01(mid), "Expensive": _clamp01(expensive)}

def size_memberships(size: float, s_min: float, s_target: float):
    s_max = s_target * 1.35
    if size <= s_min: small = 1.0
    elif size < s_target: small = (s_target - size) / (s_target - s_min)
    else: small = 0.0
    if size <= s_target: large = 0.0
    elif size < s_max: large = (size - s_target) / (s_max - s_target)
    else: large = 1.0
    if size <= s_min or size >= s_max: target = 0.0
    elif size == s_target: target = 1.0
    elif size < s_target: target = (size - s_min) / (s_target - s_min)
    else: target = (s_max - size) / (s_max - s_target)
    return {"Small": _clamp01(small), "Target": _clamp01(target), "Large": _clamp01(large)}

def photos_memberships(count: int):
    c = float(max(0, int(count)))
    if c <= 3: few = 1.0
    elif c < 6: few = (6 - c) / 3.0
    else: few = 0.0
    if c <= 6: many = 0.0
    elif c < 12: many = (c - 6) / 6.0
    else: many = 1.0
    if c <= 3 or c >= 15: some = 0.0
    elif c == 9: some = 1.0
    elif c < 9: some = (c - 3) / (9 - 3)
    else: some = (15 - c) / (15 - 9)
    return {"Few": _clamp01(few), "Some": _clamp01(some), "Many": _clamp01(many)}

def rooms_memberships(rooms: float, rmin: float, rlow: float, rhigh: float, rmax: float):
    """
    Zbiory:
      TooFew — wysoka dla r <= rmin, maleje do 0 przy rlow
      Target — trójkąt z bazą [rlow, rhigh], wierzchołek: środek (lub całe plateau gdy rlow==rhigh)
      TooMany — rośnie od rhigh do rmax i 1 powyżej rmax
    """
    x = float(max(0.0, rooms))
    # TooFew
    if x <= rmin: few = 1.0
    elif x < rlow: few = (rlow - x) / (rlow - rmin)
    else: few = 0.0
    # Target
    if rlow == rhigh:
        # "igła" na jednej wartości (np. 1 pokój dla "singiel")
        if x == rlow: target = 1.0
        elif x < rlow: target = (x - rmin) / (rlow - rmin) if rlow > rmin else 0.0
        else: target = (rmax - x) / (rmax - rhigh) if rmax > rhigh else 0.0
    else:
        if x <= rlow or x >= rhigh: target = 0.0
        elif x == (rlow + rhigh)/2.0: target = 1.0
        elif x < (rlow + rhigh)/2.0: target = (x - rlow) / ((rhigh - rlow)/2.0)
        else: target = (rhigh - x) / ((rhigh - rlow)/2.0)
    target = _clamp01(target)
    # TooMany
    if x <= rhigh: many = 0.0
    elif x < rmax: many = (x - rhigh) / (rmax - rhigh)
    else: many = 1.0
    return {"TooFew": _clamp01(few), "Target": target, "TooMany": _clamp01(many)}

def sugeno_single_input(memberships: Dict[str, float], rule_consts: Dict[str, float]) -> float:
    num = den = 0.0
    for bin_name, mu in memberships.items():
        if mu <= 0: continue
        c = float(rule_consts.get(bin_name, 0.5))
        num += mu * c
        den += mu
    return 0.0 if den == 0 else float(num / den)

# ====== POI → Low/Med/High ======
def poi_memberships(poi_score: float):
    x = _clamp01(float(poi_score))
    if x <= 0.0: low = 1.0
    elif x < 0.5: low = (0.5 - x) / 0.5
    else: low = 0.0
    if x <= 0.0 or x >= 1.0: mid = 0.0
    elif x == 0.5: mid = 1.0
    elif x < 0.5: mid = (x - 0.0) / 0.5
    else: mid = (1.0 - x) / 0.5
    if x <= 0.5: high = 0.0
    else: high = (x - 0.5) / 0.5
    return {"Low": _clamp01(low), "Mid": _clamp01(mid), "High": _clamp01(high)}

# ====== Reguły końcowe (rozszerzone o ROOMS) ======
Rule = Tuple[Dict[str, str], float]

FINAL_RULES: Dict[str, List[Rule]] = {
    "rodzinny": [
        ({"POI":"High","PRICE":"Cheap","SIZE":"Target","ROOMS":"Target"}, 1.00),
        ({"POI":"High","PRICE":"Mid","SIZE":"Target","ROOMS":"Target"},   0.92),
        ({"POI":"High","ROOMS":"TooFew"},                                 0.55),
        ({"POI":"High","SIZE":"Small"},                                   0.60),
        ({"POI":"Mid","PRICE":"Cheap","SIZE":"Target","ROOMS":"Target"},  0.85),
        ({"PRICE":"Expensive","POI":"Low"},                               0.20),
        ({"POI":"High"},                                                  0.75),
        ({"POI":"Low"},                                                   0.35),
    ],
    "studencki": [
        ({"POI":"High","PRICE":"Cheap","ROOMS":"Target"},                 1.00),
        ({"POI":"High","PRICE":"Mid","ROOMS":"Target"},                   0.90),
        ({"POI":"Mid","PRICE":"Cheap","ROOMS":"Target"},                  0.86),
        ({"POI":"High","ROOMS":"TooMany"},                                0.60),
        ({"PRICE":"Expensive","POI":"Low"},                               0.15),
        ({"POI":"High"},                                                  0.80),
        ({"POI":"Low"},                                                   0.30),
    ],
    "singiel": [
        ({"POI":"High","PRICE":"Cheap","ROOMS":"Target"},                 0.97),
        ({"POI":"High","PRICE":"Mid","ROOMS":"Target"},                   0.88),
        ({"POI":"High","SIZE":"Target","ROOMS":"Target"},                 0.90),
        ({"POI":"Low","PRICE":"Expensive"},                               0.20),
        ({"POI":"High","ROOMS":"TooMany"},                                0.60),
        ({"POI":"High"},                                                  0.78),
        ({"POI":"Low"},                                                   0.35),
    ],
    "wlasciciel_psa": [
        ({"POI":"High","PRICE":"Cheap","SIZE":"Large","ROOMS":"Target"},  1.00),
        ({"POI":"High","PRICE":"Mid","SIZE":"Large","ROOMS":"Target"},    0.90),
        ({"POI":"High","SIZE":"Target","ROOMS":"Target"},                 0.88),
        ({"POI":"Low","PRICE":"Expensive"},                               0.20),
        ({"POI":"High"},                                                  0.80),
        ({"POI":"Low"},                                                   0.30),
    ],
    "uniwersalny": [
        ({"POI":"High","PRICE":"Cheap","SIZE":"Target","ROOMS":"Target"}, 0.98),
        ({"POI":"High","PRICE":"Mid","SIZE":"Target","ROOMS":"Target"},   0.90),
        ({"POI":"Mid","PRICE":"Cheap","ROOMS":"Target"},                  0.82),
        ({"POI":"Low","PRICE":"Expensive"},                               0.22),
        ({"POI":"High"},                                                  0.76),
        ({"POI":"Low"},                                                   0.35),
    ],
}

def rule_activation(rule_cond: Dict[str, str],
                    mu_poi: Dict[str,float],
                    mu_price: Dict[str,float],
                    mu_size: Dict[str,float],
                    mu_photos: Dict[str,float],
                    mu_rooms: Dict[str,float]) -> float:
    mus = []
    if "POI" in rule_cond: mus.append(mu_poi.get(rule_cond["POI"], 0.0))
    if "PRICE" in rule_cond: mus.append(mu_price.get(rule_cond["PRICE"], 0.0))
    if "SIZE" in rule_cond: mus.append(mu_size.get(rule_cond["SIZE"], 0.0))
    if "PHOTOS" in rule_cond: mus.append(mu_photos.get(rule_cond["PHOTOS"], 0.0))
    if "ROOMS" in rule_cond: mus.append(mu_rooms.get(rule_cond["ROOMS"], 0.0))
    if not mus:
        return 0.0
    return float(min(mus))

def fuzzy_final_attractiveness(profile: str,
                               poi_score: float,
                               price: float, p10: float, p50: float, p95: float,
                               size: float, s_min: float, s_target: float,
                               photos_count: int,
                               rooms: float, rmin: float, rlow: float, rhigh: float, rmax: float,
                               fallback_weighted: float) -> float:
    mu_poi = poi_memberships(poi_score)
    mu_price = price_memberships(price, p10, p50, p95)
    mu_size = size_memberships(size, s_min, s_target)
    mu_photos = photos_memberships(photos_count)
    mu_rooms = rooms_memberships(rooms, rmin, rlow, rhigh, rmax)
    num = den = 0.0
    for cond, const in FINAL_RULES[profile]:
        alpha = rule_activation(cond, mu_poi, mu_price, mu_size, mu_photos, mu_rooms)
        if alpha <= 0:
            continue
        num += alpha * const
        den += alpha
    if den <= 1e-6:
        return fallback_weighted
    return float(num / den)

# ====== I/O ======
def _normalize_offers(offers_path: Path) -> pd.DataFrame:
    df = pd.read_csv(offers_path).copy()
    if "offer_id" not in df.columns:
        raise ValueError("offers.csv musi mieć kolumnę 'offer_id'")
    out = pd.DataFrame({"apt_id": df["offer_id"].astype(str)})

    # price_pln_m2
    if "price_per_m2" in df.columns:
        out["price_pln_m2"] = pd.to_numeric(df["price_per_m2"], errors="coerce")
    elif {"price_amount", "area_m2", "price_currency"}.issubset(df.columns):
        m = (df["price_currency"].astype(str).upper() == "PLN")
        with np.errstate(divide="ignore", invalid="ignore"):
            out["price_pln_m2"] = np.where(
                m,
                pd.to_numeric(df["price_amount"], errors="coerce") / pd.to_numeric(df["area_m2"], errors="coerce"),
                np.nan
            )
    else:
        out["price_pln_m2"] = np.nan

    # size_m2
    out["size_m2"] = pd.to_numeric(df.get("area_m2", np.nan), errors="coerce")

    # rooms
    if "rooms" in df.columns:
        out["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    else:
        out["rooms"] = np.nan

    # photos_count (brak w offers.csv -> 0)
    out["photos_count"] = 0

    return out

def load_inputs(scores_path: Path, offers_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(scores_path)
    required = {"apt_id", "kategoria", "poi_feature"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn w apartments_poi_scores.csv: {missing}")
    apts = _normalize_offers(offers_path)
    return scores, apts

def fill_missing_attributes(apts: pd.DataFrame, seed=123) -> pd.DataFrame:
    """Wypełnia WYŁĄCZNIE braki; nie nadpisuje danych z offers.csv."""
    rng = np.random.default_rng(seed)
    out = apts.copy()

    # price
    if "price_pln_m2" not in out.columns or out["price_pln_m2"].isna().all():
        base = rng.normal(12500, 1800, size=len(out))
        base = np.clip(base, 8000, 20000)
        out["price_pln_m2"] = np.where(out.get("price_pln_m2").notna(), out["price_pln_m2"], base.round(0))
    else:
        mask = out["price_pln_m2"].isna()
        if mask.any():
            base = rng.normal(12500, 1800, size=mask.sum())
            base = np.clip(base, 8000, 20000)
            out.loc[mask, "price_pln_m2"] = base.round(0)

    # size
    if "size_m2" not in out.columns or out["size_m2"].isna().all():
        size = rng.normal(48, 14, size=len(out))
        size = np.clip(size, 18, 85)
        out["size_m2"] = size.round(1)
    else:
        mask = out["size_m2"].isna()
        if mask.any():
            size = rng.normal(48, 14, size=mask.sum())
            size = np.clip(size, 18, 85)
            out.loc[mask, "size_m2"] = size.round(1)

    # rooms — jeśli brak, heurystycznie z metrażu
    if "rooms" not in out.columns:
        out["rooms"] = np.nan
    mask_r = out["rooms"].isna()
    if mask_r.any():
        # prosta reguła: <=28→1, <=45→2, <=65→3, <=85→4, else 5
        bins = np.array([0, 28, 45, 65, 85, 1e9], dtype=float)
        vals = np.array([1, 2, 3, 4, 5], dtype=float)
        sizes = out.loc[mask_r, "size_m2"].astype(float).to_numpy()
        idx = np.digitize(sizes, bins, right=True) - 1
        est = vals[np.clip(idx, 0, len(vals)-1)]
        out.loc[mask_r, "rooms"] = est

    # photos
    if "photos_count" not in out.columns:
        out["photos_count"] = 0

    return out

# ====== GŁÓWNA LOGIKA ======
def compute_blocks_and_final(scores: pd.DataFrame, apts: pd.DataFrame) -> pd.DataFrame:
    # quantyle ceny
    p10 = float(np.percentile(apts["price_pln_m2"], 10))
    p50 = float(np.percentile(apts["price_pln_m2"], 50))
    p95 = float(np.percentile(apts["price_pln_m2"], 95))

    # pivot POI
    piv = scores.pivot_table(index="apt_id", columns="kategoria", values="poi_feature", aggfunc="mean").fillna(0.0)
    piv.columns.name = None
    all_cats = sorted(set().union(*[set(d.keys()) for d in W_POI.values()]))
    for cat in all_cats:
        if cat not in piv.columns:
            piv[cat] = 0.0

    rows = []
    for apt_id, row in piv.iterrows():
        apt_row = apts.loc[apts["apt_id"] == apt_id]
        if apt_row.empty:
            continue

        price  = float(apt_row["price_pln_m2"].iloc[0])
        size   = float(apt_row["size_m2"].iloc[0])
        photos = int(apt_row["photos_count"].iloc[0])
        rooms  = float(apt_row["rooms"].iloc[0])

        for prof, weights in W_POI.items():
            # 1) POI
            cats = list(weights.keys())
            vals = [float(row.get(c, 0.0)) for c in cats]
            wts  = [float(weights[c]) for c in cats]
            poi_score = weighted_mean_signed(vals, wts)

            # 2) Fuzzy 1D bloków
            smin, starget = SIZE_TARGET[prof]
            rmin, rlow, rhigh, rmax = ROOMS_TARGET_RANGES[prof]

            price_s  = sugeno_single_input(price_memberships(price, p10, p50, p95), PRICE_RULES[prof])
            m2_s     = sugeno_single_input(size_memberships(size, smin, starget), SIZE_RULES[prof])
            photo_s  = sugeno_single_input(photos_memberships(photos),          PHOTOS_RULES[prof])
            rooms_s  = sugeno_single_input(rooms_memberships(rooms, rmin, rlow, rhigh, rmax), ROOMS_RULES[prof])

            # 3) Fallback — ważenie bloków (z ROOMS)
            wblk = WB[prof]
            num_fb = (
                wblk["POI"]*poi_score +
                wblk["CENA"]*price_s +
                wblk["M2"]*m2_s +
                wblk["ROOMS"]*rooms_s +
                wblk["ZDJ"]*photo_s
            )
            den_fb = sum(wblk.values())
            fallback = float(num_fb/den_fb if den_fb > 0 else 0.0)

            # 4) Końcowe fuzzy (Sugeno wielowymiarowe)
            final = fuzzy_final_attractiveness(
                prof,
                poi_score,
                price, p10, p50, p95,
                size, smin, starget,
                photos,
                rooms, rmin, rlow, rhigh, rmax,
                fallback_weighted=fallback
            )

            # 5) Uzasadnienia dla POI (TOP±)
            contrib = pd.Series({c: weights[c]*row.get(c, 0.0) for c in cats})
            top_plus = ", ".join([f"{c}(+{contrib[c]:.2f})" for c in contrib.sort_values(ascending=False).head(3).index])
            top_minus = ", ".join([f"{c}({contrib[c]:.2f})" for c in contrib.sort_values().head(3).index])

            rows.append({
                "apt_id": apt_id,
                "profile": prof,
                "POI": round(poi_score, 4),
                "CENA": round(price_s, 4),
                "M2": round(m2_s, 4),
                "ROOMS": round(rooms_s, 4),
                "ZDJ": round(photo_s, 4),
                "ATRAKCYJNOSC": round(final, 4),
                "price_pln_m2": round(price, 0),
                "size_m2": round(size, 1),
                "rooms": int(round(rooms)),
                "photos_count": photos,
                "TOP_PLUS": top_plus,
                "TOP_MINUS": top_minus,
            })
    return pd.DataFrame(rows)

def save_rankings(df: pd.DataFrame, out_dir: Path, topn=30):
    out_all = out_dir / "attractiveness_by_profile.csv"
    df.to_csv(out_all, index=False, encoding="utf-8")
    for prof in df["profile"].unique():
        sub = df[df["profile"] == prof].sort_values("ATRAKCYJNOSC", ascending=False)
        sub.head(topn).to_csv(out_dir / f"ranking_{prof}.csv", index=False, encoding="utf-8")

def main():
    if len(sys.argv) < 3:
        print("Użycie: python compute_attractiveness_profiles.py PATH_TO_apartments_poi_scores.csv PATH_TO_offers.csv")
        sys.exit(1)
    scores_path = Path(sys.argv[1])
    offers_path = Path(sys.argv[2])
    out_dir = scores_path.parent

    scores, apts_raw = load_inputs(scores_path, offers_path)
    apts = fill_missing_attributes(apts_raw, seed=123)
    out = compute_blocks_and_final(scores, apts)
    save_rankings(out, out_dir)

    print("[OK] Zapisano:")
    print(f"- {out_dir/'attractiveness_by_profile.csv'}")
    for prof in W_POI.keys():
        print(f"- {out_dir/f'ranking_{prof}.csv'} (TOP mieszkania dla profilu '{prof}')")

if __name__ == "__main__":
    main()
