# compute_finattractiveness_profiles.py
# Użycie:
#   python compute_finattractiveness_profiles.py .\data\apartments_poi_scores.csv .\data\apartments_out.json

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass

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

# ====== Wagi bloków ======
WB: Dict[str, Dict[str, float]] = {
    "rodzinny":      {"POI": 0.30, "CENA": 0.18, "M2": 0.32, "ROOMS": 0.15, "ZDJ": 0.05},
    "studencki":     {"POI": 0.38, "CENA": 0.34, "M2": 0.12, "ROOMS": 0.11, "ZDJ": 0.05},
    "singiel":       {"POI": 0.25, "CENA": 0.35, "M2": 0.20, "ROOMS": 0.15, "ZDJ": 0.05},
    "wlasciciel_psa":{"POI": 0.42, "CENA": 0.18, "M2": 0.22, "ROOMS": 0.13, "ZDJ": 0.05},
    "uniwersalny":   {"POI": 0.28, "CENA": 0.24, "M2": 0.24, "ROOMS": 0.15, "ZDJ": 0.09},
}

# ====== Parametry bloków ======
SIZE_TARGET = {
    "rodzinny":       (25, 65),
    "studencki":      (20, 35),
    "singiel":        (20, 45),
    "wlasciciel_psa": (25, 55),
    "uniwersalny":    (22, 55),
}

ROOMS_TARGET_RANGES = {
    "rodzinny":       (2.0, 3.0, 4.0, 5.0),
    "studencki":      (1.0, 1.0, 2.0, 3.0),
    "singiel":        (1.0, 1.0, 1.0, 2.0),
    "wlasciciel_psa": (2.0, 2.0, 3.0, 4.0),
    "uniwersalny":    (2.0, 2.0, 3.0, 4.0),
}

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

ROOMS_RULES = {
    "rodzinny":      {"TooFew": 0.30, "Target": 1.00, "TooMany": 0.60},
    "studencki":     {"TooFew": 0.55, "Target": 0.95, "TooMany": 0.50},
    "singiel":       {"TooFew": 0.60, "Target": 1.00, "TooMany": 0.50},
    "wlasciciel_psa":{"TooFew": 0.40, "Target": 1.00, "TooMany": 0.60},
    "uniwersalny":   {"TooFew": 0.45, "Target": 0.95, "TooMany": 0.60},
}

STYLE_SCORES = {
    "rodzinny":      {"old": 0.70, "modern": 0.90, "unknown": 0.80},
    "studencki":     {"old": 0.50, "modern": 1.00, "unknown": 0.75},
    "singiel":       {"old": 0.50, "modern": 1.00, "unknown": 0.75},
    "wlasciciel_psa":{"old": 0.60, "modern": 0.90, "unknown": 0.75},
    "uniwersalny":   {"old": 0.65, "modern": 0.90, "unknown": 0.80},
}

# ====== Opisy słowne do tagów ======
POI_DESC = {
    "High": "bardzo dobra lokalizacja względem usług",
    "Mid":  "umiarkowanie dobra lokalizacja",
    "Low":  "słaba lokalizacja względem usług",
}

PRICE_DESC = {
    "Cheap":      "niska cena za metr",
    "Mid":        "średnia cena za metr",
    "Expensive":  "wysoka cena za metr",
}

SIZE_DESC = {
    "Small":  "zbyt mały metraż",
    "Target": "odpowiedni metraż",
    "Large":  "duży metraż",
}

ROOMS_DESC = {
    "TooFew":  "zbyt mała liczba pokoi",
    "Target":  "odpowiednia liczba pokoi",
    "TooMany": "zbyt duża liczba pokoi",
}

STYLE_DESC = {
    "modern":  "nowoczesny styl mieszkania",
    "old":     "starszy, klasyczny styl mieszkania",
    "unknown": "neutralny styl mieszkania",
}


def _argmax_label(d: Dict[str, float]) -> str:
    return max(d.items(), key=lambda kv: kv[1])[0]

# ====== Progi cenowe ======
EXPENSIVE_CITIES = {
    "warszawa", "kraków", "krakow",
    "gdańsk", "gdansk",
    "wrocław", "wroclaw",
}

PRICE_THRESHOLDS = {
    "expensive_city": {"cheap_max": 11500, "mid_min": 11500, "mid_max": 14000, "expensive_min": 14000},
    "normal_city":    {"cheap_max": 8500, "mid_min": 8500, "mid_max": 11000, "expensive_min": 11000},
}

@dataclass
class PriceFuzzyConfig:
    cheap_max: float
    mid_min: float
    mid_max: float
    expensive_min: float

def get_city_tier(city: str | None) -> str:
    if not city:
        return "normal_city"
    return "expensive_city" if str(city).lower() in EXPENSIVE_CITIES else "normal_city"

# ====== Helpery ======
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def weighted_mean_signed(values, weights):
    v = np.array(values, float)
    w = np.array(weights, float)
    if np.all(w == 0):
        return 0.0
    return float((v * w).sum() / np.abs(w).sum())

# ====== Membership functions ======
def price_memberships(price: float, cfg: PriceFuzzyConfig) -> Dict[str, float]:
    x = float(price)

    if x <= cfg.cheap_max: cheap = 1.0
    elif x >= cfg.mid_min: cheap = 0.0
    else: cheap = (cfg.mid_min - x) / (cfg.mid_min - cfg.cheap_max)

    if x <= cfg.cheap_max or x >= cfg.expensive_min: mid = 0.0
    elif cfg.mid_min <= x <= cfg.mid_max: mid = 1.0
    elif x < cfg.mid_min: mid = (x - cfg.cheap_max) / (cfg.mid_min - cfg.cheap_max)
    else: mid = (cfg.expensive_min - x) / (cfg.expensive_min - cfg.mid_max)

    if x <= cfg.mid_max: exp = 0.0
    elif x >= cfg.expensive_min: exp = 1.0
    else: exp = (x - cfg.mid_max) / (cfg.expensive_min - cfg.mid_max)

    return {"Cheap": _clamp01(cheap), "Mid": _clamp01(mid), "Expensive": _clamp01(exp)}

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

def rooms_memberships(rooms: float, rmin, rlow, rhigh, rmax):
    x = float(rooms)

    if x <= rmin: few = 1.0
    elif x < rlow: few = (rlow - x) / (rlow - rmin)
    else: few = 0.0

    if x <= rmin or x >= rmax: tgt = 0.0
    elif x < rlow: tgt = (x - rmin) / (rlow - rmin)
    elif x <= rhigh: tgt = 1.0
    else: tgt = (rmax - x) / (rmax - rhigh)

    if x <= rhigh: many = 0.0
    elif x < rmax: many = (x - rhigh) / (rmax - rhigh)
    else: many = 1.0

    return {"TooFew": _clamp01(few), "Target": _clamp01(tgt), "TooMany": _clamp01(many)}

def poi_memberships(x: float):
    x = _clamp01(x)
    if x <= 0: low = 1.0
    elif x < 0.5: low = (0.5 - x) / 0.5
    else: low = 0.0

    if x <= 0 or x >= 1: mid = 0.0
    elif x == 0.5: mid = 1.0
    elif x < 0.5: mid = x / 0.5
    else: mid = (1 - x) / 0.5

    if x <= 0.5: high = 0.0
    else: high = (x - 0.5) / 0.5

    return {"Low": low, "Mid": mid, "High": high}

# ====== Reguły końcowe ======
Rule = Tuple[Dict[str,str], float]

FINAL_RULES = {
    "rodzinny": [
        ({"POI":"High","PRICE":"Cheap","SIZE":"Target","ROOMS":"Target"},1.00),
        ({"POI":"High","PRICE":"Mid","SIZE":"Target","ROOMS":"Target"},0.92),
        ({"POI":"High","ROOMS":"TooFew"},0.55),
        ({"POI":"High","SIZE":"Small"},0.60),
        ({"POI":"Mid","PRICE":"Cheap","SIZE":"Target","ROOMS":"Target"},0.85),
        ({"PRICE":"Expensive","POI":"Low"},0.20),
        ({"POI":"High"},0.75),
        ({"POI":"Low"},0.35),
    ],
    "studencki": [
        ({"POI":"High","PRICE":"Cheap","ROOMS":"Target"},1.00),
        ({"POI":"High","PRICE":"Mid","ROOMS":"Target"},0.90),
        ({"POI":"Mid","PRICE":"Cheap","ROOMS":"Target"},0.86),
        ({"POI":"High","ROOMS":"TooMany"},0.60),
        ({"PRICE":"Expensive","POI":"Low"},0.15),
        ({"POI":"High"},0.80),
        ({"POI":"Low"},0.30),
    ],
    "singiel": [
        ({"POI":"High","PRICE":"Cheap","ROOMS":"Target"},0.97),
        ({"POI":"High","PRICE":"Mid","ROOMS":"Target"},0.88),
        ({"POI":"High","SIZE":"Target","ROOMS":"Target"},0.90),
        ({"POI":"Low","PRICE":"Expensive"},0.20),
        ({"POI":"High","ROOMS":"TooMany"},0.60),
        ({"POI":"High"},0.78),
        ({"POI":"Low"},0.35),
    ],
    "wlasciciel_psa": [
        ({"POI":"High","PRICE":"Cheap","SIZE":"Large","ROOMS":"Target"},1.00),
        ({"POI":"High","PRICE":"Mid","SIZE":"Large","ROOMS":"Target"},0.90),
        ({"POI":"High","SIZE":"Target","ROOMS":"Target"},0.88),
        ({"POI":"Low","PRICE":"Expensive"},0.20),
        ({"POI":"High"},0.80),
        ({"POI":"Low"},0.30),
    ],
    "uniwersalny": [
        ({"POI":"High","PRICE":"Cheap","SIZE":"Target","ROOMS":"Target"},0.98),
        ({"POI":"High","PRICE":"Mid","SIZE":"Target","ROOMS":"Target"},0.90),
        ({"POI":"Mid","PRICE":"Cheap","ROOMS":"Target"},0.82),
        ({"POI":"Low","PRICE":"Expensive"},0.22),
        ({"POI":"High"},0.76),
        ({"POI":"Low"},0.35),
    ],
}

def rule_activation(cond, mu_poi, mu_price, mu_size, mu_rooms):
    mus = []
    if "POI" in cond: mus.append(mu_poi[cond["POI"]])
    if "PRICE" in cond: mus.append(mu_price[cond["PRICE"]])
    if "SIZE" in cond: mus.append(mu_size[cond["SIZE"]])
    if "ROOMS" in cond: mus.append(mu_rooms[cond["ROOMS"]])
    return min(mus) if mus else 0.0

def fuzzy_final_attractiveness(
    profile, poi_score, price, price_cfg, size, s_min, s_target,
    rooms, rmin, rlow, rhigh, rmax, fallback
):
    mu_poi = poi_memberships(poi_score)
    mu_price = price_memberships(price, price_cfg)
    mu_size = size_memberships(size, s_min, s_target)
    mu_rooms = rooms_memberships(rooms, rmin, rlow, rhigh, rmax)

    num = den = 0.0
    for cond, const in FINAL_RULES[profile]:
        a = rule_activation(cond, mu_poi, mu_price, mu_size, mu_rooms)
        if a > 0:
            num += a * const
            den += a
    return fallback if den == 0 else num/den

# ====== Styl ======
def style_score_for_profile(profile: str, raw):
    key = str(raw).lower() if isinstance(raw, str) else "unknown"
    if key not in ("old","modern"): key = "unknown"
    return STYLE_SCORES[profile].get(key, STYLE_SCORES[profile]["unknown"])

# ====== Dane wejściowe ======
def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"apt_id","kategoria","poi_feature"}
    if not req.issubset(df.columns):
        raise ValueError(f"Brak kolumn: {req - set(df.columns)}")
    return df.groupby(["apt_id","kategoria"],as_index=False).agg({"poi_feature":"mean"})

def _extract_city(df):
    for col in ["city","miejscowosc","miejscowość","location_city","locality"]:
        if col in df.columns:
            return df[col]
    return pd.Series([None]*len(df))

def load_apartments_from_json(path: Path) -> pd.DataFrame:
    df = pd.read_json(path)
    if "source_id" not in df.columns:
        raise ValueError("Brak source_id w JSON")

    out = pd.DataFrame()
    out["apt_id"] = df["source_id"].astype(str)
    out["city"]   = _extract_city(df)

    if "price_per_m2" in df.columns and df["price_per_m2"].notna().any():
        out["price_pln_m2"] = pd.to_numeric(df["price_per_m2"],errors="coerce")
    else:
        price = pd.to_numeric(df.get("price",np.nan),errors="coerce")
        area  = pd.to_numeric(df.get("footage",np.nan),errors="coerce")
        curr  = df.get("currency","PLN").astype(str).str.upper()
        out["price_pln_m2"] = np.where(curr=="PLN", price/area, np.nan)

    out["size_m2"] = pd.to_numeric(df.get("footage",np.nan),errors="coerce")
    out["rooms"]   = pd.to_numeric(df.get("room_num",np.nan),errors="coerce")
    out["photo_style"] = df.get("photo_style",None)

    return out.drop_duplicates("apt_id")

# ====== Główna logika ======
def compute_blocks_and_final(scores, apts):
    piv = scores.pivot_table(index="apt_id",columns="kategoria",values="poi_feature",aggfunc="mean").fillna(0)
    piv.columns.name = None

    all_cats = sorted({c for d in W_POI.values() for c in d})
    for c in all_cats:
        if c not in piv.columns:
            piv[c] = 0.0

    rows = []
    for apt_id, poi_row in piv.iterrows():
        apt = apts.loc[apts["apt_id"]==apt_id]
        if apt.empty:
            continue
        apt = apt.iloc[0]

        price = float(apt["price_pln_m2"])
        size  = float(apt["size_m2"])
        rooms = float(apt["rooms"])
        style_raw = apt["photo_style"]
        city = apt["city"]

        tier = get_city_tier(city)
        price_cfg = PriceFuzzyConfig(**PRICE_THRESHOLDS[tier])

        for prof in W_POI.keys():

            # 1) POI weighted sum
            cats = list(W_POI[prof].keys())
            vals = [poi_row[c] for c in cats]
            wts  = [W_POI[prof][c] for c in cats]
            poi_score = weighted_mean_signed(vals, wts)

            # 2) Sugeno 1D
            smin, starget = SIZE_TARGET[prof]
            rmin, rlow, rhigh, rmax = ROOMS_TARGET_RANGES[prof]

            price_s = np.average(
                [PRICE_RULES[prof][k] for k in ["Cheap","Mid","Expensive"]],
                weights=list(price_memberships(price,price_cfg).values())
            )
            size_s = np.average(
                [SIZE_RULES[prof][k] for k in ["Small","Target","Large"]],
                weights=list(size_memberships(size,smin,starget).values())
            )
            rooms_s = np.average(
                [ROOMS_RULES[prof][k] for k in ["TooFew","Target","TooMany"]],
                weights=list(rooms_memberships(rooms,rmin,rlow,rhigh,rmax).values())
            )

            # 3) styl
            style_s = style_score_for_profile(prof, style_raw)

            # 4) fallback – pełny (POI, CENA, M2, ROOMS, ZDJ)
            w = WB[prof]
            num_fb_full = (w["POI"]*poi_score + w["CENA"]*price_s + w["M2"]*size_s +
                           w["ROOMS"]*rooms_s + w["ZDJ"]*style_s)
            fallback_full = num_fb_full / sum(w.values())

            # 5) wynik Sugeno
            final_sugeno = fuzzy_final_attractiveness(
                prof, poi_score, price, price_cfg,
                size, smin, starget,
                rooms, rmin, rlow, rhigh, rmax,
                fallback_full
            )

            # 6) dodatkowy „miękki” wpływ liczby pokoi i stylu
            num_rs = w["ROOMS"]*rooms_s + w["ZDJ"]*style_s
            den_rs = w["ROOMS"] + w["ZDJ"]
            fallback_rs = num_rs / den_rs if den_rs > 0 else 0.0

            alpha = 0.8  # 80% decyzji z reguł, 20% z bloków ROOMS+ZDJ
            final = alpha * final_sugeno + (1.0 - alpha) * fallback_rs

            # 7) wersja float + uint16
            atrak_float = float(round(final, 4))
            atrak_u16 = int(
                np.clip(
                    round(final * 65535.0),
                    0,
                    65535
                )
            )

            # 8) uzasadnienia POI
            contrib = pd.Series({c: W_POI[prof][c]*poi_row[c] for c in cats})
            top_plus = ", ".join(f"{c}(+{contrib[c]:.2f})" for c in contrib.nlargest(3).index)
            top_minus = ", ".join(f"{c}({contrib[c]:.2f})" for c in contrib.nsmallest(3).index)

            rows.append({
                "apt_id": apt_id,
                "profile": prof,
                "POI": round(poi_score,4),
                "CENA": round(price_s,4),
                "M2": round(size_s,4),
                "ROOMS": round(rooms_s,4),
                "ZDJ": round(style_s,4),
                "ATRAKCYJNOSC": atrak_float,
                "ATRAKCYJNOSC_U16": atrak_u16,
                "price_pln_m2": price,
                "size_m2": size,
                "rooms": int(round(rooms)),
                "photo_style": style_raw,
                "STYLE_SCORE": round(style_s,4),
                "city": city,
                "PRICE_TIER": tier,
                "TOP_PLUS": top_plus,
                "TOP_MINUS": top_minus,
            })

    return pd.DataFrame(rows)

# ====== Opisy słowne dla profilu 'uniwersalny' ======
def make_descriptions_uniwersalny(df: pd.DataFrame, out_dir: Path) -> None:
    df_u = df[df["profile"] == "uniwersalny"].copy()
    if df_u.empty:
        return

    def _build(row: pd.Series) -> str:
        try:
            apt_id = str(row["apt_id"])
            profile = str(row["profile"])

            mu_poi = poi_memberships(float(row["POI"]))
            poi_label = _argmax_label(mu_poi)
            poi_desc = POI_DESC.get(poi_label, poi_label)

            city = row.get("city", None)
            tier = get_city_tier(city)
            price_cfg = PriceFuzzyConfig(**PRICE_THRESHOLDS[tier])
            price_val = float(row["price_pln_m2"])
            mu_price = price_memberships(price_val, price_cfg)
            price_label = _argmax_label(mu_price)
            price_desc = PRICE_DESC.get(price_label, price_label)

            smin, starget = SIZE_TARGET["uniwersalny"]
            size_val = float(row["size_m2"])
            mu_size = size_memberships(size_val, smin, starget)
            size_label = _argmax_label(mu_size)
            size_desc = SIZE_DESC.get(size_label, size_label)

            rmin, rlow, rhigh, rmax = ROOMS_TARGET_RANGES["uniwersalny"]
            rooms_val = float(row["rooms"])
            mu_rooms = rooms_memberships(rooms_val, rmin, rlow, rhigh, rmax)
            rooms_label = _argmax_label(mu_rooms)
            rooms_desc = ROOMS_DESC.get(rooms_label, rooms_label)

            style_raw = row.get("photo_style", "")
            if isinstance(style_raw, str) and style_raw.strip():
                skey = style_raw.strip().lower()
                style_text = STYLE_DESC.get(skey, STYLE_DESC["unknown"])
            else:
                style_text = STYLE_DESC["unknown"]

            atrak = float(row.get("ATRAKCYJNOSC", 0.0))

            return (
                f"{apt_id}, {profile}, {style_text}, "
                f"{poi_desc}, {price_desc}, {size_desc}, {rooms_desc}, "
                f"wynik = {atrak:.2f}"
            )
        except Exception:
            return ""

    df_u["INTERPRETACJA"] = df_u.apply(_build, axis=1)
    desc = df_u[["apt_id", "INTERPRETACJA"]].copy()
    desc = desc[desc["INTERPRETACJA"] != ""].drop_duplicates("apt_id")

    if desc.empty:
        return

    out_path = out_dir / "descriptions_uniwersalny.csv"
    desc.to_csv(out_path, index=False, encoding="utf-8")
    print("[OK] Zapisano opisy (uniwersalny):", out_path)

def save_rankings(df, out_dir: Path, topn=30):
    df.to_csv(out_dir/"attractiveness_by_profile.csv",index=False,encoding="utf-8")
    for prof in df["profile"].unique():
        df[df["profile"]==prof].sort_values("ATRAKCYJNOSC",ascending=False)\
            .head(topn).to_csv(out_dir/f"ranking_{prof}.csv",index=False)

    make_descriptions_uniwersalny(df, out_dir)

def main():
    if len(sys.argv)<3:
        print("Użycie: python compute_finattractiveness_profiles.py <apartments_poi_scores.csv> <apartments_out.json>")
        sys.exit(1)

    scores_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    out_dir = scores_path.parent

    scores = load_scores(scores_path)
    apts = load_apartments_from_json(json_path)
    out = compute_blocks_and_final(scores, apts)
    save_rankings(out, out_dir)

    print("[OK] Zapisano wyniki w:")
    print(out_dir/"attractiveness_by_profile.csv")
    for prof in W_POI.keys():
        print(out_dir/f"ranking_{prof}.csv")

if __name__ == "__main__":
    main()
