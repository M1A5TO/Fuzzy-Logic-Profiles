# Final version
from __future__ import annotations

import os
import re
import json
import unicodedata
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List

import numpy as np

# =========================
#  CONFIG
# =========================
TIME_UNIT = os.getenv("TIME_UNIT", "seconds")  # "seconds" | "minutes"
ALPHA = float(os.getenv("ALPHA", "0.9"))  # default higher -> POI-driven

DB_MIN = 0
DB_MAX = 100

DEBUG_APT_ID = os.getenv("DEBUG_APT_ID")  # e.g. "1" or None

PRICE_PENALTY_STEP = float(os.getenv("PRICE_PENALTY_STEP", "2000"))          # PLN/m2 per step
PRICE_PENALTY_PER_STEP = float(os.getenv("PRICE_PENALTY_PER_STEP", "0.01"))  # score in 0..1 subtracted per step
PRICE_PENALTY_MAX = float(os.getenv("PRICE_PENALTY_MAX", "0.08"))            # cap (0..1)

# =========================
#  MAPOWANIE OSM -> LOGIC
# =========================
OSM_TO_LOGIC = {
    "supermarket": "sklep",
    "convenience": "sklep",
    "bakery": "sklep",
    "bus_stop": "przystanek_autobus",
    "tram_stop": "przystanek_tramwaj",
    "clinic_hospital": "szpital_przychodnia",
    "pharmacy": "apteka",
    "playground": "plac_zabaw",
    "kinder_childcare": "szkola_przedszkole",
    "school": "szkola_przedszkole",
    "university": "uczelnia",
    "park": "park",
    "library": "biblioteka",
    "nightclub": "klub",
    "pub": "pub",
    "pet_shop": "sklep_zoologiczny",
    "veterinary": "weterynarz",
    "rail_station": "stacja_kolej_metro",
    "fitness_centre": "silownia",
    "parcel_locker": "paczkomat",
}

# =========================
#  WAGI / REGUŁY
# =========================
HIGH, MID, LOW, ZERO = 1.0, 0.65, 0.35, 0.0

W_POI: Dict[str, Dict[str, float]] = {
    "rodzinny": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "szkola_przedszkole": HIGH, "plac_zabaw": MID, "park": HIGH,
        "biblioteka": MID, "silownia": LOW,
        "weterynarz": MID, "sklep_zoologiczny": MID,
        "klub": -0.5, "pub": -0.3,
        "paczkomat": LOW,
    },
    "studencki": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "uczelnia": HIGH, "biblioteka": HIGH, "silownia": HIGH,
        "klub": HIGH, "pub": HIGH,
        "park": MID,
        "szkola_przedszkole": ZERO, "plac_zabaw": ZERO,
        "paczkomat": MID,
    },
    "singiel": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
         "silownia": HIGH, "klub": HIGH, "pub": HIGH,
        "park": MID, "biblioteka": LOW,
        "szkola_przedszkole": ZERO, "plac_zabaw": ZERO,
        "paczkomat": MID,
    },
    "wlasciciel_psa": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "park": HIGH, "weterynarz": HIGH, "sklep_zoologiczny": HIGH,
        "silownia": LOW, "bank_atm": MID,
        "klub": -0.2, "pub": -0.1,
        "szkola_przedszkole": LOW, "plac_zabaw": LOW, "biblioteka": LOW,
        "paczkomat": LOW,
    },
    "uniwersalny": {
        "szpital_przychodnia": HIGH, "apteka": HIGH, "sklep": HIGH,
        "przystanek_autobus": HIGH, "stacja_kolej_metro": HIGH, "przystanek_tramwaj": HIGH,
        "park": MID, "biblioteka": MID, "silownia": MID,
        "szkola_przedszkole": MID, "plac_zabaw": MID,
        "klub": LOW, "pub": LOW,
        "weterynarz": MID, "sklep_zoologiczny": MID,
        "paczkomat": MID, "uczelnia": MID,
    },
}

# POI-dominant weights (sum ~ 1.0 each profile)
WB: Dict[str, Dict[str, float]] = {
    "rodzinny":       {"POI": 0.55, "CENA": 0.15, "M2": 0.15, "ROOMS": 0.10, "ZDJ": 0.05},
    "studencki":      {"POI": 0.60, "CENA": 0.20, "M2": 0.07, "ROOMS": 0.08, "ZDJ": 0.05},
    "singiel":        {"POI": 0.55, "CENA": 0.20, "M2": 0.10, "ROOMS": 0.10, "ZDJ": 0.05},
    "wlasciciel_psa": {"POI": 0.65, "CENA": 0.12, "M2": 0.10, "ROOMS": 0.08, "ZDJ": 0.05},
    "uniwersalny":    {"POI": 0.55, "CENA": 0.15, "M2": 0.12, "ROOMS": 0.10, "ZDJ": 0.08},
}

EXPENSIVE_CITIES = {"warszawa", "krakow", "gdansk", "wroclaw"}

CITY_ALIASES = {
    "gda??sk": "gdansk",
    "gda�sk": "gdansk",
    "gdańsk": "gdansk",
    "gdansk": "gdansk",

    "wroc??aw": "wroclaw",
    "wroc�aw": "wroclaw",
    "wrocław": "wroclaw",
    "wroclaw": "wroclaw",

    "krak??w": "krakow",
    "krak�w": "krakow",
    "kraków": "krakow",
    "krakow": "krakow",

    "warszawa": "warszawa",
}

def norm_city(city: str | None) -> str:
    if not city:
        return ""
    s = str(city).strip().lower()
    s = re.sub(r"\s+", " ", s)

    if s in CITY_ALIASES:
        return CITY_ALIASES[s]

    s_nfkd = unicodedata.normalize("NFKD", s)
    s_ascii = "".join(ch for ch in s_nfkd if not unicodedata.combining(ch))

    s_ascii = re.sub(r"[^a-z\s\-]", "", s_ascii)
    s_ascii = re.sub(r"[\s\-]+", " ", s_ascii).strip()

    if s_ascii in CITY_ALIASES:
        return CITY_ALIASES[s_ascii]

    return s_ascii

def get_city_tier(city: str | None) -> str:
    # FIX: do not use prefix heuristic (it breaks e.g. "Starogard Gdański")
    c = norm_city(city)
    return "expensive_city" if c in EXPENSIVE_CITIES else "normal_city"

PRICE_THRESHOLDS = {
    "expensive_city": {"cheap_max": 11500, "mid_min": 11500, "mid_max": 14000, "expensive_min": 14000},
    "normal_city":    {"cheap_max": 8500,  "mid_min": 8500,  "mid_max": 11000, "expensive_min": 11000},
}

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
    "singiel":        (1.0, 1.0, 2.0, 3.0),
    "wlasciciel_psa": (2.0, 2.0, 3.0, 4.0),
    "uniwersalny":    (2.0, 2.0, 3.0, 4.0),
}

PRICE_RULES = {
    "rodzinny":       {"Cheap": 1.00, "Mid": 0.60, "Expensive": 0.20},
    "studencki":      {"Cheap": 1.00, "Mid": 0.55, "Expensive": 0.10},
    "singiel":        {"Cheap": 0.90, "Mid": 0.65, "Expensive": 0.25},
    "wlasciciel_psa": {"Cheap": 0.95, "Mid": 0.60, "Expensive": 0.20},
    "uniwersalny":    {"Cheap": 0.95, "Mid": 0.60, "Expensive": 0.25},
}

SIZE_RULES = {
    "rodzinny":       {"Small": 0.20, "Target": 1.00, "Large": 0.70},
    "studencki":      {"Small": 0.30, "Target": 0.95, "Large": 0.60},
    "singiel":        {"Small": 0.35, "Target": 0.90, "Large": 0.65},
    "wlasciciel_psa": {"Small": 0.25, "Target": 1.00, "Large": 0.75},
    "uniwersalny":    {"Small": 0.30, "Target": 0.95, "Large": 0.70},
}

ROOMS_RULES = {
    "rodzinny":       {"TooFew": 0.30, "Target": 1.00, "TooMany": 0.60},
    "studencki":      {"TooFew": 0.55, "Target": 0.95, "TooMany": 0.50},
    "singiel":        {"TooFew": 0.60, "Target": 0.95, "TooMany": 0.50},
    "wlasciciel_psa": {"TooFew": 0.40, "Target": 1.00, "TooMany": 0.60},
    "uniwersalny":    {"TooFew": 0.45, "Target": 0.95, "TooMany": 0.60},
}

# =========================
#  STYLE scoring
# =========================
STYLE_SCORES: Dict[str, Dict[str, float]] = {
    "rodzinny": {
        "MODERN": 0.85, "CLASSIC": 0.90, "INDUSTRIAL": 0.55, "SCANDINAVIAN": 0.92,
        "MINIMALIST": 0.80, "VINTAGE": 0.70, "OTHER": 0.75, "UNKNOWN": 0.78,
    },
    "studencki": {
        "MODERN": 0.88, "CLASSIC": 0.70, "INDUSTRIAL": 0.92, "SCANDINAVIAN": 0.80,
        "MINIMALIST": 0.86, "VINTAGE": 0.78, "OTHER": 0.75, "UNKNOWN": 0.78,
    },
    "singiel": {
        "MODERN": 0.92, "CLASSIC": 0.78, "INDUSTRIAL": 0.88, "SCANDINAVIAN": 0.86,
        "MINIMALIST": 0.90, "VINTAGE": 0.82, "OTHER": 0.78, "UNKNOWN": 0.80,
    },
    "wlasciciel_psa": {
        "MODERN": 0.82, "CLASSIC": 0.86, "INDUSTRIAL": 0.65, "SCANDINAVIAN": 0.90,
        "MINIMALIST": 0.78, "VINTAGE": 0.74, "OTHER": 0.76, "UNKNOWN": 0.78,
    },
    "uniwersalny": {
        "MODERN": 0.88, "CLASSIC": 0.85, "INDUSTRIAL": 0.75, "SCANDINAVIAN": 0.90,
        "MINIMALIST": 0.86, "VINTAGE": 0.80, "OTHER": 0.78, "UNKNOWN": 0.80,
    },
}

def norm_style(style: Any) -> str:
    if style is None:
        return "UNKNOWN"
    s = str(style).strip().upper()
    if not s:
        return "UNKNOWN"
    if s in {"MODERN", "CLASSIC", "INDUSTRIAL", "SCANDINAVIAN", "MINIMALIST", "VINTAGE", "OTHER"}:
        return s
    return "OTHER"

PROFILE_TO_BACKEND_FIELD = {
    "studencki": "student_attractiveness",
    "singiel": "single_attractiveness",
    "wlasciciel_psa": "dog_owner_attractiveness",
    "rodzinny": "family_attractiveness",
    "uniwersalny": "universal_attractiveness",
}

# =========================
#  FUZZY HELPERS
# =========================
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def parse_time_to_minutes(t: float, unit: str) -> float:
    t = float(t)
    return t / 60.0 if unit == "seconds" else t

def time_fuzzy(t_min: float) -> float:
    t = float(t_min)
    if t <= 5:
        return 1.0
    if t <= 10:
        return 1.0 - 0.4 * ((t - 5) / 5)
    if t <= 15:
        return 0.6 - 0.3 * ((t - 10) / 5)
    return 0.0

def weighted_mean_signed(values, weights, eps: float = 1e-9) -> float:
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    den = float(np.abs(w).sum())
    if den < eps:
        return 0.0
    return float((v * w).sum() / den)

@dataclass
class PriceFuzzyConfig:
    cheap_max: float
    mid_min: float
    mid_max: float
    expensive_min: float

def price_memberships(price: float, cfg: PriceFuzzyConfig) -> Dict[str, float]:
    x = float(price)

    if x <= cfg.cheap_max:
        cheap = 1.0
    elif x >= cfg.mid_min:
        cheap = 0.0
    else:
        cheap = (cfg.mid_min - x) / (cfg.mid_min - cfg.cheap_max)

    if x <= cfg.cheap_max or x >= cfg.expensive_min:
        mid = 0.0
    elif cfg.mid_min <= x <= cfg.mid_max:
        mid = 1.0
    elif x < cfg.mid_min:
        mid = (x - cfg.cheap_max) / (cfg.mid_min - cfg.cheap_max)
    else:
        mid = (cfg.expensive_min - x) / (cfg.expensive_min - cfg.mid_max)

    if x <= cfg.mid_max:
        exp = 0.0
    elif x >= cfg.expensive_min:
        exp = 1.0
    else:
        exp = (x - cfg.mid_max) / (cfg.expensive_min - cfg.mid_max)

    return {"Cheap": _clamp01(cheap), "Mid": _clamp01(mid), "Expensive": _clamp01(exp)}

def size_memberships(size: float, s_min: float, s_target: float):
    s_max = s_target * 1.35

    if size <= s_min:
        small = 1.0
    elif size < s_target:
        small = (s_target - size) / (s_target - s_min)
    else:
        small = 0.0

    if size <= s_target:
        large = 0.0
    elif size < s_max:
        large = (size - s_target) / (s_max - s_target)
    else:
        large = 1.0

    if size <= s_min or size >= s_max:
        target = 0.0
    elif size == s_target:
        target = 1.0
    elif size < s_target:
        target = (size - s_min) / (s_target - s_min)
    else:
        target = (s_max - size) / (s_max - s_target)

    return {"Small": _clamp01(small), "Target": _clamp01(target), "Large": _clamp01(large)}

def rooms_memberships(rooms: float, rmin, rlow, rhigh, rmax):
    x = float(rooms)

    if x <= rmin:
        few = 1.0
    elif x < rlow:
        few = (rlow - x) / (rlow - rmin)
    else:
        few = 0.0

    if x <= rmin or x >= rmax:
        tgt = 0.0
    elif x < rlow:
        tgt = (x - rmin) / (rlow - rmin)
    elif x <= rhigh:
        tgt = 1.0
    else:
        tgt = (rmax - x) / (rmax - rhigh)

    if x <= rhigh:
        many = 0.0
    elif x < rmax:
        many = (x - rhigh) / (rmax - rhigh)
    else:
        many = 1.0

    return {"TooFew": _clamp01(few), "Target": _clamp01(tgt), "TooMany": _clamp01(many)}

def poi_memberships(x: float):
    x = _clamp01(x)

    if x <= 0:
        low = 1.0
    elif x < 0.5:
        low = (0.5 - x) / 0.5
    else:
        low = 0.0

    if x <= 0 or x >= 1:
        mid = 0.0
    elif x == 0.5:
        mid = 1.0
    elif x < 0.5:
        mid = x / 0.5
    else:
        mid = (1 - x) / 0.5

    if x <= 0.5:
        high = 0.0
    else:
        high = (x - 0.5) / 0.5

    return {"Low": low, "Mid": mid, "High": high}

def price_excess_penalty(price_m2: float, cfg: PriceFuzzyConfig) -> float:
    x = float(price_m2)
    if x <= cfg.expensive_min:
        return 0.0

    excess = x - float(cfg.expensive_min)
    step = max(1.0, float(PRICE_PENALTY_STEP))
    per = max(0.0, float(PRICE_PENALTY_PER_STEP))

    penalty = (excess / step) * per
    return float(np.clip(penalty, 0.0, float(PRICE_PENALTY_MAX)))

# =========================
#  OPISY (ENUM) - zawsze wg "uniwersalny"
# =========================
def _label_from_mu(mu: Dict[str, float], order: tuple[str, ...]) -> str:
    best = order[0]
    best_v = float(mu.get(best, 0.0))
    for k in order[1:]:
        v = float(mu.get(k, 0.0))
        if v > best_v:
            best, best_v = k, v
    return best

def make_poi_desc_enum(poi_score: float) -> str:
    mu = poi_memberships(_clamp01(float(poi_score)))
    lab = _label_from_mu(mu, ("High", "Mid", "Low"))
    if lab == "High":
        return "HIGH"
    if lab == "Mid":
        return "MEDIUM"
    return "LOW"

def make_price_desc_enum(price_m2: float, price_cfg: PriceFuzzyConfig) -> str:
    mu = price_memberships(price_m2, price_cfg)
    lab = _label_from_mu(mu, ("Cheap", "Mid", "Expensive"))
    if lab == "Cheap":
        return "CHEAP"
    if lab == "Mid":
        return "AVERAGE"
    return "EXPENSIVE"

def make_size_desc_enum(size_m2: float, smin: float, starget: float) -> str:
    mu = size_memberships(size_m2, smin, starget)
    lab = _label_from_mu(mu, ("Target", "Large", "Small"))
    if lab == "Target":
        return "MEDIUM"
    if lab == "Large":
        return "LARGE"
    return "SMALL"

# =========================
#  FINAL RULES (Sugeno)
# =========================
Rule = Tuple[Dict[str, str], float]
FINAL_RULES: Dict[str, List[Rule]] = {
    "rodzinny": [
        ({"POI": "High", "PRICE": "Cheap", "SIZE": "Target", "ROOMS": "Target"}, 1.00),
        ({"POI": "High", "PRICE": "Mid",   "SIZE": "Target", "ROOMS": "Target"}, 0.92),
        ({"POI": "High", "ROOMS": "TooFew"}, 0.55),
        ({"POI": "High", "SIZE": "Small"}, 0.60),
        ({"POI": "Mid",  "PRICE": "Cheap", "SIZE": "Target", "ROOMS": "Target"}, 0.85),
        ({"PRICE": "Expensive", "POI": "Low"}, 0.20),
        ({"POI": "High"}, 0.75),
        ({"POI": "Low"}, 0.35),
    ],
    "studencki": [
        ({"POI": "High", "PRICE": "Cheap", "ROOMS": "Target"}, 1.00),
        ({"POI": "High", "PRICE": "Mid",   "ROOMS": "Target"}, 0.90),
        ({"POI": "Mid",  "PRICE": "Cheap", "ROOMS": "Target"}, 0.86),
        ({"POI": "High", "ROOMS": "TooMany"}, 0.60),
        ({"PRICE": "Expensive", "POI": "Low"}, 0.15),
        ({"POI": "High"}, 0.80),
        ({"POI": "Low"}, 0.30),
    ],
    "singiel": [
        ({"POI": "High", "PRICE": "Cheap", "ROOMS": "Target"}, 0.97),
        ({"POI": "High", "PRICE": "Mid",   "ROOMS": "Target"}, 0.88),
        ({"POI": "High", "SIZE": "Target", "ROOMS": "Target"}, 0.90),
        ({"POI": "Low", "PRICE": "Expensive"}, 0.20),
        ({"POI": "High", "ROOMS": "TooMany"}, 0.60),
        ({"POI": "High"}, 0.78),
        ({"POI": "Low"}, 0.35),
    ],
    "wlasciciel_psa": [
        ({"POI": "High", "PRICE": "Cheap", "SIZE": "Large", "ROOMS": "Target"}, 1.00),
        ({"POI": "High", "PRICE": "Mid",   "SIZE": "Large", "ROOMS": "Target"}, 0.90),
        ({"POI": "High", "SIZE": "Target", "ROOMS": "Target"}, 0.88),
        ({"POI": "Low", "PRICE": "Expensive"}, 0.20),
        ({"POI": "High"}, 0.80),
        ({"POI": "Low"}, 0.30),
    ],
    "uniwersalny": [
        ({"POI": "High", "PRICE": "Cheap", "SIZE": "Target", "ROOMS": "Target"}, 0.98),
        ({"POI": "High", "PRICE": "Mid",   "SIZE": "Target", "ROOMS": "Target"}, 0.90),
        ({"POI": "Mid",  "PRICE": "Cheap", "ROOMS": "Target"}, 0.82),
        ({"POI": "Low", "PRICE": "Expensive"}, 0.22),
        ({"POI": "High"}, 0.76),
        ({"POI": "Low"}, 0.35),
    ],
}

def rule_activation(cond, mu_poi, mu_price, mu_size, mu_rooms):
    mus = []
    if "POI" in cond:
        mus.append(mu_poi[cond["POI"]])
    if "PRICE" in cond:
        mus.append(mu_price[cond["PRICE"]])
    if "SIZE" in cond:
        mus.append(mu_size[cond["SIZE"]])
    if "ROOMS" in cond:
        mus.append(mu_rooms[cond["ROOMS"]])
    return min(mus) if mus else 0.0

def fuzzy_final_attractiveness(
    profile: str,
    poi_score: float,
    price: float,
    price_cfg: PriceFuzzyConfig,
    size: float,
    s_min: float,
    s_target: float,
    rooms: float,
    rmin: float,
    rlow: float,
    rhigh: float,
    rmax: float,
    fallback: float,
) -> float:
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
    return fallback if den == 0 else (num / den)

def score_to_db_int(x01: float) -> int:
    val = int(round(float(x01) * DB_MAX))
    return int(np.clip(val, DB_MIN, DB_MAX))

# =========================
#  OFFLINE: load POI relations with time_to_poi
# =========================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_poi_relations(poi_rel_files: List[str]) -> Dict[int, List[dict]]:
    """
    Expected format per file: list[ { apartment_id, time_to_poi, poi:{category,...} or category } ]
    """
    rels_by_apt: Dict[int, List[dict]] = {}
    for p in poi_rel_files:
        data = load_json(p)
        if not isinstance(data, list):
            raise ValueError(f"{p}: expected a JSON list")
        for rel in data:
            apt_id = int(rel.get("apartment_id"))
            rels_by_apt.setdefault(apt_id, []).append(rel)
    return rels_by_apt

def compute_poi_score_for_profile_offline(apt_id: int, profile: str, rels_by_apt: Dict[int, List[dict]]) -> float:
    """
    Offline POI score.
    Penalizes missing categories by including all W_POI[profile] categories with feat=0.0.
    """
    rels = rels_by_apt.get(int(apt_id), []) or []
    mapped_best: Dict[str, float] = {}

    for rel in rels:
        poi_obj = rel.get("poi") if isinstance(rel, dict) else None
        osm_cat = poi_obj.get("category") if isinstance(poi_obj, dict) else None
        if osm_cat is None:
            osm_cat = rel.get("category")

        if osm_cat not in OSM_TO_LOGIC:
            continue

        logic_cat = OSM_TO_LOGIC[osm_cat]
        t_raw = rel.get("time_to_poi")
        if t_raw is None:
            continue

        t_min = parse_time_to_minutes(t_raw, TIME_UNIT)
        feat = time_fuzzy(t_min)
        if feat <= 0.0:
            continue

        mapped_best[logic_cat] = max(mapped_best.get(logic_cat, 0.0), float(feat))

    vals: List[float] = []
    wts: List[float] = []
    for cat, w in W_POI[profile].items():
        vals.append(float(mapped_best.get(cat, 0.0)))
        wts.append(float(w))

    return weighted_mean_signed(vals, wts)

def compute_scores_offline(apt: dict, rels_by_apt: Dict[int, List[dict]]) -> Dict[str, Any]:
    apt_id = int(apt["id"])
    debug = (DEBUG_APT_ID is not None and str(apt_id) == str(DEBUG_APT_ID))

    price_m2 = float(apt.get("price_per_m2") or 0.0)
    size_m2 = float(apt.get("footage") or 0.0)
    rooms = float(apt.get("room_num") or 0.0)
    city = apt.get("city")

    tier = get_city_tier(city)
    price_cfg = PriceFuzzyConfig(**PRICE_THRESHOLDS[tier])

    # --- OPISY: zawsze wg profilu "uniwersalny" ---
    poi_u = compute_poi_score_for_profile_offline(apt_id, "uniwersalny", rels_by_apt)
    smin_u, starget_u = SIZE_TARGET["uniwersalny"]

    poi_desc = make_poi_desc_enum(poi_u)
    price_desc = make_price_desc_enum(price_m2, price_cfg)
    size_desc = make_size_desc_enum(size_m2, smin_u, starget_u)

    out: Dict[str, Any] = {}
    style_key = norm_style(apt.get("style"))

    for prof, field in PROFILE_TO_BACKEND_FIELD.items():
        poi_score = compute_poi_score_for_profile_offline(apt_id, prof, rels_by_apt)

        smin, starget = SIZE_TARGET[prof]
        rmin, rlow, rhigh, rmax = ROOMS_TARGET_RANGES[prof]

        mu_price = price_memberships(price_m2, price_cfg)
        mu_size = size_memberships(size_m2, smin, starget)
        mu_rooms = rooms_memberships(rooms, rmin, rlow, rhigh, rmax)

        price_s = float(np.average(
            [PRICE_RULES[prof][k] for k in ["Cheap", "Mid", "Expensive"]],
            weights=[mu_price["Cheap"], mu_price["Mid"], mu_price["Expensive"]],
        ))
        size_s = float(np.average(
            [SIZE_RULES[prof][k] for k in ["Small", "Target", "Large"]],
            weights=[mu_size["Small"], mu_size["Target"], mu_size["Large"]],
        ))
        rooms_s = float(np.average(
            [ROOMS_RULES[prof][k] for k in ["TooFew", "Target", "TooMany"]],
            weights=[mu_rooms["TooFew"], mu_rooms["Target"], mu_rooms["TooMany"]],
        ))

        style_s = float(STYLE_SCORES.get(prof, {}).get(style_key, STYLE_SCORES.get(prof, {}).get("UNKNOWN", 0.8)))

        w = WB[prof]
        wsum = float(sum(w.values()))

        fallback_full = (
            w["POI"] * poi_score
            + w["CENA"] * price_s
            + w["M2"] * size_s
            + w["ROOMS"] * rooms_s
            + w["ZDJ"] * style_s
        ) / wsum

        final_sugeno = fuzzy_final_attractiveness(
            prof, poi_score, price_m2, price_cfg,
            size_m2, smin, starget,
            rooms, rmin, rlow, rhigh, rmax,
            fallback_full
        )

        # FIX: mix with fallback_full (not ROOMS+STYLE only) -> POI still dominates
        final01 = float(ALPHA * final_sugeno + (1.0 - ALPHA) * fallback_full)

        pen = price_excess_penalty(price_m2, price_cfg)
        final01 = float(np.clip(final01 - pen, 0.0, 1.0))

        if debug:
            print(f"\n[DBG] apt={apt_id} prof={prof} tier={tier} city={city!r}")
            print(f"  price_m2={price_m2:.0f} size_m2={size_m2:.2f} rooms={rooms:.1f}")
            print(f"  style_raw={apt.get('style')!r} style_key={style_key} style_s={style_s:.3f}")
            print(f"  poi_score={poi_score:.3f}")
            print(f"  mu_price={mu_price}  price_s={price_s:.3f}")
            print(f"  size_s={size_s:.3f} rooms_s={rooms_s:.3f}")
            print(f"  fallback_full={fallback_full:.3f} final_sugeno={final_sugeno:.3f}")
            print(f"  penalty={pen:.3f} ALPHA={ALPHA:.2f} final01={final01:.3f} db={score_to_db_int(final01)}")
            print(f"  enum_desc: poi_desc={poi_desc} price_desc={price_desc} size_desc={size_desc}")

        out[field] = score_to_db_int(final01)

    out["poi_desc"] = poi_desc
    out["price_desc"] = price_desc
    out["size_desc"] = size_desc
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Offline apartment scoring (POI-dominant).")
    ap.add_argument("--apartments", required=True, help="Path to apartments JSON (list).")
    ap.add_argument(
        "--poi-rels", nargs="+", required=True,
        help="Paths to POI relation JSON files (lists with time_to_poi)."
    )
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    apartments = load_json(args.apartments)
    if not isinstance(apartments, list):
        raise ValueError("apartments JSON must be a list")

    rels_by_apt = build_poi_relations(args.poi_rels)

    updated = []
    for apt in apartments:
        if not isinstance(apt, dict) or "id" not in apt:
            continue
        scores = compute_scores_offline(apt, rels_by_apt)
        apt2 = dict(apt)
        apt2.update(scores)
        updated.append(apt2)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print(f"[OK] scored {len(updated)} apartments -> {args.out}")

if __name__ == "__main__":
    main()
