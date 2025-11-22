# save as: make_explained_rankings.py
# Usage:
#   python make_explained_rankings.py C:\Users\antek\PycharmProjects\PythonProjectMiasto\data\attractiveness_by_profile.csv
#
# Wejście: attractiveness_by_profile.csv (z compute_finattractiveness_profiles.py)
#   wymagane kolumny (per wiersz = mieszkanie × profil):
#     profile, POI, price_pln_m2, size_m2, ATRAKCYJNOSC
#   opcjonalne:
#     rooms        – jeśli brak lub NaN, zostanie oszacowane z size_m2
#     photo_style  – "old"/"modern"/inne
#     STYLE_SCORE  – liczba w [0,1] opisująca jakość stylu (używana tylko opisowo)

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

Rule = Tuple[Dict[str, str], float]

# ====== FINAL RULES (jak w compute_finattractiveness_profiles.py — z ROOMS, bez PHOTOS) ======
FINAL_RULES: Dict[str, List[Rule]] = {
    "rodzinny": [
        ({"POI": "High", "PRICE": "Cheap", "SIZE": "Target", "ROOMS": "Target"}, 1.00),
        ({"POI": "High", "PRICE": "Mid",  "SIZE": "Target", "ROOMS": "Target"}, 0.92),
        ({"POI": "High", "ROOMS": "TooFew"},                                 0.55),
        ({"POI": "High", "SIZE": "Small"},                                   0.60),
        ({"POI": "Mid",  "PRICE": "Cheap", "SIZE": "Target", "ROOMS": "Target"}, 0.85),
        ({"PRICE": "Expensive", "POI": "Low"},                               0.20),
        ({"POI": "High"},                                                    0.75),
        ({"POI": "Low"},                                                     0.35),
    ],
    "studencki": [
        ({"POI": "High", "PRICE": "Cheap", "ROOMS": "Target"},               1.00),
        ({"POI": "High", "PRICE": "Mid",   "ROOMS": "Target"},               0.90),
        ({"POI": "Mid",  "PRICE": "Cheap", "ROOMS": "Target"},               0.86),
        ({"POI": "High", "ROOMS": "TooMany"},                                0.60),
        ({"PRICE": "Expensive", "POI": "Low"},                               0.15),
        ({"POI": "High"},                                                    0.80),
        ({"POI": "Low"},                                                     0.30),
    ],
    "singiel": [
        ({"POI": "High", "PRICE": "Cheap", "ROOMS": "Target"},               0.97),
        ({"POI": "High", "PRICE": "Mid",   "ROOMS": "Target"},               0.88),
        ({"POI": "High", "SIZE": "Target", "ROOMS": "Target"},               0.90),
        ({"POI": "Low",  "PRICE": "Expensive"},                              0.20),
        ({"POI": "High", "ROOMS": "TooMany"},                                0.60),
        ({"POI": "High"},                                                    0.78),
        ({"POI": "Low"},                                                     0.35),
    ],
    "wlasciciel_psa": [
        ({"POI": "High", "PRICE": "Cheap", "SIZE": "Large",  "ROOMS": "Target"}, 1.00),
        ({"POI": "High", "PRICE": "Mid",   "SIZE": "Large",  "ROOMS": "Target"}, 0.90),
        ({"POI": "High", "SIZE": "Target", "ROOMS": "Target"},                 0.88),
        ({"POI": "Low",  "PRICE": "Expensive"},                                0.20),
        ({"POI": "High"},                                                      0.80),
        ({"POI": "Low"},                                                       0.30),
    ],
    "uniwersalny": [
        ({"POI": "High", "PRICE": "Cheap", "SIZE": "Target", "ROOMS": "Target"}, 0.98),
        ({"POI": "High", "PRICE": "Mid",   "SIZE": "Target", "ROOMS": "Target"}, 0.90),
        ({"POI": "Mid",  "PRICE": "Cheap", "ROOMS": "Target"},                  0.82),
        ({"POI": "Low",  "PRICE": "Expensive"},                                 0.22),
        ({"POI": "High"},                                                       0.76),
        ({"POI": "Low"},                                                        0.35),
    ],
}

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

ROOMS_RULES = {
    "rodzinny":       {"TooFew": 0.30, "Target": 1.00, "TooMany": 0.60},
    "studencki":      {"TooFew": 0.55, "Target": 0.95, "TooMany": 0.50},
    "singiel":        {"TooFew": 0.60, "Target": 1.00, "TooMany": 0.50},
    "wlasciciel_psa": {"TooFew": 0.40, "Target": 1.00, "TooMany": 0.60},
    "uniwersalny":    {"TooFew": 0.45, "Target": 0.95, "TooMany": 0.60},
}

# ====== Fuzzy helpery (spójne z compute_finattractiveness_profiles.py) ======
def _clamp01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

def poi_memberships(poi_score: float):
    x = _clamp01(float(poi_score))
    if x <= 0.0:
        low = 1.0
    elif x < 0.5:
        low = (0.5 - x) / 0.5
    else:
        low = 0.0

    if x <= 0.0 or x >= 1.0:
        mid = 0.0
    elif x == 0.5:
        mid = 1.0
    elif x < 0.5:
        mid = (x - 0.0) / 0.5
    else:
        mid = (1.0 - x) / 0.5

    if x <= 0.5:
        high = 0.0
    else:
        high = (x - 0.5) / 0.5

    return {"Low": _clamp01(low), "Mid": _clamp01(mid), "High": _clamp01(high)}

def price_memberships(price: float, p10: float, p50: float, p95: float):
    if price <= p10:
        cheap = 1.0
    elif price < p50:
        cheap = (p50 - price) / (p50 - p10)
    else:
        cheap = 0.0

    if price <= p50:
        expensive = 0.0
    elif price < p95:
        expensive = (price - p50) / (p95 - p50)
    else:
        expensive = 1.0

    if price <= p10 or price >= p95:
        mid = 0.0
    elif price == p50:
        mid = 1.0
    elif price < p50:
        mid = (price - p10) / (p50 - p10)
    else:
        mid = (p95 - price) / (p95 - p50)

    return {
        "Cheap": _clamp01(cheap),
        "Mid": _clamp01(mid),
        "Expensive": _clamp01(expensive),
    }

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

    return {
        "Small": _clamp01(small),
        "Target": _clamp01(target),
        "Large": _clamp01(large),
    }

def rooms_memberships(rooms: float, rmin: float, rlow: float, rhigh: float, rmax: float):
    try:
        x = float(rooms)
    except Exception:
        x = np.nan
    if np.isnan(x):
        return {"TooFew": 0.0, "Target": 0.0, "TooMany": 0.0}

    # TooFew
    if x <= rmin:
        few = 1.0
    elif x < rlow:
        few = (rlow - x) / max(rlow - rmin, 1e-9)
    else:
        few = 0.0

    # Target (trapez z plateau)
    if rlow == rhigh:
        # przypadek "igły" (np. singiel: dokładnie 1 pokój)
        if x == rlow:
            target = 1.0
        elif x < rlow and rlow > rmin:
            target = (x - rmin) / max(rlow - rmin, 1e-9)
        elif x > rhigh and rmax > rhigh:
            target = (rmax - x) / max(rmax - rhigh, 1e-9)
        else:
            target = 0.0
    else:
        if x <= rmin or x >= rmax:
            target = 0.0
        elif x < rlow:
            target = (x - rmin) / max(rlow - rmin, 1e-9)
        elif x <= rhigh:
            target = 1.0
        else:
            target = (rmax - x) / max(rmax - rhigh, 1e-9)

    # TooMany
    if x <= rhigh:
        many = 0.0
    elif x < rmax:
        many = (x - rhigh) / max(rmax - rhigh, 1e-9)
    else:
        many = 1.0

    return {
        "TooFew":  float(max(0.0, min(1.0, few))),
        "Target":  float(max(0.0, min(1.0, target))),
        "TooMany": float(max(0.0, min(1.0, many))),
    }

def rule_activation(
    rule_cond: Dict[str, str],
    mu_poi: Dict[str, float],
    mu_price: Dict[str, float],
    mu_size: Dict[str, float],
    mu_rooms: Dict[str, float],
) -> float:
    mus = []
    if "POI"   in rule_cond: mus.append(mu_poi.get(rule_cond["POI"], 0.0))
    if "PRICE" in rule_cond: mus.append(mu_price.get(rule_cond["PRICE"], 0.0))
    if "SIZE"  in rule_cond: mus.append(mu_size.get(rule_cond["SIZE"], 0.0))
    if "ROOMS" in rule_cond: mus.append(mu_rooms.get(rule_cond["ROOMS"], 0.0))
    return float(min(mus)) if mus else 0.0

# ====== Pomocnicze: heurystyka do rooms, gdy brak w CSV ======
def _estimate_rooms_from_size(size_m2: float) -> float:
    """Zgrubna estymacja jak w compute_finattractiveness_profiles.fill_missing_attributes."""
    try:
        s = float(size_m2)
    except Exception:
        return np.nan
    if np.isnan(s):
        return np.nan
    if s <= 28:
        return 1.0
    if s <= 45:
        return 2.0
    if s <= 65:
        return 3.0
    if s <= 85:
        return 4.0
    return 5.0

# ====== Główna funkcja ======
def main(in_csv: Path, out_dir: Path):
    df = pd.read_csv(in_csv)

    # Minimalne kolumny (bez photos_count – zdjęcia nas nie obchodzą ilościowo)
    needed = {"profile", "POI", "price_pln_m2", "size_m2", "ATRAKCYJNOSC"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn w {in_csv.name}: {sorted(missing)}")

    # Kwantyle ceny (globalnie — spójnie z compute_finattractiveness_profiles.py)
    p10 = float(np.percentile(df["price_pln_m2"], 10))
    p50 = float(np.percentile(df["price_pln_m2"], 50))
    p95 = float(np.percentile(df["price_pln_m2"], 95))

    # rooms: jeśli brak/NaN → estymacja z metrażu
    if "rooms" not in df.columns:
        df["rooms"] = np.nan
    mask_nan_rooms = df["rooms"].isna()
    if mask_nan_rooms.any():
        df.loc[mask_nan_rooms, "rooms"] = df.loc[mask_nan_rooms, "size_m2"].apply(_estimate_rooms_from_size)

    for profile, rules in FINAL_RULES.items():
        if profile not in df["profile"].unique():
            continue

        smin, starget = SIZE_TARGET[profile]
        rmin, rlow, rhigh, rmax = ROOMS_TARGET_RANGES[profile]

        sub = df[df["profile"] == profile].copy()
        explained = []

        for _, row in sub.iterrows():
            mu_poi   = poi_memberships(row["POI"])
            mu_price = price_memberships(row["price_pln_m2"], p10, p50, p95)
            mu_size  = size_memberships(row["size_m2"], smin, starget)
            mu_rooms = rooms_memberships(row.get("rooms", np.nan), rmin, rlow, rhigh, rmax)

            lines = []
            for cond, const in rules:
                alpha = rule_activation(cond, mu_poi, mu_price, mu_size, mu_rooms)
                # bierzemy tylko sensownie "silne" reguły
                if alpha > 0.2:
                    parts = ", ".join(f"{k}={v}" for k, v in cond.items())
                    lines.append(f"{parts} → {const:.2f} (α={alpha:.2f})")

            row_out = row.copy()

            expl = "; ".join(lines) if lines else "(brak silnych reguł)"

            # Dołącz opis stylu, jeśli dostępny
            if "photo_style" in row_out.index and "STYLE_SCORE" in row_out.index:
                style = str(row_out["photo_style"])
                style_score = row_out["STYLE_SCORE"]
                if style and style.lower() != "nan":
                    try:
                        style_score_f = float(style_score)
                        style_part = f" styl mieszkania: {style} (ocena stylu={style_score_f:.2f})"
                    except Exception:
                        style_part = f" styl mieszkania: {style}"
                    if expl and expl != "(brak silnych reguł)":
                        expl = expl + "; " + style_part
                    else:
                        expl = style_part

            row_out["EXPLANATION"] = expl
            explained.append(row_out)

        sub_exp = pd.DataFrame(explained)
        # TOP-5 i BOTTOM-5 (dla szybkiego przeglądu)
        top5 = sub_exp.sort_values("ATRAKCYJNOSC", ascending=False).head(5)
        bottom5 = sub_exp.sort_values("ATRAKCYJNOSC", ascending=True).head(5)
        out = pd.concat([top5, bottom5], ignore_index=True)

        out_path = out_dir / f"ranking_{profile}_explained.csv"
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_explained_rankings.py /path/to/attractiveness_by_profile.csv [/optional/output_dir]")
        sys.exit(1)
    in_csv = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else in_csv.parent
    main(in_csv, out_dir)
