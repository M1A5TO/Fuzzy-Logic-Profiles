# save as: make_explained_rankings.py
# Usage:
#   python make_explained_rankings.py C:\Users\antek\PycharmProjects\PythonProjectMiasto\data\attractiveness_by_profile.csv

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

Rule = Tuple[Dict[str, str], float]

# ====== FINAL RULES (takie jak w compute_attractiveness_profiles.py) ======
FINAL_RULES: Dict[str, List[Rule]] = {
    "rodzinny": [
        ({"POI":"High","PRICE":"Cheap","SIZE":"Target"}, 1.00),
        ({"POI":"High","PRICE":"Mid","SIZE":"Target"},   0.90),
        ({"POI":"High","PRICE":"Cheap","SIZE":"Large"},  0.85),
        ({"POI":"Mid","PRICE":"Cheap","SIZE":"Target"},  0.80),
        ({"POI":"High","SIZE":"Small"},                  0.60),
        ({"PRICE":"Expensive","POI":"Low"},              0.20),
        ({"POI":"Mid","PRICE":"Expensive","SIZE":"Small"}, 0.30),
        ({"POI":"High","PRICE":"Cheap"},                 0.88),
        ({"POI":"High"},                                 0.75),
        ({"POI":"Low"},                                  0.35),
    ],
    "studencki": [
        ({"POI":"High","PRICE":"Cheap"},                 1.00),
        ({"POI":"High","PRICE":"Mid"},                   0.90),
        ({"POI":"Mid","PRICE":"Cheap"},                  0.85),
        ({"PRICE":"Expensive","POI":"Low"},              0.15),
        ({"POI":"High","SIZE":"Small"},                  0.70),
        ({"POI":"Mid","SIZE":"Large"},                   0.55),
        ({"POI":"High"},                                 0.80),
        ({"POI":"Low"},                                  0.30),
    ],
    "singiel": [
        ({"POI":"High","PRICE":"Cheap"},                 0.95),
        ({"POI":"High","PRICE":"Mid"},                   0.85),
        ({"POI":"High","SIZE":"Target"},                 0.90),
        ({"POI":"Mid","PRICE":"Cheap"},                  0.80),
        ({"POI":"Low","PRICE":"Expensive"},              0.20),
        ({"POI":"High","SIZE":"Small"},                  0.85),
        ({"POI":"High"},                                 0.78),
        ({"POI":"Low"},                                  0.35),
    ],
    "wlasciciel_psa": [
        ({"POI":"High","PRICE":"Cheap","SIZE":"Large"},  1.00),
        ({"POI":"High","PRICE":"Mid","SIZE":"Large"},    0.90),
        ({"POI":"High","SIZE":"Target"},                 0.88),
        ({"POI":"Mid","PRICE":"Cheap","SIZE":"Target"},  0.80),
        ({"POI":"Low","PRICE":"Expensive"},              0.20),
        ({"POI":"High"},                                 0.80),
        ({"POI":"Low"},                                  0.30),
    ],
    "uniwersalny": [
        ({"POI":"High","PRICE":"Cheap","SIZE":"Target"}, 0.98),
        ({"POI":"High","PRICE":"Mid","SIZE":"Target"},   0.90),
        ({"POI":"Mid","PRICE":"Cheap"},                  0.80),
        ({"POI":"Low","PRICE":"Expensive"},              0.22),
        ({"POI":"High"},                                 0.76),
        ({"POI":"Low"},                                  0.35),
    ],
}

SIZE_TARGET: Dict[str, Tuple[float, float]] = {
    "rodzinny": (25, 65),
    "studencki": (20, 45),
    "singiel": (20, 45),
    "wlasciciel_psa": (25, 55),
    "uniwersalny": (22, 55),
}

# ====== Fuzzy helpery (identyczne jak w compute_... ) ======
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

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

def rule_activation(rule_cond: Dict[str, str],
                    mu_poi: Dict[str,float],
                    mu_price: Dict[str,float],
                    mu_size: Dict[str,float],
                    mu_photos: Dict[str,float]) -> float:
    mus = []
    if "POI" in rule_cond: mus.append(mu_poi.get(rule_cond["POI"], 0.0))
    if "PRICE" in rule_cond: mus.append(mu_price.get(rule_cond["PRICE"], 0.0))
    if "SIZE" in rule_cond: mus.append(mu_size.get(rule_cond["SIZE"], 0.0))
    if "PHOTOS" in rule_cond: mus.append(mu_photos.get(rule_cond["PHOTOS"], 0.0))
    return float(min(mus)) if mus else 0.0

# ====== Główna funkcja ======
def main(in_csv: Path, out_dir: Path):
    df = pd.read_csv(in_csv)

    # Kwantyle ceny (na bazie całej tabeli, spójnie z compute_...)
    p10 = float(np.percentile(df["price_pln_m2"], 10))
    p50 = float(np.percentile(df["price_pln_m2"], 50))
    p95 = float(np.percentile(df["price_pln_m2"], 95))

    for profile, rules in FINAL_RULES.items():
        smin, starget = SIZE_TARGET[profile]
        sub = df[df["profile"] == profile].copy()
        explained = []

        for _, row in sub.iterrows():
            mu_poi    = poi_memberships(row["POI"])
            mu_price  = price_memberships(row["price_pln_m2"], p10, p50, p95)
            mu_size   = size_memberships(row["size_m2"], smin, starget)
            mu_photos = photos_memberships(row["photos_count"])

            lines = []
            for cond, const in rules:
                alpha = rule_activation(cond, mu_poi, mu_price, mu_size, mu_photos)
                if alpha > 0.2:
                    parts = ", ".join(f"{k}={v}" for k, v in cond.items())
                    lines.append(f"{parts} → {const:.2f} (α={alpha:.2f})")

            row = row.copy()
            row["EXPLANATION"] = "; ".join(lines) if lines else "(brak silnych reguł)"
            explained.append(row)

        sub_exp = pd.DataFrame(explained)
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
