# save as: add_interpretation_to_rankings.py
# Usage:
#   python add_interpretation_to_rankings.py C:\Users\antek\PycharmProjects\PythonProjectMiasto\data

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- Fuzzy helpery (takie jak w compute_...) ---
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

# --- Profile i docelowe metraże (jak w oryginalnym kodzie) ---
SIZE_TARGET = {
    "rodzinny": (25, 65),
    "studencki": (20, 45),
    "singiel": (20, 45),
    "wlasciciel_psa": (25, 55),
    "uniwersalny": (22, 55),
}

# --- Mapy opisów słownych ---
POI_DESC = {"High": "dobra lokalizacja", "Mid": "umiarkowana lokalizacja", "Low": "słaba lokalizacja"}
PRICE_DESC = {"Cheap": "tanie mieszkanie", "Mid": "umiarkowanie tanie", "Expensive": "drogie mieszkanie"}
SIZE_DESC = {"Small": "za małe", "Target": "idealny metraż", "Large": "za duże"}

def interpret_row(row, p10, p50, p95, smin, starget):
    mu_poi = poi_memberships(row["POI"])
    mu_price = price_memberships(row["price_pln_m2"], p10, p50, p95)
    mu_size = size_memberships(row["size_m2"], smin, starget)

    poi_label = max(mu_poi, key=mu_poi.get)
    price_label = max(mu_price, key=mu_price.get)
    size_label = max(mu_size, key=mu_size.get)

    desc = f"{POI_DESC[poi_label]} ({poi_label}), {PRICE_DESC[price_label]} ({price_label}), {SIZE_DESC[size_label]} ({size_label}), wynik = {row['ATRAKCYJNOSC']:.2f}"
    return desc

def main(folder: Path):
    files = list(folder.glob("ranking_*_explained.csv"))
    if not files:
        print("Brak plików ranking_*_explained.csv w folderze.")
        return

    # Ustalamy globalne kwantyle ceny (z wszystkich profili razem)
    all_data = [pd.read_csv(f) for f in files]
    all_df = pd.concat(all_data, ignore_index=True)
    p10 = float(np.percentile(all_df["price_pln_m2"], 10))
    p50 = float(np.percentile(all_df["price_pln_m2"], 50))
    p95 = float(np.percentile(all_df["price_pln_m2"], 95))

    for file in files:
        df = pd.read_csv(file)
        profile = df["profile"].iloc[0]
        smin, starget = SIZE_TARGET[profile]
        df["INTERPRETACJA"] = df.apply(lambda r: interpret_row(r, p10, p50, p95, smin, starget), axis=1)
        df.to_csv(file, index=False, encoding="utf-8")
        print(f"[OK] Dodano interpretację do {file.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python add_interpretation_to_rankings.py /path/to/folder")
        sys.exit(1)
    main(Path(sys.argv[1]))
