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

    return {"Cheap": _clamp01(cheap), "Mid": _clamp01(mid), "Expensive": _clamp01(expensive)}

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

def rooms_memberships(rooms: float, rmin: float, rlow: float, rhigh: float, rmax: float):
    """
    Fuzzy z plateau:
      - Target = 1.0 na całym [rlow, rhigh] (brzegi wliczone),
      - liniowy najazd 0→1 od rmin do rlow,
      - liniowy zjazd 1→0 od rhigh do rmax.
      - TooFew maleje z 1.0 (≤ rmin) do 0.0 przy rlow,
      - TooMany rośnie z 0.0 (≤ rhigh) do 1.0 przy rmax.
    """
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
        elif x <= rhigh:  # plateau
            target = 1.0
        else:  # x > rhigh
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

# --- Profile i docelowe metraże (jak w oryginalnym kodzie) ---
SIZE_TARGET = {
    "rodzinny": (25, 65),
    "studencki": (20, 45),
    "singiel": (20, 45),
    "wlasciciel_psa": (25, 55),
    "uniwersalny": (22, 55),
}

# --- Docelowa liczba pokoi per profil (dla ROOMS) ---
# Format: (min_ok, target_low, target_high, max_ok)
ROOMS_TARGET_RANGES = {
    "rodzinny":       (2.0, 3.0, 4.0, 5.0),  # 3–4 idealne
    "studencki":      (1.0, 1.0, 2.0, 3.0),  # 1–2 idealne
    "singiel":        (1.0, 1.0, 1.0, 2.0),  # 1 idealnie
    "wlasciciel_psa": (2.0, 2.0, 3.0, 4.0),  # 2–3 idealne
    "uniwersalny":    (2.0, 2.0, 3.0, 4.0),  # 2–3 idealne
}

# --- Mapy opisów słownych ---
POI_DESC    = {"High": "dobra lokalizacja", "Mid": "umiarkowana lokalizacja", "Low": "słaba lokalizacja"}
PRICE_DESC  = {"Cheap": "tanie mieszkanie", "Mid": "umiarkowanie tanie", "Expensive": "drogie mieszkanie"}
SIZE_DESC   = {"Small": "za małe", "Target": "idealny metraż", "Large": "może być za duże"}
ROOMS_DESC  = {"TooFew": "za mało pokoi", "Target": "odpowiednia liczba pokoi", "TooMany": "może być za dużo pokoi"}

STYLE_DESC = {
    "modern": "nowoczesny styl mieszkania",
    "old": "starszy / klasyczny styl mieszkania",
}

# --- Heurystyka: oszacuj rooms z metrażu, gdy brak ---
def _estimate_rooms_from_size(size_m2: float) -> float:
    try:
        s = float(size_m2)
    except Exception:
        return np.nan
    if np.isnan(s):
        return np.nan
    if s <= 28: return 1.0
    if s <= 45: return 2.0
    if s <= 65: return 3.0
    if s <= 85: return 4.0
    return 5.0

def interpret_row(row, p10, p50, p95, smin, starget, rmin, rlow, rhigh, rmax):
    mu_poi   = poi_memberships(row["POI"])
    mu_price = price_memberships(row["price_pln_m2"], p10, p50, p95)
    mu_size  = size_memberships(row["size_m2"], smin, starget)

    # rooms: jeśli brak/NaN → estymuj z size_m2 tylko na potrzeby interpretacji
    rooms_val = row.get("rooms", np.nan)
    if pd.isna(rooms_val):
        rooms_val = _estimate_rooms_from_size(row["size_m2"])
    mu_rooms = rooms_memberships(rooms_val, rmin, rlow, rhigh, rmax)

    poi_label   = max(mu_poi, key=mu_poi.get)
    price_label = max(mu_price, key=mu_price.get)
    size_label  = max(mu_size, key=mu_size.get)
    rooms_label = max(mu_rooms, key=mu_rooms.get)

    # Styl mieszkania (jeśli kolumny istnieją)
    style_part = ""
    style = row.get("photo_style") if "photo_style" in row.index else None
    style_score = row.get("STYLE_SCORE") if "STYLE_SCORE" in row.index else None

    if isinstance(style, str) and style.strip():
        style_key = style.strip().lower()
        desc_style = STYLE_DESC.get(style_key, f"styl mieszkania: {style_key}")
        if style_score is not None and not pd.isna(style_score):
            try:
                style_score_f = float(style_score)
                style_part = f", {desc_style} (STYLE={style_key}, score={style_score_f:.2f})"
            except Exception:
                style_part = f", {desc_style} (STYLE={style_key})"
        else:
            style_part = f", {desc_style} (STYLE={style_key})"

    desc = (
        f"{POI_DESC[poi_label]} ({poi_label}), "
        f"{PRICE_DESC[price_label]} ({price_label}), "
        f"{SIZE_DESC[size_label]} ({size_label}), "
        f"{ROOMS_DESC[rooms_label]} ({rooms_label}, rooms={int(round(rooms_val)) if not pd.isna(rooms_val) else 'brak'})"
        f"{style_part}, "
        f"wynik = {row['ATRAKCYJNOSC']:.2f}"
    )
    return desc

def main(folder: Path):
    files = list(folder.glob("ranking_*_explained.csv"))
    if not files:
        print("Brak plików ranking_*_explained.csv w folderze.")
        return

    # Ustalamy globalne kwantyle ceny (z wszystkich profili razem)
    all_data = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    if not all_data:
        print("Pliki ranking_*_explained.csv są puste.")
        return

    all_df = pd.concat(all_data, ignore_index=True)
    # Walidacja podstawowych kolumn
    for col in ["price_pln_m2", "size_m2", "POI", "ATRAKCYJNOSC", "profile"]:
        if col not in all_df.columns:
            raise ValueError(f"Brakuje wymaganej kolumny '{col}' w ranking_*_explained.csv")

    p10 = float(np.percentile(all_df["price_pln_m2"], 10))
    p50 = float(np.percentile(all_df["price_pln_m2"], 50))
    p95 = float(np.percentile(all_df["price_pln_m2"], 95))

    for file in files:
        df = pd.read_csv(file)
        if df.empty:
            continue
        if "profile" not in df.columns:
            print(f"[WARN] Pomijam {file.name} — brak kolumny 'profile'.")
            continue

        profile = str(df["profile"].iloc[0])
        if profile not in SIZE_TARGET or profile not in ROOMS_TARGET_RANGES:
            print(f"[WARN] Pomijam {file.name} — nieznany profil '{profile}'.")
            continue

        smin, starget = SIZE_TARGET[profile]
        rmin, rlow, rhigh, rmax = ROOMS_TARGET_RANGES[profile]

        # Uzupełnij kolumnę rooms, jeśli nie istnieje (nie nadpisujemy istniejących wartości)
        if "rooms" not in df.columns:
            df["rooms"] = np.nan

        df["INTERPRETACJA"] = df.apply(
            lambda r: interpret_row(r, p10, p50, p95, smin, starget, rmin, rlow, rhigh, rmax),
            axis=1
        )
        df.to_csv(file, index=False, encoding="utf-8")
        print(f"[OK] Dodano interpretację (z ROOMS + STYLE) do {file.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python add_interpretation_to_rankings.py /path/to/folder")
        sys.exit(1)
    main(Path(sys.argv[1]))
