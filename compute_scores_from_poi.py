# compute_scores_from_poi.py
# Użycie:
#   python compute_scores_from_poi.py "C:\path\gdynia_poi_filtered.csv" "C:\path\offers.csv"
#
# Wejście:
#   - POI CSV z kolumnami min. [kategoria, lat, lon]
#   - OFFERS CSV z kolumnami min. [offer_id, lat, lon] (+ opcjonalnie price_per_m2, area_m2, price_amount, price_currency, rooms, url, itp.)
#
# Wyjście:
#   - apartments_offers.csv            (lista mieszkań z ofert, zmapowana do [apt_id, lat, lon, price_pln_m2, size_m2, photos_count])
#   - apartments_poi_scores.csv        (wyniki per mieszkanie x kategoria)
#   - apartments_summary.csv           (zestawienie zbiorcze per mieszkanie)

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ---------- PARAMETRY GLOBALNE ----------

# promień do coverage (metry)
RADIUS_M = 1200.0

# progi trapezu dla distance_score (a <= b <= c), w metrach
# (te progi wykorzystujemy też do fuzzification dystansu: Near/Mid/Far)
DIST_THRESHOLDS: Dict[str, Tuple[float, float, float]] = {
    # krytyczne
    "szpital_przychodnia": (400, 1200, 2000),
    "apteka":              (400, 1200, 2000),
    "sklep":               (400, 1200, 2000),
    "przystanek_autobus":  (300, 1000, 1800),
    "stacja_kolej_metro":  (500, 1500, 2500),
    "przystanek_tramwaj":  (400, 1200, 2000),
    # rekreacja/edukacja
    "park":                (600, 1500, 2500),
    "biblioteka":          (600, 1500, 2500),
    "silownia":            (500, 1500, 2500),
    "kawiarnia_restauracja": (400, 1200, 2000),
    "klub":                (600, 1500, 2500),
    "pub":                 (600, 1500, 2500),
    "plac_zabaw":          (500, 1200, 2000),
    "szkola_przedszkole":  (500, 1200, 2000),
    "uczelnia":            (500, 1500, 2500),
    # usługi
    "weterynarz":          (600, 1500, 2500),
    "sklep_zoologiczny":   (600, 1500, 2500),
    "fryzjer":             (600, 1500, 2500),
    "bank_atm":            (600, 1500, 2500),
}

# skale coverage (k – im mniejsze, tym szybciej nasyca się do 1)
COVERAGE_K: Dict[str, float] = {
    "szpital_przychodnia": 1.0,
    "apteka":              2.0,
    "sklep":               2.0,
    "przystanek_autobus":  2.0,
    "stacja_kolej_metro":  1.0,
    "przystanek_tramwaj":  1.5,
    "park":                1.5,
    "biblioteka":          1.0,
    "silownia":            1.5,
    "kawiarnia_restauracja": 1.5,
    "klub":                1.0,
    "pub":                 1.0,
    "plac_zabaw":          1.0,
    "szkola_przedszkole":  1.0,
    "uczelnia":            1.0,
    "weterynarz":          1.0,
    "sklep_zoologiczny":   1.0,
    "fryzjer":             1.5,
    "bank_atm":            1.5,
}

# ---------- FUNKCJE GEO ----------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0  # m
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ---------- METRYKI POMOCNICZE ----------

def trapezoid_distance_score(d, a, b, c):
    d = float(d)
    if d <= a: return 1.0
    if d <= b: return (b - d) / (b - a)
    if d <= c: return 0.5 * (c - d) / (c - b)
    return 0.0

def coverage_score(n, k):
    k = max(float(k), 1e-6)
    return float(1 - np.exp(-n / k))

# ---------- FUZZY (Sugeno 0-order) DLA POI ----------

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def dist_memberships(d: float, a: float, b: float, c: float):
    if d <= a: near = 1.0
    elif d <= b: near = (b - d) / (b - a)
    else: near = 0.0
    if d <= a or d >= c: mid = 0.0
    elif d == b: mid = 1.0
    elif d < b: mid = (d - a) / (b - a)
    else: mid = (c - d) / (c - b)
    if d <= b: far = 0.0
    elif d <= c: far = (d - b) / (c - b)
    else: far = 1.0
    return {"Near": _clamp01(near), "Mid": _clamp01(mid), "Far": _clamp01(far)}

def cov_memberships(n: int, k: float):
    k = max(float(k), 1e-6)
    n = float(max(0, n))
    if n <= 0: low = 1.0
    elif n < k: low = (k - n) / k
    else: low = 0.0
    left, peak, right = 0.5 * k, 1.0 * k, 1.5 * k
    if n <= left or n >= right: mid = 0.0
    elif n == peak: mid = 1.0
    elif n < peak: mid = (n - left) / (peak - left)
    else: mid = (right - n) / (right - peak)
    if n <= k: high = 0.0
    elif n < 2.0 * k: high = (n - k) / k
    else: high = 1.0
    return {"Low": _clamp01(low), "Mid": _clamp01(mid), "High": _clamp01(high)}

def fuzzy_poi_utility(best_dist: float, cnt_in_R: int,
                      a: float, b: float, c: float, k: float,
                      category: str) -> float:
    μd = dist_memberships(best_dist, a, b, c)
    μc = cov_memberships(cnt_in_R, k)
    C = {
        ("Near", "High"): 1.00, ("Near", "Mid"): 0.80, ("Near", "Low"): 0.50,
        ("Mid",  "High"): 0.70, ("Mid",  "Mid"): 0.50, ("Mid",  "Low"): 0.30,
        ("Far",  "High"): 0.45, ("Far",  "Mid"): 0.25, ("Far",  "Low"): 0.20,
    }
    penalty_scale = 1.0
    if category in {"pub", "klub"}:
        penalty_scale = 0.6
    if category in {"szkola_przedszkole", "plac_zabaw"}:
        penalty_scale = 0.85
    num = den = 0.0
    for d_bin, μd_v in μd.items():
        if μd_v <= 0: continue
        for c_bin, μc_v in μc.items():
            if μc_v <= 0: continue
            alpha = min(μd_v, μc_v)
            const = C.get((d_bin, c_bin), 0.30) * penalty_scale
            num += alpha * const
            den += alpha
    return 0.0 if den == 0.0 else float(num / den)

# ---------- I/O ----------

def load_poi(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["kategoria", "lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"Brak kolumny '{col}' w {csv_path}")
    df = df.dropna(subset=["kategoria", "lat", "lon"])
    return df

def load_offers(offers_path: Path) -> pd.DataFrame:
    """Mapuje offers.csv → [apt_id, lat, lon, price_pln_m2?, size_m2?, photos_count?]."""
    df = pd.read_csv(offers_path).copy()
    for col in ["offer_id", "lat", "lon"]:
        if col not in df.columns:
            raise ValueError("offers.csv musi mieć co najmniej kolumny: offer_id, lat, lon")

    out = pd.DataFrame({
        "apt_id": df["offer_id"].astype(str),
        "lat": df["lat"].astype(float),
        "lon": df["lon"].astype(float),
    })
    # size_m2
    if "area_m2" in df.columns:
        out["size_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    # price_pln_m2
    price_pln_m2 = None
    if "price_per_m2" in df.columns:
        price_pln_m2 = pd.to_numeric(df["price_per_m2"], errors="coerce")
    elif {"price_amount", "area_m2", "price_currency"}.issubset(df.columns):
        m = (df["price_currency"].astype(str).str.upper() == "PLN")
        with np.errstate(divide="ignore", invalid="ignore"):
            price_pln_m2 = pd.Series(np.where(
                m,
                pd.to_numeric(df["price_amount"], errors="coerce") / pd.to_numeric(df["area_m2"], errors="coerce"),
                np.nan
            ))
    if price_pln_m2 is not None:
        out["price_pln_m2"] = price_pln_m2

    # photos_count (jeśli w przyszłości dorzucisz)
    out["photos_count"] = 0

    out = out.dropna(subset=["lat", "lon"])
    return out

# ---------- LICZENIE ----------

def compute_scores_for_apartment(apt_lat, apt_lon, pois_by_cat: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for cat, pts in pois_by_cat.items():
        if pts.size == 0:
            rows.append({"kategoria": cat, "best_dist_m": np.inf, "count_in_R": 0,
                         "distance_score": 0.0, "coverage_score": 0.0, "poi_feature": 0.0})
            continue
        dists = haversine_m(apt_lat, apt_lon, pts[:, 0], pts[:, 1])
        best = float(np.min(dists))
        cnt = int(np.sum(dists <= RADIUS_M))
        a, b, c = DIST_THRESHOLDS.get(cat, (600, 1500, 2500))
        k = COVERAGE_K.get(cat, 1.5)
        dscore = trapezoid_distance_score(best, a, b, c)
        cscore = coverage_score(cnt, k)
        feature = fuzzy_poi_utility(best, cnt, a, b, c, k, cat)
        rows.append({
            "kategoria": cat,
            "best_dist_m": best,
            "count_in_R": cnt,
            "distance_score": dscore,
            "coverage_score": cscore,
            "poi_feature": feature,
        })
    return pd.DataFrame(rows)

def main(poi_path: Path, offers_path: Path):
    # 1) POI
    poi = load_poi(poi_path)
    cats = sorted(poi["kategoria"].unique().tolist())

    # 2) Oferty → mieszkania
    apartments = load_offers(offers_path)

    # 3) indeks POI per kategoria: tablice [lat, lon]
    pois_by_cat: Dict[str, np.ndarray] = {}
    for cat in cats:
        sub = poi.loc[poi["kategoria"] == cat, ["lat", "lon"]].to_numpy(dtype=float)
        pois_by_cat[cat] = sub

    # 4) liczenie per mieszkanie
    all_rows = []
    for _, apt in apartments.iterrows():
        apt_id, alat, alon = apt["apt_id"], float(apt["lat"]), float(apt["lon"])
        df_scores = compute_scores_for_apartment(alat, alon, pois_by_cat)
        df_scores.insert(0, "apt_id", apt_id)
        df_scores.insert(1, "apt_lat", alat)
        df_scores.insert(2, "apt_lon", alon)
        all_rows.append(df_scores)

    scores = pd.concat(all_rows, ignore_index=True)

    # 5) podsumowanie per mieszkanie
    summary = scores.groupby("apt_id").agg(
        apt_lat=("apt_lat", "first"),
        apt_lon=("apt_lon", "first"),
        poi_feature_mean=("poi_feature", "mean"),
        poi_feature_median=("poi_feature", "median"),
        poi_feature_max=("poi_feature", "max"),
    ).reset_index()

    tmp = scores.set_index(["apt_id", "kategoria"])["poi_feature"]
    top3 = tmp.groupby(level=0, group_keys=False).apply(
        lambda s: ", ".join([f"{cat}({val:.2f})" for cat, val in sorted(s.items(), key=lambda x: x[1], reverse=True)[:3]])
    ).reset_index(name="top3_kategorie")
    summary = summary.merge(top3, on="apt_id", how="left")

    # 6) zapisy
    out_dir = poi_path.parent
    apartments_path = out_dir / "apartments_offers.csv"
    scores_path = out_dir / "apartments_poi_scores.csv"
    summary_path = out_dir / "apartments_summary.csv"

    apartments.to_csv(apartments_path, index=False, encoding="utf-8")
    scores.to_csv(scores_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"[OK] Zapisano:")
    print(f"- {apartments_path}")
    print(f"- {scores_path}")
    print(f"- {summary_path}")
    print("\nPrzykład TOP-3 kategorii (pierwsze 5 mieszkań):")
    print(summary[["apt_id", "top3_kategorie"]].head(5).to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Użycie: python compute_scores_from_poi.py C:\\ścieżka\\gdynia_poi_filtered.csv C:\\ścieżka\\offers.csv")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
