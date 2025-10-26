# compute_scores_from_poi.py
# Użycie:
#   python compute_scores_from_poi.py "C:\ścieżka\gdynia_poi_filtered.csv"
#
# Wejście: CSV z kolumnami min. [kategoria, lat, lon]
# Wyjście:
#   - apartments_simulated.csv            (lista mieszkań losowych w bbox POI)
#   - apartments_poi_scores.csv          (wyniki per mieszkanie x kategoria)
#   - apartments_summary.csv             (zestawienie zbiorcze per mieszkanie)

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ---------- PARAMETRY GLOBALNE ----------

# promień do coverage (metry)
RADIUS_M = 1200.0

# miksowanie odległości i pokrycia
ALPHA = 0.6  # 0.6 * distance_score + 0.4 * coverage_score

# liczba mieszkań do zasymulowania
N_APTS = 40
RANDOM_SEED = 42

# progi trapezu dla distance_score (a <= b <= c), w metrach
# możesz personalizować per kategorię (tu rozsądne defaulty)
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

# parametry saturacji coverage (im mniejsze k, tym szybciej rośnie do 1)
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


# ---------- FUNKCJE POMOCNICZE ----------

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Odległość w metrach między (lat1,lon1) i (lat2,lon2) — wektorowo (lat2/lon2 mogą być tablicami).
    """
    R = 6371000.0  # promień Ziemi (m)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)

    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def trapezoid_distance_score(d, a, b, c):
    """
    Funkcja trapezowa w metrach: <=a:1, a->b: maleje 1->0, b->c: maleje 0->0 (z lekkim ogonem 0.5*),
    >c: 0. Tutaj uproszczona: dwa odcinki malejące do 0.
    """
    d = float(d)
    if d <= a:
        return 1.0
    if d <= b:
        return (b - d) / (b - a)
    if d <= c:
        # słabsza strefa: 0.5 -> 0 (możesz zmienić na łagodniejszą)
        return 0.5 * (c - d) / (c - b)
    return 0.0


def coverage_score(n, k):
    k = max(float(k), 1e-6)
    return float(1 - np.exp(-n / k))


# ---------- PIPELINE ----------

def load_poi(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # sanity
    for col in ["kategoria", "lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"Brak kolumny '{col}' w {csv_path}")
    df = df.dropna(subset=["kategoria", "lat", "lon"])
    return df


def simulate_apartments(bbox, n=N_APTS, seed=RANDOM_SEED) -> pd.DataFrame:
    """
    Generuje losowe mieszkania w bbox (min_lat, min_lon, max_lat, max_lon).
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    rng = np.random.default_rng(seed)
    lats = rng.uniform(min_lat, max_lat, size=n)
    lons = rng.uniform(min_lon, max_lon, size=n)
    df = pd.DataFrame({
        "apt_id": [f"A{i:03d}" for i in range(n)],
        "lat": lats,
        "lon": lons,
    })
    return df


def compute_scores_for_apartment(apt_lat, apt_lon, pois_by_cat: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Liczy metryki per kategoria dla jednego mieszkania.
    Zwraca DataFrame: [kategoria, best_dist_m, count_in_R, distance_score, coverage_score, poi_feature]
    """
    rows = []
    for cat, pts in pois_by_cat.items():
        if pts.size == 0:
            rows.append({
                "kategoria": cat,
                "best_dist_m": np.inf,
                "count_in_R": 0,
                "distance_score": 0.0,
                "coverage_score": 0.0,
                "poi_feature": 0.0,
            })
            continue

        # odległości do wszystkich POI w danej kategorii (wektorowo)
        dists = haversine_m(apt_lat, apt_lon, pts[:, 0], pts[:, 1])  # pts: [lat, lon]
        best = float(np.min(dists))
        cnt = int(np.sum(dists <= RADIUS_M))

        a, b, c = DIST_THRESHOLDS.get(cat, (600, 1500, 2500))
        k = COVERAGE_K.get(cat, 1.5)

        dscore = trapezoid_distance_score(best, a, b, c)
        cscore = coverage_score(cnt, k)
        feature = ALPHA * dscore + (1 - ALPHA) * cscore

        rows.append({
            "kategoria": cat,
            "best_dist_m": best,
            "count_in_R": cnt,
            "distance_score": dscore,
            "coverage_score": cscore,
            "poi_feature": feature,
        })

    return pd.DataFrame(rows)


def main(csv_path: Path):
    # 1) POI
    poi = load_poi(csv_path)
    cats = sorted(poi["kategoria"].unique().tolist())

    # 2) bbox
    min_lat, max_lat = poi["lat"].min(), poi["lat"].max()
    min_lon, max_lon = poi["lon"].min(), poi["lon"].max()
    bbox = (min_lat, min_lon, max_lat, max_lon)

    # 3) dane symulacyjne mieszkań
    apartments = simulate_apartments(bbox, n=N_APTS, seed=RANDOM_SEED)

    # 4) indeks POI per kategoria: tablice [lat, lon]
    pois_by_cat: Dict[str, np.ndarray] = {}
    for cat in cats:
        sub = poi.loc[poi["kategoria"] == cat, ["lat", "lon"]].to_numpy(dtype=float)
        pois_by_cat[cat] = sub

    # 5) liczenie per mieszkanie
    all_rows = []
    for _, apt in apartments.iterrows():
        apt_id, alat, alon = apt["apt_id"], float(apt["lat"]), float(apt["lon"])
        df_scores = compute_scores_for_apartment(alat, alon, pois_by_cat)
        df_scores.insert(0, "apt_id", apt_id)
        df_scores.insert(1, "apt_lat", alat)
        df_scores.insert(2, "apt_lon", alon)
        all_rows.append(df_scores)

    scores = pd.concat(all_rows, ignore_index=True)

    # 6) podsumowanie per mieszkanie (średnia z poi_feature po wszystkich kategoriach)
    summary = scores.groupby("apt_id").agg(
        apt_lat=("apt_lat", "first"),
        apt_lon=("apt_lon", "first"),
        poi_feature_mean=("poi_feature", "mean"),
        poi_feature_median=("poi_feature", "median"),
        poi_feature_max=("poi_feature", "max"),
    ).reset_index()

    # top-3 kategorie per mieszkanie (dla interpretowalności)
    def topk(series, k=3):
        # series: poi_feature z MultiIndex (apt_id,kategoria) — tu zrobimy merge niżej
        return ", ".join(series.nlargest(k).index.get_level_values("kategoria"))

    # przygotuj MultiIndex
    tmp = scores.set_index(["apt_id", "kategoria"])["poi_feature"]
    # zbierz top-3 jako DataFrame
    top3 = tmp.groupby(level=0, group_keys=False).apply(lambda s: ", ".join(
        [f"{cat}({val:.2f})" for cat, val in sorted(s.items(), key=lambda x: x[1], reverse=True)[:3]]
    )).reset_index(name="top3_kategorie")

    summary = summary.merge(top3, on="apt_id", how="left")

    # 7) zapisy
    out_dir = csv_path.parent
    apartments_path = out_dir / "apartments_simulated.csv"
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
    if len(sys.argv) < 2:
        print("Użycie: python compute_scores_from_poi.py C:\\ścieżka\\gdynia_poi_filtered.csv")
        sys.exit(1)
    main(Path(sys.argv[1]))
