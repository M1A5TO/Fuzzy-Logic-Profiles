# extract_poi_pyrosm.py
# Użycie:
#   python extract_poi_pyrosm.py "C:\Users\antek\PycharmProjects\PythonProjectMiasto\data\gdynia_5km.pbf"

import sys
from pathlib import Path
from typing import Optional, Dict, Any

import geopandas as gpd
from pyrosm import OSM


# --- Filtry OSM pod Twoje kategorie ---
CUSTOM_FILTER = {
    "amenity": [
        "school", "kindergarten", "childcare",
        "hospital", "clinic", "pharmacy",
        "library",
        "university", "college",
        "veterinary",
        "bank", "atm",
        "cafe", "restaurant",
        "nightclub", "pub",
    ],
    "leisure": [
        "park", "playground", "fitness_centre",
    ],
    "shop": [
        "supermarket", "convenience", "bakery", "hairdresser", "pet",
    ],
    # pojedynczo pobieramy tram_stop i bus_stop,
    # a metro/SKM/pociągi rozpoznamy później warunkiem 2-tagowym
    "railway": ["tram_stop"],
    "highway": ["bus_stop"],
}

# Kolumny, które chcemy zachować (nie wszystkie zawsze występują)
KEEP_COLS = [
    "id", "osm_type", "name",
    "amenity", "leisure", "shop",
    "railway", "highway", "public_transport",
    "lat", "lon", "geometry",
]


def ensure_cols(gdf: gpd.GeoDataFrame, cols) -> gpd.GeoDataFrame:
    for c in cols:
        if c not in gdf.columns:
            gdf[c] = None
    return gdf


def to_point_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Ujednolicenie do punktów (centroid dla way/relation)
    # i ustawienie/utrzymanie WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    else:
        gdf = gdf.to_crs(4326)

    # centroid działa dla geometrii powierzchniowych/liniowych;
    # dla Point zwróci ten sam punkt
    gdf["geometry"] = gdf["geometry"].centroid
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf


def is_train_station(tags: Dict[str, Any]) -> bool:
    # Warunek 2-tagowy: public_transport ∈ {station,halt} AND railway ∈ {station,halt}
    railway = tags.get("railway")
    pub = tags.get("public_transport")
    return (railway in {"station", "halt"}) and (pub in {"station", "halt"})


def classify_row(row: Dict[str, Optional[str]]) -> Optional[str]:
    # Najpierw stacje kolej/metro z warunkiem 2-tagowym:
    if is_train_station({"railway": row.get("railway"), "public_transport": row.get("public_transport")}):
        return "stacja_kolej_metro"

    # Potem reszta kategorii:
    if row.get("railway") == "tram_stop":
        return "przystanek_tramwaj"
    if row.get("highway") == "bus_stop":
        return "przystanek_autobus"

    amenity = row.get("amenity")
    leisure = row.get("leisure")
    shop = row.get("shop")

    if amenity in {"hospital", "clinic"}:
        return "szpital_przychodnia"
    if amenity == "pharmacy":
        return "apteka"
    if shop in {"supermarket", "convenience", "bakery"}:
        return "sklep"
    if amenity in {"university", "college"}:
        return "uczelnia"
    if leisure == "fitness_centre":
        return "silownia"
    if amenity == "nightclub":
        return "klub"
    if amenity == "pub":
        return "pub"
    if amenity in {"cafe", "restaurant"}:
        return "kawiarnia_restauracja"
    if amenity == "library":
        return "biblioteka"
    if leisure == "park":
        return "park"
    if leisure == "playground":
        return "plac_zabaw"
    if amenity in {"school", "kindergarten", "childcare"}:
        return "szkola_przedszkole"
    if amenity == "veterinary":
        return "weterynarz"
    if shop == "pet":
        return "sklep_zoologiczny"
    if shop == "hairdresser":
        return "fryzjer"
    if amenity in {"bank", "atm"}:
        return "bank_atm"

    return None


def main(pbf_path: Path) -> None:
    assert pbf_path.exists(), f"Nie znaleziono pliku: {pbf_path}"
    out_dir = pbf_path.parent

    # 1) Wczytanie POI przez pyrosm z filtrami
    osm = OSM(str(pbf_path))
    gdf = osm.get_pois(custom_filter=CUSTOM_FILTER)

    if gdf is None or len(gdf) == 0:
        print("Brak POI dla zadanych filtrów. Sprawdź filtr lub plik PBF.")
        return

    # 2) Centroidy + WGS84 + lon/lat
    gdf = to_point_centroid(gdf)

    # 3) Upewnij się, że mamy kolumny do klasyfikacji, nawet jeśli brak w podzbiorze
    gdf = ensure_cols(gdf, KEEP_COLS)

    # 4) Klasyfikacja do Twoich kategorii
    gdf["kategoria"] = gdf.apply(lambda r: classify_row({
        "amenity": r.get("amenity"),
        "leisure": r.get("leisure"),
        "shop": r.get("shop"),
        "railway": r.get("railway"),
        "highway": r.get("highway"),
        "public_transport": r.get("public_transport"),
    }), axis=1)

    gdf = gdf[gdf["kategoria"].notna()].copy()

    # 5) Selekcja kolumn do zapisu
    keep = [c for c in KEEP_COLS if c in gdf.columns] + ["kategoria"]
    keep = list(dict.fromkeys(keep))  # deduplikacja z zachowaniem kolejności
    gdf = gdf[keep]

    # 6) Zapis
    geojson_path = out_dir / "gdynia_poi_filtered.geojson"
    csv_path = out_dir / "gdynia_poi_filtered.csv"

    # GeoJSON (punktowe geometrie)
    gdf_geo = gdf.copy()
    # geopandas wymaga kolumny 'geometry' jako aktywnej geometrii
    if "geometry" not in gdf_geo.columns:
        gdf_geo = gpd.GeoDataFrame(gdf_geo, geometry=gpd.points_from_xy(gdf_geo["lon"], gdf_geo["lat"]), crs="EPSG:4326")
    else:
        gdf_geo = gpd.GeoDataFrame(gdf_geo, geometry=gdf_geo["geometry"], crs="EPSG:4326")

    gdf_geo.to_file(geojson_path, driver="GeoJSON")

    # CSV bez kolumny geometry
    gdf.drop(columns=[c for c in ["geometry"] if c in gdf.columns], inplace=True)
    gdf.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"Zapisano {len(gdf)} rekordów:")
    print(f"- {geojson_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python extract_poi_pyrosm.py C:\\ścieżka\\do\\gdynia_5km.pbf")
        sys.exit(1)

    main(Path(sys.argv[1]))
