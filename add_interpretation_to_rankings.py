#   python add_interpretation_to_rankings.py C:\Users\antek\PycharmProjects\PythonProjectMiasto\data
from __future__ import annotations
import sys
from pathlib import Path
import re
import pandas as pd

# --- Mapy opisów słownych ---
POI_DESC    = {"High": "dobra lokalizacja",
               "Mid": "umiarkowana lokalizacja",
               "Low": "słaba lokalizacja"}

PRICE_DESC  = {"Cheap": "tanie mieszkanie",
               "Mid": "umiarkowanie tanie",
               "Expensive": "drogie mieszkanie"}

SIZE_DESC   = {"Small": "za małe",
               "Target": "idealny metraż",
               "Large": "może być za duże"}

ROOMS_DESC  = {"TooFew": "za mało pokoi",
               "Target": "odpowiednia liczba pokoi",
               "TooMany": "może być za dużo pokoi"}

STYLE_DESC = {
    "modern": "nowoczesny styl mieszkania",
    "old":    "starszy / klasyczny styl mieszkania",
}


def _extract_label(text: str, name: str, default: str) -> str:
    """
    Z tekstu EXPLANATION wyciąga np. 'POI=Mid' → 'Mid'.
    """
    if not isinstance(text, str):
        return default
    pattern = rf"{name}\s*=\s*([A-Za-z]+)"
    m = re.search(pattern, text)
    if not m:
        return default
    return m.group(1)


def build_interpretation(row) -> str:
    """
    Zwraca:
    apt_id, profile, Mieszkanie w stylu X, [POI], [PRICE], [SIZE], [ROOMS], wynik = yy.yy
    """
    expl = row.get("EXPLANATION", "")

    # etykiety fuzzy
    poi   = _extract_label(expl, "POI", "Mid")
    price = _extract_label(expl, "PRICE", "Mid")
    size  = _extract_label(expl, "SIZE", "Target")
    rooms = _extract_label(expl, "ROOMS", "Target")

    poi_desc   = POI_DESC.get(poi, poi)
    price_desc = PRICE_DESC.get(price, price)
    size_desc  = SIZE_DESC.get(size, size)
    rooms_desc = ROOMS_DESC.get(rooms, rooms)

    # styl
    style_key = row.get("photo_style", "")
    if isinstance(style_key, str) and style_key.strip():
        style_key = style_key.strip().lower()
        style_text = f"Mieszkanie w stylu {style_key}"
    else:
        style_text = "Mieszkanie"

    # identyfikacja
    apt_id  = row.get("apt_id", "")
    profile = row.get("profile", "")

    # wynik
    atrak = row.get("ATRAKCYJNOSC", "")
    try:
        score = f"{float(atrak):.2f}"
    except:
        score = str(atrak)

    return (
        f"{apt_id}, "
        f"{profile}, "
        f"{style_text}, "
        f"{poi_desc}, "
        f"{price_desc}, "
        f"{size_desc}, "
        f"{rooms_desc}, "
        f"wynik = {score}"
    )


def process_file(path: Path, out_folder: Path) -> None:
    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Pomijam pusty plik {path.name}")
        return

    if "profile" not in df.columns:
        print(f"[WARN] Brak kolumny profile w {path.name}")
        return

    profile = str(df["profile"].iloc[0]).strip()
    out_path = out_folder / f"description_{profile}.csv"

    df_out = pd.DataFrame()
    df_out["INTERPRETACJA"] = df.apply(build_interpretation, axis=1)
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Zapisano {out_path.name}")


def main(folder: Path):
    files = list(folder.glob("ranking_*_explained.csv"))
    if not files:
        print("Brak plików ranking_*_explained.csv w folderze.")
        return

    for f in files:
        process_file(f, folder)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python generate_descriptions.py /path/to/folder")
        sys.exit(1)
    main(Path(sys.argv[1]))
