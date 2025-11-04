import pandas as pd, numpy as np, json

# 2.1 POI muszą mieć współrzędne
poi = pd.read_csv(r"C:\Users\antek\PycharmProjects\PythonProjectMiasto\data\gdynia_poi_filtered.csv")
assert not poi.empty, "Plik POI jest pusty"
assert poi[['lat','lon']].notna().all().all(), "Brak współrzędnych w POI"
assert set(['kategoria']).issubset(poi.columns), "Brakuje kolumny 'kategoria'"

# 2.2 Wyniki scores – każdy apt_id × kategoria ma wpis
scores = pd.read_csv(r"C:\Users\antek\PycharmProjects\PythonProjectMiasto\data\apartments_poi_scores.csv")
assert {'apt_id','kategoria','best_dist_m','count_in_R','poi_feature'} <= set(scores.columns), "Brak kolumn w scores"
assert (scores['best_dist_m'] >= 0).all(), "Ujemna odległość w scores"
assert (scores['count_in_R'] >= 0).all(), "Ujemny count_in_R"
assert ((scores['poi_feature'] >= 0) & (scores['poi_feature'] <= 1)).all(), "poi_feature poza zakresem [0,1]"

# 2.3 Atrakcyjność – pola w [0,1]
atr = pd.read_csv(r"C:\Users\antek\PycharmProjects\PythonProjectMiasto\data\attractiveness_by_profile.csv")
for col in ['POI','CENA','M2','ZDJ','ATRAKCYJNOSC']:
    assert ((atr[col] >= 0) & (atr[col] <= 1)).all(), f"{col} poza zakresem [0,1]"

print("✅ Wszystkie testy przeszły poprawnie.")
