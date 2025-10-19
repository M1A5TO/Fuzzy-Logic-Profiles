import numpy as np

# --- Funkcje jak w Twoim kodzie ---

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def trapezoid_distance_score(d, a, b, c):
    if d <= a: return 1.0
    if d <= b: return (b - d) / (b - a)
    if d <= c: return 0.5 * (c - d) / (c - b)
    return 0.0

def coverage_score(n, k):
    return float(1 - np.exp(-n / max(float(k), 1e-6)))

def poi_feature_from(best_dist_m, count_in_R, thresholds=(400,1200,2000), k=2.0, alpha=0.6):
    a,b,c = thresholds
    dscore = trapezoid_distance_score(best_dist_m, a,b,c)
    cscore = coverage_score(count_in_R, k)
    return alpha*dscore + (1-alpha)*cscore

def weighted_mean_signed(values, weights):
    v = np.array(values, float); w = np.array(weights, float)
    return float((v*w).sum() / np.abs(w).sum())

def robust_price_score(price, p10, p95):
    if p95 <= p10: return 0.5
    return float(np.clip((p95 - price) / (p95 - p10), 0, 1))

def size_score(size_m2, s_min, s_target):
    if s_target <= s_min: return 0.5
    return float(np.clip((size_m2 - s_min) / (s_target - s_min), 0, 1))

def photos_score(n):
    return float(1 - np.exp(-n/5.0))

# --- Dane syntetyczne (równik; łatwe metry) ---

# Apartamenty
A1 = (0.0, 0.0)     # A1 na (lat=0, lon=0)
A2 = (0.0, 0.01)    # A2 ~1113 m na wschód

# POI: dwie kategorie: sklep, apteka
# sklepy: jeden idealnie przy A1, drugi ~0.005 lon (~556 m)
shops = np.array([
    [0.0, 0.0],     # przy A1
    [0.0, 0.005],   # ~556 m od A1; ~557 m od A2 (best dla A2)
])
# apteka: ~0.02 lon (~2226 m) na wschód od A1; ~1113 m od A2
pharmacies = np.array([
    [0.0, 0.02]
])

RADIUS_M = 1200.0
THR = (400, 1200, 2000)   # progi trapezu jak w Twoim kodzie
K_cov = 2.0               # nasycanie coverage
ALPHA = 0.6               # miks distance/coverage

# Wagi POI dla profilu studenckiego (tu bierzemy tylko 2 kategorie, obie "high"=1.0)
W_stud = {"sklep": 1.0, "apteka": 1.0}

# Wagi bloków dla profilu studenckiego
WB = {"POI": 0.40, "CENA": 0.35, "M2": 0.20, "ZDJ": 0.05}

# Parametry cen/metrażu/zdjęć do testu
p10, p95 = 8000.0, 20000.0
price_A1, price_A2 = 10000.0, 16000.0   # CENA ~ 0.8333 i 0.3333
size_A1, size_A2 = 45.0, 20.0           # M2 dla stud: s_min=20, s_target=45 -> 1.0 i 0.0
photos_A1, photos_A2 = 10, 0            # ZDJ: ~0.8647 i 0.0

def _best_and_count(apt, pts):
    dists = haversine_m(apt[0], apt[1], pts[:,0], pts[:,1])
    return float(dists.min()), int((dists <= RADIUS_M).sum())

def _poi_score_for_apartment(apt):
    # sklep
    d_shop, c_shop = _best_and_count(apt, shops)
    feat_shop = poi_feature_from(d_shop, c_shop, THR, K_cov, ALPHA)
    # apteka
    d_ph, c_ph = _best_and_count(apt, pharmacies)
    feat_ph = poi_feature_from(d_ph, c_ph, THR, K_cov, ALPHA)
    # średnia ważona (obie wagi = 1.0)
    poi = weighted_mean_signed([feat_shop, feat_ph], [W_stud["sklep"], W_stud["apteka"]])
    return feat_shop, feat_ph, poi

def _final_atr(poi, price, size, photos):
    cena = robust_price_score(price, p10, p95)
    m2   = size_score(size, 20.0, 45.0)      # profil studencki
    zdj  = photos_score(photos)
    num = WB["POI"]*poi + WB["CENA"]*cena + WB["M2"]*m2 + WB["ZDJ"]*zdj
    den = sum(WB.values())
    return poi, cena, m2, zdj, float(num/den)

def test_golden_A1():
    # A1: sklep (0m i drugi 556m) -> feat ~0.85285; apteka (2226m) -> feat = 0
    feat_shop, feat_ph, poi = _poi_score_for_apartment(A1)
    poi_expected = (feat_shop + 0.0) / 2.0
    assert abs(feat_ph - 0.0) < 1e-6
    assert 0.84 < feat_shop < 0.86
    assert abs(poi - poi_expected) < 1e-6

    poi, cena, m2, zdj, atr = _final_atr(poi, price_A1, size_A1, photos_A1)
    # CENA ≈ (20000-10000)/(20000-8000)=10000/12000=0.8333; M2=1.0; ZDJ≈0.8646647
    assert 0.82 < cena < 0.85
    assert m2 == 1.0
    assert 0.86 - 1e-3 < zdj < 0.87 + 1e-3

    # ATR ręcznie: 0.4*POI + 0.35*0.8333 + 0.2*1.0 + 0.05*0.8647 ≈ 0.705 ± 0.01
    assert 0.695 <= atr <= 0.715

def test_golden_A2():
    # A2: sklep best ~557m, count=2; apteka best ~1113m, count=1
    feat_shop, feat_ph, poi = _poi_score_for_apartment(A2)
    # sanity zakresów
    assert 0.70 < feat_shop < 0.77
    assert 0.20 < feat_ph < 0.25
    assert 0.45 < poi < 0.50

    poi, cena, m2, zdj, atr = _final_atr(poi, price_A2, size_A2, photos_A2)
    # CENA ≈ (20000-16000)/12000 = 0.3333; M2=0.0; ZDJ=0.0
    assert 0.32 < cena < 0.35
    assert m2 == 0.0 and zdj == 0.0

    # ATR: 0.4*POI + 0.35*0.3333 + 0.2*0 + 0.05*0 ≈ ~0.30–0.32
    assert 0.29 <= atr <= 0.33
