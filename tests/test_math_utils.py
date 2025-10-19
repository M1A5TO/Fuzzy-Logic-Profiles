import numpy as np

# --- minimalne kopie funkcji (identyczne z Twoimi skryptami) ---

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
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

def weighted_mean_signed(values, weights):
    v = np.array(values, float)
    w = np.array(weights, float)
    if np.all(w == 0): return 0.0
    return float((v*w).sum() / np.abs(w).sum())

# --- TESTY ---

def test_haversine_zero_distance():
    assert haversine_m(54.0, 18.0, 54.0, 18.0) == 0.0

def test_haversine_known_degree_latitude():
    d = haversine_m(0, 0, 1, 0)  # ~111.2 km
    assert 110000 < d < 112500

def test_trapezoid_monotonicity():
    a, b, c = 400, 1200, 2000
    assert trapezoid_distance_score(100, a, b, c) == 1.0
    assert trapezoid_distance_score(800, a, b, c) > trapezoid_distance_score(1500, a, b, c) > trapezoid_distance_score(2500, a, b, c)

def test_trapezoid_edges():
    a, b, c = 400, 1200, 2000
    assert trapezoid_distance_score(400, a, b, c) == 1.0
    assert trapezoid_distance_score(1200, a, b, c) == 0.0  # na styku górnej części pierwszego odcinka
    assert trapezoid_distance_score(2000, a, b, c) == 0.0

def test_coverage_saturation_and_nonnegative():
    k = 2.0
    assert coverage_score(0, k) == 0.0
    assert 0.55 < coverage_score(2, k) < 0.7
    assert coverage_score(10, k) > 0.99

def test_weighted_mean_signed_positive_and_negative_weights():
    vals = [1.0, 0.5, 0.0]
    wts  = [ 1.0,-0.5, 0.0]
    out = weighted_mean_signed(vals, wts)
    # Ujemna waga powinna obniżyć wynik względem wersji bez kary
    out_no_penalty = weighted_mean_signed([1.0, 0.5], [1.0, 0.0])
    assert 0 <= out <= 1
    assert out < out_no_penalty
