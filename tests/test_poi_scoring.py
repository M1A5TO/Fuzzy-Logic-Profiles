import numpy as np
import pandas as pd

# Kopie funkcji z compute_scores_from_poi.py
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

ALPHA = 0.6
def poi_feature(best_dist_m, count_in_R, thresholds=(400,1200,2000), k=2.0, alpha=ALPHA):
    a,b,c = thresholds
    d = trapezoid_distance_score(best_dist_m, a,b,c)
    cov = coverage_score(count_in_R, k)
    return alpha*d + (1-alpha)*cov

def test_poi_feature_reacts_to_distance_and_coverage():
    # dwa scenariusze: (blisko, mało) vs (daleko, dużo)
    close_few = poi_feature(best_dist_m=200, count_in_R=1, thresholds=(400,1200,2000), k=2.0)
    far_many  = poi_feature(best_dist_m=1400, count_in_R=6, thresholds=(400,1200,2000), k=2.0)
    # obie sytuacje mogą być atrakcyjne; upewnijmy się, że wynik jest sensowny
    assert 0.5 <= close_few <= 1.0
    assert 0.4 <= far_many <= 1.0

def test_best_dist_and_count_behaviour():
    # mieszkanie A: blisko jednej apteki
    apt = (54.3867, 18.6331)
    poi_near = np.array([[54.3880, 18.6335]])  # bardzo blisko
    dists = haversine_m(apt[0], apt[1], poi_near[:,0], poi_near[:,1])
    assert dists.min() < 200  # rząd setek metrów
    # liczenie w promieniu 1200 m (tu: 1)
    count_in_R = int(np.sum(dists <= 1200))
    assert count_in_R == 1

def test_increase_radius_increases_count():
    # proste punkty w linii
    apt = (0.0, 0.0)
    pois = np.array([[0.0, 0.01], [0.0, 0.02], [0.0, 0.03]])  # ~1.1km, 2.2km, 3.3km na równiku
    dists = haversine_m(apt[0], apt[1], pois[:,0], pois[:,1])
    R1 = 1500
    R2 = 3000
    c1 = int(np.sum(dists <= R1))
    c2 = int(np.sum(dists <= R2))
    assert c1 < c2
