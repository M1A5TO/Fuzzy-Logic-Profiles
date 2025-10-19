# Fuzzy-Logic-Profiles

    
# 🏙️ 15-Minute City – Fuzzy Logic Evaluation System

This repository contains a **fuzzy logic–based evaluation system** that measures the **attractiveness of apartments** in the context of the *15-Minute City* concept.  
The project integrates **geospatial data from OpenStreetMap**, **distance analysis**, and **profile-dependent fuzzy rules** to assess how well each apartment fits various lifestyle profiles (e.g. family, student, pet owner).

---

## 📘 Project Overview

The goal is to evaluate residential locations by simulating access to essential Points of Interest (POIs) — such as schools, parks, supermarkets, public transport, and more — within a **15-minute radius (≈1200 m)**.  
A **fuzzy multi-criteria model** converts these spatial and contextual features into an overall *attractiveness score (0 – 1)* for each profile.

---

## 🌍 Profiles

Each apartment is evaluated from the perspective of **five user profiles**, each emphasizing different priorities:

| Profile | High Weight | Medium Weight | Low / Zero Weight |
|----------|--------------|----------------|-------------------|
| **Family** | Schools, parks, shops, medical services | Libraries | Clubs, bars |
| **Student** | University, gyms, cafes, nightlife | Restaurants | Schools, police |
| **Single** | Bars, cafes, restaurants, gyms | — | — |
| **Pet Owner** | Parks, veterinary clinics, pet shops | — | — |
| **Universal** | Balanced mix of all POIs | — | — |

---

## 🧠 Fuzzy Logic Model

The overall attractiveness `A` for each apartment is computed as a **Sugeno-type weighted aggregation**:

\[
A = w_{POI}·POI + w_{price}·CENA + w_{size}·M2 + w_{photos}·ZDJ
\]

Each partial component is normalized to [0 – 1]:

| Symbol | Meaning | Description |
|---------|----------|-------------|
| `POI` | Environmental attractiveness | Weighted fuzzy score of surrounding POIs |
| `CENA` | Price attractiveness | Normalized based on percentiles (cheap → 1.0) |
| `M2` | Size attractiveness | Based on preferred area range for profile |
| `ZDJ` | Photo confidence | Increases with the number of available photos |

---

## 🧩 POI Scoring Logic

For each apartment–POI pair, the system computes:

1. **Distance component**  
   Using the [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula)  
   → mapped through a trapezoidal fuzzy function  
   → full score for ≤ 400 m, fades out to zero beyond 2 km.

2. **Coverage component**  
   Measures how many POIs of the same type exist within 1200 m:  
   \[
   f_{cov}(n) = 1 - e^{-n / k}
   \]
   (where `k` controls saturation).

3. **Final POI feature:**  
   \[
   POI_{feature} = α·f_{dist} + (1-α)·f_{cov}
   \]
   with α = 0.6 by default.

Negative weights (e.g. for clubs in *family* profile) act as **penalties**, reducing the overall score.

---

## 🧮 Data Workflow

| Step | Script | Description |
|------|---------|-------------|
| **1. Extract POIs** | `extract_poi_pyrosm.py` | Loads `.pbf` file (e.g. Gdynia 5 km radius) using **Pyrosm**, filters and classifies POIs (e.g. schools, parks, supermarkets, metro stations). |
| **2. Compute POI Scores** | `compute_scores_from_poi.py` | Calculates nearest distances, counts within radius, and fuzzy POI scores for each apartment. |
| **3. Compute Attractiveness** | `compute_attractiveness_profiles.py` | Applies profile-specific weights, normalization of price/size/photos, and generates final attractiveness CSV. |

Output files:

---

## 🧭 Example Output

| apt_id | profile | POI | CENA | M2 | ZDJ | ATRACTYJNOŚĆ | TOP_PLUS | TOP_MINUS |
|--------|----------|-----|------|----|------|--------------|-----------|------------|
| A000 | **Family** | 0.614 | 0.990 | 0.83 | 0.86 | **0.79** | sklep(+1.00), szkola_przedszkole(+0.99) | pub(-0.24), klub(-0.15) |
| A000 | **Student** | 0.659 | 0.990 | 1.00 | 0.86 | **0.85** | przystanek_autobus(+1.00), tramwaj(+0.86) | galeria(0.00), szkoła(0.00) |
| A000 | **Pet Owner** | 0.594 | 0.990 | 1.00 | 0.86 | **0.81** | park(+0.99) | klub(-0.06) |

---

## ✅ Validation & Testing

### 1. Unit Tests
Implemented using `pytest`.  
All fundamental functions are tested:
- `haversine_m()` distance accuracy  
- `trapezoid_distance_score()` monotonicity  
- `coverage_score()` saturation  
- `weighted_mean_signed()` with negative weights  
- `POI feature` reaction to distance/coverage  
- OSM classification rules (`stacja_kolej_metro` requires both tags)

✅ **All 8 tests passed**.

### 2. Integration (Golden) Test
A minimal synthetic scenario with known geometry and expected scores confirms:
- correct metric distance calculation (~111 km per degree),
- correct fuzzy distance decay,
- correct aggregation across `POI`, `CENA`, `M2`, and `ZDJ`.

✅ **Full end-to-end validation passed.**

### 3. Logical & Spatial Validation
- Cross-profile comparison: each profile prefers different locations.  
- Family profile penalizes noisy areas (clubs/pubs).  
- Heatmaps match known “family-friendly” vs. “student” districts in Gdynia and Gdańsk.

---

## 🧭 Dependencies

```bash
python >= 3.10
pyrosm
osmium
pandas
numpy
pytest
matplotlib  # optional for visualization
