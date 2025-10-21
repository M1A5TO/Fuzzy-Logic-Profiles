# Fuzzy-Logic-Profiles

    
# 15-Minute City – Fuzzy Logic Evaluation System

This repository contains a **fuzzy logic–based evaluation system** that measures the **attractiveness of apartments** in the context of the *15-Minute City* concept.  
The project integrates **geospatial data from OpenStreetMap**, **distance analysis**, and **profile-dependent fuzzy rules** to assess how well each apartment fits various lifestyle profiles (e.g. family, student, singel, pet owner).

---

##  Project Overview

The goal is to evaluate residential locations by simulating access to essential Points of Interest (POIs) — such as schools, parks, supermarkets, public transport, and more — within a **15-minute radius (≈1200 m)**.  
A **fuzzy multi-criteria model** converts these spatial and contextual features into an overall *attractiveness score (0 – 1)* for each profile.

---

##  Profiles

Each apartment is evaluated from the perspective of **five user profiles**, each emphasizing different priorities:

| Profile | High Weight | Medium Weight | Low / Zero Weight |
|----------|--------------|----------------|-------------------|
| **Family** | Schools, parks, shops, medical services | Libraries | Clubs, bars |
| **Student** | University, gyms, cafes, nightlife | Restaurants | Schools, police |
| **Single** | Bars, cafes, restaurants, gyms | — | — |
| **Pet Owner** | Parks, veterinary clinics, pet shops | — | — |
| **Universal** | Balanced mix of all POIs | — | — |

---

##  Fuzzy Logic Model

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

##  POI Scoring Logic

For each apartment–POI pair, the system computes:

1. **Distance component**  (temporary solution)

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
