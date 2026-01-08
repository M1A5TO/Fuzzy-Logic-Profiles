# Apartment Scoring (15-Minute City)
## POI-Dominant Fuzzy Logic System

This repository contains an **apartment attractiveness scoring system** inspired by the *15-minute city* concept.  
The system computes **profile-specific attractiveness scores (0–100)** using:

- accessibility of Points of Interest (POIs) measured by walking time,
- price per square meter (city-aware thresholds),
- apartment size and number of rooms,
- interior style,
- **fuzzy logic inference (Sugeno model)**.

The solution is fully deterministic, explainable.

---

## Key Features

- **POI-dominant scoring** – urban accessibility is the primary driver of the result
- **End-to-end fuzzy logic** – fuzzification → rule inference → Sugeno defuzzification
- **Multiple user profiles** with independent preferences and rules
- **Easy tuning** – weights, rules, thresholds, penalties

---

## User Profiles

| Profile | Output field |
|------|------|
| `rodzinny` (family) | `family_attractiveness` |
| `studencki` (student) | `student_attractiveness` |
| `singiel` (single) | `single_attractiveness` |
| `wlasciciel_psa` (dog owner) | `dog_owner_attractiveness` |
| `uniwersalny` (universal) | `universal_attractiveness` |

Additionally, interpretative labels are generated:

- `poi_desc` ∈ `{LOW, MEDIUM, HIGH}`
- `price_desc` ∈ `{CHEAP, AVERAGE, EXPENSIVE}`
- `size_desc` ∈ `{SMALL, MEDIUM, LARGE}`

All descriptive labels are computed **using the `uniwersalny` profile** to ensure consistency across profiles.

---
## Fuzzy Logic Methodology (TL;DR)

The apartment attractiveness score is computed using a **POI-dominant fuzzy logic system** based on a
**zero-order Sugeno inference model**.

### Overview

1. **Fuzzification**
   - POI accessibility is mapped from walking time to a proximity score in `[0, 1]`.
   - Price, apartment size, and number of rooms are fuzzified into linguistic sets
     (`Cheap/Mid/Expensive`, `Small/Target/Large`, `TooFew/Target/TooMany`).

2. **Rule-Based Inference**
   - Each user profile defines expert rules (e.g. *high POI + low price → high attractiveness*).
   - Rule activation uses the **minimum operator** (logical AND).

3. **Sugeno Aggregation**
   - Rules return constant outputs.
   - Final Sugeno result is the **activation-weighted average** of all fired rules.

4. **Fallback Stabilization**
   - A weighted average of POI, price, size, room, and style scores is always computed
     to ensure stable results.

5. **Final Mixing**
   - Sugeno and fallback scores are combined:
     ```
     final = ALPHA × Sugeno + (1 − ALPHA) × fallback
     ```
   - High `ALPHA` values make the system strongly POI-driven.

6. **Price Penalty**
   - Additional penalty is applied if price exceeds the city-specific “expensive” threshold.

7. **Scaling**
   - The final score is normalized to an integer range **0–100**.

This design provides a **fully explainable**, **offline**, and **profile-aware** apartment
scoring mechanism aligned with the *15-minute city* concept.

