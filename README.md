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
## Profile Weights and Preferences

Each profile is defined by two complementary configuration layers:

### 1) POI preference weights (`W_POI`)

`W_POI[profile]` is a dictionary of **logic POI categories** mapped to **signed weights**:

- **positive weights** → the profile *wants* the amenity nearby (increases POI score)
- **negative weights** → the profile *dislikes* the amenity nearby (penalizes POI score)

This allows the same POI evidence to be interpreted differently across profiles.  
Example intuition:
- For `rodzinny` (family), *schools*, *healthcare*, and *parks* are strongly positive, while *clubs* and *pubs* are negative.
- For `studencki` (student), nightlife and universities may be strongly positive.

The POI layer is computed **offline** from `time_to_poi` relations and then aggregated using a **signed weighted mean** (see the scoring pipeline below).

### 2) Component mixing weights (`WB`)

`WB[profile]` defines the fallback (stabilization) mixture weights for:

- `POI`   — POI score (accessibility-driven)
- `CENA`  — price score (from fuzzy memberships)
- `M2`    — size score (from fuzzy memberships)
- `ROOMS` — room count score (from fuzzy memberships)
- `ZDJ`   — style score (used as the “interior/style” component)

The design enforces **POI dominance** by assigning the largest share to `POI` in each profile’s `WB`.

---

## Style Scoring

The apartment `style` is normalized into one of the supported keys:

- `MODERN`, `CLASSIC`, `INDUSTRIAL`, `SCANDINAVIAN`, `MINIMALIST`, `VINTAGE`, `OTHER`, `UNKNOWN`

Each profile provides a lookup table:

- `STYLE_SCORES[profile][style_key] ∈ [0..1]`

Behavior:
- If the input `style` is missing or empty → the system uses `UNKNOWN`
- If the input `style` is present but not recognized → it maps to `OTHER`

The resulting `style_s` participates in the fallback mixture via `WB[profile]["ZDJ"]`.

---

## Price Excess Penalty

In addition to fuzzy price evaluation, the pipeline applies an explicit **excess penalty** if:

- `price_per_m2 > expensive_min` (where `expensive_min` is the city-tier-specific “expensive” threshold)

The penalty grows linearly with the exceedance and is controlled by:

- `PRICE_PENALTY_STEP` (PLN/m² per step)
- `PRICE_PENALTY_PER_STEP` (score in `[0..1]` subtracted per step)
- `PRICE_PENALTY_MAX` (cap in `[0..1]`)

This prevents extremely overpriced apartments from ranking too high even if their POI accessibility is excellent.

---

## How the Score Is Computed

The pipeline computes the final score in eight major steps.

### 1) POI Feature from Time

Each POI relation provides `time_to_poi`, converted to minutes:

- if `TIME_UNIT=seconds` → `t_min = t / 60`
- if `TIME_UNIT=minutes` → `t_min = t`

A fuzzy proximity feature is computed (exactly as in `time_fuzzy()`):

- `t ≤ 5`       → `feat = 1.0`
- `5 < t ≤ 10`  → linearly decreases from `1.0` to `0.6`
- `10 < t ≤ 15` → linearly decreases from `0.6` to `0.3`
- `t > 15`      → `feat = 0.0`

Only POIs with `feat > 0` contribute to scoring.

---

### 2) Offline POI Score per Profile (Signed Weighted Mean)

For each apartment and profile:

1. Map OSM categories → logic categories (`OSM_TO_LOGIC`).
2. For each logic category, keep the **best** (maximum) `feat` across all relations.
3. Evaluate all categories listed in `W_POI[profile]`:
   - if a category is missing → it contributes `0.0`
4. Compute a **signed weighted mean**:
   - values are in `[0..1]`
   - weights may be positive or negative
   - normalization uses `sum(|w|)` so negative weights act as true penalties

This yields `poi_score` (conceptually in `[-1..1]`), used as the POI signal for inference and mixing.

---

### 3) Fuzzification of Price / Size / Rooms

For each profile, the system computes membership degrees:

**Price memberships** (city-tier-aware):
- `Cheap`, `Mid`, `Expensive`

**Size memberships** (profile-specific target):
- `Small`, `Target`, `Large`

**Rooms memberships** (profile-specific target range):
- `TooFew`, `Target`, `TooMany`

All membership values are clamped to `[0..1]`.

---

### 4) Component Scores (Price/Size/Rooms/Style)

Membership degrees are converted into scalar component scores in `[0..1]`:

- `price_s` computed as a weighted average of `PRICE_RULES[profile]` using price memberships
- `size_s` computed as a weighted average of `SIZE_RULES[profile]` using size memberships
- `rooms_s` computed as a weighted average of `ROOMS_RULES[profile]` using rooms memberships
- `style_s` read from `STYLE_SCORES[profile]` using the normalized `style`

---

### 5) Fallback Score

A fallback score is always computed to stabilize inference:

```text
fallback_full =
  (wPOI   * poi_score +
   wCENA  * price_s +
   wM2    * size_s +
   wROOMS * rooms_s +
   wZDJ   * style_s) / sum(w)
```
This guarantees a valid output even if no Sugeno rule fires.

## 6) Sugeno Inference (Rule Base)

Each **profile** defines its own list of **zero-order Sugeno rules**.

### Antecedents (Linguistic Labels)

Each rule antecedent is a conjunction of linguistic conditions:

- **POI** ∈ {`Low`, `Mid`, `High`}
- **PRICE** ∈ {`Cheap`, `Mid`, `Expensive`}
- **SIZE** ∈ {`Small`, `Target`, `Large`}
- **ROOMS** ∈ {`TooFew`, `Target`, `TooMany`}

### Consequent

- Each rule consequent is a **constant value**  
  \[
  c \in [0, 1]
  \]
- No defuzzification is required at the rule level (zero-order Sugeno).

### Rule Activation

For a given apartment and profile, each rule is activated as:

\[
\text{activation}_i = \min \bigl(
\mu_{\text{POI}},
\mu_{\text{PRICE}},
\mu_{\text{SIZE}},
\mu_{\text{ROOMS}}
\bigr)
\]

where \(\mu\) denotes the membership value of the corresponding linguistic label.

### Sugeno Aggregation

The final Sugeno output for a profile is computed as a weighted average:

\[
\text{final\_sugeno} =
\frac{\sum_i \text{activation}_i \cdot c_i}
     {\sum_i \text{activation}_i}
\]

### Fallback Rule

If no rule fires:

\[
\sum_i \text{activation}_i = 0
\]

then:

\[
\text{final\_sugeno} = \text{fallback\_full}
\]

---

## 7) Final Mixing and Penalty

The final continuous score in the \([0, 1]\) range is obtained by mixing the
Sugeno result with the fallback score:

\[
\text{final01} =
\alpha \cdot \text{final\_sugeno}
+ (1 - \alpha) \cdot \text{fallback\_full}
\]

where:

- \(\alpha \in [0, 1]\) is a global mixing coefficient.

### Price Excess Penalty

A price-based penalty is then applied:

\[
\text{final01} = \text{clamp01}(\text{final01} - \text{penalty})
\]

where `clamp01(x)` restricts the value to the \([0, 1]\) interval.

---

## 8) Scaling to 0–100

The final profile score is scaled to an integer range:

1. Scale:
   \[
   \text{score} = \text{final01} \cdot 100
   \]

2. Round to nearest integer

3. Clamp:
   \[
   \text{score} \in [0, 100]
   \]

This value is written as the **final profile score** in the output JSON.

---

## Interpretative Labels (Enums)

The system additionally produces **human-readable descriptors**:

- `poi_desc` ∈ {`LOW`, `MEDIUM`, `HIGH`}
- `price_desc` ∈ {`CHEAP`, `AVERAGE`, `EXPENSIVE`}
- `size_desc` ∈ {`SMALL`, `MEDIUM`, `LARGE`}

### Design Choice

- Descriptors are computed **once per apartment**
- They always use the **universal profile**
- They are derived by selecting the linguistic label with the **highest membership value**
- The descriptors are **profile-independent** and ensure consistent interpretation across all scoring profiles
