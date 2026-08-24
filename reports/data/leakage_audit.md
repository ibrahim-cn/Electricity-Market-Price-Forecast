# Feature Availability and Data Leakage Audit

**Dataset:** `data/processed/merged/merged_energy_weather.parquet`  
**Shape:** 35064 rows × 103 columns (loaded and counted; file was not modified)  
**Target:** `price day ahead`  
**Audit type:** classification only. No model, no split, no feature matrix, no column deletion.

Column-level machine-readable catalog: `reports/data/feature_availability.csv` (103/103 columns).

This audit classifies **the value stored on the same row as target hour t**. Lagged features do not exist in the parquet yet; they are discussed as a *future* architecture, not implemented.

---

## 1. Information boundary

The model predicts the **day-ahead market price for delivery hour t** *before delivery day D occurs*.

A stricter operational reading, consistent with OMIE Mercado Diario, is:

- Delivery day = **D** (the calendar day that contains hour t, in Europe/Madrid).
- Forecast origin **τ** is on **D-1**, and for a true ex-ante auction model **τ ≤ D-1 12:00 CET** (typical gate closure). All 24 hourly prices of D are determined in **one** session and published together (~12:40–13:00 CET on D-1).
- The parquet does **not** contain bid, gate, or publication timestamps. The 12:00 CET figure is market structure, not a field in the file. The binding rule used below is the user-stated boundary: **no actual observation that occurs at delivery hour t**.

### What may be known at τ

| Information | Known at τ? |
|---|---|
| Calendar of hour t (hour, weekday, month, holiday, DST) | Yes — deterministic |
| `price day ahead` for hours on day D (including t) | **No — this is the target / same auction** |
| `price day ahead` for delivery days ≤ D-1 | Yes — D-1’s curve was published on D-2 |
| `price actual` at t | **No** |
| Same-hour `generation *`, `total load actual`, weather at t | **No** |
| TSO day-ahead load / wind / solar **forecasts for hour t** | Typically yes, if published before τ — **not verified in-file** |
| Actuals with timestamp ≤ τ minus operational delay | Yes in principle |
| Actuals on D-1 after τ (afternoon of the bidding day) | **No** |

### Consequence for lags

“t−1, t−24, t−48, t−168 allowed where logically appropriate” does **not** mean every lag of every series is safe.

- A lag is allowed only if the **underlying event was already observed or published at τ**.
- If the entire day-D curve is predicted at one origin on D-1 noon, then **t−24 actuals for evening hours of D** are actuals on D-1 evening — **not yet observed** at τ.
- Conservative historical actuals: **t−48 and t−168** (complete prior days).
- `price day ahead` lags of **t−24 / t−48 / t−168** are safer than actuals at t−24, because yesterday’s *auction* was published on D-2.

---

## 2. Automatic classification of all 103 columns

Every parquet column was classified by name and role. Zero columns left unclassified.

| category | count | meaning for the **row-aligned** value at t |
|---|---:|---|
| `target` | 1 | Label y_t; not a feature |
| `forbidden_leakage` | 22 | Same-hour realized price / load / generation |
| `potentially_valid_day_ahead` | 3 | Named day-ahead forecasts for hour t |
| `ambiguous` | 75 | Same-hour weather observations (and 3h precip) |
| `identifier` | 2 | Clock fields; index / calendar source only |

No column was silently moved from ambiguous → usable.

---

## 3. Group A — Definitely forbidden / leakage

These are **forbidden as features at time t** (the value on the same row as the target).

### A1. `price actual` — critical

- Different series from `price day ahead` (not a duplicate).
- Realized during/after delivery hour t, **after** the day-ahead auction.
- Project rule: **never a feature**, including lags, rolling stats, or encodings.

### A2. Same-hour `total load actual` — high

- Realized national demand at t.
- Metered on the delivery day; unknown at D-1 gate.

### A3. Same-hour `generation *` — high (20 columns)

Realized mix at t. Dispatch happens on D, after clearing.

- `generation biomass`
- `generation fossil brown coal/lignite`
- `generation fossil coal-derived gas`
- `generation fossil gas`
- `generation fossil hard coal`
- `generation fossil oil`
- `generation fossil oil shale`
- `generation fossil peat`
- `generation geothermal`
- `generation hydro pumped storage consumption`
- `generation hydro run-of-river and poundage`
- `generation hydro water reservoir`
- `generation marine`
- `generation nuclear`
- `generation other`
- `generation other renewable`
- `generation solar`
- `generation waste`
- `generation wind offshore`
- `generation wind onshore`

Near-constant zeros (coal-derived gas, oil shale, peat, geothermal, marine, wind offshore) are still **leakage if used at t**. They are not “safe because they are zero.”

### A4. Target-derived / future-looking constructs (not in the file, still forbidden)

Do not create later:

- `price day ahead` at t as a feature
- any hour of day D’s `price day ahead` used to predict another hour of D (same auction)
- rolling / expanding stats of the target that include t or future hours
- target encoding of hour/weekday using the **full** dataset
- any feature that uses rows with `timestamp_utc > τ`

### A5. Future target values

`y_{t+k}` for k ≥ 0 in the same auction, or any future delivery day’s unpublished price.

---

## 4. Group B — Potentially valid day-ahead information

### B1. Columns already in the parquet (contemporaneous forecast *for* t)

| column | why potentially valid | residual risk |
|---|---|---|
| `total load forecast` | Named day-ahead demand forecast for hour t | Publication time not in file |
| `forecast solar day ahead` | Named day-ahead solar forecast for t | Same |
| `forecast wind onshore day ahead` | Named day-ahead onshore-wind forecast for t | Same |

These are **not** actuals. They are the only row-aligned production/demand-like fields that can be used at t **if** we accept the name as evidence they were produced for the day-ahead process.

They stay **medium** risk: a post-clearing revised forecast would leak; we cannot verify vintage.

### B2. Allowed in principle, **not present** as columns yet

- Historical `price day ahead` (see §7)
- Calendar / time features derived from `timestamp_utc` (see proposed Group 1)
- Historical generation, load, weather **with timestamp ≤ τ**
- Appropriately lagged demand / generation / weather

Do not implement these in this audit.

---

## 5. Group C — Ambiguous (do not silently include)

### C1. Same-hour weather (75 columns)

All `*_Barcelona|_Bilbao|_Madrid|_Seville|_Valencia` weather fields are **observations at t**, not weather forecasts.

Families: `temp`, `temp_min`, `temp_max`, `pressure`, `humidity`, `wind_speed`, `wind_deg`, `rain_1h`, `rain_3h`, `snow_3h`, `clouds_all`, `weather_id`, `weather_main`, `weather_description`, `weather_icon`.

Why ambiguous, not automatically valid:

- No issue / publication time in the dataset.
- Observed weather at delivery hour t is not known before day D.
- Using them as if they were day-ahead weather forecasts would be leakage or nowcasting.

**Decision:** do not put same-hour weather into the model. Lagged weather is a later, explicit Group 3 choice.

### C2. Extra ambiguity: `rain_3h_*`, `snow_3h_*`

3-hour window direction is undocumented. The window may include time after t. Higher risk than `rain_1h`.

### C3. Same-hour actual generation and load

Listed again because they sit in both “forbidden as contemporaneous features” and “ambiguous as lagged features”:

- As **X_t**: forbidden (Group A).
- As **X_{t−k}**: only valid if t−k ≤ τ minus delay. t−1 on day D is not available under this boundary. t−24 is only safe for hours of D that map to D-1 hours already elapsed at τ.

### C4. Anything with unknown publication time

Includes the three forecast columns (potential, not proven) and all weather. Forecasts are in Group B because the **column names** claim day-ahead vintage. Weather names claim observations.

---

## 6. Special columns

### `price day ahead` — TARGET

- Completeness: 0 missing; min 2.06; max 101.99 (from prior quality report).
- Availability: published after the D-1 auction for all 24 hours of D together.
- Use: **y only**.

### `price actual` — FORBIDDEN

- Never in X. Not a substitute target for this project.

### `total load actual` vs `total load forecast`

| | `total load actual` | `total load forecast` |
|---|---|---|
| Nature | Realized demand at t | Forecast for t |
| Same-hour use | Forbidden | Potentially valid |
| Missing in parquet | 14 | 0 |

### `generation *` vs `forecast * day ahead`

| | generation actuals | `forecast solar/wind onshore day ahead` |
|---|---|---|
| Nature | Realized at t | Forecast for t |
| Same-hour use | Forbidden | Potentially valid |

There is **no** day-ahead forecast column for nuclear, hydro, gas, coal, etc. Those series can enter only as **lags**, not as t.

### Weather

Observed, city-level, aligned to t. Ambiguous contemporaneously. No NWP forecast columns exist in this parquet.

### Identifiers

`time` (original offset string) and `timestamp_utc` (UTC-aware). Known in advance. **Not** raw numeric features. Source of calendar features only.

---

## 7. Can historical `price day ahead` be used?

**Yes**, if and only if the lag belongs to a **previous auction**, already published at τ.

OMIE clears **all 24 hours of D in one session**. Therefore intra-day lags of the target on day D leak the same session.

| lag | Same hour last… | Acceptable? | Why |
|---|---|---|---|
| t | current hour | **No** | Target |
| t−1 | previous hour | **No as a blanket lag** | For hours 1–23 of D, t−1 is another hour of the **same** auction (all published together). Only hour 00 of D has t−1 = hour 23 of D−1 (published D−2). A single lag-1 feature would leak for most hours. |
| t−24 | previous delivery day, same hour | **Yes** | D−1 curve published on D−2, before τ on D−1 |
| t−48 | two days back | **Yes** | Prior auction |
| t−72 | three days back | **Yes** | Prior auction |
| t−168 | previous week, same hour | **Yes** | Weekly seasonality; published well before τ |
| rolling mean ending at t | — | **No** | Includes y_t |
| rolling mean ending at t−24 | — | **Yes, later** | Uses only published auctions |
| hour-of-week target mean on all rows | — | **No** | Full-sample target encoding / future leakage |

**Recommended target lags (not implemented):** `t−24`, `t−48`, `t−168`.  
**Not recommended:** `t−1` of `price day ahead`.

Do **not** build lags of `price actual`.

---

## 8. Methods this project will not use

- Random train/test split, KFold, or row shuffle (breaks time order; can put future hours in train).
- Target encoding on the full dataset.
- Features computed with future rows (center-aligned rolling, bidirectional fill, global z-score using test period).
- Same-hour actuals treated as “explanatory nowcast” for a day-ahead target.

When splitting is eventually done: **time-based** cut on `timestamp_utc` only. Not done in this audit.

---

## 9. Proposed leakage-safe feature architecture

**Not implemented.** Design only.

### GROUP 1 — Calendar features

Derive later from `timestamp_utc` (Europe/Madrid local clock), never from the target:

- hour of day, day of week, month
- weekend flag
- optional Spanish public holiday flag (external calendar, known in advance)
- DST / UTC offset (already in raw `time`; do not drop DST)
- optional cyclic encodings (sin/cos hour, sin/cos day-of-year)

### GROUP 2 — Day-ahead forecasts / information available before delivery

Use at t only these existing columns (publication time still unverified):

- `total load forecast`
- `forecast solar day ahead`
- `forecast wind onshore day ahead`

Do **not** add same-hour weather or same-hour generation here.

### GROUP 3 — Lagged historical information

Create later, with `timestamp_utc` shifts only (no shuffle):

- `price day ahead` at **t−24, t−48, t−168**
- optional: those three forecast series at t−24 (previous day’s forecast vintage)
- `total load actual` and selected `generation *` at **t−48, t−168** (safer than t−24 under a D−1 noon origin)
- weather observations at **t−48, t−168** (same-hour weather stays out)

If a single origin τ = D−1 12:00 is enforced strictly, any lag whose source timestamp is after τ is invalid — including some t−24 actuals.

### FORBIDDEN FEATURES

Do not use now or later as X:

1. `price actual` (any lag, any transform)
2. `price day ahead` at t (target)
3. `price day ahead` at any other hour of the same delivery day D
4. `total load actual` at t
5. every `generation *` column at t
6. every same-hour weather column (`temp_*` … `weather_icon_*`)
7. rolling/expanding statistics that include t or future times
8. full-dataset target encodings
9. raw `time` / `timestamp_utc` as numeric model inputs
10. any feature requiring `price actual`

Ambiguous items (same-hour weather; t−1 target; t−24 actuals) are **not** in Groups 1–2. They enter Group 3 only under an explicit lag-and-origin rule.

---

## 10. Catalog (compact)

Full rows: `reports/data/feature_availability.csv`.

Fields: `column_name`, `category`, `data_type`, `target_related`, `leakage_risk`, `availability_status`, `contemporaneous_use_at_t`, `lagged_use`, `reasoning`.

| category | columns |
|---|---|
| target | `price day ahead` |
| identifier | `time`, `timestamp_utc` |
| potentially_valid_day_ahead | `total load forecast`, `forecast solar day ahead`, `forecast wind onshore day ahead` |
| forbidden_leakage | `price actual`, `total load actual`, 20 × `generation *` |
| ambiguous | 75 weather columns |

---

## 11. Issues requiring attention (next phase, not this one)

1. Confirm REE/OMIE publication times for the three forecast columns if a paper-grade claim of “available before gate” is required.
2. Choose a single forecast origin (D−1 12:00 vs “any time before day D”) before building lags.
3. Same-hour weather stays out until a lagged policy is written down.
4. Remaining energy NaNs (6 generation hours, 14 load hours) interact with lags; no new imputation here.
5. Weather outliers remain in the file; unused while weather is excluded from contemporaneous X.

---

## 12. Recommended next step

Write a **leakage-safe feature specification** (still no model): calendar extractors, the three day-ahead forecast columns, and explicit lag table (`t−24/48/168` for target; `t−48/168` for actuals). Then a time-based split on `timestamp_utc`. Do not fit a model until that spec is locked.

**Parquet and raw CSVs were not modified.**
