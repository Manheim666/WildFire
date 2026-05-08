# MANHEIM — Wildfire Prediction & Risk Intelligence for Azerbaijan

> End-to-end machine learning pipeline that predicts wildfire risk across 16 Azerbaijani cities using satellite fire detections, weather reanalysis data, and a **role-based blended ensemble**. Produces daily (30-day) and hourly (168-hour) risk forecasts with an interactive web dashboard. **Catches 71% of fire-days** with a recall-first threshold.
>
> **Final metrics (test set, 2025):** Recall=0.714 · Precision=0.262 · F1=0.383 · F2=0.530 · PR-AUC=0.360 · ROC-AUC=0.870

**Built for presentation to the Ministry of Ecology and Natural Resources of Azerbaijan.**

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Notebooks](#notebooks)
5. [Project Structure](#project-structure)
6. [Models & Methodology](#models--methodology)
7. [Dashboard](#dashboard)
8. [Cities Covered](#cities-covered)
9. [Data Sources](#data-sources)
10. [Setup & Installation](#setup--installation)
11. [Glossary](#glossary)
12. [License](#license)

---

## Overview

**MANHEIM** ingests multi-source environmental data — Open-Meteo weather reanalysis (ERA5/ERA5-Land), NASA FIRMS satellite fire detections (MODIS + VIIRS), terrain elevation, vegetation indices, and land-cover statistics — then engineers 200+ predictive features and trains recall-optimized classifiers to forecast wildfire ignition probability for each of 16 cities across Azerbaijan.

The system operates at two temporal resolutions:

- **Daily pipeline** — 30-day rolling forecast with 14 models (8 baselines + 3 Optuna-tuned + 3 ensembles), role-based blended ensemble selection, isotonic probability calibration, and SHAP explainability
- **Hourly pipeline** — 168-hour (7-day) forecast with daytime-masked labels for clean signal separation

Risk levels are classified into four tiers: **Low** (< 15%), **Moderate** (15–35%), **High** (35–60%), **Extreme** (> 60%). The threshold is tuned to **maximize recall** — in wildfire early warning, missing a real fire is far more dangerous than a false alarm.

---

## Key Features

- **Multi-source data fusion** — weather, satellite fire, terrain, vegetation, land cover, population
- **200+ engineered features** — FWI family (FFMC, DMC, DC, ISI, BUI), VPD, dew point, heat index, drought proxy, dry-spell tracking, lag/rolling aggregates (1–30 day), Prophet seasonal residuals, cyclical time encodings, historical fire rates
- **Focused model selection** — 1 baseline (LogReg) + 3 strong candidates (CatBoost, XGBoost, LightGBM) + Optuna tuning + Role-Blended Ensemble as champion
- **Role-Based Blended Ensemble** — classifies each model as recall-specialist (2× weight), precision-specialist (1×), or balanced (1.5×); applies consensus boost and recall safety net
- **Bayesian hyperparameter optimization** — Optuna with 100 trials per GBT model, precision-constrained search
- **Recall-first threshold tuning** — F2-maximized at threshold=0.085 (not default 0.50); validated on held-out 2025 data
- **Isotonic probability calibration** — predicted probabilities match observed fire frequencies
- **SHAP explainability** — every prediction can be decomposed into feature contributions
- **3-way temporal split** — train (< 2024), validation (2024), test (≥ 2025) — test data never seen during training
- **Interactive web dashboard** — Leaflet risk map, daily/hourly toggle, forecast strip, detail panel, Plotly charts, filterable table
- **Folium + Plotly visualizations** — date-selectable HTML maps, animated dashboards, climate trend figures
- **Colab + local compatible** — runs identically on Google Colab and local JupyterLab/VS Code

---

## Pipeline Architecture

```
 ┌────────────────────────────────────────────────────────────────┐
 │              MANHEIM Pipeline — Run NB1 → NB6                  │
 │                                                                │
 │  NB1  Data Ingestion                                           │
 │  ├── Open-Meteo Archive API (ERA5 + ERA5-Land, 2012–present)   │
 │  ├── NASA FIRMS (MODIS C6.1 + VIIRS C2, 3 sensors)            │
 │  ├── Open-Elevation + GEE vegetation indices                   │
 │  └── → master_daily.parquet, master_hourly.parquet             │
 │          │                                                     │
 │          ▼                                                     │
 │  NB2  EDA & Feature Engineering                                │
 │  ├── 200+ features: FWI, VPD, lags, rolling, Prophet residuals │
 │  ├── Outlier detection, fire-weather hypothesis tests          │
 │  └── → engineered_daily.parquet, engineered_hourly.parquet     │
 │          │                                                     │
 │     ┌────┴──────────────────┐                                  │
 │     ▼                       ▼                                  │
 │  NB3  Weather Forecast   NB4  Wildfire Detection               │
 │  ├── Prophet + XGBoost   ├── 14 models (8 base + 3 Optuna     │
 │  ├── 30-day + 168-hour   │   + 3 ensembles incl. RoleBlend)   │
 │  └── stacking ensemble   ├── SHAP + isotonic calibration       │
 │                           └── recall-first, prec ≥ 0.30        │
 │     │                       │                                  │
 │     └───────────┬───────────┘                                  │
 │                 ▼                                              │
 │  NB5  Risk Prediction & Dashboard                              │
 │  ├── 30-day + 168-hour wildfire risk per city                  │
 │  ├── Folium maps, Plotly animated dashboards                   │
 │  └── JSON/CSV export → web dashboard                           │
 │                 │                                              │
 │                 ▼                                              │
 │  NB6  Climate Report                                           │
 │  └── Trend analysis, forecast vs history, risk outlook         │
 │                                                                │
 │  src/  Shared Python Module                                    │
 │  └── config · features · modeling · evaluation · visualization │
 └────────────────────────────────────────────────────────────────┘
```

---

## Notebooks

| # | Notebook | Purpose | Runtime |
|---|----------|---------|---------|
| 1 | `01_Data_Ingestion.ipynb` | Ingest weather, fire, terrain, vegetation data for 16 cities | ~5–15 min |
| 2 | `02_EDA_FeatureEngineering.ipynb` | EDA, hypothesis tests, 200+ feature engineering (daily + hourly) | ~3–5 min |
| 3 | `03_Weather_TimeSeries.ipynb` | Prophet + XGBoost stacking forecasts (30-day daily + 168h hourly) | ~60–120 min |
| 4 | `04_Wildfire_Detection.ipynb` | 14-model classification (incl. MLP + RoleBlend ensemble), Optuna 100 trials, SHAP, calibration | ~30–60 min |
| 5 | `05_Risk_Prediction_Dashboard.ipynb` | Risk scoring, Folium/Plotly maps, dashboard JSON export | ~2–5 min |
| 6 | `06_Climate_Report.ipynb` | Climate trend analysis, forecast vs history, risk outlook | ~1–2 min |

**Run in order: NB1 → NB2 → NB3 → NB4 → NB5 → NB6.** Each notebook auto-detects the project root.

---

## Project Structure

```
WildFire-Prediction/
├── notebooks/                        Run sequentially: NB1 → NB6
│   ├── 01_Data_Ingestion.ipynb
│   ├── 02_EDA_FeatureEngineering.ipynb
│   ├── 03_Weather_TimeSeries.ipynb
│   ├── 04_Wildfire_Detection.ipynb
│   ├── 05_Risk_Prediction_Dashboard.ipynb
│   └── 06_Climate_Report.ipynb
│
├── src/                              Shared Python module
│   ├── config.py                     Paths, constants, city coordinates
│   ├── features.py                   FWI, VPD, lags, rolling, anomaly features
│   ├── modeling.py                   Model factory functions
│   ├── evaluation.py                 Metrics, threshold tuning, leaderboards
│   ├── visualization.py              Plotting helpers (confusion matrices, PR, SHAP)
│   ├── utils.py                      Data I/O utilities
│   └── prediction_pipeline.py        End-to-end scoring pipeline (RoleBlendedEnsemble, Optuna, MLP)
│
├── data/
│   ├── raw/                          Open-Meteo cache, FIRMS archives, legacy CSVs
│   ├── processed/                    Engineered parquet files (master, engineered, fires)
│   └── reference/                    Static geography, city coordinates
│
├── models/
│   ├── wildfire/                     Fire models, manifests, feature lists
│   ├── weather/                      Weather forecast model bundles
│   └── prophet_cache/                Cached Prophet models per city/variable
│
├── outputs/                          Pipeline artefacts (forecasts, risk scores)
├── reports/
│   ├── figures/                      Publication-quality figures
│   ├── maps/                         Interactive Folium/Plotly HTML maps
│   └── metrics/                      CSV leaderboards and summaries
│
├── dashboard/                        Standalone web dashboard (HTML/JS/CSS)
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── data/                         JSON data files for the dashboard
│
├── docs/                             Documentation & GitHub Pages
│   ├── model_card.md                 Model Card (architecture, metrics, ethics)
│   ├── executive_summary.md          Non-technical summary for stakeholders
│   ├── technical_summary.md          Full technical report
│   └── presentation_talking_points.md  Speaking notes for demos
│
├── scripts/                          Standalone analysis scripts
│   ├── generate_presentation_outputs.py  All plots + CSV outputs
│   └── threshold_analysis.py         Threshold sweep & selection
│
├── run_pipeline.py                   Single-command pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Models & Methodology

### Wildfire Detection (NB4)

**Objective:** Maximize recall while maintaining reasonable precision. Missing a wildfire is far more costly than a false alarm.

| Stage | Detail |
|-------|--------|
| **Data split** | Train (< 2024), Validation (2024), Test (≥ 2025) — strict temporal separation |
| **Feature pruning** | Remove near-zero-variance + highly correlated (r > 0.95) features |
| **Base models** | LogisticRegression (baseline), CatBoost, XGBoost, LightGBM |
| **Class weighting** | Cost-sensitive learning with `scale_pos_weight` / `class_weight='balanced'` |
| **Optuna tuning (3)** | 100 trials each for XGBoost, LightGBM, CatBoost; objective: `0.6×Recall + 0.4×F1` |
| **Ensembles (3)** | SoftVoting (avg top-3), Stacking (LogReg meta-learner), **RoleBlend** (weighted blend of all 11 models) |
| **RoleBlend strategy** | Recall-specialists 2× weight, precision-specialists 1×, balanced 1.5×; consensus boost (+10% if ≥60% agree); recall safety net |
| **Threshold tuning** | Precision ≥ 0.30, Recall ≥ 0.70; fallback cascade relaxes precision to 0.20 then maximises F1 |
| **Calibration** | Isotonic regression on validation set → meaningful probability scores |
| **Explainability** | SHAP TreeExplainer with top-25 feature importance |
| **Overfitting guard** | Train-vs-val F1 gap < 15%; flagged otherwise |

### Latest Pipeline Results

| Model | Recall | Precision | F1 | PR-AUC | Role |
|-------|-------:|----------:|---:|-------:|------|
| **RoleBlend_All** | **0.714** | **0.262** | **0.383** | **0.360** | **Selected** |
| SoftVoting_Top3 | 0.685 | 0.269 | 0.387 | 0.358 | Ensemble |
| CatBoost | 0.622 | 0.294 | 0.400 | 0.356 | Baseline |
| Stacking_Top3 | 0.575 | 0.308 | 0.401 | 0.367 | Ensemble |
| XGBoost | 0.591 | 0.309 | 0.406 | 0.351 | Baseline |
| MLP_Neural | 0.638 | 0.269 | 0.379 | 0.339 | Baseline |

### Hourly Pipeline (NB4 Part B)

- Daytime label masking: relabels nighttime fire-hours to 0 (eliminates noisy labels broadcast from daily resolution)
- Separate Optuna-tuned CatBoost + XGBoost models
- 70 hourly-specific features including hourly lag/rolling with `h` suffix
- Daily-anchored ceiling caps hourly probabilities to prevent over-prediction
- Historical city medians for feature imputation (not zero-fill)

### Weather Forecasting (NB3)

- Prophet (yearly/weekly/daily seasonality) + XGBoost (recursive multi-step) stacking per city per variable
- 8-model comparison: Ridge, ElasticNet, RF, ExtraTrees, HistGBR, XGB, LGB, CatBoost
- 128+ model bundles (16 cities × 8 weather variables)

### Evaluation Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Recall** | ≥ 0.70 | Must catch real fires |
| **Precision** | ≥ 0.30 | Raised from 0.10 to reduce false alarms |
| **PR-AUC** | ≥ 0.20 | Primary ranking metric for imbalanced data |
| **Overfit gap** | < 0.15 | Train-vs-val F1 difference |
| **Selection score** | `0.35×PR-AUC + 0.30×F1 + 0.20×Recall + 0.15×Precision` | Composite ranking |

---

## Dashboard

The interactive web dashboard provides real-time wildfire risk monitoring:

- **Daily / Hourly toggle** — switch between 30-day and 168-hour forecasts
- **Leaflet risk map** — colour-coded markers for all 16 cities
- **Forecast strip** — scrollable daily/hourly risk cards
- **Detail panel** — city-specific weather conditions and risk breakdown
- **Plotly charts** — risk trend and weather condition plots
- **Filterable table** — sort by date, city, risk level

**Live dashboard:** The dashboard is published from the `docs/` folder by the GitHub Actions workflow in `.github/workflows/pages.yml`.

To enable it on GitHub:

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Set **Source** to **GitHub Actions**.
4. Run the **Deploy dashboard to GitHub Pages** workflow, or push to `main` / `master`.

The deployed page will be available at:

```text
https://<your-github-username>.github.io/<repository-name>/
```

Serve locally with:

```bash
cd dashboard && python3 -m http.server 8765
```

---

## Cities Covered

16 cities across Azerbaijan's diverse climate zones:

| City | Lat | Lon | Climate Zone |
|------|----:|----:|-------------|
| Baku | 40.41 | 49.87 | Semi-arid coastal; highest fire rate |
| Shabran | 41.21 | 48.99 | Northeastern; major fire events 2021–22 |
| Ganja | 40.68 | 46.36 | Western highland |
| Mingachevir | 40.76 | 47.06 | Central lowland (Kura River) |
| Shirvan | 39.93 | 48.93 | Kura-Araz lowland; dry climate |
| Lankaran | 38.75 | 48.85 | Southern subtropical coastal |
| Shaki | 41.20 | 47.17 | Northern foothill; forested |
| Nakhchivan | 39.21 | 45.41 | Exclave; arid continental |
| Yevlakh | 40.62 | 47.15 | Central plains; dry lowland |
| Quba | 41.36 | 48.53 | Northern mountains |
| Khachmaz | 41.46 | 48.81 | Northeastern Caspian coast |
| Gabala | 41.00 | 47.85 | High elevation; dense forest |
| Shamakhi | 40.63 | 48.64 | Mountain plateau |
| Jalilabad | 39.21 | 48.30 | Southern lowland; agricultural |
| Barda | 40.37 | 47.13 | Karabakh region; central lowland |
| Zaqatala | 41.63 | 46.64 | Northwestern mountain-foothill; lowest fire rate |

Fire labels are aggregated daily within a **20 km radius** of each city centroid using NASA FIRMS detections.

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| **Open-Meteo Archive** | Hourly weather reanalysis (ERA5 + ERA5-Land, 2012–present) | Free, no API key |
| **Open-Meteo Forecast** | 16-day ahead hourly weather | Free, no API key |
| **NASA FIRMS** | Active fire detections (MODIS C6.1 + VIIRS C2) | Free archive CSVs |
| **Open-Elevation** | Terrain elevation + derived slope | Free, no API key |
| **Google Earth Engine** | MODIS burned area, Sentinel-2 NDVI/NDBI | Free (GEE account) |
| **Reference CSVs** | Land cover, urban density, population, roads | Static local files |

### Important Notes

- Fire risk = probability of a FIRMS-detected hotspot, **not** burn area or severity
- Probabilities are isotonically calibrated — use the threshold from `model_manifest.json`
- Weather forecast accuracy degrades beyond day 7; days 1–7 are most reliable
- Evaluate with **PR-AUC and recall**, not accuracy (class imbalance ~8–10% fire-day prevalence)

---

## Setup & Installation

### Local

```bash
git clone https://github.com/your-repo/WildFire-Prediction.git
cd WildFire-Prediction
pip install -r requirements.txt
jupyter lab notebooks/
```

### Run Full Pipeline

```bash
python run_pipeline.py                  # all 6 notebooks
python run_pipeline.py --from 4         # start from NB04
python run_pipeline.py --only 4         # NB04 only
python -m src.pipeline.run_full_pipeline # alternative entry point
```

### Generate Presentation Outputs

```bash
python scripts/generate_presentation_outputs.py  # metrics, plots, city summary
python scripts/threshold_analysis.py              # threshold sweep analysis
```

### Dashboard

```bash
cd dashboard && python3 -m http.server 8765
# Open http://localhost:8765
```

### Google Colab

1. Upload the project folder to Google Drive
2. Open any notebook — the first cell auto-mounts Drive and detects the project root
3. Run notebooks in order: **NB1 → NB2 → NB3 → NB4 → NB5 → NB6**

### Environment Variable (optional)

```bash
export MANHEIM_ROOT=/absolute/path/to/project
```

### Git LFS

Large data files (`.parquet`, `.csv`, `.pkl`) are tracked via Git LFS:

```bash
git lfs install && git lfs pull
```

### Dependencies

| Group | Key Packages |
|-------|-------------|
| **Core** | pandas, numpy, pyarrow, joblib, tqdm |
| **Ingestion** | openmeteo-requests, requests-cache, retry-requests |
| **EDA** | scipy, statsmodels, prophet, matplotlib, seaborn |
| **ML** | scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn, optuna, shap, MLPClassifier (sklearn) |
| **Visualization** | folium, plotly |

---

## Glossary

| Term | Description |
|------|-------------|
| **FWI** | Fire Weather Index — composite wildfire danger metric |
| **FFMC** | Fine Fuel Moisture Code — surface litter dryness |
| **DMC/DC** | Duff Moisture / Drought Code — subsurface drought indicators |
| **ISI/BUI** | Initial Spread Index / Buildup Index — fire spread and fuel availability |
| **VPD** | Vapor Pressure Deficit — atmospheric dryness (higher = more fire risk) |
| **FIRMS** | NASA Fire Information for Resource Management System |
| **MODIS/VIIRS** | Satellite sensors for thermal anomaly detection (~1 km / ~375 m resolution) |
| **PR-AUC** | Precision-Recall Area Under the Curve — primary metric for imbalanced classification |
| **SHAP** | SHapley Additive exPlanations — model interpretability method |
| **Isotonic calibration** | Non-parametric probability calibration for reliable risk scores |

---

## Presentation Outputs

All files needed for stakeholder presentations are in `outputs/` and `reports/figures/`:

| File | Description |
|------|-------------|
| `outputs/final_metrics.json` | Final model performance |
| `outputs/final_threshold.json` | Selected threshold + justification |
| `outputs/final_predictions.csv` | Per-sample test predictions |
| `outputs/city_risk_summary.csv` | City-level recall/precision/fires |
| `outputs/forecast_30_days.csv` | 30-day wildfire risk forecast |
| `reports/figures/confusion_matrix.png` | Confusion matrix |
| `reports/figures/precision_recall_curve.png` | PR curve |
| `reports/figures/roc_curve.png` | ROC curve |
| `reports/figures/feature_importance.png` | Top-25 features |
| `reports/figures/city_recall.png` | Recall by city chart |
| `reports/figures/threshold_tradeoff.png` | Threshold sweep plot |
| `docs/model_card.md` | Model Card |
| `docs/executive_summary.md` | Executive Summary |
| `docs/technical_summary.md` | Technical Report |
| `docs/presentation_talking_points.md` | Speaking Notes |

---

## Limitations

1. **Precision vs recall trade-off:** No threshold achieves recall ≥ 0.70 AND precision ≥ 0.30 simultaneously
2. **City variation:** Low-fire cities (Nakhchivan, Zaqatala, Shabran) have poor recall
3. **Feature dominance:** Historical fire counts are the strongest features — model partially learns "fires beget fires"
4. **Forecast horizon:** Weather forecast accuracy degrades beyond 7–10 days
5. **Ignition source:** Model predicts fire-weather conditions, not human-caused ignitions

---

## Future Improvements

1. Real-time weather API integration for automated daily predictions
2. Satellite vegetation freshness (real-time NDVI) for fuel dryness
3. Expand to rural areas and additional cities
4. Mobile alert system for field personnel
5. Validate against Ministry of Ecology incident records
6. Periodic retraining as climate patterns evolve

---

## License

This project is intended for research and environmental monitoring purposes. Data sources are publicly available under their respective terms of use.
