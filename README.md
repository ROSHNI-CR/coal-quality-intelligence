---
title: Coal Quality Intelligence
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Coal Quality Intelligence Platform
## Coal India Limited — Environmental Intelligence Engine

A full-stack AI platform that quantifies how environmental conditions influence coal quality (GCV, Moisture, Ash) before sampling, enabling proactive quality management across 305 mapped mines.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the platform
COAL_EIE_DB_PATH=coal_eie_production_complete.db uvicorn backend.app.main:app --reload

# 3. Open browser
http://localhost:8000
```

---

## Project Structure

```
coal_india/
├── README.md
├── requirements.txt
├── coal_eie_production_complete.db     ← Production database (all modules)
├── frontend_integrated.html            ← Complete single-file frontend
└── backend/
    ├── models/                         ← Trained ML model artifacts (joblib)
    │   ├── gcv_lightgbm_run4.joblib
    │   ├── moisture_lightgbm_run4.joblib
    │   └── ash_lightgbm_run4.joblib
    └── app/
        ├── main.py                     ← FastAPI entry point
        ├── api/                        ← REST API routers
        │   ├── analytics_routes.py     ← KPIs, GIS layer, subsidiary data
        │   ├── environmental_routes.py ← Weather, KB, recommendations
        │   ├── prediction_routes.py    ← Module 5 predictions
        │   ├── explainability_routes.py← Module 6 narratives + alerts
        │   ├── scenario_routes.py      ← Module 7 what-if scenarios
        │   └── report_routes.py        ← Module 9 CSV/JSON exports
        ├── db/                         ← Schema migrations + seed scripts
        └── services/                   ← Business logic
            ├── environmental/          ← Module 2: weather ingestion
            ├── recommendation/         ← Module 3: rule engine
            ├── influence_quantification/ ← Module 4: ML benchmarking
            ├── prediction/             ← Module 5: model deployment
            ├── explainability/         ← Module 6: SHAP + narratives
            └── scenario/               ← Module 7: simulator
```

---

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| 0 | Data Foundation — 593 mines, 91,303 sampling records | ✅ |
| 1 | Dashboard Shell — dark enterprise UI | ✅ |
| 2 | Environmental Variable Layer — 101,973 real weather records, 305 mines | ✅ |
| 3 | Environmental Knowledge Base — 19 variables, recommendation engine | ✅ |
| 4 | Influence Quantification — GroupKFold ML benchmarking, SHAP importance | ✅ |
| 5 | Prediction Engine — 3 deployed LightGBM models, OOF intervals | ✅ |
| 6 | Explainable AI — local attribution + 4-question narratives | ✅ |
| 7 | Scenario Simulator — what-if environmental analysis | ✅ |
| 8 | GIS Intelligence — real Leaflet map, 305 mine markers | ✅ |
| 9 | Reports & Exports — CSV/JSON downloads | ✅ |

---

## API Endpoints

```
GET  /api/analytics/kpis
GET  /api/analytics/environmental-influence?target_metric=gcv&top_n=6
GET  /api/analytics/subsidiary-performance
GET  /api/analytics/gcv-trend
GET  /api/analytics/high-risk-mines?top_n=10
GET  /api/analytics/mine-stats/{mine_code}
GET  /api/analytics/mines/gis-layer
GET  /api/analytics/model-metadata
GET  /api/environmental/ingestion-status
GET  /api/environmental/knowledge-base
GET  /api/environmental/mines/{mine_code}/timeseries
GET  /api/environmental/recommendations/{mine_code}
GET  /api/predictions/models
GET  /api/predictions/{mine_code}?date=YYYY-MM-DD&target=gcv
GET  /api/predictions/{mine_code}/all?date=YYYY-MM-DD
GET  /api/predictions/backtest/summary
GET  /api/explain/{mine_code}?date=YYYY-MM-DD&target=moisture
GET  /api/explain/alerts/active
POST /api/scenario/{mine_code}?label=Custom
GET  /api/scenario/{mine_code}/presets
GET  /api/reports/national-summary?fmt=csv
GET  /api/reports/mine-profiles?fmt=csv
GET  /api/reports/influence-rankings?target=gcv&fmt=csv
GET  /api/reports/prediction-backtest?fmt=csv
```

---

## Data

- **593 total mines** (305 mapped with real GPS coordinates)
- **91,303 sampling records** (Apr 2025 – Mar 2026)
- **101,973 weather records** — 100% real Open-Meteo Archive API, 0% synthetic
- **ML models**: LightGBM (GroupKFold R²: GCV=0.347, Moisture=0.359, Ash=0.230)

---

## Requirements

```
requests>=2.31
pandas>=2.0
numpy>=1.26
scipy>=1.11
scikit-learn>=1.3
fastapi>=0.110
uvicorn[standard]>=0.27

# Optional — auto-detected, better performance if installed:
xgboost>=2.0
lightgbm>=4.0
shap>=0.44
```
