# 🌾 AI-Powered Crop Yield Prediction & Optimization

A complete, runnable machine-learning project that (1) **predicts crop yield**
from soil, weather, satellite, and farm-management data, and (2)
**optimizes controllable inputs** (fertilizer, irrigation, pesticide, planting
density) to maximize predicted yield, optionally within a cost budget.

---

## 1. Problem Statement

Farmers and agribusinesses need to decide, before and during a growing
season, how much fertilizer/water/pesticide to apply and at what density to
plant — decisions that materially affect yield, cost, and environmental
impact. This project builds a data-driven decision support system:

1. **Predict** expected yield (tons/hectare) given field conditions.
2. **Optimize** the controllable inputs to maximize yield (or maximize
   yield-per-dollar within a budget).

---

## 2. Project Structure

```
crop_yield_project/
├── data/
│   ├── generate_data.py         # synthetic dataset generator (swap for real data)
│   └── crop_yield_data.csv      # 6,000-row generated dataset
├── src/
│   ├── train_model.py           # trains & compares 3 models, saves the best
│   ├── optimize.py              # grid-search optimizer over controllable inputs
│   └── predict.py               # CLI: predict + recommend for one field
├── models/
│   ├── crop_yield_model.pkl     # trained sklearn pipeline (preprocessing + model)
│   ├── metrics.json             # evaluation metrics for all candidate models
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── predicted_vs_actual.png
├── app_streamlit.py             # interactive web dashboard (run with `streamlit run`)
├── requirements.txt
└── README.md
```

---

## 3. Data

`data/generate_data.py` produces a **synthetic but agronomically realistic**
dataset (6,000 rows) covering 5 crops (Wheat, Rice, Maize, Soybean, Cotton),
5 regions, and 4 soil types, with:

- **Weather**: rainfall, temperature, humidity, sunlight hours
- **Soil**: pH, nitrogen/phosphorus/potassium, organic matter
- **Remote sensing**: NDVI (vegetation health index)
- **Management (controllable)**: fertilizer, irrigation, pesticide, planting density
- **Target**: yield (tons/hectare)

Yield is generated from realistic response curves (e.g., a Gaussian
response to rainfall/temperature around each crop's ideal, diminishing
returns on fertilizer, NPK balance effects) so the trained models learn
patterns that mirror real agronomy.

> **To use real data**: replace `crop_yield_data.csv` with your own records
> (e.g., from USDA NASS, FAO, Kaggle "Crop Yield Prediction" datasets, or
> your own farm sensors/satellite feeds) using the same column names, or
> update `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` in `train_model.py`.

---

## 4. Models & Results

Three models were trained and compared (80/20 train/test split, 5-fold CV
on the winner):

| Model              | MAE  | RMSE | R²    |
|---------------------|------|------|-------|
| Linear Regression   | 0.28 | 0.37 | 0.913 |
| Random Forest        | 0.23 | 0.30 | 0.941 |
| **Gradient Boosting** | **0.23** | **0.30** | **0.942** |

**Gradient Boosting** was selected (5-fold CV R² = 0.940 ± 0.001 — stable,
not overfit). See `models/feature_importance.png` for which variables drive
yield most (typically NDVI, rainfall/irrigation, and fertilizer dominate).

---

## 5. Setup

```bash
cd crop_yield_project
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

---

## 6. Usage

### Step 1 — Generate data (already done, re-run to regenerate)
```bash
python3 data/generate_data.py
```

### Step 2 — Train the model
```bash
python3 src/train_model.py
```

### Step 3a — Predict + get a recommendation (CLI)
```bash
python3 src/predict.py \
  --crop Wheat --region North --soil_type Loamy \
  --rainfall_mm 420 --avg_temp_c 19 --humidity_pct 55 \
  --sunlight_hours 7 --soil_ph 6.8 \
  --soil_nitrogen_kg_ha 60 --soil_phosphorus_kg_ha 40 --soil_potassium_kg_ha 45 \
  --soil_organic_matter_pct 3.0 --fertilizer_kg_ha 140 --irrigation_mm 250 \
  --pesticide_l_ha 2 --ndvi 0.62 --planting_density_k_per_ha 85 --days_to_harvest 130
```
Output includes the predicted yield **and** the top optimized input
combination with expected yield uplift and added cost.

### Step 3b — Interactive dashboard
```bash
streamlit run app_streamlit.py
```
Adjust sliders for field conditions and see live predictions, the
optimizer's top-5 recommended input combos, and sensitivity ("what-if")
curves for each controllable input.

---

## 7. How the Optimizer Works

`src/optimize.py` treats the trained model as a **simulator**: for a fixed
set of conditions (crop, region, soil, weather, remote-sensing readings), it
grid-searches over the controllable inputs — fertilizer, irrigation,
pesticide, planting density — and returns the combination(s) with the
highest predicted yield, optionally filtered to a per-hectare cost budget
using illustrative unit costs (edit `COST_PER_UNIT` in `optimize.py` for
your local prices).

This is intentionally simple and explainable (a full factorial grid search).
For larger search spaces or continuous optimization, swap in:
- `scipy.optimize.minimize` (continuous, gradient-based)
- Bayesian optimization (`scikit-optimize`, `Optuna`) for expensive-to-evaluate objectives
- Reinforcement learning for **sequential** decisions across a growing season (e.g., weekly irrigation decisions)

---

## 8. Extending This Project

| Direction | How |
|---|---|
| Real satellite imagery | Pull NDVI from Sentinel-2/Landsat via Google Earth Engine or Copernicus API |
| Real weather | NASA POWER API, NOAA, or OpenWeather historical/forecast endpoints |
| Time-series forecasting | Add an LSTM/temporal model using weekly weather + NDVI trajectories instead of season aggregates |
| Field-zone precision | Cluster a field into management zones (soil sampling grid) and optimize per-zone |
| Deployment | Wrap `predict.py` logic in a FastAPI service; the Streamlit app already gives a ready UI |
| Uncertainty | Add prediction intervals (quantile regression or `GradientBoostingRegressor(loss="quantile")`) so recommendations show a confidence range, not just a point estimate |

---

## 9. Disclaimer

The bundled dataset is **synthetic**, generated to have realistic
statistical structure for demonstrating the full pipeline. Retrain on real
regional data before using this for actual farm decisions — real yield
depends on many local factors (pest/disease pressure, cultivar genetics,
extreme weather events) not fully captured here.
