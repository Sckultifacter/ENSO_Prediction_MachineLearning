# ENSO Forecasting using Machine Learning

## Overview

This project focuses on forecasting the Niño 3.4 index, a key indicator of the El Niño–Southern Oscillation (ENSO), using multiple machine learning approaches. The study evaluates predictive skill across different lead times and examines how performance is influenced by both data-driven methods and underlying physical processes.

A comparative analysis is conducted across linear, nonlinear, and sequence-based models, along with physics-informed feature engineering.

---

## Objectives

* Predict Niño 3.4 SST anomalies at multiple lead times (1–15 months)
* Compare different machine learning models for ENSO forecasting
* Incorporate physics-informed features to improve prediction
* Analyze predictability limits such as the Spring Predictability Barrier

---

## Dataset

The model uses a multivariate monthly time-series dataset combining:

* Niño 3.4 index (central Pacific SST anomaly)
* Southern Oscillation Index (SOI)
* Outgoing Longwave Radiation (OLR)
* Indian Ocean Sea Surface Temperature

These variables represent key components of the coupled ocean–atmosphere system governing ENSO.

---

## Models Implemented

### Ridge Regression

* Linear baseline model
* Provides stable and interpretable results
* Performs well with physics-informed features

### Random Forest

* Nonlinear ensemble model
* Captures complex feature interactions
* Sensitive to noise in long-range forecasting

### Support Vector Regression (SVR)

* Kernel-based model for nonlinear regression
* Effective in capturing medium-range dependencies
* Provides a balance between bias and variance

### Long Short-Term Memory (LSTM)

* Deep learning sequence model
* Designed to capture temporal dependencies in time-series data
* Useful for modeling sequential patterns in climate dynamics

---

## Physics-Informed Features

To incorporate domain knowledge, additional features are engineered:

* Rolling mean of Niño 3.4 (ocean thermal memory)
* ENSO phase indicator (El Niño / La Niña)
* SST gradient (Indian Ocean vs Pacific)
* Seasonal encoding (sin/cos of month)

These features help represent ocean–atmosphere coupling and seasonal variability.

---

## Methodology

1. Data preprocessing and merging
2. Feature engineering using lagged variables (t−1 to t−6)
3. Temporal train-test split
4. Model training for multiple lead times (1–15 months)
5. Evaluation using:

   * Root Mean Squared Error (RMSE)
   * Mean Absolute Error (MAE)
   * Pearson Correlation
6. Visualization using interactive Plotly outputs

---

## Results Summary

* High predictive skill at short lead times (1–3 months)
* Moderate performance at medium lead times (~6 months)
* Significant degradation at longer lead times (9–15 months)

Physics-informed features improve performance, particularly for linear models, by incorporating structured climate signals.

---

## Key Insight

Prediction skill declines at longer lead times due to intrinsic variability in the climate system, particularly the Spring Predictability Barrier. This highlights the limitations of purely data-driven approaches and the importance of integrating physical understanding.

---

## Project Structure

```id="structure02"
ENSO/
├── data/
│   ├── olr.txt
│   ├── soi.txt
│   ├── sst_india.csv
│   ├── sstoi.indices.txt
│
├── outputs/
│   ├── ridge/
│   ├── rf/
│   ├── svr/
│   └── lstm/
│
├── src/
│   ├── ridge/
│   │   ├── ridge.py
│   │   └── ridge_phy.py
│   │
│   ├── rf/
│   │   ├── random_forest.py
│   │   └── rf_physics.py
│   │
│   ├── svr/
│   │   └── svr.py
│   │
│   ├── lstm/
│   │   └── lstm.py
│
├── README.md
├── requirements.txt
```

---

## How to Run

Install dependencies:

```id="run01"
pip install -r requirements.txt
```

Run models:

```id="run02"
python src/ridge/ridge.py
python src/ridge/ridge_phy.py
python src/rf/random_forest.py
python src/rf/rf_physics.py
python src/svr/svr.py
python src/lstm/lstm.py
```

Outputs are saved as interactive HTML files in the `outputs/` directory.

---

## Conclusion

This project demonstrates that machine learning models can effectively capture short-term ENSO dynamics, but their predictive skill is fundamentally limited at longer lead times. Incorporating physics-based features improves model performance and interpretability, reinforcing the importance of combining data-driven and physical approaches in climate forecasting.

---
