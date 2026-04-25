# Physics-Informed Neural ODE for Production Decline Curve Analysis
### Gulf of Mexico Basin | INEOS Energy × Imperial College London

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/data-BOEM%20Public-green.svg)](https://www.data.boem.gov/)

---

## Overview

This repository contains the full implementation of a **Physics-Informed Neural ODE (PINODE)** framework for production decline curve analysis (DCA), developed as part of an industry-academic research collaboration between **INEOS Energy** and **Imperial College London**.

The project applies Neural Ordinary Differential Equations — constrained by reservoir physics — to model and forecast oil production decline across **1,500+ well completions** in the Gulf of Mexico (GoM) basin. It benchmarks PINODE against six classical DCA models and incorporates **automated changepoint detection** to handle non-stationary production behaviour.

The dataset is sourced from the publicly available **BOEM (Bureau of Ocean Energy Management)** production database.

---

## Motivation

Traditional decline curve methods (Arps, Duong, etc.) assume smooth, stationary production behaviour — an assumption that rarely holds in real offshore wells, where workovers, shut-ins, and facility constraints create abrupt regime changes. This project addresses two core limitations:

1. **Non-stationarity**: Production signals are segmented using automated changepoint detection (PELT algorithm) before model fitting, allowing each decline segment to be analysed independently.
2. **Physics violation**: Pure data-driven models can produce physically implausible forecasts (e.g., negative rates, unbounded growth). PINODE enforces reservoir physics directly in the loss function — constraining cumulative production, decline rate behaviour, and late-time flattening.

---

## Methods

### 1. Classical DCA Benchmarks (`decline_curves_base_model.py`)
Six decline models implemented and benchmarked:
- **Arps**: Exponential, Hyperbolic, Harmonic
- **Stretched Exponential Decline (SED)**
- **Ilk Power Law**
- **Duong** (fracture-dominated decline)

Each model is fitted using `scipy.optimize.curve_fit` with R² and MSE evaluation. An interactive widget allows per-well model comparison with dynamic changepoint selection.

### 2. Changepoint Detection
Automated segmentation using the **PELT (Pruned Exact Linear Time)** algorithm (`ruptures` library) with an RBF cost function. Detected changepoints identify production regime shifts — workovers, artificial lift installation, facility events — enabling localised decline analysis on each segment.

### 3. Physics-Informed Neural ODE (`pinode_with_cpd.py`)
The core contribution of this project. The ODE governing hyperbolic decline is:

```
dq/dt = -b · q^(d+1)
```

Where `b` (decline rate) and `d` (curvature) are **learnable parameters** optimised end-to-end via backpropagation through the ODE solver (`torchdiffeq`).

**Physics-informed loss function** combines:
- **Data loss**: MSE between predicted and observed production rates
- **Cumulative production constraint**: Simpson's rule integration ensures predicted EUR matches observed cumulative production
- **Second derivative penalty**: Discourages non-physical concavity in the decline curve
- **Late-time flattening constraint**: Prevents unrealistic terminal decline rates
- **L2 regularisation**: Keeps `b` and `d` within physically meaningful bounds

**Training setup**:
- Optimiser: Adam with Cosine Annealing Warm Restarts scheduler
- Input scaling: MinMaxScaler on both time and rate axes
- Forecasting: 12-step ahead extrapolation beyond the last observed data point

### 4. Supplementary Models (`xgboost_model.py`, `tabnet_model.py`)
XGBoost and TabNet models trained on engineered production and reservoir features for comparative forecasting benchmarks.

### 5. Well Clustering (`clustering.ipynb`)
DBSCAN clustering on reservoir and production attributes (G&G features) to identify wells with similar decline behaviour — supporting field development planning and analogue selection.

---

## Repository Structure

```
├── pinode_with_cpd.py          # Core PINODE model with changepoint integration
├── decline_curves_base_model.py # Classical DCA benchmarks (6 models)
├── xgboost_model.py            # XGBoost forecasting benchmark
├── tabnet_model.py             # TabNet forecasting benchmark
├── data_import.py              # Data loading and preprocessing pipeline
├── clustering.ipynb            # Well clustering (DBSCAN)
├── forecasting.ipynb           # Forecasting experiments and comparisons
├── curvefit.ipynb              # Interactive curve fitting exploration
├── changepointdetection.ipynb  # Changepoint detection analysis
└── README.md
```

---

## Data

All analysis uses the **BOEM Gulf of Mexico production database** — a publicly available dataset containing monthly production records for offshore well completions.

**Preprocessing pipeline** (`data_import.py`):
- Filter completions with fewer than 60 months of history (insufficient for decline analysis)
- Remove completions where >40% of recorded days have zero production (non-producing wells)
- Deduplicate on `Days_Elapsed`, retaining peak rate per day
- Group by `Well_Completion_Name`, `BOEM_FIELD`, `LEASE`

---

## Installation

```bash
git clone https://github.com/gems-at3823/PINODE-DCA-GoA.git
cd irp-workbooks-progress
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch 2.0+, torchdiffeq, ruptures, scipy, scikit-learn, pandas, matplotlib, ipywidgets

**GPU recommended** for PINODE training (CUDA 12.0+).

---

## Usage

### Run PINODE with changepoint detection
```python
from pinode_with_cpd import main
main()  # Launches interactive well selector and change point slider
```

### Run classical DCA benchmarks
```python
from decline_curves_base_model import main
main()  # Interactive multi-model comparison per well completion
```

**Note**: Set `%matplotlib inline` in Jupyter before running interactive widgets.

---

## Key Results

- PINODE consistently outperformed traditional Arps models on non-stationary wells — particularly those with mid-life workovers or artificial lift transitions
- Changepoint-segmented fitting improved R² significantly on wells with multiple production regimes compared to whole-history fitting
- Physics constraints in the loss function eliminated non-physical forecasts (negative terminal rates, unbounded decline) that appeared in unconstrained ML baselines

---

## Acknowledgements

This project was developed in collaboration with the **Subsurface Data Science team at INEOS Energy** and supervised by faculty at **Imperial College London** as part of the MSc Geo-Energy with Machine Learning and Data Science programme (2023–2024).

---

## Contact

**Akhil Toram**  
Petroleum Engineer | Subsurface Data Science  
[akhil.toram07@gmail.com](mailto:akhil.toram07@gmail.com) | [LinkedIn](https://www.linkedin.com/in/akhil-t-07125410a/) | [GitHub](https://github.com/gems-at3823)
