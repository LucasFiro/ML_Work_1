# Project 1

![Python](https://img.shields.io/badge/Python-3.9-blue)

This repository contains the code for Project 1 of the EPFL Machine Learning course (CS-433): predicting coronary heart disease (MICHD) from the BRFSS 2015 health survey with machine learning methods implemented from scratch in NumPy, including data cleaning, an L2-regularized logistic regression with an F1-optimized decision threshold, and the comparison, cross-validation and ablation experiments reported in our paper.
---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/LucasFiro/ML_Work_1.git
cd ML_Work_1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional: verify that required versions are installed

---

## Repository Structure
- `implementations.py` — the 6 required methods (GD/SGD on MSE, least squares, ridge,
  logistic and regularized logistic regression) and additional models (L1 logistic
  regression, KNN, Gaussian naive Bayes)
- `run.py` — end-to-end script producing our best submission
- `compare.py` — model comparison (hyperparameter grid and F1-optimal threshold for
  each model on the same split); writes `results/summary.csv`, `results/results.json`,
  the F1-vs-threshold curves and one submission file per model
- `helpers.py` — course helper (`create_csv_submission`)
- `Pipeline/Python/`
  - `load_data.py` — CSV loading with a `.npz` cache, labels converted to {0, 1}
  - `analysis.py` — exploratory statistics, special-code cleaning and preprocessing
  - `tools.py` — train/validation split, ridge solver, metrics, threshold search
  - `model.py` — comparison of several models on the same split
  - `cross_validation.py` — k-fold cross-validation of λ
  - `ablation.py` — ablation study and baselines reported in the paper
  - `plotting.py`, `main.py` — figures and exploratory pipeline
- `results/` — outputs of `compare.py`
- `figures/` — cross-validation and ablation results, figure of the report
- `Pipeline/Jupyter/` — scratch notebook
- `Data/` — local data folder (not versioned)
- `requirements.txt` — Python dependencies (`numpy`, `matplotlib` for plots only)
- `.gitignore` — files and folders not versioned (dataset, cache, virtual environment)

---

## Data description

The data comes from the **BRFSS 2015** (Behavioral Risk Factor Surveillance System), a
health-related telephone survey run by the CDC on U.S. residents
([source](https://www.cdc.gov/brfss/annual_data/annual_2015.html)). The task is a binary
classification: predict whether a person has **coronary heart disease or has had a
heart attack (MICHD)** from health and lifestyle features.

| File | Content | Shape |
|---|---|---|
| `x_train.csv` | Training features (+ `Id` column) | 328,135 × 321 |
| `y_train.csv` | Training labels, `-1` / `1` in the CSV | 328,135 |
| `x_test.csv` | Test features (labels hidden, scored on AIcrowd) | 109,379 × 321 |
| `sample_submission.csv` | Expected submission format | – |


## Usage:
> **Note**: Be careful with compatibility issue of data.
Run Python scripts with argument parsing. 

### Arguments for `main.py`:
"load", "explore","plot","features", "model", "all"
- `default`: all
- `load`: Load the dataset
- `explore`: Perform exploratory data analysis on the training data
- `plot`: Generate and save visualization for analysis
- `features`: Perform feature extraction
- `model`: Run model


---

## 👤 Authors

```text
Célia Budelot, Manon Cesbron Darnaud, Lucas Firouzi
```
