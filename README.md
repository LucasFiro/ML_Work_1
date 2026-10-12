# First ML lab

![Python](https://img.shields.io/badge/Python-3.12-blue)

This repository contains the code for ...

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
- `Data/`: Folder for data storage (data needs to be downloaded locally and is ignored by git
    - `temp`: File present just to create Data folder, useless
- `Pipeline/`: Code for the training and analysis pipeline
  - `Jupyter/`
    - `Temp_Notebook.ipynb` : Notebook to try things
  - `Python/`
    - `analysis.py`: Metrics computation
    - `load_data.py`: Data loading
    - `main.py`: Script to run the full pipeline
    - `model.py`: ML model for predictions
    - `plotting.py`: Plot generation
    - `tools.py`: Additional helper
- `.gitignore`: Used to record which folder and file ignore
- `README.md`: README file
- `requirements.txt`: Requirements to run the code in this repository

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
> ⚠️ **Note**: Be careful with compatibility issue of data.
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
