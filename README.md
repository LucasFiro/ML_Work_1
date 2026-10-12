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

- `Pipeline/`: Code for the training and analysis pipeline
  - `test.csv`
  - `temp_notebook.ipynb`:
  - `Python/`
    - `analysis.py`: Metrics computation
    - `load_data.py`: Data loading
    - `main.py`: Script to run the full pipeline
    - `plotting.py`: Plot generation
    - `tools.py`: Additional helper
  - `__pycache__`: contains the cache for the code (not important for the analysis)
- `README.md`: README file
- `requirements.txt`: Requirements to run the code in this repository

---

## Data description:
Heat-shock images in zebrafishes marked with Her1-YFP reporter.
The images were acquired using the SPIM based microscope Viventis LS1 Live.
Each embryo is screened at ambient temperature and at heat-shock temperature (40 degrees celcius).
The data is composed of a csv file containing the x,y and z coordinates and a tiff file containing the 3d image.

## Data Annotation:
Data annotation was made using FIJI built in multiselection tool. It is however important to filter the data as it also adapt the scale of the FIJI annotation to the real pixel count of the tiff file.

## Usage:
> ⚠️ **Note**: Be careful with compatibility issue of data with spotiflow.
Run Python scripts with argument parsing. 

### Arguments for `main.py`:
"finetune", "zero_shot_prediction","both","none", "filter"
- `none`:
- `filter`: Filter the data to make it compatible with spotiflow
- `finetune`: Run the pipeline for finetuning
- `zero-shot`: Run the pipeline for zero-shot
- `both`: Run both model and compare them

### Regarding Spotiflow:

Most of the analysis was performed using the code above but modification of the csv file for making them compatible to spotiflow starting with the fiji tools are done using command line

---

## 👤 Author

```text
Lucas Firouzi
```
