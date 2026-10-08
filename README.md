# Predictive Maintenance Using Machine Learning

## Team Members
- Aiham Bawaneh
- Mahmoud Bani-Hani
- Mohammad Aljabali
- Reem Aldelki
- Shahed Alkhawaga

## Business Problem
Unexpected machine downtime causes production delays, costly emergency repairs,
and can turn minor equipment issues into major breakdowns. This project explores
whether machine learning can predict a machine failure before it happens, giving
maintenance teams time to act early instead of reacting after the fact.

## Objectives
1. Predict whether a machine will experience a failure
2. Identify the main factors associated with machine failure
3. Provide an understandable explanation for the prediction
4. Suggest an appropriate maintenance action

## Dataset
- **Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository
- **Size:** 10,000 rows, 14 columns
- **Features used (6):** Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear
- **Target variable:** `Machine failure` (0 = no failure, 1 = failure)
- **Note:** The dataset is synthetic but designed to reflect realistic industrial predictive maintenance patterns. The failure rate is highly imbalanced (3.39% of records are failures), which we will account for in the modeling stage.

**Citation:**
AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

---

## Milestone 2 — Data Understanding & Cleaning

The original dataset was already clean (0 missing values, 0 duplicates). To practice a
full cleaning workflow, we introduced realistic data issues into a copy of it
(`not_clean_data.csv`, 10,200 rows) and then cleaned it step by step.

### Data quality findings
- **Missing values:** 7,864 cells across the 5 numeric columns
- **Duplicates:** 200 rows sharing the same `UDI` (147 exact copies + 53 near-duplicates with a blank or corrupted cell)
- **Impossible values:** 30 readings (6 per sensor), e.g. air temperature of 999 K, negative torque and speed, tool wear of -20 min
- **Incorrect data types:** `Rotational speed` and `Tool wear` loaded as `float64` although they are whole numbers
- **Checked and clean:** `Type` and `Product ID` (no blanks, valid categories)

### Cleaning decisions
| # | Decision | Reason |
|---|---|---|
| 1 | Removed duplicates by `UDI`, keeping the most complete row (200 rows removed) | `duplicated()` missed 53 rows because their copies differed in a blank or corrupted cell. Removing them first keeps later statistics undistorted |
| 2 | Set values outside physical limits to `NaN` (30 values) | IQR clipping is blind to physics: it treated -20 min of wear as valid (its lower bound was -112) and replaced 999 K with an invented boundary number |
| 3 | Filled missing cells (7,709) with the column median | 56% of rows had at least one blank, so dropping rows would decimate the data. The median is robust to skew |
| 4 | Converted `Rotational speed` and `Tool wear` to `int64` | A column with blanks is stored as `float64` and stays that way after filling. We restored the true whole-number type |
| 5 | Kept natural outliers | Extreme values are symptoms of failure: 88.7% of Torque outliers are failures (vs 3.39% overall) |

Physical limits used: Air temperature 280–320 K, Process temperature 290–330 K,
Rotational speed 1–5,000 rpm, Torque 0.1–100 Nm, Tool wear 0–300 min.

### Before vs After
| Metric | Before | After |
|---|---|---|
| Total rows | 10,200 | 10,000 |
| Missing cells | 7,864 | 0 |
| Rows sharing a `UDI` | 200 | 0 |
| Impossible values | 30 | 0 (set to `NaN`, then median-filled) |
| Speed & Wear type | float64 | int64 |
| Machine failure rate | 3.35% | 3.39% |

### Remaining data quality issues (documented, not fixed)
- **Failure flags vs readings:** in 224 rows the filled sensor values no longer reproduce the original `HDF`, `PWF` or `OSF` flag. Median-filling gives a stand-in, not the true reading, so we kept the original labels.
- **Known quirk of the public dataset:** 27 rows where `Machine failure` does not match the five failure flags. It exists in the original UCI file.
- **Class imbalance (3.39% failures):** a property of the problem, to be handled at the modeling stage.

---

## Repository Contents
- `ms1_show_data_code.ipynb` — data loading and initial exploration (Milestone 1)
- `ai4i2020.csv` — the original dataset
- `dirty_data_code.ipynb` — creates `not_clean_data.csv` by injecting data issues (Milestone 2)
- `not_clean_data.csv` — the raw working file with data issues (Milestone 2)
- `ms2_clean_data.ipynb` — full cleaning notebook with a Decision / Reason for every step (Milestone 2)
- `clean_data.csv` — the final clean dataset, 10,000 rows (Milestone 2)

## How to Run
1. Install the requirements: `pip install pandas numpy matplotlib`
2. Open the notebooks in Jupyter Notebook or Google Colab, with the CSV files in the same folder (or update the path in the first cell)
3. Run `ms1_show_data_code.ipynb` for the Milestone 1 exploration
4. Run `ms2_clean_data.ipynb` to reproduce `clean_data.csv` from `not_clean_data.csv`

## Status
- Milestone 1: Dataset selection, loading, and initial exploration — complete
- Milestone 2: Data understanding and cleaning — complete
