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

The original dataset had no missing values and no duplicates. To practice a
full cleaning workflow, we introduced realistic data issues (missing values and
duplicate rows) into a copy of it (`ai4i2020_dirty.csv`, 10,100 rows) and then
cleaned it step by step. The statistical outliers are natural to the original data.

### Data quality findings
- **Missing values:** 4,040 cells across the 5 numeric columns, affecting 3,437 rows (about 34%)
- **Duplicates:** 100 duplicate rows, all exact copies of existing rows (100 `UDI` values appear twice)
- **Statistical outliers (1.5×IQR rule):** 394 values in `Rotational speed` and 66 in `Torque` (460 in total); none in the other three numeric columns
- **Checked and clean:** `Type` and `Product ID` (no blanks, valid categories)

### Cleaning decisions
The steps run in this order in the notebook.

| # | Decision | Reason |
|---|---|---|
| 1 | Treated outliers with IQR clipping: for each numeric sensor column, computed Q1, Q3 and IQR, then capped values to `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` with `clip()` | To prevent extreme statistical outliers from skewing the machine learning model while preserving the core data distribution. No rows were removed in this step |
| 2 | Filled missing cells (4,040) with the column mean using `fillna(mean)` | About 34% of rows had at least one blank, so dropping rows would remove a third of the dataset. The mean preserves the average of the sensor readings |
| 3 | Removed duplicate rows with `drop_duplicates()` (100 rows removed) | All 100 duplicates were exact copies, so comparing full rows removes them safely. This step runs last, leaving one row per `UDI` |

### Before vs After
| Metric | Before | After |
|---|---|---|
| Total rows | 10,100 | 10,000 |
| Missing cells | 4,040 | 0 |
| Duplicate rows | 100 | 0 |
| Outlier values (1.5×IQR rule) | 460 | 0 (capped to the limits) |
| Machine failure rate | 3.37% | 3.39% |

### Remaining data quality issues (documented, not fixed)
- **Failure flags vs readings:** in 213 rows (2.1%), the imputed or clipped sensor values no longer reproduce the original `HDF`, `PWF` or `OSF` flag. Mean-filling and clipping give statistical stand-ins, not the true readings, so we kept the original labels.
- **Known quirk of the public dataset:** 27 rows where `Machine failure` does not match the five failure flags. It exists in the original UCI file.
- **Class imbalance (3.39% failures):** a property of the problem, to be handled at the modeling stage.
- **Open question — effect of capping on the failure signal:** only 66 rows have extreme `Torque` (outside 13.09–66.99 Nm), yet 58 of them (88%) are failures, versus 3.39% overall, and they hold about 17% of all failures. Capping these values may have weakened this warning sign. At the modeling stage we will compare results with and without capping (using Recall) and consider keeping the original values or adding an `is_outlier` flag.

---

## Repository Contents
- `ms1_show_data_code.ipynb` — data loading and initial exploration (Milestone 1)
- `ai4i2020.csv` — the original dataset
- `dirty_data_code.ipynb` — creates the dirty copy of the dataset by injecting data issues (Milestone 2)
- `ai4i2020_dirty.csv` — the raw working file with data issues, 10,100 rows (Milestone 2)
- `code_for_cleandatepresentation.ipynb` — cleaning notebook: IQR clipping, mean imputation, duplicate removal (Milestone 2)
- `cleaned_data.csv` — the final clean dataset, 10,000 rows (Milestone 2)

## How to Run
1. Install the requirements: `pip install pandas matplotlib`
2. Open the notebooks in Jupyter Notebook or Google Colab, with the CSV files in the same folder (or update the path in the first cell)
3. Run `ms1_show_data_code.ipynb` for the Milestone 1 exploration
4. Run `code_for_cleandatepresentation.ipynb` to reproduce `cleaned_data.csv` from `ai4i2020_dirty.csv` (set the file path in the first cell to `ai4i2020_dirty.csv`)

## Status
- Milestone 1: Dataset selection, loading, and initial exploration — complete
- Milestone 2: Data understanding and cleaning — complete
