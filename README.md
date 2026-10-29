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
- **Original size:** 10,000 rows, 14 columns
- **Working file from Milestone 3 on:** `AI4I_with_synthetic_data_cleaned_data.csv` — 12,500 rows, 14 columns (the 10,000 cleaned original rows plus 2,500 synthetic failure rows)
- **Features used (9):** Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear, plus 3 engineered features (`Temp_Diff`, `Mechanical_Power_W`, `Strain_Index`)
- **Target variable:** `Machine failure` (0 = no failure, 1 = failure)
- **Note:** The dataset is synthetic but designed to reflect realistic industrial predictive maintenance patterns. The original failure rate is highly imbalanced (3.39% of records are failures). The 12,500-row working file has a 22.71% failure rate because 2,500 synthetic failure rows were added, so that rate is artificial and probabilities should be read as relative risk.

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
- **Class imbalance (3.39% failures):** a property of the problem. Synthetic failure rows were added for Milestone 3 (see below); the remaining imbalance is handled at the modeling stage.
- **Open question — effect of capping on the failure signal:** only 66 rows have extreme `Torque` (outside 13.09–66.99 Nm), yet 58 of them (88%) are failures, versus 3.39% overall, and they hold about 17% of all failures. Capping these values may have weakened this warning sign. At the modeling stage we will compare results with and without capping (using Recall) and consider keeping the original values or adding an `is_outlier` flag.

---

## Milestone 3 — EDA, Statistics & Data Preprocessing

**Goal:** explore the data, then produce a dataset that is fully numeric, standardized, and split into train and test without leakage, ready for modeling in Milestone 4.

### Dataset used
We use **one file only**: `AI4I_with_synthetic_data_cleaned_data.csv` (12,500 rows: the cleaned original data plus 2,500 synthetic failure rows). We do not use the 15,000-row file and we do not merge the two files, because their first 10,000 rows are shared. Merging would repeat rows, so a copy of the same row could land in both train and test (data leakage).

| Subset | Rows | Failures | Failure rate |
|---|---|---|---|
| Original (`UDI` ≤ 10,000) | 10,000 | 339 | 3.39% |
| Synthetic (`UDI` > 10,000) | 2,500 | 2,500 | 100% |
| Whole file | 12,500 | 2,839 | 22.71% |

Sanity check before analysis: shape (12,500 × 14), 0 missing values, 0 duplicate rows, 2 text columns (`Product ID`, `Type`).

### Part 1 — EDA & statistics (`1_EDA_and_Statistics_Notebook.ipynb`)

| Step | What we did | Why |
|---|---|---|
| 0. Sanity check | Shape, nulls, duplicates, data types | Confirm we opened the right, clean file |
| 1. Descriptive statistics | Mean, median, mode, variance, std, Q1, Q3, P90 and skewness for the 5 sensors | The mean alone can mislead; skew shows symmetric vs skewed features and the variance gap justifies standardization |
| 2. Probability & imbalance | Failure rate by subset; conditional probabilities | Understand where 22.71% comes from and turn patterns into actionable thresholds |
| 3. Six charts | Pie, bar, histograms, box plots, scatter, line chart | Each chart carries a stated takeaway, not just a picture |
| 4. Correlation | Heatmap and interpretation | Find relationships, redundancy, and state what correlation does not prove |

**Key insights**
1. Torque above 50 Nm raises failure risk 3.46× (51.16% vs 14.79%), which supports an early warning at 50 Nm.
2. Tool wear above 200 min raises risk 3.11× (56.55% vs 18.16%), which supports replacing the tool before minute 200. Risk stays around 12–20% up to 175 min, then jumps (33%, 52%, 79%).
3. Rotational speed and torque are inversely related (r ≈ −0.72, consistent with power = torque × speed); failures cluster at low speed and high torque.
4. Product type (L/M/H) is a weak failure indicator on its own (failure rate 24.4%, 18.8%, 23.6%).
5. Feature scales differ hugely (variance 30,918 for speed vs 2.56 for process temperature), so standardization is required.
6. The 22.71% failure rate comes from over-sampling (original 3.39%), so probabilities are read as relative risk.

**Correlation (not causation).** Torque ↔ failure r ≈ 0.26 (the strongest link with the target, but only weak to moderate), speed ↔ torque r ≈ −0.72, air ↔ process temperature r ≈ 0.63. Correlation shows variables move together; it does not show direction or rule out a third factor.

### Part 2 — Preprocessing pipeline (`2_Data_Preprocessing_Pipeline.ipynb`)

The order is fixed, because any step that learns statistics from the data (the scaler) must come **after** the split, while steps that work row by row are safe before it.

`Feature Engineering → Feature Selection → Encoding → Train/Test Split → Scaling (fit on train only) → Final Check`

**1. Feature engineering** (computed from the same row only, so no leakage)

| Feature | Formula | Physical reason | Correlation with target |
|---|---|---|---|
| `Temp_Diff` | Process temp − Air temp | Cooling efficiency; a small gap means weak cooling (linked to HDF) | −0.12 |
| `Mechanical_Power_W` | Torque × rpm × 2π/60 | Mechanical power in watts; very low or high power means a power failure (PWF) | 0.22 |
| `Strain_Index` | Torque × Tool wear | Cumulative stress: high torque on a worn tool (linked to OSF) | 0.31 |

`Mechanical_Power_W` and `Torque` have a correlation of 0.93, so power is partly redundant with torque (fine for trees; keep in mind for linear models).

**2. Feature selection** (every drop is backed by evidence in the notebook)

| Column | Decision | Evidence |
|---|---|---|
| `UDI` | Dropped | All 2,500 rows above 10,000 are synthetic failures, so a model could read the row number |
| `Product ID` | Dropped | 10,000 unique IDs in 12,500 rows (one repeats 115 times), so the same ID could sit in train and test; its L/M/H letter already exists in `Type` |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Dropped | `Machine failure` equals any of them in 99.78% of rows, so they are the answer itself (target leakage) |

Result: 12,500 × 10 (9 features + target).

**3. Encoding.** `Type` is mapped explicitly to `L=0, M=1, H=2` (ordinal: L < M < H) instead of `LabelEncoder`, which sorts alphabetically and would give H=0, L=1, M=2.

**4. Train/test split.** 80/20 with `stratify=y` and `random_state=42` → 10,000 train rows and 2,500 test rows, with failure rates 22.71% and 22.72%. Checks before scaling: index overlap 0 and identical feature rows in both sets 0.

**5. Scaling.** `StandardScaler` on the 8 continuous columns (`Type` is left as is). It is fitted on the training set only (`fit_transform`) and then applied to the test set (`transform`). The test-set mean and standard deviation are close to, but not exactly, 0 and 1, which shows the test set was not used to fit the scaler.

**6. Final readiness check.** 8 of 8 assertions pass: all columns numeric, no nulls, zero index overlap, no ID or leakage columns left, train mean ≈ 0, train std ≈ 1, stratified split (difference under 0.2%), train + test = full data.

### Before vs After (preprocessing)
| Metric | Before | After |
|---|---|---|
| Columns (with target) | 14 | 10 |
| Category columns | 2 text columns (`Product ID`, `Type`) | `Type` → 0/1/2; `Product ID` dropped |
| Feature scale | Raw units (variance 30,918 vs 2.56) | Standardized (mean 0, std 1 on train) |
| Rows train / test | 0 / 0 | 10,000 / 2,500 (no overlap) |

### Outputs for Milestone 4
- `train_ready_milestone3.csv` — 10,000 × 10, scaled
- `test_ready_milestone3.csv` — 2,500 × 10, scaled with the training statistics
- `scaler_milestone3.joblib` — the fitted scaler, to transform any new data the same way

### Limitation (documented)
The synthetic rows were generated on the full data **before** the train/test split. We checked that no identical rows appear in both train and test and that synthetic rows are not exact copies of original rows, but synthetic rows built from original rows may still land in both sets. A better approach is to split first and over-sample the training set only. Until then, test scores in Milestone 4 may be slightly higher than real-world performance, and evaluation should rely on class-aware metrics (Precision, Recall, F1, ROC-AUC) rather than accuracy.

---

## Repository Contents

```
README.md
milestone1/
├── ms1_data/
│   └── ai4i2020_original_data.csv
├── ms1_notebooks/
│   └── show_data_code.ipynb
└── ms1_presentations/
    └── predictive_maintenance.pptx
milestone2/
├── ms2_data/
│   ├── ai4i2020_dirty.csv
│   └── cleaned_data.csv
├── ms2_notebooks/
│   ├── clean_data_code.ipynb
│   └── dirty_data_code.ipynb
└── ms2_presentations/
    └── data_cleaning.pptx
milestone3/
├── ms3_data/
│   └── synthetic_data/
│       ├── AI4I_with_synthetic_data.csv
│       ├── AI4I_with_synthetic_data_1000.csv
│       ├── AI4I_with_synthetic_data_1000_cleaned_data.csv
│       └── AI4I_with_synthetic_data_cleaned_data.csv
├── ms3_notebooks/
│   ├── 1_EDA_and_Statistics_Notebook.ipynb
│   ├── 2_Data_Preprocessing_Pipeline.ipynb
│   ├── scaler_milestone3.joblib
│   ├── synthetic_data_code.ipynb
│   ├── test_ready_milestone3.csv
│   └── train_ready_milestone3.csv
└── ms3_presentations/
    └── eda_preprocessing.pptx
```

### Milestone 1
- `milestone1/ms1_notebooks/show_data_code.ipynb` — data loading and initial exploration
- `milestone1/ms1_data/ai4i2020_original_data.csv` — the original dataset
- `milestone1/ms1_presentations/predictive_maintenance.pptx` — Milestone 1 presentation

### Milestone 2
- `milestone2/ms2_notebooks/dirty_data_code.ipynb` — creates the dirty copy of the dataset by injecting data issues
- `milestone2/ms2_data/ai4i2020_dirty.csv` — the raw working file with data issues, 10,100 rows
- `milestone2/ms2_notebooks/clean_data_code.ipynb` — cleaning notebook: IQR clipping, mean imputation, duplicate removal
- `milestone2/ms2_data/cleaned_data.csv` — the final clean dataset, 10,000 rows
- `milestone2/ms2_presentations/data_cleaning.pptx` — Milestone 2 presentation

### Milestone 3
- `milestone3/ms3_notebooks/synthetic_data_code.ipynb` — code for the synthetic data (files in `ms3_data/synthetic_data/`)
- `milestone3/ms3_data/synthetic_data/AI4I_with_synthetic_data_cleaned_data.csv` — the working dataset used in Milestone 3, 12,500 rows
- `milestone3/ms3_data/synthetic_data/AI4I_with_synthetic_data.csv`, `AI4I_with_synthetic_data_1000.csv` and `AI4I_with_synthetic_data_1000_cleaned_data.csv` — the other synthetic-data files (not used in the Milestone 3 analysis)
- `milestone3/ms3_notebooks/1_EDA_and_Statistics_Notebook.ipynb` — exploratory analysis, statistics, probabilities, six charts and correlation
- `milestone3/ms3_notebooks/2_Data_Preprocessing_Pipeline.ipynb` — feature engineering, selection, encoding, split, scaling and readiness check
- `milestone3/ms3_notebooks/train_ready_milestone3.csv`, `test_ready_milestone3.csv` — the final train and test sets for modeling
- `milestone3/ms3_notebooks/scaler_milestone3.joblib` — the fitted `StandardScaler`
- `milestone3/ms3_presentations/eda_preprocessing.pptx` — Milestone 3 presentation

## How to Run
1. Install the requirements: `pip install pandas numpy matplotlib seaborn scikit-learn joblib`
2. Open the notebooks in Jupyter Notebook or Google Colab. The notebooks read their CSV by file name, so keep the CSV next to the notebook or update the path in the first cell (for Milestone 3, the input is `milestone3/ms3_data/synthetic_data/AI4I_with_synthetic_data_cleaned_data.csv`)
3. Milestone 1: run `milestone1/ms1_notebooks/show_data_code.ipynb`
4. Milestone 2: run `milestone2/ms2_notebooks/clean_data_code.ipynb` to reproduce `cleaned_data.csv` from `ai4i2020_dirty.csv` (set the file path in the first cell to `ai4i2020_dirty.csv`)
5. Milestone 3: run `1_EDA_and_Statistics_Notebook.ipynb` for the analysis, then `2_Data_Preprocessing_Pipeline.ipynb` to regenerate `train_ready_milestone3.csv`, `test_ready_milestone3.csv` and `scaler_milestone3.joblib`

## Status
- Milestone 1: Dataset selection, loading, and initial exploration — complete
- Milestone 2: Data understanding and cleaning — complete
- Milestone 3: EDA, statistics and data preprocessing (train/test sets ready for modeling) — complete
