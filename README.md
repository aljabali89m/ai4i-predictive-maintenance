# Predictive Maintenance Using Machine Learning

## Team Members

- Aiham Bawaneh
- Mahmoud Bani-Hani
- Mohammad Aljabali
- Reem Aldelki
- Shahed Alkhawaga

## Business Problem

Unexpected machine downtime causes production delays, costly emergency repairs, and can turn minor equipment issues into major breakdowns. This project explores whether machine learning can predict a machine failure before it happens, giving maintenance teams time to act early instead of reacting after the fact.

## Objectives

- Predict whether a machine will experience a failure
- Identify the main factors associated with machine failure
- Provide an understandable explanation for the prediction
- Suggest an appropriate maintenance action

## Dataset

- **Source:** AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository
- **Original size:** 10,000 rows, 14 columns
- **Working file from Milestone 3 on:** `AI4I_with_synthetic_data_cleaned_data.csv` — 12,500 rows, 14 columns (the 10,000 cleaned original rows plus 2,500 synthetic failure rows)
- **Features used (9):** Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear, plus 3 engineered features (Temp_Diff, Mechanical_Power_W, Strain_Index)
- **Target variable:** Machine failure (0 = no failure, 1 = failure)
- **Note:** The dataset is synthetic but designed to reflect realistic industrial predictive maintenance patterns. The original failure rate is highly imbalanced (3.39% of records are failures). The 12,500-row working file has a 22.71% failure rate because 2,500 synthetic failure rows were added, so that rate is artificial and probabilities should be read as relative risk.
- **Citation:** AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

---

## Milestone 2 — Data Understanding & Cleaning

The original dataset had no missing values and no duplicates. To practice a full cleaning workflow, we introduced realistic data issues (missing values and duplicate rows) into a copy of it (`ai4i2020_dirty.csv`, 10,100 rows) and then cleaned it step by step. The statistical outliers are natural to the original data.

### Data quality findings

- **Missing values:** 4,040 cells across the 5 numeric columns, affecting 3,437 rows (about 34%)
- **Duplicates:** 100 duplicate rows, all exact copies of existing rows (100 UDI values appear twice)
- **Statistical outliers (1.5×IQR rule):** 394 values in Rotational speed and 66 in Torque (460 in total); none in the other three numeric columns
- **Checked and clean:** Type and Product ID (no blanks, valid categories)

### Cleaning decisions

The steps run in this order in the notebook.

| # | Decision | Reason |
|---|----------|--------|
| 1 | Treated outliers with IQR clipping: for each numeric sensor column, computed Q1, Q3 and IQR, then capped values to [Q1 − 1.5×IQR, Q3 + 1.5×IQR] with `clip()` | To prevent extreme statistical outliers from skewing the machine learning model while preserving the core data distribution. No rows were removed in this step |
| 2 | Filled missing cells (4,040) with the column mean using `fillna(mean)` | About 34% of rows had at least one blank, so dropping rows would remove a third of the dataset. The mean preserves the average of the sensor readings |
| 3 | Removed duplicate rows with `drop_duplicates()` (100 rows removed) | All 100 duplicates were exact copies, so comparing full rows removes them safely. This step runs last, leaving one row per UDI |

### Before vs After

| Metric | Before | After |
|--------|--------|-------|
| Total rows | 10,100 | 10,000 |
| Missing cells | 4,040 | 0 |
| Duplicate rows | 100 | 0 |
| Outlier values (1.5×IQR rule) | 460 | 0 (capped to the limits) |
| Machine failure rate | 3.37% | 3.39% |

### Remaining data quality issues (documented, not fixed)

- **Failure flags vs readings:** in 213 rows (2.1%), the imputed or clipped sensor values no longer reproduce the original HDF, PWF or OSF flag. Mean-filling and clipping give statistical stand-ins, not the true readings, so we kept the original labels.
- **Known quirk of the public dataset:** 27 rows where Machine failure does not match the five failure flags. It exists in the original UCI file.
- **Class imbalance (3.39% failures):** a property of the problem. Synthetic failure rows were added for Milestone 3 (see below); the remaining imbalance is handled at the modeling stage.
- **Open question — effect of capping on the failure signal:** only 66 rows have extreme Torque (outside 13.09–66.99 Nm), yet 58 of them (88%) are failures, versus 3.39% overall, and they hold about 17% of all failures. Capping these values may have weakened this warning sign. We planned to compare results with and without capping (using Recall) at the modeling stage; this comparison was **not performed in Milestone 4** and remains a follow-up (see Milestone 4, Next steps).

---

## Milestone 3 — EDA, Statistics & Data Preprocessing

**Goal:** explore the data, then produce a dataset that is fully numeric, standardized, and split into train and test without leakage, ready for modeling in Milestone 4.

### Dataset used

We use one file only: `AI4I_with_synthetic_data_cleaned_data.csv` (12,500 rows: the cleaned original data plus 2,500 synthetic failure rows). We do not use the 15,000-row file and we do not merge the two files, because their first 10,000 rows are shared. Merging would repeat rows, so a copy of the same row could land in both train and test (data leakage).

| Subset | Rows | Failures | Failure rate |
|--------|------|----------|--------------|
| Original (UDI ≤ 10,000) | 10,000 | 339 | 3.39% |
| Synthetic (UDI > 10,000) | 2,500 | 2,500 | 100% |
| Whole file | 12,500 | 2,839 | 22.71% |

Sanity check before analysis: shape (12,500 × 14), 0 missing values, 0 duplicate rows, 2 text columns (Product ID, Type).

### Part 1 — EDA & statistics (`1_EDA_and_Statistics_Notebook.ipynb`)

| Step | What we did | Why |
|------|-------------|-----|
| 0. Sanity check | Shape, nulls, duplicates, data types | Confirm we opened the right, clean file |
| 1. Descriptive statistics | Mean, median, mode, variance, std, Q1, Q3, P90 and skewness for the 5 sensors | The mean alone can mislead; skew shows symmetric vs skewed features and the variance gap justifies standardization |
| 2. Probability & imbalance | Failure rate by subset; conditional probabilities | Understand where 22.71% comes from and turn patterns into actionable thresholds |
| 3. Six charts | Pie, bar, histograms, box plots, scatter, line chart | Each chart carries a stated takeaway, not just a picture |
| 4. Correlation | Heatmap and interpretation | Find relationships, redundancy, and state what correlation does not prove |

**Key insights**

- Torque above 50 Nm raises failure risk 3.46× (51.16% vs 14.79%), which supports an early warning at 50 Nm.
- Tool wear above 200 min raises risk 3.11× (56.55% vs 18.16%), which supports replacing the tool before minute 200. Risk stays around 12–20% up to 175 min, then jumps (33%, 52%, 79%).
- Rotational speed and torque are inversely related (r ≈ −0.72, consistent with power = torque × speed); failures cluster at low speed and high torque.
- Product type (L/M/H) is a weak failure indicator on its own (failure rate 24.4%, 18.8%, 23.6%).
- Feature scales differ hugely (variance 30,918 for speed vs 2.56 for process temperature), so standardization is required.
- The 22.71% failure rate comes from over-sampling (original 3.39%), so probabilities are read as relative risk.
- **Correlation (not causation).** Torque ↔ failure r ≈ 0.26 (the strongest link with the target, but only weak to moderate), speed ↔ torque r ≈ −0.72, air ↔ process temperature r ≈ 0.63. Correlation shows variables move together; it does not show direction or rule out a third factor.

### Part 2 — Preprocessing pipeline (`2_Data_Preprocessing_Pipeline.ipynb`)

The order is fixed, because any step that learns statistics from the data (the scaler) must come after the split, while steps that work row by row are safe before it.

**Feature Engineering → Feature Selection → Encoding → Train/Test Split → Scaling (fit on train only) → Final Check**

**1. Feature engineering** (computed from the same row only, so no leakage)

| Feature | Formula | Physical reason | Correlation with target |
|---------|---------|-----------------|-------------------------|
| Temp_Diff | Process temp − Air temp | Cooling efficiency; a small gap means weak cooling (linked to HDF) | −0.12 |
| Mechanical_Power_W | Torque × rpm × 2π/60 | Mechanical power in watts; very low or high power means a power failure (PWF) | 0.22 |
| Strain_Index | Torque × Tool wear | Cumulative stress: high torque on a worn tool (linked to OSF) | 0.31 |

Mechanical_Power_W and Torque have a correlation of 0.93, so power is partly redundant with torque (fine for trees; keep in mind for linear models).

**2. Feature selection** (every drop is backed by evidence in the notebook)

| Column | Decision | Evidence |
|--------|----------|----------|
| UDI | Dropped | All 2,500 rows above 10,000 are synthetic failures, so a model could read the row number |
| Product ID | Dropped | 10,000 unique IDs in 12,500 rows (one repeats 115 times), so the same ID could sit in train and test; its L/M/H letter already exists in Type |
| TWF, HDF, PWF, OSF, RNF | Dropped | Machine failure equals any of them in 99.78% of rows, so they are the answer itself (target leakage) |

Result: 12,500 × 10 (9 features + target).

**3. Encoding.** Type is mapped explicitly to L=0, M=1, H=2 (ordinal: L < M < H) instead of LabelEncoder, which sorts alphabetically and would give H=0, L=1, M=2.

**4. Train/test split.** 80/20 with `stratify=y` and `random_state=42` → 10,000 train rows and 2,500 test rows, with failure rates 22.71% and 22.72%. Checks before scaling: index overlap 0 and identical feature rows in both sets 0.

**5. Scaling.** StandardScaler on the 8 continuous columns (Type is left as is). It is fitted on the training set only (`fit_transform`) and then applied to the test set (`transform`). The test-set mean and standard deviation are close to, but not exactly, 0 and 1, which shows the test set was not used to fit the scaler.

**6. Final readiness check.** 8 of 8 assertions pass: all columns numeric, no nulls, zero index overlap, no ID or leakage columns left, train mean ≈ 0, train std ≈ 1, stratified split (difference under 0.2%), train + test = full data.

**Before vs After (preprocessing)**

| Metric | Before | After |
|--------|--------|-------|
| Columns (with target) | 14 | 10 |
| Category columns | 2 text columns (Product ID, Type) | Type → 0/1/2; Product ID dropped |
| Feature scale | Raw units (variance 30,918 vs 2.56) | Standardized (mean 0, std 1 on train) |
| Rows train / test | 0 / 0 | 10,000 / 2,500 (no overlap) |

**Outputs for Milestone 4**

- `train_ready_milestone3.csv` — 10,000 × 10, scaled
- `test_ready_milestone3.csv` — 2,500 × 10, scaled with the training statistics
- `scaler_milestone3.joblib` — the fitted scaler, to transform any new data the same way

**Limitation (documented)**

The synthetic rows were generated on the full data before the train/test split. We checked that no identical rows appear in both train and test and that synthetic rows are not exact copies of original rows, but synthetic rows built from original rows may still land in both sets. A better approach is to split first and over-sample the training set only. Until then, test scores in Milestone 4 may be higher than real-world performance (see the Milestone 4 limitation below), and evaluation should rely on class-aware metrics (Precision, Recall, F1) rather than accuracy.

---

## Milestone 4 — Model Training & Evaluation

**Goal:** train several classification models on the Milestone 3 data, compare them fairly, choose one, and evaluate it honestly on unseen data. Notebook: `milestone4/ms4_notebooks/milestone4.ipynb`. Presentation: `milestone4/ms4_presentations/makina_sense_model.pptx`.

### Input data

| Item | Value |
|------|-------|
| Training set | `train_ready_milestone3.csv` — 10,000 rows × 10 columns |
| Test set | `test_ready_milestone3.csv` — 2,500 rows × 10 columns, kept unseen until the final score |
| Features / target | 9 features (Type, 5 sensors, 3 engineered) / Machine failure |
| Failure rate | 22.71% (train) and 22.72% (test) |

There is no separate validation set: stratified 5-fold cross-validation on the training set plays that role, so every training row is used for both fitting and validation.

### Approach

| Step | What we did | Why |
|------|-------------|-----|
| 1. Load and separate | Read the two prepared CSV files; split features (X) from the target (y) | The files are already encoded, scaled and leak-free from Milestone 3 |
| 2. Define five models | Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting | Different model families show whether more complex decision boundaries help |
| 3. Cross-validate | `StratifiedKFold(5, shuffle=True, random_state=42)`, the same folds for every model | A fair comparison that keeps the failure proportion in each fold |
| 4. Select | Highest mean cross-validated F1-score | The test set is not used for the choice |
| 5. Refit and test once | Fit on all training rows, score on the 2,500 unseen test rows | An honest estimate on data the model has not seen |
| 6. Interpret | Confusion matrix, classification report, feature importance, error analysis | Understand what the model gets right, wrong, and why |

### Models

| Model | Setup | Why it is in the comparison |
|-------|-------|-----------------------------|
| Logistic Regression | `StandardScaler` in a Pipeline, `class_weight="balanced"`, `max_iter=1000` | Explainable linear baseline |
| KNN | `StandardScaler` in a Pipeline, k = 5 | Tests whether similar operating conditions fail alike |
| Decision Tree | `class_weight="balanced"` | Readable non-linear if/then rules |
| Random Forest | 100 trees, `class_weight="balanced"` | Stable ensemble that captures feature interactions |
| Gradient Boosting | 100 estimators | Trees built sequentially, each correcting earlier errors |

All models use `random_state=42`.

### Metrics and why

| Metric | Question it answers | Why we use it |
|--------|--------------------|---------------|
| Accuracy | How many predictions are correct overall? | Overview only: always predicting "no failure" already scores about 77% |
| Precision | How many raised alarms are real? | Higher precision means fewer false alarms |
| Recall | How many real failures did we catch? | Missed failures are the costly mistake in maintenance |
| F1-score | Balance of precision and recall | Our model-selection criterion |
| Balanced Accuracy | Average recall over both classes | Not fooled by the class imbalance |

### Results — cross-validation (5-fold, training set)

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 |
|-------|-------------|--------------|-----------|-------|
| **Random Forest** | 0.9705 | 0.9670 | 0.9009 | **0.9327** |
| Gradient Boosting | 0.9658 | 0.9701 | 0.8767 | 0.9208 |
| Decision Tree | 0.9462 | 0.8836 | 0.8794 | 0.8812 |
| KNN | 0.9353 | 0.9558 | 0.7499 | 0.8402 |
| Logistic Regression | 0.6993 | 0.4074 | 0.7111 | 0.5179 |

### Results — unseen test set (2,500 rows)

| Model | Accuracy | Precision | Recall | F1 | Balanced Accuracy |
|-------|----------|-----------|--------|----|-------------------|
| **Random Forest** | 0.9708 | 0.9626 | 0.9067 | **0.9338** | 0.9482 |
| Gradient Boosting | 0.9668 | 0.9691 | 0.8820 | 0.9235 | 0.9369 |
| Decision Tree | 0.9428 | 0.8645 | 0.8873 | 0.8758 | 0.9232 |
| KNN | 0.9316 | 0.9501 | 0.7377 | 0.8305 | 0.8631 |
| Logistic Regression | 0.7040 | 0.4119 | 0.7077 | 0.5207 | 0.7053 |

The test ranking matches the cross-validation ranking, so the choice is not an accident of the folds.

### Final model: Random Forest

- Highest mean CV F1 (0.9327) and highest CV recall (0.9009) of the five models.
- CV F1 (0.9327) and test F1 (0.9338) are almost identical, which gives no sign of overfitting.
- Gradient Boosting is close (CV F1 0.9208) and slightly more precise on the test set (0.9691 vs 0.9626). Random Forest catches more failures (test recall 0.9067 vs 0.8820), which matters more for maintenance, so recall broke the tie. The margin between the two is small.
- Logistic Regression is far behind: failures depend on non-linear combinations of conditions, which a linear boundary cannot separate.

### Confusion matrix (Random Forest, test set)

| | Predicted: no failure | Predicted: failure |
|---|---|---|
| **Actual: no failure (1,932)** | 1,912 (true negatives) | 20 (false alarms) |
| **Actual: failure (568)** | 53 (missed failures) | 515 (caught failures) |

- 2,427 of 2,500 predictions are correct and 73 are wrong.
- Failure class: precision 0.96, recall 0.91, F1 0.93. No-failure class: precision 0.97, recall 0.99, F1 0.98.
- 53 of the 73 errors are missed failures, which are the costly kind. Lowering the decision threshold would catch more failures at the price of more false alarms (see Next steps).

### Feature importance (Random Forest built-in)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Rotational speed | 0.2006 |
| 2 | Temp_Diff (engineered) | 0.1951 |
| 3 | Mechanical_Power_W (engineered) | 0.1593 |
| 4 | Torque | 0.1447 |
| 5 | Tool wear | 0.1276 |
| 6 | Strain_Index (engineered) | 0.0617 |
| 7 | Air temperature | 0.0555 |
| 8 | Process temperature | 0.0492 |
| 9 | Type | 0.0063 |

Two of the top three features were engineered in Milestone 3. Importance shows what the model uses to predict, not what causes failures. Mechanical_Power_W is built from torque and speed, so these features share importance.

### Challenges and limitations

| Challenge | What we did | Planned improvement |
|-----------|-------------|---------------------|
| Few original failures (3.39%) | Used class weights where supported; judged with F1 and Recall | Collect more observed, real failure cases |
| Synthetic rows affect evaluation | Reported the dataset composition and this limitation | Evaluate separately on original-only data |
| Default model settings | Used identical folds for a fair baseline comparison | Tune hyperparameters with Grid Search |
| 53 missed failures | Reviewed Recall alongside Precision | Tune the decision threshold to the cost of a miss |

**Main limitation: the test score reflects the prepared dataset, not real machines.** The 2,500 synthetic rows are all failures and were generated before the split. Mapping the test rows back to `UDI` shows that about 87% of the failures in the test set (494 of 568) are synthetic, so the headline metrics are likely higher than the performance on real failures. The notebook does not report an original-only test score. Read the results as performance on this prepared dataset and validate on more real failures before any real-world use.

Other note: the scaler that produced the final CSV files was fitted on the whole training set, so the cross-validation folds share a tiny amount of scaling information (negligible for tree models). The notebook also places a `StandardScaler` inside the Pipeline for Logistic Regression and KNN.

### Next steps

- Evaluate on original-only test rows, and split first, then over-sample the training set only.
- Tune hyperparameters (Grid Search) for the top models.
- Tune the decision threshold to the cost of a missed failure.
- Compare results with and without outlier capping (the open question from Milestone 2).
- Collect more observed, real failure cases.

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
│   ├── scaler_milestone3.joblib
│   └── synthetic_data_code.ipynb
└── ms3_presentations/
    └── eda_preprocessing.pptx
milestone4/
├── ms4_data/
│   ├── test_ready_milestone3.csv
│   └── train_ready_milestone3.csv
├── ms4_notebooks/
│   └── milestone4.ipynb
└── ms4_presentations/
    └── makina_sense_model.pptx
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
- `2_Data_Preprocessing_Pipeline.ipynb` — feature engineering, selection, encoding, split, scaling and readiness check (it produced the train/test files below)
- `milestone3/ms3_notebooks/scaler_milestone3.joblib` — the fitted StandardScaler
- `milestone3/ms3_presentations/eda_preprocessing.pptx` — Milestone 3 presentation

### Milestone 4

- `milestone4/ms4_data/train_ready_milestone3.csv`, `test_ready_milestone3.csv` — the final train and test sets from Milestone 3, used as the input for modeling
- `milestone4/ms4_notebooks/milestone4.ipynb` — five models, stratified 5-fold cross-validation, test evaluation, confusion matrix, feature importance and error analysis
- `milestone4/ms4_presentations/makina_sense_model.pptx` — Milestone 4 presentation

## How to Run

1. Install the requirements: `pip install pandas numpy matplotlib seaborn scikit-learn joblib`
2. Open the notebooks in Jupyter Notebook or Google Colab. The notebooks read their CSV from a path in the first cells, so update that path to where the files are on your computer (for Milestone 3, the input is `milestone3/ms3_data/synthetic_data/AI4I_with_synthetic_data_cleaned_data.csv`).
3. **Milestone 1:** run `milestone1/ms1_notebooks/show_data_code.ipynb`
4. **Milestone 2:** run `milestone2/ms2_notebooks/clean_data_code.ipynb` to reproduce `cleaned_data.csv` from `ai4i2020_dirty.csv` (set the file path in the first cell to `ai4i2020_dirty.csv`)
5. **Milestone 3:** run `1_EDA_and_Statistics_Notebook.ipynb` for the analysis, then `2_Data_Preprocessing_Pipeline.ipynb` to regenerate `train_ready_milestone3.csv`, `test_ready_milestone3.csv` and `scaler_milestone3.joblib`
6. **Milestone 4:** run `milestone4/ms4_notebooks/milestone4.ipynb`. In the loading cell, set the two paths to `milestone4/ms4_data/train_ready_milestone3.csv` and `milestone4/ms4_data/test_ready_milestone3.csv`. The notebook trains and compares the five models and prints the tables and figures reported above. Results can differ slightly in the last decimals between scikit-learn versions.

## Status

- **Milestone 1:** Dataset selection, loading, and initial exploration — complete
- **Milestone 2:** Data understanding and cleaning — complete
- **Milestone 3:** EDA, statistics and data preprocessing (train/test sets ready for modeling) — complete
- **Milestone 4:** Model training and evaluation (five models compared with cross-validation; Random Forest selected) — complete
- **Next:** final project presentation (Demo Day)