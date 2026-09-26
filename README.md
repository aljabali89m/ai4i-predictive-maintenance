# Predictive Maintenance Using Machine Learning

## Team Members
- Shahed Alkhawaga
- Aiham Bawaneh
- Mohammad Aljabali
- Mahmoud Bani-Hani
- Reem Aldelki

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
- **Size:** 10,000 rows
- **Features used (6):** Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear
- **Target variable:** `Machine failure` (0 = no failure, 1 = failure)
- **Note:** The dataset is synthetic but designed to reflect realistic industrial predictive maintenance patterns. The failure rate is highly imbalanced (~3.4% of records are failures), which we will account for in later milestones.

**Citation:**
AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

## Repository Contents
- `Project_G7.ipynb` — data loading and initial exploration (Milestone 1)
- `ai4i2020.csv` — the raw dataset

## How to Run
1. Open `notebook.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells to load and explore the dataset

## Status
Milestone 1: Dataset selection, loading, and initial exploration — complete.
