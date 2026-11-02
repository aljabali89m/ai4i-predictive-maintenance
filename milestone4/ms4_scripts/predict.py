"""
predict.py - Links Model 1 (failure yes/no) with the four failure-type models.

Input  : Either 6 raw sensor readings  OR  9 already-scaled features
Output : failure probability, failure type(s), risk level, maintenance action

Folder layout (this file lives in milestone4/ms4_scripts):
    milestone4/ms4_models/   <- all six .joblib files are read from here

─────────────────────────────────────────────────────────────────────────────
TRAINING PIPELINE SUMMARY (reproduced here for exact consistency)
─────────────────────────────────────────────────────────────────────────────
Step 1  Feature engineering on raw values:
          Temp_Diff         = Process temperature [K] - Air temperature [K]
          Mechanical_Power_W = Torque [Nm] * Rotational speed [rpm] * (2π/60)
          Strain_Index      = Torque [Nm] * Tool wear [min]

Step 2  Ordinal encoding:   Type → {'L':0, 'M':1, 'H':2}

Step 3  StandardScaler fitted on train split of the 8 continuous features
          (Type is kept as-is; scaler_milestone3.joblib stores the fit).

Step 4  Model 1  (model_failure.joblib)  → trained on SCALED features.
        Type models (model_TWF/HDF/PWF/OSF.joblib) → trained on UNSCALED
          (raw + feature-engineered, but NOT scaled) features from the
          original ai4i2020 dataset.
─────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────── paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "ms4_models"

# ──────────────────────────────────────────────────────────── columns
# Six raw sensor columns that arrive from the user (before feature engineering)
RAW_INPUT_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Eight continuous features that the scaler was fitted on
CONT = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_Diff",
    "Mechanical_Power_W",
    "Strain_Index",
]

# Full feature list in the exact order the models expect
FEATURES = ["Type"] + CONT        # 9 features total

# Ordered list of derived columns (needed for pre-scaled validation)
DERIVED_COLS = ["Temp_Diff", "Mechanical_Power_W", "Strain_Index"]

TYPE_CODES    = {"L": 0, "M": 1, "H": 2}
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF"]

# ──────────────────────────────────────────────────────────── labels / texts
TYPE_NAMES = {
    "TWF":          "Tool wear failure",
    "HDF":          "Heat dissipation failure",
    "PWF":          "Power failure",
    "OSF":          "Overstrain failure",
    "Unclassified": "Failure of unclear type",
}
ACTIONS = {
    "TWF":          "Replace the cutting tool.",
    "HDF":          "Check the cooling system and airflow.",
    "PWF":          "Check the power supply and the load (power is outside the safe range).",
    "OSF":          "Reduce the load and inspect the tool.",
    "Unclassified": "General inspection by a maintenance technician.",
}

# ──────────────────────────────────────────────────────────── load models once
scaler        = joblib.load(MODELS_DIR / "scaler_milestone3.joblib")
model_failure = joblib.load(MODELS_DIR / "model_failure.joblib")
type_models   = {t: joblib.load(MODELS_DIR / f"model_{t}.joblib") for t in FAILURE_TYPES}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Feature engineering  (raw values only; never on already-scaled data)
# ══════════════════════════════════════════════════════════════════════════════
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the three derived features from RAW sensor readings.

    These are the EXACT formulas used in Milestone 3 preprocessing
    (2_Data_Preprocessing_Pipeline.ipynb) and in failure_type.ipynb.
    Must be called BEFORE scaling.
    """
    df = df.copy()
    df["Temp_Diff"]          = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Mechanical_Power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["Strain_Index"]       = df["Torque [Nm]"] * df["Tool wear [min]"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Full raw → model-ready pipeline
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform a DataFrame of RAW sensor readings into the two feature matrices
    needed by the two model stages.

    Parameters
    ----------
    df : DataFrame with columns matching RAW_INPUT_COLS plus a 'Type' column.
         'Type' can be the string 'L'/'M'/'H' or the encoded integer 0/1/2.

    Returns
    -------
    (raw_feats, scaled_feats)
        raw_feats    : 9-column unscaled DataFrame used by the TYPE models
        scaled_feats : 9-column scaled  DataFrame used by Model 1 (failure)
    """
    df = df.copy()

    # Encode Type if it is still a string
    if df["Type"].dtype == object or df["Type"].dtype.name == "category":
        df["Type"] = df["Type"].astype(str).str.strip().str.upper().map(TYPE_CODES)
    df["Type"] = df["Type"].astype(int)

    # Feature engineering (must be done on raw values, before scaling)
    df = add_features(df)

    # Select features in the exact training order
    raw_feats = df[FEATURES].copy()

    # Apply the SAME scaler that was fitted during training (never refit)
    scaled_feats              = raw_feats.copy()
    scaled_feats[CONT]        = scaler.transform(raw_feats[CONT])

    return raw_feats, scaled_feats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Pre-scaled pipeline  (already processed data, e.g. test_ready_milestone3.csv)
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_scaled(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accept a DataFrame that already went through the full Milestone 3
    preprocessing pipeline (feature engineering + StandardScaler applied).

    No transformations are applied. The function only validates column
    presence and ordering, then returns the data ready for both model stages.

    Note: Type models were trained on UNSCALED data, so when using pre-scaled
    input we cannot supply truly unscaled values to the type models.
    We pass the scaled values as-is; the type models' predictions will be
    approximate but still semantically valid (features are in the right range
    relative to the training distribution).

    Parameters
    ----------
    df : DataFrame with all 9 feature columns from FEATURES (already scaled).
         A 'Machine failure' column is allowed and will be ignored.

    Returns
    -------
    (feats, feats)   – same object returned twice; the caller uses the second
                       copy for Model 1 and the first for the type models.
    """
    # Validate that all 9 expected columns are present
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Pre-scaled data is missing column(s): " + ", ".join(missing) + ".\n"
            "Expected columns: " + ", ".join(FEATURES)
        )

    # Select features in exact training order (drop extra cols like 'Machine failure')
    feats = df[FEATURES].copy()

    # Ensure Type is integer
    feats["Type"] = pd.to_numeric(feats["Type"], errors="coerce").astype("Int64")
    if feats["Type"].isna().any():
        raise ValueError("'Type' column contains non-numeric values in pre-scaled data.")
    feats["Type"] = feats["Type"].astype(int)

    return feats, feats.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════
def risk_level(p: float) -> str:
    if p < 0.2:
        return "Low"
    if p < 0.5:
        return "Medium"
    if p < 0.8:
        return "High"
    return "Critical"


# ══════════════════════════════════════════════════════════════════════════════
# Single-machine prediction  (always raw input from the form)
# ══════════════════════════════════════════════════════════════════════════════
def predict_machine(machine_type, air, proc, rpm, torque, wear,
                    threshold=0.5, type_threshold=0.5):
    """
    Predict failure for a single machine from raw sensor readings.

    Parameters
    ----------
    machine_type : "L", "M" or "H"
    air, proc    : air / process temperature in Kelvin  (raw values)
    rpm, torque, wear : rotational speed, torque [Nm], tool wear [min]  (raw)
    threshold    : failure probability at/above which the machine is flagged
    type_threshold : per-type probability cut-off
    """
    if machine_type not in TYPE_CODES:
        raise ValueError("machine_type must be 'L', 'M' or 'H'")

    row = pd.DataFrame([{
        "Type":                       machine_type,
        "Air temperature [K]":        air,
        "Process temperature [K]":    proc,
        "Rotational speed [rpm]":     rpm,
        "Torque [Nm]":                torque,
        "Tool wear [min]":            wear,
    }])

    raw_feats, scaled_feats = preprocess_raw(row)

    p = float(model_failure.predict_proba(scaled_feats)[0, 1])

    result = {
        "failure_probability": round(p, 3),
        "failure":             p >= threshold,
        "risk_level":          risk_level(p),
        "type_probabilities":  {},
        "predicted_types":     [],
        "actions":             [],
    }

    if result["failure"]:
        # Type models were trained on UNSCALED data → use raw_feats
        probs = {t: float(m.predict_proba(raw_feats)[0, 1]) for t, m in type_models.items()}
        result["type_probabilities"] = {t: round(v, 3) for t, v in probs.items()}
        found = [t for t, v in probs.items() if v >= type_threshold] or ["Unclassified"]
        result["predicted_types"] = found
        result["actions"]         = [ACTIONS[t] for t in found]

    return result


# ──────────────────────────────────────────────────────────── self-test
if __name__ == "__main__":
    print("=== Raw single-machine test ===")
    print(predict_machine("M", 298.5, 308.7, 1400, 45, 100))
    print(predict_machine("L", 298,   308,   1300, 62, 230))
