"""
predict.py - links Model 1 (failure yes/no) with the four failure-type models.

Input : 6 raw readings of a machine
Output: failure probability, failure type(s), risk level, maintenance action

Folder layout (this file lives in milestone4/ms4_scripts):
    milestone4/ms4_models/   <- all six .joblib files are read from here
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "ms4_models"

# ------------------------------------------------------------------ columns
CONT = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]", "Temp_Diff", "Mechanical_Power_W", "Strain_Index"]
FEATURES = ["Type"] + CONT              # same order as train_ready_milestone3.csv
TYPE_CODES = {"L": 0, "M": 1, "H": 2}
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF"]

# ------------------------------------------------------------------ texts for the user
TYPE_NAMES = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "Unclassified": "Failure of unclear type",
}
ACTIONS = {
    "TWF": "Replace the cutting tool.",
    "HDF": "Check the cooling system and airflow.",
    "PWF": "Check the power supply and the load (power is outside the safe range).",
    "OSF": "Reduce the load and inspect the tool.",
    "Unclassified": "General inspection by a maintenance technician.",
}

# ------------------------------------------------------------------ load models once
scaler = joblib.load(MODELS_DIR / "scaler_milestone3.joblib")
model_failure = joblib.load(MODELS_DIR / "model_failure.joblib")
type_models = {t: joblib.load(MODELS_DIR / f"model_{t}.joblib") for t in FAILURE_TYPES}


def add_features(df):
    """Same three engineered features as Milestone 3."""
    df = df.copy()
    df["Temp_Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Mechanical_Power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["Strain_Index"] = df["Torque [Nm]"] * df["Tool wear [min]"]
    return df


def risk_level(p):
    if p < 0.2:
        return "Low"
    if p < 0.5:
        return "Medium"
    if p < 0.8:
        return "High"
    return "Critical"


def predict_machine(machine_type, air, proc, rpm, torque, wear,
                    threshold=0.5, type_threshold=0.5):
    """
    machine_type : "L", "M" or "H"
    air, proc    : air / process temperature in Kelvin
    rpm, torque, wear : rotational speed, torque [Nm], tool wear [min]
    threshold    : failure probability at or above which we call it a failure
    """
    if machine_type not in TYPE_CODES:
        raise ValueError("machine_type must be 'L', 'M' or 'H'")

    row = pd.DataFrame([{
        "Type": TYPE_CODES[machine_type],
        "Air temperature [K]": air, "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear,
    }])

    raw = add_features(row)[FEATURES]                  # UNSCALED -> type models
    scaled = raw.copy()
    scaled[CONT] = scaler.transform(raw[CONT])         # SCALED   -> Model 1

    p = float(model_failure.predict_proba(scaled)[0, 1])
    result = {
        "failure_probability": round(p, 3),
        "failure": p >= threshold,
        "risk_level": risk_level(p),
        "type_probabilities": {},
        "predicted_types": [],
        "actions": [],
    }

    if result["failure"]:                              # ask the specialists only now
        probs = {t: float(m.predict_proba(raw)[0, 1]) for t, m in type_models.items()}
        result["type_probabilities"] = {t: round(v, 3) for t, v in probs.items()}
        found = [t for t, v in probs.items() if v >= type_threshold] or ["Unclassified"]
        result["predicted_types"] = found
        result["actions"] = [ACTIONS[t] for t in found]
    return result


if __name__ == "__main__":
    print(predict_machine("M", 298.5, 308.7, 1400, 45, 100))
    print(predict_machine("L", 298, 308, 1300, 62, 230))
