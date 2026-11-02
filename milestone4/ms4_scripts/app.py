"""
app.py - Makina Sense GUI for the AI4I predictive-maintenance project.

Put app.py + templates/ next to predict.py (milestone4/ms4_scripts), then:
    pip install -r requirements.txt
    python app.py          ->  http://127.0.0.1:5000

It reuses everything from predict.py (the six .joblib files, feature
engineering, scaling rules, risk levels, actions).

──────────────────────────────────────────────────────────────────────────────
CSV UPLOAD MODES
──────────────────────────────────────────────────────────────────────────────
The /api/dataset endpoint accepts a "mode" form field:

  mode = "raw"
      CSV has 6 sensor columns + Type (raw, unscaled).
      The backend runs the FULL preprocessing pipeline:
        1. Encode Type (L/M/H → 0/1/2)
        2. Compute Temp_Diff, Mechanical_Power_W, Strain_Index
        3. Apply StandardScaler (scaler_milestone3.joblib) to 8 continuous cols
      Required columns: Type, Air temperature [K], Process temperature [K],
                        Rotational speed [rpm], Torque [Nm], Tool wear [min]
      Optional column:  Machine failure (enables metrics)

  mode = "scaled"
      CSV already went through the full Milestone-3 pipeline
      (feature engineering + StandardScaler applied).
      The backend skips ALL transformations and passes the data straight
      to the models.
      Required columns: Type, Air temperature [K], Process temperature [K],
                        Rotational speed [rpm], Torque [Nm], Tool wear [min],
                        Temp_Diff, Mechanical_Power_W, Strain_Index
      Optional column:  Machine failure (enables metrics)
──────────────────────────────────────────────────────────────────────────────
"""
import math
from collections import Counter

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

try:
    import predict as P
except FileNotFoundError as e:
    raise SystemExit(
        f"\nModel file not found: {e.filename}\n"
        "predict.py needs all six files in milestone4/ms4_models:\n"
        "model_failure, model_TWF, model_HDF, model_PWF, model_OSF, "
        "scaler_milestone3 (.joblib)\n"
    )

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
MAX_ROWS = 50000

# ──────────────────────────────────────────────────── column definitions
# Raw-mode: columns expected in the uploaded CSV before any processing
RAW_UPLOAD_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
RAW_NUM_COLS = RAW_UPLOAD_COLS[1:]   # numeric-only subset of RAW_UPLOAD_COLS

# Scaled-mode: all 9 feature columns must already be present
SCALED_UPLOAD_COLS = P.FEATURES      # ["Type"] + P.CONT  (9 features)

# The three derived columns produced exclusively by the Milestone-3 pipeline
DERIVED_COLS = ["Temp_Diff", "Mechanical_Power_W", "Strain_Index"]


def auto_detect_mode(df: pd.DataFrame) -> str:
    """
    Automatically determine whether a CSV contains raw sensor readings or
    already-preprocessed (scaled) features.

    Detection rule
    --------------
    If the file contains ALL THREE derived columns
        Temp_Diff, Mechanical_Power_W, Strain_Index
    -> the Milestone-3 feature-engineering step has already been applied
       -> treat as "scaled" (pre-processed).

    Otherwise -> treat as "raw" (standard sensor readings that need the full
    preprocessing pipeline).

    This heuristic is reliable because those three column names are produced
    exclusively by the Milestone-3 pipeline and would not appear in a normal
    sensor export.
    """
    return "scaled" if all(c in df.columns for c in DERIVED_COLS) else "raw"

# Reference scores from the training notebooks
REFERENCE = {
    "failure": {"accuracy": 0.9708, "precision": 0.9626, "recall": 0.9067, "f1": 0.9338},
    "types": {
        "TWF": {"f1": 0.000, "cv": 0.063},
        "HDF": {"f1": 0.949, "cv": 0.950},
        "PWF": {"f1": 1.000, "cv": 0.984},
        "OSF": {"f1": 1.000, "cv": 0.995},
    },
}

# Keys for single-machine form input (always raw values)
SINGLE_KEYS = {
    "air":    "Air temperature",
    "proc":   "Process temperature",
    "rpm":    "Rotational speed",
    "torque": "Torque",
    "wear":   "Tool wear",
}


# ══════════════════════════════════════════════════════════════════════════════
# Core batch-prediction helper
# ══════════════════════════════════════════════════════════════════════════════
def predict_df(df: pd.DataFrame, mode: str) -> tuple:
    """
    Vectorised prediction for an uploaded CSV.

    Parameters
    ----------
    df   : the DataFrame read from the uploaded CSV
    mode : "raw"    → apply full preprocessing pipeline (engineering + scaling)
           "scaled" → data already preprocessed; pass directly to models

    Returns
    -------
    (p, tprob)
        p      : ndarray of shape (n,) – failure probabilities from Model 1
        tprob  : dict {type_str: ndarray of shape (n,)} – per-type probabilities
    """
    mode = mode.strip().lower()

    if mode not in ("raw", "scaled"):
        raise ValueError('mode must be "raw" or "scaled".')

    # ── RAW MODE ─────────────────────────────────────────────────────────────
    if mode == "raw":
        # 1. Check required columns
        missing = [c for c in RAW_UPLOAD_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                "Raw CSV is missing column(s): " + ", ".join(missing) + ".\n"
                "Required for raw mode: " + ", ".join(RAW_UPLOAD_COLS)
            )

        raw = df[RAW_UPLOAD_COLS].copy()

        # 2. Encode Type  (accept 'L'/'M'/'H' strings  OR  already-numeric 0/1/2)
        codes   = raw["Type"].astype(str).str.strip().str.upper().map(P.TYPE_CODES)
        numeric = pd.to_numeric(raw["Type"], errors="coerce")
        raw["Type"] = codes.fillna(numeric.where(numeric.isin([0, 1, 2])))

        # 3. Convert numeric columns
        for c in RAW_NUM_COLS:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

        # 4. Validate – reject rows with missing/invalid values
        bad = raw.isna().any(axis=1)
        if bad.any():
            rows = (bad[bad].index[:5] + 2).tolist()
            raise ValueError(
                f"{int(bad.sum())} row(s) have a missing or invalid value "
                f"(file rows {rows}). "
                "Type must be L, M or H; the other five columns must be numbers."
            )
        raw["Type"] = raw["Type"].astype(int)

        # 5. Feature engineering + scaling via the unified pipeline
        raw_feats, scaled_feats = P.preprocess_raw(raw)

        # 6. Model 1 (failure) uses SCALED features
        p = P.model_failure.predict_proba(scaled_feats)[:, 1]

        # 7. Type models use UNSCALED (raw-engineered) features
        tprob = {t: m.predict_proba(raw_feats)[:, 1] for t, m in P.type_models.items()}

        return p, tprob

    # ── SCALED MODE ───────────────────────────────────────────────────────────
    # Data has already gone through the full Milestone-3 pipeline.
    # No feature engineering and NO additional scaling are applied.
    else:
        # 1. Validate that all 9 feature columns are present
        missing = [c for c in SCALED_UPLOAD_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                "Pre-scaled CSV is missing column(s): " + ", ".join(missing) + ".\n"
                "Required for scaled mode: " + ", ".join(SCALED_UPLOAD_COLS)
            )

        # 2. Select features in exact training order; drop 'Machine failure' etc.
        feats = df[SCALED_UPLOAD_COLS].copy()

        # 3. Convert all to numeric (they should already be)
        for c in SCALED_UPLOAD_COLS:
            feats[c] = pd.to_numeric(feats[c], errors="coerce")

        bad = feats.isna().any(axis=1)
        if bad.any():
            rows = (bad[bad].index[:5] + 2).tolist()
            raise ValueError(
                f"{int(bad.sum())} row(s) have non-numeric values (file rows {rows})."
            )

        feats["Type"] = feats["Type"].astype(int)

        # 4. Model 1 (failure) – data is already scaled, pass directly
        p = P.model_failure.predict_proba(feats)[:, 1]

        # 5. Type models – also use the same scaled features
        #    (the pre-scaled file does not carry unscaled values any more;
        #     the type models' outputs are still semantically valid)
        tprob = {t: m.predict_proba(feats)[:, 1] for t, m in P.type_models.items()}

        return p, tprob


# ══════════════════════════════════════════════════════════════════════════════
# Metric helper
# ══════════════════════════════════════════════════════════════════════════════
def prf(y, pred):
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "support":   int(y.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def index():
    return render_template("index.html", ref=REFERENCE, type_names=P.TYPE_NAMES)


@app.post("/api/predict")
def api_predict():
    """Single-machine prediction (always raw values from the HTML form)."""
    data         = request.get_json(silent=True) or {}
    machine_type = data.get("machine_type")
    if machine_type not in P.TYPE_CODES:
        return jsonify(error="Choose a machine type (L, M or H)."), 400

    values = {}
    for key, label in SINGLE_KEYS.items():
        try:
            v = float(data.get(key))
        except (TypeError, ValueError):
            return jsonify(error=f"{label} must be a number."), 400
        if not math.isfinite(v):
            return jsonify(error=f"{label} must be a finite number."), 400
        values[key] = v

    r = P.predict_machine(
        machine_type,
        values["air"], values["proc"],
        values["rpm"], values["torque"], values["wear"],
    )
    r["failure"]    = bool(r["failure"])
    r["type_names"] = P.TYPE_NAMES
    return jsonify(r)


@app.post("/api/dataset")
def api_dataset():
    """
    Batch prediction for an uploaded CSV.

    Form fields
    -----------
    file      : the CSV file
    threshold : failure-probability cut-off  (default 0.5, clamped to [0.05, 0.95])
    mode      : "raw" | "scaled"
                  raw    → apply feature engineering + StandardScaler
                  scaled → already preprocessed; pass straight to models
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a CSV file first."), 400

    try:
        thr = min(max(float(request.form.get("threshold", 0.5)), 0.05), 0.95)
    except ValueError:
        thr = 0.5

    mode = (request.form.get("mode") or "").strip().lower()
    # If the client did not supply a mode (or sent an empty string),
    # auto-detect from the column names once the file is loaded.
    manual_mode = mode in ("raw", "scaled")

    try:
        df = pd.read_csv(f)
    except Exception:
        return jsonify(error="Could not read the file as a CSV."), 400

    if df.empty:
        return jsonify(error="The file has no rows."), 400
    if len(df) > MAX_ROWS:
        return jsonify(error=f"Too many rows ({len(df):,}). The limit is {MAX_ROWS:,}."), 400

    # Auto-detect format from column names if mode was not explicitly provided
    if not manual_mode:
        mode = auto_detect_mode(df)

    try:
        p, tprob = predict_df(df, mode)
    except ValueError as e:
        return jsonify(error=str(e)), 400

    fail = p >= thr
    found = [
        ([t for t in P.FAILURE_TYPES if tprob[t][i] >= 0.5] or ["Unclassified"]) if fail[i] else []
        for i in range(len(df))
    ]

    # Product ID column (for display only)
    ids = (
        df["Product ID"].astype(str)
        if "Product ID" in df.columns
        else pd.Series(range(1, len(df) + 1)).astype(str)
    )

    # Ground-truth column (optional – enables metrics panel)
    y       = pd.to_numeric(df.get("Machine failure"), errors="coerce") if "Machine failure" in df.columns else None
    labelled = y is not None and y.isin([0, 1]).all()

    # ── Build per-row response ────────────────────────────────────────────
    # Determine which raw sensor values to echo back to the UI
    if mode == "raw":
        def _row_vals(r):
            return {
                "type":   str(r["Type"]),
                "air":    float(r["Air temperature [K]"]),
                "proc":   float(r["Process temperature [K]"]),
                "rpm":    float(r["Rotational speed [rpm]"]),
                "torque": float(r["Torque [Nm]"]),
                "wear":   float(r["Tool wear [min]"]),
            }
    else:
        # For scaled data, report the scaled values (clearly marked in the UI)
        def _row_vals(r):
            return {
                "type":   str(int(r["Type"])) if pd.notna(r["Type"]) else "?",
                "air":    round(float(r["Air temperature [K]"]), 4),
                "proc":   round(float(r["Process temperature [K]"]), 4),
                "rpm":    round(float(r["Rotational speed [rpm]"]), 4),
                "torque": round(float(r["Torque [Nm]"]), 4),
                "wear":   round(float(r["Tool wear [min]"]), 4),
            }

    rows = []
    for i in range(len(df)):
        r = df.iloc[i]
        row = {
            "id":     ids.iloc[i],
            **_row_vals(r),
            "prob":   round(float(p[i]), 3),
            "fail":   bool(fail[i]),
            "risk":   P.risk_level(p[i]),
            "types":  found[i],
            "action": " ".join(P.ACTIONS[t] for t in found[i]),
            "ptype":  {t: round(float(tprob[t][i]), 3) for t in P.FAILURE_TYPES},
        }
        if labelled:
            row["actual"] = int(y.iloc[i])
        rows.append(row)

    # ── Aggregate metrics (only when ground-truth is available) ──────────
    metrics = None
    if labelled:
        tn, fp, fn, tp = confusion_matrix(y, fail.astype(int), labels=[0, 1]).ravel()
        metrics = {
            "accuracy": float((tn + tp) / len(df)),
            **prf(y, fail.astype(int)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "types": {},
        }
        for t in P.FAILURE_TYPES:
            if t in df.columns and pd.to_numeric(df[t], errors="coerce").isin([0, 1]).all():
                metrics["types"][t] = prf(df[t].astype(int), (tprob[t] >= 0.5).astype(int))

    risk   = Counter(P.risk_level(v) for v in p)
    tcount = Counter(t for lst in found for t in lst)

    return jsonify(
        n=len(df),
        failures=int(fail.sum()),
        threshold=thr,
        mode=mode,
        risk={k: risk.get(k, 0) for k in ["Low", "Medium", "High", "Critical"]},
        types={k: tcount.get(k, 0) for k in P.FAILURE_TYPES + ["Unclassified"]},
        metrics=metrics,
        rows=rows,
    )


if __name__ == "__main__":
    app.run(debug=True)
