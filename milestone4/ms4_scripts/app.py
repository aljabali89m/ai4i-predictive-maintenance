"""
app.py - Makina Sense GUI for the AI4I predictive-maintenance project.

Put app.py + templates/ next to predict.py (milestone4/ms4_scripts), then:
    pip install -r requirements.txt
    python app.py          ->  http://127.0.0.1:5000

It reuses everything from predict.py (the six .joblib files, feature
engineering, scaling rules, risk levels, actions), so results match
predict_machine() exactly - the dataset path is just the vectorised version.
"""
import math
from collections import Counter

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

try:
    import predict as P
except FileNotFoundError as e:
    raise SystemExit(f"\nModel file not found: {e.filename}\n"
                     "predict.py needs all six files in milestone4/ms4_models:\n"
                     "model_failure, model_TWF, model_HDF, model_PWF, model_OSF, scaler_milestone3 (.joblib)\n")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
MAX_ROWS = 50000

RAW_COLS = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
NUM_COLS = RAW_COLS[1:]

# Reference scores copied from failure_detection.ipynb and failure_type.ipynb.
REFERENCE = {
    "failure": {"accuracy": 0.9708, "precision": 0.9626, "recall": 0.9067, "f1": 0.9338},
    "types": {"TWF": {"f1": 0.000, "cv": 0.063}, "HDF": {"f1": 0.949, "cv": 0.950},
              "PWF": {"f1": 1.000, "cv": 0.984}, "OSF": {"f1": 1.000, "cv": 0.995}},
}
SINGLE_LIMITS = {"air": ("Air temperature", 250, 400), "proc": ("Process temperature", 250, 400),
                 "rpm": ("Rotational speed", 0, 10000), "torque": ("Torque", 0, 200),
                 "wear": ("Tool wear", 0, 1000)}


def predict_df(df):
    """Vectorised version of predict_machine(): returns (failure_prob, {type: prob})."""
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing))
    raw = df[RAW_COLS].copy()
    codes = raw["Type"].astype(str).str.strip().str.upper().map(P.TYPE_CODES)
    numeric = pd.to_numeric(raw["Type"], errors="coerce")
    raw["Type"] = codes.fillna(numeric.where(numeric.isin([0, 1, 2])))
    for c in NUM_COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    bad = raw.isna().any(axis=1)
    if bad.any():
        rows = (bad[bad].index[:5] + 2).tolist()
        raise ValueError(f"{int(bad.sum())} row(s) have a missing or invalid value (file rows {rows}). "
                         "Type must be L, M or H; the other five columns must be numbers.")
    raw["Type"] = raw["Type"].astype(int)
    raw = P.add_features(raw)[P.FEATURES]               # unscaled -> type models
    scaled = raw.copy()
    scaled[P.CONT] = P.scaler.transform(raw[P.CONT])    # scaled   -> Model 1
    p = P.model_failure.predict_proba(scaled)[:, 1]
    tprob = {t: m.predict_proba(raw)[:, 1] for t, m in P.type_models.items()}
    return p, tprob


def prf(y, pred):
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "support": int(y.sum())}


@app.get("/")
def index():
    return render_template("index.html", ref=REFERENCE, type_names=P.TYPE_NAMES)


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    machine_type = data.get("machine_type")
    if machine_type not in P.TYPE_CODES:
        return jsonify(error="Choose a machine type (L, M or H)."), 400
    values = {}
    for key, (label, lo, hi) in SINGLE_LIMITS.items():
        try:
            v = float(data.get(key))
        except (TypeError, ValueError):
            return jsonify(error=f"{label} must be a number."), 400
        if not math.isfinite(v) or not lo <= v <= hi:
            return jsonify(error=f"{label} must be between {lo} and {hi}."), 400
        values[key] = v
    r = P.predict_machine(machine_type, values["air"], values["proc"],
                          values["rpm"], values["torque"], values["wear"])
    r["failure"] = bool(r["failure"])
    r["type_names"] = P.TYPE_NAMES
    return jsonify(r)


@app.post("/api/dataset")
def api_dataset():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a CSV file first."), 400
    try:
        thr = min(max(float(request.form.get("threshold", 0.5)), 0.05), 0.95)
    except ValueError:
        thr = 0.5
    try:
        df = pd.read_csv(f)
    except Exception:
        return jsonify(error="Could not read the file as a CSV."), 400
    if df.empty:
        return jsonify(error="The file has no rows."), 400
    if len(df) > MAX_ROWS:
        return jsonify(error=f"Too many rows ({len(df):,}). The limit is {MAX_ROWS:,}."), 400
    try:
        p, tprob = predict_df(df)
    except ValueError as e:
        return jsonify(error=str(e)), 400

    fail = p >= thr
    found = [([t for t in P.FAILURE_TYPES if tprob[t][i] >= 0.5] or ["Unclassified"]) if fail[i] else []
             for i in range(len(df))]
    ids = df["Product ID"].astype(str) if "Product ID" in df.columns else pd.Series(range(1, len(df) + 1)).astype(str)

    y = pd.to_numeric(df["Machine failure"], errors="coerce") if "Machine failure" in df.columns else None
    labelled = y is not None and y.isin([0, 1]).all()

    rows = []
    for i in range(len(df)):
        r = df.iloc[i]
        rows.append({
            "id": ids.iloc[i], "type": str(r["Type"]), "air": float(r[RAW_COLS[1]]), "proc": float(r[RAW_COLS[2]]),
            "rpm": float(r[RAW_COLS[3]]), "torque": float(r[RAW_COLS[4]]), "wear": float(r[RAW_COLS[5]]),
            "prob": round(float(p[i]), 3), "fail": bool(fail[i]), "risk": P.risk_level(p[i]), "types": found[i],
            "action": " ".join(P.ACTIONS[t] for t in found[i]),
            "ptype": {t: round(float(tprob[t][i]), 3) for t in P.FAILURE_TYPES},
            **({"actual": int(y.iloc[i])} if labelled else {}),
        })

    metrics = None
    if labelled:
        tn, fp, fn, tp = confusion_matrix(y, fail.astype(int), labels=[0, 1]).ravel()
        metrics = {"accuracy": float((tn + tp) / len(df)), **prf(y, fail.astype(int)),
                   "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "types": {}}
        for t in P.FAILURE_TYPES:        # specialists scored on all rows, like the notebook
            if t in df.columns and pd.to_numeric(df[t], errors="coerce").isin([0, 1]).all():
                metrics["types"][t] = prf(df[t].astype(int), (tprob[t] >= 0.5).astype(int))

    risk = Counter(P.risk_level(v) for v in p)
    tcount = Counter(t for lst in found for t in lst)
    return jsonify(n=len(df), failures=int(fail.sum()), threshold=thr,
                   risk={k: risk.get(k, 0) for k in ["Low", "Medium", "High", "Critical"]},
                   types={k: tcount.get(k, 0) for k in P.FAILURE_TYPES + ["Unclassified"]},
                   metrics=metrics, rows=rows)


if __name__ == "__main__":
    app.run(debug=True)
