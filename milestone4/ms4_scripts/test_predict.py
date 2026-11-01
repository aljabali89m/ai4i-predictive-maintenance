"""Run:  python test_predict.py   (from inside ms4_scripts)"""
from pathlib import Path
import pandas as pd
from predict import predict_machine

ORIGINAL = Path(__file__).resolve().parent.parent.parent / "milestone1" / "ms1_data" / "ai4i2020_original_data.csv"

print("=== 1) Three hand-made examples ===")
for args in [("M", 298.5, 308.7, 1400, 45, 100),
             ("L", 298, 308, 1300, 62, 230),
             ("M", 300, 309, 2800, 15, 50)]:
    r = predict_machine(*args)
    print(args, "->", r["failure_probability"], r["risk_level"], r["predicted_types"])

print("\n=== 2) Real failure rows from the original file ===")
df = pd.read_csv(ORIGINAL)
for t in ["HDF", "PWF", "OSF", "TWF"]:
    rows = df[(df[t] == 1) & (df[[x for x in ["TWF","HDF","PWF","OSF","RNF"] if x != t]].sum(axis=1) == 0)].head(20)
    hits = 0
    for _, r in rows.iterrows():
        out = predict_machine(r["Type"], r["Air temperature [K]"], r["Process temperature [K]"],
                              r["Rotational speed [rpm]"], r["Torque [Nm]"], r["Tool wear [min]"])
        hits += t in out["predicted_types"]
    print(f"{t}: type found correctly in {hits} of {len(rows)} real {t}-only failures")

print("\n=== 3) Same input twice gives the same output ===")
a = predict_machine("L", 298, 308, 1300, 62, 230)
b = predict_machine("L", 298, 308, 1300, 62, 230)
print("OK" if a == b else "DIFFERENT!")
