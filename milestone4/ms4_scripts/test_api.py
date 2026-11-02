"""
Quick integration test for the /api/dataset endpoint.
Tests both RAW and SCALED modes.
"""
import json
import urllib.request

BASE = "http://127.0.0.1:5000"


def multipart_post(url, fields, files):
    boundary = "TestBoundary42"
    body_parts = []

    for name, value in fields.items():
        body_parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )

    for name, (filename, data, ctype) in files.items():
        header = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode()
        body_parts.append(header + data + b"\r\n")

    body_parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(body_parts)
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


# ── Test 1: /api/predict (single raw machine) ─────────────────────────────
print("=" * 55)
print("Test 1: /api/predict (single raw machine)")
payload = json.dumps({
    "machine_type": "M", "air": 298.5, "proc": 308.7,
    "rpm": 1400, "torque": 45, "wear": 100
}).encode()
req = urllib.request.Request(
    f"{BASE}/api/predict", data=payload,
    headers={"Content-Type": "application/json"}, method="POST"
)
with urllib.request.urlopen(req) as r:
    d = json.load(r)
print(f"  failure_probability : {d['failure_probability']}")
print(f"  risk_level          : {d['risk_level']}")
print(f"  PASS: not all predicted failure = {not d['failure']}")


# ── Test 2: /api/dataset mode=scaled (test_ready_milestone3.csv) ──────────
print()
print("=" * 55)
print("Test 2: /api/dataset  mode=scaled  (test_ready_milestone3.csv)")
with open("milestone4/ms4_data/test_ready_milestone3.csv", "rb") as f:
    csv_bytes = f.read()

d = multipart_post(
    f"{BASE}/api/dataset",
    fields={"threshold": "0.5", "mode": "scaled"},
    files={"file": ("test_ready_milestone3.csv", csv_bytes, "text/csv")}
)
print(f"  mode                : {d.get('mode')}")
print(f"  n                   : {d['n']}")
print(f"  failures            : {d['failures']}")
print(f"  risk                : {d['risk']}")
m = d.get("metrics")
if m:
    print(f"  Accuracy            : {round(m['accuracy'], 4)}")
    print(f"  F1                  : {round(m['f1'], 4)}")
print(f"  PASS: not all failure= {d['failures'] < d['n']}")


# ── Test 3: /api/dataset mode=raw (original CSV) ──────────────────────────
print()
print("=" * 55)
print("Test 3: /api/dataset  mode=raw  (ai4i2020_original_data.csv, first 200 rows)")
import io, pandas as pd
orig = pd.read_csv("milestone1/ms1_data/ai4i2020_original_data.csv").head(200)
raw_cols = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"]
sub = orig[raw_cols]
buf = io.BytesIO()
sub.to_csv(buf, index=False)
csv_bytes2 = buf.getvalue()

d2 = multipart_post(
    f"{BASE}/api/dataset",
    fields={"threshold": "0.5", "mode": "raw"},
    files={"file": ("raw_sample.csv", csv_bytes2, "text/csv")}
)
print(f"  mode                : {d2.get('mode')}")
print(f"  n                   : {d2['n']}")
print(f"  failures            : {d2['failures']}")
m2 = d2.get("metrics")
if m2:
    print(f"  Accuracy            : {round(m2['accuracy'], 4)}")
    print(f"  F1                  : {round(m2['f1'], 4)}")
print(f"  PASS: not all failure= {d2['failures'] < d2['n']}")
print()
print("ALL TESTS PASSED ✅")
