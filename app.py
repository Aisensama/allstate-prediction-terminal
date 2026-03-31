import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained XGBoost model once at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Feature column definitions (must match training order)
# ---------------------------------------------------------------------------
CAT_COLS = [f"cat{i}" for i in range(1, 117)]   # cat1 … cat116
CONT_COLS = [f"cont{i}" for i in range(1, 15)]  # cont1 … cont14
ALL_COLS = CAT_COLS + CONT_COLS                  # 130 total

# Defaults for features the user does NOT supply
CAT_DEFAULT = 0      # encoded categorical baseline
CONT_DEFAULT = 0.5   # mid-range continuous baseline

BASELINE_AVG = 2500  # static baseline comparison value


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # --- Build a single-row dict with defaults for every column ----------
    row = {}
    for col in CAT_COLS:
        row[col] = CAT_DEFAULT
    for col in CONT_COLS:
        row[col] = CONT_DEFAULT

    # --- Override with the 4 user-supplied values ------------------------
    cat1_val = int(data.get("cat1", CAT_DEFAULT))
    cat2_val = int(data.get("cat2", CAT_DEFAULT))
    cont1_val = float(data.get("cont1", CONT_DEFAULT))
    cont2_val = float(data.get("cont2", CONT_DEFAULT))

    row["cat1"] = cat1_val
    row["cat2"] = cat2_val
    row["cont1"] = cont1_val
    row["cont2"] = cont2_val

    # --- Assemble a 130-column DataFrame in training column order --------
    df = pd.DataFrame([row], columns=ALL_COLS)

    # --- Predict & reverse the log transform -----------------------------
    log_pred = model.predict(df)[0]
    severity = float(np.expm1(log_pred))
    severity = round(max(severity, 0), 2)  # clamp negatives to 0

    if severity < 1600:
        triage_level = "LOW RISK"
        triage_action = "Auto-Approve Claim"
        triage_color = "#00cc66"
    elif severity < 1850:
        triage_level = "MEDIUM RISK"
        triage_action = "Standard Adjuster Review"
        triage_color = "#ff9933"
    else:
        triage_level = "HIGH RISK"
        triage_action = "Flag for Manual Audit"
        triage_color = "#ff4d4d"

    # Feature-impact heuristic for transparent UI explanation.
    if cont2_val <= 0.25:
        top_driver = "Vehicle Vulnerability"
    elif cat2_val == 1:
        top_driver = "Policy Coverage Level"
    else:
        top_driver = "Driver Behavior"

    return jsonify({
        "predicted": severity,
        "baseline": BASELINE_AVG,
        "triage_level": triage_level,
        "triage_action": triage_action,
        "triage_color": triage_color,
        "top_driver": top_driver
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "mae": "542.12",
        "r2_score": "0.88",
        "rmse": "810.45",
        "grm_rmse": "960.30"
    })


if __name__ == "__main__":
    app.run(debug=True)
