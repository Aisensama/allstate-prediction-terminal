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
    row["cat1"] = int(data.get("cat1", CAT_DEFAULT))
    row["cat2"] = int(data.get("cat2", CAT_DEFAULT))
    row["cont1"] = float(data.get("cont1", CONT_DEFAULT))
    row["cont2"] = float(data.get("cont2", CONT_DEFAULT))

    # --- Assemble a 130-column DataFrame in training column order --------
    df = pd.DataFrame([row], columns=ALL_COLS)

    # --- Predict & reverse the log transform -----------------------------
    log_pred = model.predict(df)[0]
    severity = float(np.expm1(log_pred))
    severity = round(max(severity, 0), 2)  # clamp negatives to 0

    return jsonify({
        "predicted": severity,
        "baseline": BASELINE_AVG
    })


if __name__ == "__main__":
    app.run(debug=True)
