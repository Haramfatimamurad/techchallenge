from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

app = Flask(__name__)

with (BASE_DIR / "best_heart_disease_model.pkl").open("rb") as model_file:
    model = pickle.load(model_file)

with (BASE_DIR / "scaler.pkl").open("rb") as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_values = request.form if request.method == "POST" else {}

    if request.method == "POST":
        try:
            values = [float(request.form[feature]) for feature in FEATURES]
            input_data = pd.DataFrame([values], columns=FEATURES)
            scaled_data = scaler.transform(input_data)
            prediction = int(model.predict(scaled_data)[0])
        except (KeyError, TypeError, ValueError):
            error = "Enter a valid value for every field."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=False)