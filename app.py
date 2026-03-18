from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib

app = Flask(__name__)


# Load model
model = xgb.XGBClassifier()
model.load_model("heatwave_model.json")


# SHAP explainer
explainer = shap.TreeExplainer(model)


# Load weights
weights_df = pd.read_csv("shap_weights.csv")
weights = weights_df["Weight"].values


# Load scalers
shap_scaler = joblib.load("shap_scaler.pkl")
shsi_scaler = joblib.load("shsi_scaler.pkl")


features = [
"2-meter air temperature",
"Relative Humidity (in %)",
"10-meter zonal wind",
"10-meter meridional wind",
"Mean sea level pressure",
"Surface solar radiation downwards",
"Minimum 2-meter air temperature"
]


def classify_heatwave(x):

    if x < 0.25:
        return "No Heatwave"

    elif x < 0.50:
        return "Mild"

    elif x < 0.75:
        return "Moderate"

    else:
        return "Severe"



@app.route("/", methods=["GET","POST"])
def index():

    prediction = None
    severity = None
    shsi = None
    values = {}

    if request.method == "POST":

        for f in features:
            values[f] = float(request.form[f])

        X = np.array(list(values.values())).reshape(1,-1)

        # Heatwave prediction
        pred = model.predict(X)[0]

        # SHAP values
        shap_values = explainer.shap_values(X)

        shap_norm = shap_scaler.transform(np.abs(shap_values))

        SHSI = np.dot(shap_norm, weights)

        SHSI = shsi_scaler.transform(SHSI.reshape(-1,1))[0][0]

        severity = classify_heatwave(SHSI)

        prediction = "Heatwave" if pred==1 else "No Heatwave"

        shsi = round(SHSI,3)

    return render_template(
        "index.html",
        features=features,
        prediction=prediction,
        severity=severity,
        shsi=shsi,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)