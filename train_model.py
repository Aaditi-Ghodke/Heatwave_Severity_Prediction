import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


# ==============================
# Load Dataset
# ==============================

df = pd.read_excel("Dataset_2000_to_2025_final.xlsx")

df["time"] = pd.to_datetime(df["time"])


# ==============================
# Create Heatwave Target
# ==============================

threshold = df["Maximum 2-meter air temperature"].quantile(0.95)

df["heatwave"] = (df["Maximum 2-meter air temperature"] > threshold).astype(int)


# ==============================
# Features
# ==============================

features = [
"2-meter air temperature",
"Relative Humidity (in %)",
"10-meter zonal wind",
"10-meter meridional wind",
"Mean sea level pressure",
"Surface solar radiation downwards",
"Minimum 2-meter air temperature"
]

X = df[features]
y = df["heatwave"]


# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# ==============================
# Train XGBoost
# ==============================

model = xgb.XGBClassifier(
n_estimators=400,
max_depth=6,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8
)

model.fit(X_train, y_train)


# ==============================
# Accuracy
# ==============================

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)


# ==============================
# SHAP
# ==============================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# Feature importance
importance = np.abs(shap_values).mean(axis=0)

weights = importance / importance.sum()

weights_df = pd.DataFrame({
"Feature": X.columns,
"Weight": weights
})

weights_df.to_csv("shap_weights.csv", index=False)


# ==============================
# Normalize SHAP values
# ==============================

shap_scaler = MinMaxScaler()

shap_norm = shap_scaler.fit_transform(np.abs(shap_values))

joblib.dump(shap_scaler, "shap_scaler.pkl")


# ==============================
# SHSI Calculation
# ==============================

SHSI = np.dot(shap_norm, weights)

df["SHSI"] = SHSI


# Normalize SHSI

shsi_scaler = MinMaxScaler()

df["SHSI"] = shsi_scaler.fit_transform(df[["SHSI"]])

joblib.dump(shsi_scaler, "shsi_scaler.pkl")


# ==============================
# Severity Classification
# ==============================

def classify_heatwave(x):

    if x < 0.25:
        return "No Heatwave"

    elif x < 0.50:
        return "Mild"

    elif x < 0.75:
        return "Moderate"

    else:
        return "Severe"


df["Heatwave_Class"] = df["SHSI"].apply(classify_heatwave)


# ==============================
# Save Model
# ==============================

model.save_model("heatwave_model.json")


print("Training completed and files saved")