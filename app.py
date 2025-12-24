import streamlit as st
import pandas as pd

from src.preprocessing import compute_rolling_average
from src.frost_logic import detect_frost_windows
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Frost Risk Predictor", layout="wide")

st.title("❄️ AI-Enabled Early Frost Risk Prediction")

station_coords = {
    "S1": (22.72, 75.85),
    "S2": (22.73, 75.86),
    "S3": (22.74, 75.84),
    "S4": (22.71, 75.83),
    "S5": (22.75, 75.87)
}


# Parameters
W = st.slider("Rolling Window (hours)", 2, 6, 3)
F = st.slider("Frost Threshold (°C)", -2.0, 5.0, 0.0)
M = st.slider("Minimum Stations", 1, 5, 3)

def train_frost_model(df, threshold):
    data = df.dropna().copy()

    data["frost_label"] = (data["rolling_avg"] < threshold).astype(int)

    X = data[["temperature", "rolling_avg", "hour"]]
    y = data["frost_label"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    return model


df = pd.read_csv("data/temperature_data.csv")

df = compute_rolling_average(df, W)
# Train ML model
model = train_frost_model(df, F)

latest_hour = df["hour"].max()
latest_data = df[df["hour"] == latest_hour].copy()

latest_data["frost"] = latest_data["rolling_avg"] < F

map_data = []

for _, row in latest_data.iterrows():
    lat, lon = station_coords[row["station"]]
    map_data.append({
        "lat": lat,
        "lon": lon,
        "station": row["station"],
        "risk": "Frost Risk" if row["frost"] else "Safe"
    })

map_df = pd.DataFrame(map_data)



latest_features = latest_data[["temperature", "rolling_avg", "hour"]]

probabilities = model.predict_proba(latest_features)[:, 1]
avg_frost_probability = probabilities.mean() * 100

st.subheader("🤖 AI Frost Risk Prediction")

st.metric(
    label="Next Hour Frost Probability",
    value=f"{avg_frost_probability:.2f} %"
)

if avg_frost_probability > 70:
    st.error("🚨 High Frost Risk – Activate Protection Systems")
elif avg_frost_probability > 40:
    st.warning("⚠️ Moderate Risk – Prepare Frost Protection")
else:
    st.success("✅ Low Risk – No Immediate Action Needed")


st.subheader("📊 Temperature Data")
st.dataframe(df.head(10))

frost_windows = detect_frost_windows(df.dropna(), F, M)

st.subheader("🚨 Frost Alerts")    

if frost_windows:
    for hour, severity in frost_windows:
        st.error(f"Frost Risk at Hour {hour} | Stations affected: {severity}")
else:
    st.success("No frost risk detected")

st.subheader("🗺️ Vineyard Frost Risk Map")
st.map(map_df)


# Visualization
st.subheader("📈 Temperature Trend")
st.line_chart(
    df.pivot(index="hour", columns="station", values="temperature")
)
