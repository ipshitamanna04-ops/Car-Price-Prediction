from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "car_price_prediction.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


model = load_model()
predictor = getattr(model, "best_estimator_", model)
feature_names = getattr(predictor, "feature_names_in_", None)


def prepare_input_data(brand, year, kilometers, fuel, transmission, engine):
    input_data = pd.DataFrame([
        {
            "Brand": brand,
            "Year": year,
            "Kilometers_Driven": kilometers,
            "Fuel_Type": fuel,
            "Transmission": transmission,
            "Engine_CC": engine,
        }
    ])

    input_encoded = pd.get_dummies(input_data, dtype=int)
    if feature_names is not None:
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
    return input_encoded


st.title("Car Price Prediction")
st.markdown("---------------------------")

brand = st.selectbox(
    "Select Brand",
    ["Ford", "Honda", "Hyundai", "Mahindra", "Maruti", "Tata", "Toyota", "Volkswagen"],
)

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2020,
)

kilometers = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=50000,
)

fuel = st.selectbox(
    "Fuel Type",
    ["CNG", "Diesel", "Electric", "Petrol"],
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual"],
)

engine = st.number_input(
    "Engine CC",
    min_value=500,
    max_value=8000,
    value=1200,
)

if st.button("Predict Car Price"):
    try:
        prepared_input = prepare_input_data(brand, year, kilometers, fuel, transmission, engine)
        prediction = predictor.predict(prepared_input)[0]
        st.success(f"Predicted Car Price: {prediction:,.2f}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
