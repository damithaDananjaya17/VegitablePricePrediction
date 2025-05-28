import streamlit as st
import joblib
import numpy as np
from datetime import date

# Load trained models
models = {
    "Welimada": joblib.load("welimada_model.pkl"),
    "Bandarawela": joblib.load("bandarawela_model.pkl"),
    "Nuwara Eliya": joblib.load("nuwaraeliya_model.pkl")
}

# Load encoders for each model
encoders = {
    "Welimada": {
        "vegetable": joblib.load("welimada_vegetable_encoder.pkl"),
        "variety": joblib.load("welimada_variety_encoder.pkl"),
        "province": joblib.load("welimada_province_encoder.pkl"),
        "market": joblib.load("welimada_market_encoder.pkl"),
    },
    "Bandarawela": {
        "vegetable": joblib.load("bandarawela_vegetable_encoder.pkl"),
        "variety": joblib.load("bandarawela_variety_encoder.pkl"),
        "province": joblib.load("bandarawela_province_encoder.pkl"),
        "market": joblib.load("bandarawela_market_encoder.pkl"),
    },
    "Nuwara Eliya": {
        "vegetable": joblib.load("nuwaraeliya_vegetable_encoder.pkl"),
        "variety": joblib.load("nuwaraeliya_variety_encoder.pkl"),
        "province": joblib.load("nuwaraeliya_province_encoder.pkl"),
        "market": joblib.load("nuwaraeliya_market_encoder.pkl"),
    }
}

# Streamlit UI
st.set_page_config(page_title="Potato Price Predictor", page_icon="🥔")
st.title("🥔 Potato Price Prediction Tool")
st.markdown("Predict future potato selling prices using crop type, market, weather, and selected date.")

# Step 1: Select Market
selling_market = st.selectbox("Select Selling Market", ["Welimada", "Bandarawela", "Nuwara Eliya"])
encoder = encoders[selling_market]

# Step 2: Select Date
selected_date = st.date_input("Select Expected Selling Date", value=date.today())
month = selected_date.month
day_of_year = selected_date.timetuple().tm_yday

# Step 3: Crop Details (Dropdowns from encoder-trained classes)
vegetable = st.selectbox("Vegetable", list(encoder["vegetable"].classes_))
variety = st.selectbox("Variety", list(encoder["variety"].classes_))
province = st.selectbox("Province", list(encoder["province"].classes_))

# Step 4: Market-specific Climate Inputs
if selling_market == "Welimada":
    temperature = st.number_input("Welimada Temperature (°C)", value=25.0)
    rainfall = st.number_input("Welimada Rainfall (mm)", value=5.0)
elif selling_market == "Bandarawela":
    temperature = st.number_input("Bandarawela Temperature (°C)", value=22.0)
    rainfall = st.number_input("Bandarawela Rainfall (mm)", value=8.0)
else:
    temperature = st.number_input("Nuwara Eliya Temperature (°C)", value=20.0)
    rainfall = st.number_input("Nuwara Eliya Rainfall (mm)", value=10.0)

# Step 5: Predict Button
if st.button("Predict Price"):
    try:
        # Encode categorical fields
        veg_code = encoder["vegetable"].transform([vegetable])[0]
        var_code = encoder["variety"].transform([variety])[0]
        prov_code = encoder["province"].transform([province])[0]
        market_code = encoder["market"].transform([selling_market])[0]

        # Feature input structure: [veg, variety, temp, rainfall, province, market, month, day_of_year]
        input_features = np.array([[veg_code, var_code, temperature, rainfall, prov_code, market_code, month, day_of_year]])

        # Predict
        model = models[selling_market]
        prediction = model.predict(input_features)[0]

        # Show result
        st.success(
            f"📅 Predicted Price on {selected_date.strftime('%d %B %Y')} at **{selling_market}**: Rs. {round(prediction, 2)}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that the inputs match what your model was trained with.")
