import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import date
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

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
st.title("🥔 Vegitable Price Prediction Tool")
st.markdown("Predict future vegitable selling prices using crop type, market, weather, and selected date.")

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

# Step 4.5: Farmer Production Estimate
farmer_production_kg = st.number_input("How many kilograms of potato will you produce?", min_value=0)

# Step 5: Predict Button
if st.button("Predict Price"):
    try:
        veg_code = encoder["vegetable"].transform([vegetable])[0]
        var_code = encoder["variety"].transform([variety])[0]
        prov_code = encoder["province"].transform([province])[0]
        market_code = encoder["market"].transform([selling_market])[0]

        input_features = np.array([[veg_code, var_code, temperature, rainfall, prov_code, market_code, month, day_of_year]])

        model = models[selling_market]
        prediction = model.predict(input_features)[0]

        st.success(
            f"🔻 Predicted Price on {selected_date.strftime('%d %B %Y')} at **{selling_market}**: Rs. {round(prediction, 2)}"
        )

        price_trend_df = pd.read_csv("crop_price_trends.csv")
        avg_price_row = price_trend_df[
            (price_trend_df['vegetable'] == vegetable) &
            (price_trend_df['variety'] == variety)
        ]

        avg_price = None
        if not avg_price_row.empty:
            avg_price = avg_price_row.iloc[0]['avg_price']

            st.subheader("📊 Price Comparison")
            fig, ax = plt.subplots()
            ax.bar(["Predicted", "Average"], [prediction, avg_price], color=["skyblue", "lightgreen"])
            ax.set_ylabel("Rs.")
            st.pyplot(fig)

        if os.path.exists("historical_production.csv"):
            hist_df = pd.read_csv("historical_production.csv")
            ref = hist_df[(hist_df["vegetable"] == vegetable) & (hist_df["market"] == selling_market)]

            if not ref.empty:
                avg_kg = ref.iloc[0]["average_kg"]
                if 0.85 * avg_kg <= farmer_production_kg <= 1.15 * avg_kg:
                    st.success(f"🟢 Your planned amount is close to the average ({int(avg_kg)} kg) for {selling_market}.")

        farmer_entry = pd.DataFrame([{
            "vegetable": vegetable,
            "market": selling_market,
            "year": selected_date.year,
            "kg": farmer_production_kg
        }])
        farmer_log_path = "farmer_entries.csv"
        farmer_entry.to_csv(farmer_log_path, mode='a', header=not os.path.exists(farmer_log_path), index=False)

        if os.path.exists(farmer_log_path):
            all_entries = pd.read_csv(farmer_log_path)
            year_total = all_entries[
                (all_entries["vegetable"] == vegetable) &
                (all_entries["market"] == selling_market) &
                (all_entries["year"] == selected_date.year)
            ]["kg"].sum()

            if not ref.empty:
                st.markdown(f"📦 Farmers have submitted **{int(year_total)} kg** for {selling_market} in {selected_date.year}. Target: {int(avg_kg)} kg.")

                if year_total >= avg_kg:
                    st.warning("❌ Required total has already been met or exceeded. Oversupply risk increases.")
                    if avg_price is not None:
                        alternatives = price_trend_df[
                            (price_trend_df['avg_price'] > avg_price) &
                            (price_trend_df['vegetable'] != vegetable)
                        ].sort_values(by='avg_price', ascending=False).head(3)

                        if not alternatives.empty:
                            st.markdown("### 🌱 Due to oversupply, consider switching to one of these high-demand crops:")
                            for idx, row in alternatives.iterrows():
                                st.markdown(f"- **{row['vegetable']}** ({row['variety']}) – Avg. Price: Rs. {int(row['avg_price'])}")
                else:
                    remaining = avg_kg - year_total
                    st.info(f"✅ You can still produce. Only {int(remaining)} kg remaining to meet market demand.")

        if st.checkbox("📈 Retrain model using this input"):
            retrain_row = pd.DataFrame([{
                "vegetable": vegetable,
                "variety": variety,
                "temperature": temperature,
                "rainfall": rainfall,
                "province": province,
                "selling_market": selling_market,
                "Month": month,
                "DayOfYear": day_of_year,
                "price": prediction
            }])
            csv_path = f"{selling_market.lower()}_training_data.csv"
            retrain_row.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

            df = pd.read_csv(csv_path)
            df['vegetable'] = encoder["vegetable"].transform(df['vegetable'])
            df['variety'] = encoder["variety"].transform(df['variety'])
            df['province'] = encoder["province"].transform(df['province'])
            df['selling_market'] = encoder["market"].transform(df['selling_market'])

            X = df[['vegetable', 'variety', 'temperature', 'rainfall', 'province', 'selling_market', 'Month', 'DayOfYear']]
            y = df['price']

            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            joblib.dump(model, f"{selling_market.lower()}_model.pkl")
            models[selling_market] = model

            st.success("✅ Model retrained with new data.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that the inputs match what your model was trained with.")
