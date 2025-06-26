import pickle
import streamlit as st
import pandas as pd
from os import path

# Page settings
st.set_page_config(page_title="Beer Servings Estimation")
st.title("🍺 Beer Servings Estimation App")
st.write("Predict the total litres of pure alcohol based on beverage consumption details.")

# Load the trained model
model_path = path.join("model", "LR_model.pkl")

try:
    with open(model_path, "rb") as f:
        LR_model = pickle.load(f)
except FileNotFoundError:
    st.error(f"❌ Model file not found at: {model_path}")
    st.stop()

# Input fields (based on lowercase CSV column names)
country = st.selectbox(
    "Select a country",
    ["Germany", "USA", "India", "Brazil", "Czech Republic", "Ireland", "Japan"]
)

continent = st.selectbox(
    "Select a continent",
    ["Europe", "North America", "Asia", "South America", "Africa", "Oceania"]
)

beer = st.number_input("🍺 Beer Servings", min_value=0, value=50)
spirit = st.number_input("🥃 Spirit Servings", min_value=0, value=50)
wine = st.number_input("🍷 Wine Servings", min_value=0, value=50)

# Predict
if st.button("Predict Total Alcohol Consumption"):
    input_df = pd.DataFrame({
        'country': [country],
        'beer_servings': [beer],
        'spirit_servings': [spirit],
        'wine_servings': [wine],
        'continent': [continent]
    })

    st.write("🔍 Input Preview:")
    st.dataframe(input_df)

    try:
        pred = LR_model.predict(input_df)
        st.success(f"🎯 Estimated Total Litres of Pure Alcohol: {round(pred[0], 2)}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
