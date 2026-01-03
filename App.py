import streamlit as st

st.set_page_config(page_title="Wind Power Prediction", layout="centered")

st.title("🌬️ Wind Power Prediction App")

st.header("Enter Wind Parameters")

wind_speed = st.text_input("Wind Speed (m/s)")
temperature = st.text_input("Temperature (°C)")

uploaded_file = st.file_uploader("Upload Wind Dataset (CSV)", type=["csv"])

if st.button("Predict Power"):
    st.success("Prediction button clicked!")

st.write("App is running successfully 🚀")
