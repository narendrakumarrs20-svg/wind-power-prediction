import streamlit as st
import pandas as pd
import gdown
import os
from wind_predictor import WindPowerPredictor

st.set_page_config(page_title="Wind Power Prediction", layout="wide")

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@st.cache_data
def download_training_data():
    files = {
        "Panapatty_.2018_scada_data.csv": "1aka58ljL8Jtyy6haVagdJC-ZUdK-X2DK",
        "Panapatty_.2019_scada_data.csv": "1Clqxs3BlaSP7mbyJuqOPjmgqPzhve31q",
        "Panapatty_.2020_scada_data.csv": "1XMZ5_fTO86yutzrsIzGiocGGI8MJOZO0"
    }
    
    for filename, file_id in files.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path) or os.path.getsize(path) < 1000: # Redownload if file is missing or too small
            with st.spinner(f"Downloading {filename}..."):
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, path, quiet=False)
                
                # Verify if the download actually worked
                if os.path.exists(path) and os.path.getsize(path) < 1000:
                    st.error(f"⚠️ {filename} is too small! Check Google Drive Sharing Permissions.")

download_training_data()

st.sidebar.title("Settings")
model_type = st.sidebar.selectbox("Select Model", ["random_forest", "linear", "gradient_boosting"])
mode = st.sidebar.radio("Prediction Mode", ["Full Year", "Date Range"])

date_range = None
if mode == "Date Range":
    start = st.sidebar.date_input("Start Date")
    end = st.sidebar.date_input("End Date")
    date_range = (str(start), str(end))

st.title("🌬️ Wind Power Prediction App")
st.header("Upload Test Data")
uploaded_file = st.file_uploader("Upload 2021 Wind Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Save the uploaded test file to the data directory
    test_filename = "Panapatty_.2021_scada_data.csv"
    test_file_path = os.path.join(DATA_DIR, test_filename)
    with open(test_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Predict Power"):
        try:
            predictor = WindPowerPredictor(
                base_path=DATA_DIR,
                train_files=[
                    'Panapatty_.2018_scada_data.csv',
                    'Panapatty_.2019_scada_data.csv',
                    'Panapatty_.2020_scada_data.csv'
                ],
                test_file='Panapatty_.2021_scada_data.csv'
            )

            with st.spinner("Processing data and training model..."):
                results_df, metrics = predictor.run_prediction(
                    model_type=model_type,
                    date_range=date_range
                )

            st.success("Prediction Complete!")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("R² Score", f"{metrics['R2']:.4f}")

            st.subheader("Power Prediction Visualization")
            st.line_chart(results_df.set_index('Timestamp')[['Actual', 'Predicted']])
            st.dataframe(results_df.head(100))

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload the 2021 SCADA CSV file to begin.")