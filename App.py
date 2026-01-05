import streamlit as st
import pandas as pd
import gdown
import os
from wind_predictor import WindPowerPredictor

# 1. Page Configuration
st.set_page_config(page_title="Wind Power Prediction", layout="wide")

# 2. Setup Data Directory and Download Logic
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@st.cache_data
def download_training_data():
    """Downloads training CSVs from Google Drive using gdown."""
    # REPLACE THE IDs BELOW with your actual Google Drive File IDs
    files = {
        "Panapatty_.2018_scada_data.csv": "https://drive.google.com/drive/folders/16v2CMHX96gtsIcusB2ZamzZ-G1IaU6p0?usp=sharing",
        "Panapatty_.2019_scada_data.csv": "https://drive.google.com/drive/folders/16v2CMHX96gtsIcusB2ZamzZ-G1IaU6p0?usp=sharing",
        "Panapatty_.2020_scada_data.csv": "https://drive.google.com/drive/folders/16v2CMHX96gtsIcusB2ZamzZ-G1IaU6p0?usp=sharing"
    }
    
    for filename, file_id in files.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            with st.spinner(f"Downloading {filename} from Google Drive..."):
                url = f"https://drive.google.com/drive/folders/16v2CMHX96gtsIcusB2ZamzZ-G1IaU6p0?usp=sharing"
                gdown.download(url, path, quiet=False)

# Trigger the download at startup
download_training_data()

# 3. Sidebar UI
st.sidebar.title("Settings")
model_type = st.sidebar.selectbox("Select Model", ["random_forest", "linear", "gradient_boosting"])
mode = st.sidebar.radio("Prediction Mode", ["Full Year", "Date Range"])

date_range = None
if mode == "Date Range":
    start = st.sidebar.date_input("Start Date")
    end = st.sidebar.date_input("End Date")
    date_range = (str(start), str(end))

# 4. Main App UI
st.title("🌬️ Wind Power Prediction App")
st.write("This app uses historical SCADA data to predict wind power generation.")

st.header("Upload Test Data")
uploaded_file = st.file_uploader("Upload 2021 Wind Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Save the uploaded test file to the data directory
    test_file_path = os.path.join(DATA_DIR, "uploaded_test_data.csv")
    with open(test_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Predict Power"):
        try:
            # Initialize the Predictor from your wind_predictor.py
            predictor = WindPowerPredictor(
                base_path=DATA_DIR,
                train_files=[
                    'Panapatty_.2018_scada_data.csv',
                    'Panapatty_.2019_scada_data.csv',
                    'Panapatty_.2020_scada_data.csv'
                ],
                test_file="Panapatty_.2021_scada_data.csv"
            )

            with st.spinner("Processing data and training model..."):
                results_df, metrics = predictor.run_prediction(
                    model_type=model_type,
                    date_range=date_range
                )

            # Display Success and Metrics
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("R² Score", f"{metrics['R2']:.4f}")

            # Visualizations
            st.subheader("Power Prediction Visualization")
            st.line_chart(results_df.set_index('Timestamp')[['Actual', 'Predicted']])
            
            st.subheader("Prediction Data (First 100 rows)")
            st.dataframe(results_df.head(100))

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV file to begin.")