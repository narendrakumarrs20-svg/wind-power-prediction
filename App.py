import os
import gdown

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

files = {
    "Panapatty_.2018_scada_data.csv": "https://drive.google.com/drive/folders/16v2CMHX96gtsIcusB2ZamzZ-G1IaU6p0?usp=sharing",
    "Panapatty_.2019_scada_data.csv": "https://drive.google.com/drive/folders/16v2CMHX96gtsIcusB2ZamzZ-G1IaU6p0?usp=sharing",
    "Panapatty_.2020_scada_data.csv": "https://drive.google.com/drive/folders/16v2CMHX96gtsIcusB2ZamzZ-G1IaU6p0?usp=sharing",
    "Panapatty_.2021_scada_data.csv": "https://drive.google.com/drive/folders/16v2CMHX96gtsIcusB2ZamzZ-G1IaU6p0?usp=sharing"
}

for name, file_id in files.items():
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/drive/folders/16v2CMHX96gtsIcusB2ZamzZ-G1IaU6p0?usp=sharing", path, quiet=False)

from fastapi import FastAPI
from wind_predictor import WindPowerPredictor

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Wind Power Forecasting API Running"}

@app.get("/predict")
def predict(model_type: str = "random_forest"):
    
    predictor = WindPowerPredictor(
        base_path="data",
        train_files=[
            "Panapatty_.2018_scada_data.csv",
            "Panapatty_.2019_scada_data.csv",
            "Panapatty_.2020_scada_data.csv"
        ],
        test_file="Panapatty_.2021_scada_data.csv",
        output_dir="output"
    )

    results, metrics = predictor.run_prediction(model_type=model_type)

    return {
        "model": model_type,
        "accuracy": metrics["Accuracy_Percentage"],
        "rmse": metrics["RMSE"]
    }
