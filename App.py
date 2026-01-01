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
