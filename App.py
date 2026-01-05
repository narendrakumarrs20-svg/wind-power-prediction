from flask import Flask, jsonify, render_template
from wind_predictor import run_cloud_prediction

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/predict")
def predict():
    result = run_cloud_prediction(
        model_type="random_forest",
        date_range=None
    )
    return jsonify(result)

if __name__ == "__main__":
    app.run()
