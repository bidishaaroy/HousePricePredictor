from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from housing_api import HousingPriceAPI

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("housing_api.joblib")
housing_api = HousingPriceAPI.load(MODEL_PATH)


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/options.json")
def feature_options():
    return send_from_directory(".", "options.json")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not payload:
        return jsonify({
            "success": False,
            "error": "No prediction data was provided."
        }), 400

    model_name = payload.get("model")
    features = payload.get("features")

    if model_name not in housing_api.available_models():
        return jsonify({
            "success": False,
            "error": f"Unknown model: {model_name}"
        }), 400

    if not features:
        return jsonify({
            "success": False,
            "error": "No property features were provided."
        }), 400

    required_features = [
        "Neighborhood",
        "Year",
        "Property Type",
        "Bedrooms",
        "Bathrooms",
        "Year Built",
        "Renovation Year",
        "Garage Type",
        "Square Footage (House)",
        "Square Footage (Land)",
        "Basement",
        "Legal Units"
    ]

    missing_features = [
        feature
        for feature in required_features
        if feature not in features
    ]

    if missing_features:
        return jsonify({
            "success": False,
            "error": (
                "Missing required features: "
                + ", ".join(missing_features)
            )
        }), 400

    try:
        price = housing_api.predict(model_name, features)
    except (ValueError, RuntimeError) as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    return jsonify({
        "success": True,
        "model": model_name,
        "price": price,
        "result": f"Estimated market price: ${price:,.2f}"
    })


if __name__ == "__main__":
    app.run(debug=True)
