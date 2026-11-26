import os
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# ==========================================
# 1. LIVE CLUSTER PIPELINE (Executes on Startup)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "Datasets", "bc_last_20_years", "split_datasets", "BC_2004_to_2023.csv")

CATEGORICAL = ["Neighborhood", "Property Type", "Garage Type", "Basement"]
NUMERIC = ["Bedrooms", "Bathrooms", "Year Built", "Renovation Year", "Square Footage (House)", "Square Footage (Land)", "Legal Units"]

# Global variables to hold our trained pipeline objects
preprocessor = None
kmeans_clusterer = None
model_cluster_0 = None
model_cluster_1 = None

print("Booting up server and training Best-In-Cluster Prediction Pipeline...")

if os.path.exists(CSV_PATH):
    try:
        # Load and clean data
        df = pd.read_csv(CSV_PATH)
        for col in CATEGORICAL:
            df[col] = df[col].fillna("Missing").astype(str)
        for col in NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        X = df[CATEGORICAL + NUMERIC]
        y = df["Market Price"]

        # 1. Fit the Preprocessor (Standardization & One-Hot Encoding)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)
            ]
        )
        X_processed = preprocessor.fit_transform(X)

        # 2. Fit K-Means (k=2) to map the cluster boundaries
        kmeans_clusterer = KMeans(n_clusters=2, random_state=42, n_init="auto")
        cluster_labels = kmeans_clusterer.fit_predict(X_processed)

        # 3. Split the training data by cluster
        X_c0, y_c0 = X_processed[cluster_labels == 0], y[cluster_labels == 0]
        X_c1, y_c1 = X_processed[cluster_labels == 1], y[cluster_labels == 1]

        # 4. Train Cluster 0 Model (Polynomial Degree 2, as per your results)
        model_cluster_0 = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('regressor', LinearRegression())
        ])
        model_cluster_0.fit(X_c0, y_c0)

        # 5. Train Cluster 1 Model (Using the optimal Tree logic you just found)
        model_cluster_1 = DecisionTreeRegressor(
            max_depth=12, 
            min_samples_leaf=10, 
            min_samples_split=2, 
            random_state=42
        )
        model_cluster_1.fit(X_c1, y_c1)

        print("Cluster routing and models successfully trained! Ready for API requests.")
    except Exception as e:
        print(f"CRITICAL ERROR loading data: {e}")
else:
    print(f"CRITICAL ERROR: Could not locate the dataset at {CSV_PATH}")


# ==========================================
# 2. FLASK WEB ROUTES
# ==========================================
@app.route("/")
def home():
    return send_file("index.html")

@app.route("/options.json")
def feature_options():
    return send_from_directory(".", "options.json")

@app.route("/predict", methods=["POST"])
def predict():
    features = request.get_json(silent=True)

    if not features:
        return jsonify({"success": False, "error": "No property features were provided."}), 400

    required_features = [
        "Neighborhood", "Year", "Property Type", "Bedrooms", "Bathrooms", 
        "Year Built", "Renovation Year", "Garage Type", "Square Footage (House)", 
        "Square Footage (Land)", "Basement", "Legal Units"
    ]

    missing_features = [f for f in required_features if f not in features]
    if missing_features:
        return jsonify({"success": False, "error": "Missing required features: " + ", ".join(missing_features)}), 400

    if preprocessor is None or kmeans_clusterer is None:
        return jsonify({"success": False, "error": "Backend Error: The pipeline failed to train on startup."})

    # Convert the user's web inputs into a DataFrame
    input_df = pd.DataFrame([features])
    for col in CATEGORICAL:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
    for col in NUMERIC:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    try:
        # 1. Preprocess the single new property
        X_new_processed = preprocessor.transform(input_df)
        
        # 2. Predict which cluster this house belongs to
        assigned_cluster = kmeans_clusterer.predict(X_new_processed)[0]
        
        # 3. Route to the correct model
        if assigned_cluster == 0:
            estimated_price = model_cluster_0.predict(X_new_processed)[0]
            model_name = "Cluster 0 (Polynomial Model)"
        else:
            estimated_price = model_cluster_1.predict(X_new_processed)[0]
            model_name = "Cluster 1 (Piecewise/Tree Model)"

        # 4. Return the formatted result to the website
        result_text = f"Routed to {model_name}. Estimated Market Price: ${estimated_price:,.2f}"

    except Exception as e:
        return jsonify({"success": False, "error": f"Model Prediction Error: {str(e)}"})

    return jsonify({"success": True, "result": result_text})


if __name__ == "__main__":
    app.run(debug=True)