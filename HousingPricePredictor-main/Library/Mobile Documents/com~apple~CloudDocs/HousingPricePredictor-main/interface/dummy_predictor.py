import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# 1. Bulletproof Path Finding for Everyone in Your Group
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check both the current folder and the parent folder just in case
path_option_1 = os.path.join(BASE_DIR, "Datasets", "bc_last_20_years", "split_datasets", "BC_2004_to_2023.csv")
path_option_2 = os.path.join(BASE_DIR, "..", "Datasets", "bc_last_20_years", "split_datasets", "BC_2004_to_2023.csv")

if os.path.exists(path_option_1):
    CSV_PATH = path_option_1
elif os.path.exists(path_option_2):
    CSV_PATH = path_option_2
else:
    CSV_PATH = "MISSING"

CATEGORICAL = ["Neighborhood", "Property Type", "Garage Type", "Basement"]
NUMERIC = ["Bedrooms", "Bathrooms", "Year Built", "Renovation Year", "Square Footage (House)", "Square Footage (Land)", "Legal Units"]

# 2. Train the model on the fly
print("Loading data and training live model...")
try:
    if CSV_PATH == "MISSING":
        raise FileNotFoundError("Could not locate Datasets folder.")
        
    df = pd.read_csv(CSV_PATH)
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Missing").astype(str)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)
        ],
        remainder="drop"
    )

    live_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ])

    live_model.fit(df[CATEGORICAL + NUMERIC], df["Market Price"])
    print("Live model successfully trained!")
except Exception as e:
    print(f"CRITICAL ERROR loading data: {e}")
    df = pd.DataFrame()

# 3. The required function that Flask imports
def process_features(features: dict) -> str:
    if df.empty:
        return "Backend Error: The CSV dataset is missing. Cannot make predictions."

    input_df = pd.DataFrame([features])
    
    for col in CATEGORICAL:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
    for col in NUMERIC:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    try:
        estimated_price = live_model.predict(input_df)[0]
    except Exception as e:
        return f"Model Prediction Error: {str(e)}"

    feature_summary = "\n".join(f"{k}: {v}" for k, v in features.items())
    return (
        "Live prediction completed successfully.\n\n"
        f"Received features:\n{feature_summary}\n\n"
        f"Estimated market price: ${estimated_price:,.2f}"
    )