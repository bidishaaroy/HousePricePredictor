import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# 1. Define Features (Matching your previous phases)
FEATURES = [
    "Neighborhood", "Year", "Property Type", "Bedrooms", "Bathrooms",
    "Year Built", "Renovation Year", "Garage Type", "Square Footage (House)",
    "Square Footage (Land)", "Basement", "Legal Units"
]
CATEGORICAL_FEATURES = [
    "Neighborhood", "Year", "Property Type", "Bedrooms", "Bathrooms",
    "Garage Type", "Basement", "Legal Units"
]
NUMERIC_FEATURES = [
    "Year Built", "Renovation Year", "Square Footage (House)", "Square Footage (Land)"
]
TARGET = "Market Price"

def clean_data(df):
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Missing").astype(str)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def build_preprocessor():
    # We use a fresh preprocessor for EACH cluster to localize the scaling
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

def main():
    # (Assuming you saved your clustered data. If not, you will run your K-Means code here first to get the DataFrames with 'Cluster_ID')
    # For this script, let's assume train_df and val_df already have the 'Cluster_ID' column from Phase 3.
     # Load the newly clustered data
    data_dir = Path("./Datasets/bc_last_20_years/split_datasets")
    
    # We don't need to run clean_data() again because we cleaned it in Phase 3 before saving
    train_df = pd.read_csv(data_dir / "train_clustered.csv")
    val_df = pd.read_csv(data_dir / "val_clustered.csv")
    # Placeholder for loading the data (replace with actual K-means output logic if needed)
    # train_df = ... 
    # val_df = ...
    
    k_clusters = 4
    
    # Define the Hypothesis Classes to test
    models_to_test = {
        "Standard OLS": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso (alpha=100)": Lasso(alpha=100.0, max_iter=10000),
        "Poly Degree 2 + Ridge": Pipeline(steps=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0))
        ])
    }

    # Prepare the visualization figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    print(f"{'Cluster':<10} | {'Winning Model':<25} | {'Val MAPE':<10} | {'Val R2':<10}")
    print("-" * 65)

    for cluster_id in range(k_clusters):
        # Isolate the data for this specific cluster
        c_train = train_df[train_df['Cluster_ID'] == cluster_id]
        c_val = val_df[val_df['Cluster_ID'] == cluster_id]
        
        X_train, y_train = c_train[FEATURES], c_train[TARGET]
        X_val, y_val = c_val[FEATURES], c_val[TARGET]
        
        best_model_name = ""
        best_mape = float('inf')
        best_r2 = -float('inf')
        best_predictions = None

        # Test each hypothesis class
        for name, estimator in models_to_test.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", estimator)
            ])
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            
            mape = mean_absolute_percentage_error(y_val, preds) * 100
            r2 = r2_score(y_val, preds)
            
            if mape < best_mape:
                best_mape = mape
                best_r2 = r2
                best_model_name = name
                best_predictions = preds

        print(f"{cluster_id:<10} | {best_model_name:<25} | {best_mape:<9.2f}% | {best_r2:<9.4f}")

        # Visualization: Actual vs Predicted for the Winning Model
        ax = axes[cluster_id]
        ax.scatter(y_val, best_predictions, alpha=0.6, edgecolors='w', s=60)
        
        # Draw the perfect prediction diagonal line
        min_val = min(y_val.min(), best_predictions.min())
        max_val = max(y_val.max(), best_predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_title(f"Cluster {cluster_id} - {best_model_name}\nMAPE: {best_mape:.2f}% | $R^2$: {best_r2:.2f}")
        ax.set_xlabel("Actual Price ($)")
        ax.set_ylabel("Predicted Price ($)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.ticklabel_format(style='plain', axis='both') # Removes scientific notation

    plt.tight_layout()
    plt.savefig("linear_tournament_results.png")

   

if __name__ == "__main__":
    # Note: You will need to import/merge your K-Means clustering function here 
    # so that train_df and val_df are populated with 'Cluster_ID' before the loop runs.
    main()