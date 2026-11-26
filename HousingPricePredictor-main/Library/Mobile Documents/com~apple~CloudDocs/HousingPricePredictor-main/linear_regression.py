import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from K_Means_Clustering import get_kmeans, sweep_k
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge  # regularized, given small per-cluster n
from sklearn.base import BaseEstimator, TransformerMixin

# Basic Linear Regression model

FEATURES = [
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
    "Legal Units",
]

TARGET = "Market Price"

CATEGORICAL_FEATURES = [
    "Neighborhood",
    "Year",
    "Property Type",
    "Garage Type",
    "Basement",
]

NUMERIC_FEATURES = [
    "Bedrooms",
    "Bathrooms",
    "Legal Units",
    "Year Built",
    "Renovation Year",
    "Square Footage (House)",
    "Square Footage (Land)",
]

EXCLUDED_FILES = {"validation.csv"}

K_RANGE = range(2, 8)  # candidate k values to sweep

from sklearn.base import BaseEstimator, RegressorMixin, clone

from sklearn.model_selection import KFold

class BestPerClusterRegressor(BaseEstimator, RegressorMixin):
    """For each cluster, uses k-fold CV on the training data alone to pick
    the best-performing candidate hypothesis, then refits that winner on
    the full training data for that cluster. Validation data is never
    touched during fit() -- only used later for the final reported metric."""

    def __init__(self, candidate_pipelines, kmeans, cluster_preprocessor, n_folds=5, random_state=42):
        self.candidate_pipelines = candidate_pipelines
        self.kmeans = kmeans
        self.cluster_preprocessor = cluster_preprocessor
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, y):
        X_scaled = self.cluster_preprocessor.transform(X[FEATURES])
        cluster_ids = self.kmeans.predict(X_scaled)

        self.cluster_models_ = {}
        self.cluster_choice_ = {}

        for cid in np.unique(cluster_ids):
            mask = cluster_ids == cid
            X_c, y_c = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

            n_folds_c = min(self.n_folds, len(X_c))  # can't have more folds than rows
            if n_folds_c < 2:
                # Too few points in this cluster to cross-validate -- just fit the
                # simplest candidate directly and flag it
                name = list(self.candidate_pipelines.keys())[0]
                model = clone(self.candidate_pipelines[name]).fit(X_c, y_c)
                self.cluster_models_[cid] = model
                self.cluster_choice_[cid] = (name, None)
                print(f"  Cluster {cid}: only {len(X_c)} points, skipped CV, used '{name}'")
                continue

            kf = KFold(n_splits=n_folds_c, shuffle=True, random_state=self.random_state)

            best_name, best_score = None, float("inf")
            for name, pipeline in self.candidate_pipelines.items():
                fold_scores = []
                for train_idx, test_idx in kf.split(X_c):
                    model = clone(pipeline)
                    model.fit(X_c.iloc[train_idx], y_c.iloc[train_idx])
                    preds = model.predict(X_c.iloc[test_idx])
                    fold_scores.append(
                        mean_absolute_percentage_error(y_c.iloc[test_idx], preds)
                    )
                mean_score = np.mean(fold_scores)
                if mean_score < best_score:
                    best_name, best_score = name, mean_score

            # Refit the winning hypothesis on ALL of this cluster's training data
            winning_model = clone(self.candidate_pipelines[best_name]).fit(X_c, y_c)
            self.cluster_models_[cid] = winning_model
            self.cluster_choice_[cid] = (best_name, best_score)
            print(f"  Cluster {cid}: chose '{best_name}' (CV MAPE={best_score:.4f}, n={len(X_c)})")

        return self

    def predict(self, X):
        X_scaled = self.cluster_preprocessor.transform(X[FEATURES])
        cluster_ids = self.kmeans.predict(X_scaled)
        preds = np.empty(len(X))
        for cid in np.unique(cluster_ids):
            mask = cluster_ids == cid
            preds[mask] = self.cluster_models_[cid].predict(X[mask])
        return preds
class ClusteredRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_pipeline, kmeans, cluster_preprocessor):
        self.base_pipeline = base_pipeline
        self.kmeans = kmeans
        # this is how the hypothesis class can be varied
        self.cluster_preprocessor = cluster_preprocessor

    def fit(self, X, y):
        X_scaled = self.cluster_preprocessor.transform(X[FEATURES])
        cluster_ids = self.kmeans.predict(X_scaled)
        self.cluster_models_ = {}
        # for each cluster, fit each independently
        for cid in np.unique(cluster_ids):
            mask = cluster_ids == cid
            model = clone(self.base_pipeline)
            model.fit(X[mask], y[mask])
            self.cluster_models_[cid] = model
        return self

    def predict(self, X):
        X_scaled = self.cluster_preprocessor.transform(X[FEATURES])
        cluster_ids = self.kmeans.predict(X_scaled)
        preds = np.empty(len(X))
        for cid in np.unique(cluster_ids):
            mask = cluster_ids == cid
            preds[mask] = self.cluster_models_[cid].predict(X[mask])
        return preds

def find_data_dir():
    """Find the folder with validation.csv and the BC split CSV files."""
    script_dir = Path(__file__).resolve().parent

    possible_dirs = [
        script_dir / "Datasets" / "bc_last_20_years" / "split_datasets",
        script_dir,
        Path.cwd() / "Datasets" / "bc_last_20_years" / "split_datasets",
        Path.cwd(),
    ]

    for folder in possible_dirs:
        if (folder / "validation.csv").exists():
            return folder

    raise FileNotFoundError(
        "Error: validation.csv was not found. Put this file in the split_datasets folder "
        "or in the same folder as this script."
    )


def make_encoder():
    """Create a one-hot encoder functional for different sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_data(df):
    """Preprocess data for training/validation."""
    df = df.copy()

    # Missing categorical values treated as a category
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Missing").astype(str)

    # Missing numeric values replaced with the training median
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # K_Means_Clustering: Impute missing numeric values with 0 (or median) to prevent KMeans from crashing
        df[col] = df[col].fillna(0)
    return df

def make_linear_regression_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", make_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    return model

def make_polynomial_pipeline(degree=2, alpha=1.0):
    numeric_transform = Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transform, NUMERIC_FEATURES),
            ("categorical", make_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("regressor", Ridge(alpha=alpha)),
    ])

class PiecewiseHinge(BaseEstimator, TransformerMixin):
    """For each numeric column, adds a hinge feature max(0, x - breakpoint),
    letting the linear model learn a different slope before/after the breakpoint."""
    def __init__(self, breakpoints):
        self.breakpoints = breakpoints  # dict: {column_name: breakpoint_value}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, bp in self.breakpoints.items():
            X[f"{col}_hinge"] = np.maximum(0, X[col] - bp)
        return X


def make_piecewise_pipeline(breakpoints, alpha=1.0):
    hinge_features = [f"{col}_hinge" for col in breakpoints]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES + hinge_features),
            ("categorical", make_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("hinge", PiecewiseHinge(breakpoints)),
        ("preprocess", preprocessor),
        ("regressor", Ridge(alpha=alpha)),
    ])

    # eg: 3 cluster Year Built finding (~peak around 2000):
    # piecewise_pipeline = make_piecewise_pipeline(breakpoints={"Year Built": 2000})

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    return {
        "RMSE": math.sqrt(mse),
        "MAE": mean_absolute_error(y, predictions),
        "MAPE": mean_absolute_percentage_error(y, predictions) * 100,
        "R2": r2_score(y, predictions),
    }


def main():
    data_dir = find_data_dir()
    validation_df = clean_data(pd.read_csv(data_dir / "validation.csv"))
    X_val = validation_df[FEATURES]
    y_val = validation_df[TARGET]
    results = []
    
    # K_Means_Clustering: Add preprocessing 
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    for file_name in sorted(os.listdir(data_dir)):
        if file_name in EXCLUDED_FILES:
            continue
        if not file_name.endswith(".csv"):
            continue
        if "standardized" in file_name:
            continue
        if not file_name.startswith("BC_"):
            continue

        train_df = clean_data(pd.read_csv(data_dir / file_name))
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        # K_Means_Clustering: Scale X_train
        X_train_scaled = preprocessor.fit_transform(X_train)
        
        # We already proved k=2 is optimal, so we skip the sweep to save time
        best_k = 2
        kmeans = get_kmeans(best_k, X_train_scaled)
        # Model applied within each cluster...
        
        # Baseline model
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        baseline_train = evaluate_model(baseline, X_train, y_train)
        baseline_val = evaluate_model(baseline, X_val, y_val)

        # Linear Regression model
        linear_model = make_linear_regression_pipeline()
        linear_model.fit(X_train, y_train)
        linear_train = evaluate_model(linear_model, X_train, y_train)
        linear_val = evaluate_model(linear_model, X_val, y_val)
        
        # KMeans Clustered Linear Regression model (Degree 2 polynomial)
        poly_model = ClusteredRegressor(make_polynomial_pipeline(degree=2), kmeans, preprocessor)
        poly_model.fit(X_train, y_train)
        poly_train = evaluate_model(poly_model, X_train, y_train)
        poly_val = evaluate_model(poly_model, X_val, y_val)
        
        # KMeans Clustered Linear Regression model (Piecewise linear)
        piecewise_model = ClusteredRegressor(make_piecewise_pipeline({"Year Built": 2000}), kmeans, preprocessor)
        piecewise_model.fit(X_train, y_train)
        piecewise_train = evaluate_model(piecewise_model, X_train, y_train)
        piecewise_val = evaluate_model(piecewise_model, X_val, y_val)
        
        candidates = {
            "linear": make_linear_regression_pipeline(),
            "poly_deg2": make_polynomial_pipeline(degree=2),
            "piecewise": make_piecewise_pipeline({"Year Built": 2000}),
        }

        model = BestPerClusterRegressor(candidates, kmeans, preprocessor)
        model.fit(X_train, y_train)   # validation never enters here

        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)     
        
        results.append(
            {
                "Split": file_name,
                "Baseline Train MAPE": baseline_train["MAPE"],
                "Baseline Val MAPE": baseline_val["MAPE"],
                "Linear Train MAPE": linear_train["MAPE"],
                "Linear Val MAPE": linear_val["MAPE"],   
                "Polynomial Train MAPE": poly_train["MAPE"],
                "Polynomial Val MAPE": poly_val["MAPE"], 
                "Piecewise Train MAPE": piecewise_train["MAPE"],
                "Piecewise Val MAPE": piecewise_val["MAPE"],                                            
                "Basline Val RMSE": baseline_val["RMSE"],
                "Basline Val MAE": baseline_val["MAE"],
                "Baseline Val R2": baseline_val["R2"],        
                "Linear Val RMSE": linear_val["RMSE"],
                "Linear Val MAE": linear_val["MAE"],
                "Linear Val R2": linear_val["R2"],
                "Polynomial Val RMSE": poly_val["RMSE"],
                "Polynomial Val MAE": poly_val["MAE"],
                "Polynomial Val R2": poly_val["R2"],      
                "Piecewise Val RMSE": piecewise_val["RMSE"],
                "Piecewise Val MAE": piecewise_val["MAE"],
                "Piecewise Val R2": piecewise_val["R2"],                              
            }
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No training split files found.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("\nLinear Regression Results")
    print(results_df.to_string(index=False, formatters={
        "Baseline Train MAPE": "{:.2f}%".format,
        "Baseline Val MAPE": "{:.2f}%".format,
        "Linear Train MAPE": "{:.2f}%".format,
        "Linear Val MAPE": "{:.2f}%".format,    
        "Polynomial Train MAPE": "{:.2f}%".format,
        "Polynomial Val MAPE": "{:.2f}%".format, 
        "Piecewise Train MAPE": "{:.2f}%".format,
        "Piecewise Val MAPE": "{:.2f}%".format,                          
        "Baseline Val RMSE": "{:.2f}%".format,
        "Baseline Val MAE": "{:.2f}%".format,
        "Baseline Val R2": "{:.2f}%".format,
        "Linear Val RMSE": "${:,.2f}".format,
        "Linear Val MAE": "${:,.2f}".format,
        "Linear Val R2": "{:.4f}".format,
        "Polynomial Val RMSE": "{:.2f}%".format,
        "Polynomial Val MAE": "{:.2f}%".format,
        "Polynomial Val R2": "{:.2f}%".format,
        "Piecewise Val RMSE": "${:,.2f}".format,
        "Piecewise Val MAE": "${:,.2f}".format,
        "Piecewise Val R2": "{:.4f}".format,        
    }))

    output_path = data_dir / "linear_regression_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved at: {output_path}")


if __name__ == "__main__":
    main()
