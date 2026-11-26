"""Train per-split models and save the winning BestPerClusterRegressor
as a joblib bundle that predict.py can load."""

import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from K_Means_Clustering import get_kmeans, sweep_k
from models import (
    FEATURES,
    TARGET,
    BestPerClusterRegressor,
    ClusteredRegressor,
    clean_data,
    make_cluster_preprocessor,
    make_linear_regression_pipeline,
    make_piecewise_pipeline,
    make_polynomial_pipeline,
    tune_tree,
)

EXCLUDED_FILES = {"validation.csv"}
K_RANGE = range(2, 8)  # candidate k values to sweep


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
        "Error: validation.csv was not found. Put this file in the split_datasets "
        "folder or in the same folder as this script."
    )


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
    model_dir = data_dir / "saved_models"
    model_dir.mkdir(exist_ok=True)

    validation_df = clean_data(pd.read_csv(data_dir / "validation.csv"), drop_missing_target=True)
    X_val = validation_df[FEATURES]
    y_val = validation_df[TARGET]
    results = []

    for file_name in sorted(os.listdir(data_dir)):
        if file_name in EXCLUDED_FILES:
            continue
        if not file_name.endswith(".csv"):
            continue
        if "standardized" in file_name:
            continue
        if not file_name.startswith("BC_"):
            continue

        train_df = clean_data(pd.read_csv(data_dir / file_name), drop_missing_target=True)
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]

        preprocessor = make_cluster_preprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)

        print(f"\n=== {file_name} ===")
        print(f"Sweeping k over {list(K_RANGE)} using silhouette score...")
        best_k, scores = sweep_k(X_train_scaled, K_RANGE)
        kmeans = get_kmeans(best_k, X_train_scaled)

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        baseline_train = evaluate_model(baseline, X_train, y_train)
        baseline_val = evaluate_model(baseline, X_val, y_val)

        linear_model = make_linear_regression_pipeline()
        linear_model.fit(X_train, y_train)
        linear_train = evaluate_model(linear_model, X_train, y_train)
        linear_val = evaluate_model(linear_model, X_val, y_val)

        poly_model = ClusteredRegressor(make_polynomial_pipeline(degree=2), kmeans, preprocessor)
        poly_model.fit(X_train, y_train)
        poly_train = evaluate_model(poly_model, X_train, y_train)
        poly_val = evaluate_model(poly_model, X_val, y_val)

        piecewise_model = ClusteredRegressor(
            make_piecewise_pipeline({"Year Built": 2000}), kmeans, preprocessor
        )
        piecewise_model.fit(X_train, y_train)
        piecewise_train = evaluate_model(piecewise_model, X_train, y_train)
        piecewise_val = evaluate_model(piecewise_model, X_val, y_val)

        tree_search = tune_tree(X_train, y_train, fast=True)
        tree_model = tree_search.best_estimator_
        tree_train = evaluate_model(tree_model, X_train, y_train)
        tree_val = evaluate_model(tree_model, X_val, y_val)
        print(f"Tree params: {tree_search.best_params_} "
              f"(CV MAPE={-tree_search.best_score_ * 100:.2f}%)")

        candidates = {
            "linear": make_linear_regression_pipeline(),
            "poly_deg2": make_polynomial_pipeline(degree=2),
            "piecewise": make_piecewise_pipeline({"Year Built": 2000}),
        }
        model = BestPerClusterRegressor(candidates, kmeans, preprocessor)
        model.fit(X_train, y_train)

        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)

        bundle = {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "trained_on": file_name,
            "best_k": best_k,
            "cluster_choices": model.cluster_choice_,
            "tree_model": tree_model,
            "tree_params": tree_search.best_params_,
            "sklearn_version": sklearn.__version__,
            "val_metrics": val_metrics,
        }
        model_path = model_dir / f"best_per_cluster_{Path(file_name).stem}.joblib"
        joblib.dump(bundle, model_path)
        print(f"Saved model -> {model_path}")

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
                "Tree Train MAPE": tree_train["MAPE"],
                "Tree Val MAPE": tree_val["MAPE"],
                "BestPerCluster Train MAPE": train_metrics["MAPE"],
                "BestPerCluster Val MAPE": val_metrics["MAPE"],
                "Baseline Val RMSE": baseline_val["RMSE"],
                "Baseline Val MAE": baseline_val["MAE"],
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
                "Tree Val RMSE": tree_val["RMSE"],
                "Tree Val MAE": tree_val["MAE"],
                "Tree Val R2": tree_val["R2"],
                "Tree Best Params": str(tree_search.best_params_),
                "BestPerCluster Val RMSE": val_metrics["RMSE"],
                "BestPerCluster Val MAE": val_metrics["MAE"],
                "BestPerCluster Val R2": val_metrics["R2"],
            }
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No training split files found.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("\nResults")
    print(results_df.to_string(index=False))

    output_path = data_dir / "linear_regression_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved at: {output_path}")


if __name__ == "__main__":
    main()