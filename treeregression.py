import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree


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
    "Bedrooms",
    "Bathrooms",
    "Garage Type",
    "Basement",
    "Legal Units",
]

NUMERIC_FEATURES = [
    "Year Built",
    "Renovation Year",
    "Square Footage (House)",
    "Square Footage (Land)",
]

EXCLUDED_FILES = {
    "validation.csv",
    "training.csv",
    "linear_regression_results.csv",
    "tree_regression_results.csv",
    "train_clustered.csv",
    "val_clustered.csv",
}


def find_data_dir() -> Path:
    """Find folder that has validation.csv and split files."""
    script_dir = Path(__file__).resolve().parent

    dirs_searched = [
        script_dir / "Datasets" / "bc_last_20_years" / "split_datasets",
        script_dir,
        Path.cwd() / "Datasets" / "bc_last_20_years" / "split_datasets",
        Path.cwd(),
    ]

    for folder in dirs_searched:
        if (folder / "validation.csv").exists():
            return folder

    raise FileNotFoundError(
        "validation.csv was not found. Put this script in the repository root, "
        "the split_datasets folder, or update find_data_dir()."
    )


def make_encoder():
    """Support new and old scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data columns used by the tree model."""
    df = df.copy()

    missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Missing").astype(str)

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        median = df[column].median()
        if pd.isna(median):
            median = 0
        df[column] = df[column].fillna(median)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    return df


def make_tree_pipeline(random_state: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", make_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                DecisionTreeRegressor(
                    random_state=random_state,
                    criterion="squared_error",
                ),
            ),
        ]
    )


def tune_tree(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """
    Select tree complexity by using cross-validation on training data.
    """
    pipeline = make_tree_pipeline()

    parameter_grid = {
        "regressor__max_depth": [3, 5, 6, 8, 12, 20, None],
        "regressor__min_samples_split": [2, 5, 10, 20, 50, 100, 200, 400],
        "regressor__min_samples_leaf": [1, 2, 5, 10, 20, 50],
        "regressor__max_features": [None, "sqrt", "log2"],
    }

    n_splits = min(5, len(X_train))
    if n_splits < 2:
        raise ValueError("At least 2 training rows are required.")

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_mean_absolute_percentage_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    search.fit(X_train, y_train)
    tree = search.best_estimator_.named_steps["regressor"]
    print("actual depth:", tree.get_depth())
    print("leaves:", tree.get_n_leaves())
    print("nodes:", tree.tree_.node_count)    
    return search


def model_evaluation(model, X: pd.DataFrame, y: pd.Series) -> dict:
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    return {
        "RMSE": math.sqrt(mse),
        "MAE": mean_absolute_error(y, predictions),
        "MAPE": mean_absolute_percentage_error(y, predictions) * 100,
        "R2": r2_score(y, predictions),
        "Predictions": predictions,
    }


def save_actual_vs_predicted(
    y_true: pd.Series,
    predictions: np.ndarray,
    split_name: str,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, predictions, alpha=0.6)

    lower = min(float(y_true.min()), float(np.min(predictions)))
    upper = max(float(y_true.max()), float(np.max(predictions)))
    plt.plot([lower, upper], [lower, upper], linestyle="--")

    plt.xlabel("Actual Market Price")
    plt.ylabel("Predicted Market Price")
    plt.title(f"Decision Tree: Actual vs Predicted\n{split_name}")
    plt.ticklabel_format(style="plain", axis="both")
    plt.tight_layout()
    plt.savefig(output_dir / "tree_actual_vs_predicted.png", dpi=300)
    plt.close()


def save_mape_plot(results_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(11, 6))
    plt.plot(
        results_df["Split"],
        results_df["Tree Train MAPE"],
        marker="o",
        label="Training MAPE",
    )
    plt.plot(
        results_df["Split"],
        results_df["Tree Val MAPE"],
        marker="o",
        label="Validation MAPE",
    )

    plt.xlabel("Training Split")
    plt.ylabel("MAPE (%)")
    plt.title("Decision Tree Training vs Validation MAPE")
    plt.xticks(rotation=75, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tree_mape_by_split.png", dpi=300)
    plt.close()


def save_tree_preview(best_model: Pipeline, output_dir: Path) -> None:
    """
    Save top three levels for readable visualization.
    """
    tree_model = best_model.named_steps["regressor"]

    try:
        feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
    except AttributeError:
        feature_names = None

    plt.figure(figsize=(24, 12))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
    )
    plt.title("Decision Tree Preview (First 3 Levels)")
    plt.tight_layout()
    plt.savefig(output_dir / "tree_structure_preview.png", dpi=300)
    plt.close()

def save_tree_poster(best_model: Pipeline, output_dir: Path,
                     max_depth: int = 2, fontsize: int = 18) -> None:
    """Poster-quality render of the top levels of the fitted tree."""
    tree_model = best_model.named_steps["regressor"]

    raw_names = best_model.named_steps["preprocess"].get_feature_names_out()
    feature_names = [n.split("__", 1)[-1] for n in raw_names]

    n_leaves = 2 ** max_depth
    fig, ax = plt.subplots(figsize=(6 * n_leaves, 4.5 * (max_depth + 1)))

    plot_tree(
        tree_model,
        feature_names=feature_names,
        max_depth=max_depth,
        filled=True,
        rounded=True,
        impurity=False,        # drops the squared_error line — halves the text per box
        proportion=True,       # fractions instead of huge raw sample counts
        precision=0,
        fontsize=fontsize,
        ax=ax,
    )
    ax.set_title(f"Decision tree: top {max_depth} levels", fontsize=fontsize + 8, pad=20)

    fig.savefig(output_dir / "tree_poster.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "tree_poster.pdf", bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    data_dir = find_data_dir()
    output_dir = data_dir / "tree_results"
    output_dir.mkdir(exist_ok=True)

    validation_df = clean_data(pd.read_csv(data_dir / "validation.csv"))
    valX = validation_df[FEATURES]
    val_y = validation_df[TARGET]

    results = []
    best_overall_model = None
    best_overall_split = None
    best_overall_val_mape = float("inf")
    best_overall_predictions = None

    candidate_files = sorted(
        file_name
        for file_name in os.listdir(data_dir)
        if file_name.startswith("BC_")
        and file_name.endswith(".csv")
        and "standardized" not in file_name
        and "clustered" not in file_name
        and file_name not in EXCLUDED_FILES
    )

    for file_name in candidate_files:
        print(f"\nTraining decision tree for {file_name}")

        train_df = clean_data(pd.read_csv(data_dir / file_name))
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]

        search = tune_tree(X_train, y_train)
        model = search.best_estimator_

        trainmetrics = model_evaluation(model, X_train, y_train)
        valmetrics = model_evaluation(model, valX, val_y)

        print(f"Best parameters: {search.best_params_}")
        print(f"Train MAPE: {trainmetrics['MAPE']:.2f}%")
        print(f"Validation MAPE: {valmetrics['MAPE']:.2f}%")
        print(f"Validation RMSE: ${valmetrics['RMSE']:,.2f}")
        print(f"Validation MAE: ${valmetrics['MAE']:,.2f}")
        print(f"Validation R2: {valmetrics['R2']:.4f}")

        results.append(
            {
                "Split": file_name,
                "Training Rows": len(train_df),
                "Best Max Depth": search.best_params_["regressor__max_depth"],
                "Best Min Samples Split": search.best_params_[
                    "regressor__min_samples_split"
                ],
                "Best Min Samples Leaf": search.best_params_[
                    "regressor__min_samples_leaf"
                ],
                "Best Max Features": search.best_params_[
                    "regressor__max_features"
                ],
                "CV MAPE": -search.best_score_ * 100,
                "Tree Train MAPE": trainmetrics["MAPE"],
                "Tree Val MAPE": valmetrics["MAPE"],
                "Tree Val RMSE": valmetrics["RMSE"],
                "Tree Val MAE": valmetrics["MAE"],
                "Tree Val R2": valmetrics["R2"],
            }
        )

        if valmetrics["MAPE"] < best_overall_val_mape:
            best_overall_val_mape = valmetrics["MAPE"]
            best_overall_model = model
            best_overall_split = file_name
            best_overall_predictions = valmetrics["Predictions"]

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No BC_*.csv training files were found.")
        return

    results_path = output_dir / "tree_regression_results.csv"
    results_df.to_csv(results_path, index=False)

    save_mape_plot(results_df, output_dir)

    if (
        best_overall_model is not None
        and best_overall_predictions is not None
        and best_overall_split is not None
    ):
        save_actual_vs_predicted(
            val_y,
            best_overall_predictions,
            best_overall_split,
            output_dir,
        )
        save_tree_preview(best_overall_model, output_dir)
        save_tree_poster(best_overall_model, output_dir, max_depth=3, fontsize=18)
    print("\nDecision Tree Results")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "CV MAPE": "{:.2f}%".format,
                "Tree Train MAPE": "{:.2f}%".format,
                "Tree Val MAPE": "{:.2f}%".format,
                "Tree Val RMSE": "${:,.2f}".format,
                "Tree Val MAE": "${:,.2f}".format,
                "Tree Val R2": "{:.4f}".format,
            },
        )
    )

    print(f"\nBest validation split: {best_overall_split}")
    print(f"Best validation MAPE: {best_overall_val_mape:.2f}%")
    print(f"Save results at: {results_path}")
    print(f"Save plots at: {output_dir}")


if __name__ == "__main__":
    main()