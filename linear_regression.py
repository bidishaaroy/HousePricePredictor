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
from sklearn.tree import DecisionTreeRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

EXCLUDED_FILES = {"validation.csv"}

K_RANGE = range(2, 8)  # candidate k values to sweep

from sklearn.base import BaseEstimator, RegressorMixin, clone

from sklearn.model_selection import KFold

class BestPerClusterRegressor(BaseEstimator, RegressorMixin):
    """use k-fold CV to find the best model per cluster"""

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

class FourierFeatures(BaseEstimator, TransformerMixin):
    """For each column, adds sin/cos harmonics of a given period letting the
    linear model fit a smooth cyclical component instead of a monotone slope."""
    def __init__(self, periods, n_harmonics=2):
        self.periods = periods          # dict: {column_name: period_in_units}
        self.n_harmonics = n_harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, period in self.periods.items():
            x = pd.to_numeric(X[col], errors="coerce").fillna(0)
            X[col] = x                  # clean_data cast this to str if it was categorical
            for h in range(1, self.n_harmonics + 1):
                angle = 2 * np.pi * h * x / period
                X[f"{col}_sin{h}"] = np.sin(angle)
                X[f"{col}_cos{h}"] = np.cos(angle)
        return X


def make_periodic_pipeline(periods, n_harmonics=2, alpha=1.0):
    fourier = [
        f"{col}_{fn}{h}"
        for col in periods
        for h in range(1, n_harmonics + 1)
        for fn in ("sin", "cos")
    ]
    periodic_cols = [c for c in periods if c not in NUMERIC_FEATURES]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in periods]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES + periodic_cols + fourier),
            ("categorical", make_encoder(), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("fourier", FourierFeatures(periods, n_harmonics)),
        ("preprocess", preprocessor),
        ("regressor", Ridge(alpha=alpha)),
    ])

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


def make_tree_pipeline(max_depth=8, min_samples_leaf=5, random_state=42):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", make_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("regressor", DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )),
    ])




def plot_all_candidates(candidates, kmeans, cluster_preprocessor, X_fit, y_fit, X_eval, y_eval, split_name, output_dir, winners=None):
    """Fit every candidate on every cluster's training data, then plot each one's actual vs predicted on the eval set. Rows = clusters, cols = candidates."""
    fit_ids = kmeans.predict(cluster_preprocessor.transform(X_fit[FEATURES]))
    eval_ids = kmeans.predict(cluster_preprocessor.transform(X_eval[FEATURES]))
    cluster_list = sorted(np.unique(fit_ids))
    names = list(candidates.keys())

    fig, axes = plt.subplots(
        len(cluster_list), len(names),
        figsize=(3.4 * len(names), 3.2 * len(cluster_list)),
        squeeze=False,
    )

    for row, cid in enumerate(cluster_list):
        fit_mask = fit_ids == cid
        eval_mask = eval_ids == cid
        X_c, y_c = X_fit[fit_mask], y_fit[fit_mask]

        for col, name in enumerate(names):
            ax = axes[row][col]

            if len(X_c) < 2 or eval_mask.sum() == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"c{cid} · {name}", fontsize=9)
                continue

            try:
                fitted = clone(candidates[name]).fit(X_c, y_c)
                preds = fitted.predict(X_eval[eval_mask])
            except Exception as exc:
                ax.text(0.5, 0.5, type(exc).__name__, ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="red")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"c{cid} · {name}", fontsize=9)
                continue

            y_e = y_eval[eval_mask]
            ax.scatter(y_e, preds, alpha=0.45, s=14, edgecolors="none")

            lo = min(float(y_e.min()), float(np.min(preds)))
            hi = max(float(y_e.max()), float(np.max(preds)))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
            ax.axhline(float(y_e.mean()), color="gray", ls=":", lw=1.2)

            mape = mean_absolute_percentage_error(y_e, preds) * 100
            won = winners is not None and winners.get(cid, (None,))[0] == name
            ax.set_title(
                f"{'★ ' if won else ''}c{cid} · {name}\nn={eval_mask.sum()}, MAPE={mape:.1f}%",
                fontsize=9,
                fontweight="bold" if won else "normal",
            )
            ax.tick_params(labelsize=7)
            ax.ticklabel_format(style="plain", axis="both")
            ax.grid(True, ls="--", alpha=0.3)

    fig.suptitle(f"All candidates per cluster — {split_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / f"all_candidates_{Path(split_name).stem}.png", dpi=110)
    plt.close(fig)
class PiecewiseHinge(BaseEstimator, TransformerMixin):
    """For each numeric column, adds a hinge feature max(0, x - breakpoint), letting the linear model learn a different slope before/after the breakpoint."""
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

def plot_response_curves(candidates, kmeans, cluster_preprocessor, X_fit, y_fit, feature, split_name, output_dir, n_points=120):
    """Sweep one numeric feature, hold the rest at cluster-typical values, and draw each candidate's predicted price as a curve."""
    fit_ids = kmeans.predict(cluster_preprocessor.transform(X_fit[FEATURES]))
    cluster_list = sorted(np.unique(fit_ids))

    fig, axes = plt.subplots(1, len(cluster_list),
                             figsize=(5 * len(cluster_list), 4.2), squeeze=False)

    for col, cid in enumerate(cluster_list):
        ax = axes[0][col]
        mask = fit_ids == cid
        X_c, y_c = X_fit[mask], y_fit[mask]
        if len(X_c) < 2:
            ax.axis("off")
            continue

        grid = np.linspace(X_c[feature].min(), X_c[feature].max(), n_points)
        template = X_c.iloc[[0]].copy()
        for c in NUMERIC_FEATURES:
            template[c] = X_c[c].median()
        for c in CATEGORICAL_FEATURES:
            template[c] = X_c[c].mode().iloc[0]
        X_grid = pd.concat([template] * n_points, ignore_index=True)
        X_grid[feature] = grid

        ax.scatter(X_c[feature], y_c, alpha=0.12, s=10, color="gray")
        for name, pipeline in candidates.items():
            try:
                fitted = clone(pipeline).fit(X_c, y_c)
                ax.plot(grid, fitted.predict(X_grid), lw=1.8, label=name)
            except Exception:
                continue

        ax.set_title(f"Cluster {cid} (n={len(X_c)})")
        ax.set_xlabel(feature)
        ax.set_ylabel("Predicted price")
        ax.ticklabel_format(style="plain", axis="both")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(f"Response curves vs {feature} — {split_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / f"response_{feature.replace(' ', '_')}_{Path(split_name).stem}.png", dpi=110)
    plt.close(fig)

def plot_year_response(candidates, kmeans, cluster_preprocessor, X_fit, y_fit, split_name, output_dir):
    """Predicted price at each observed year, other features held at cluster-typical values. Year stays categorical."""
    fit_ids = kmeans.predict(cluster_preprocessor.transform(X_fit[FEATURES]))
    cluster_list = sorted(np.unique(fit_ids))

    fig, axes = plt.subplots(1, len(cluster_list),
                             figsize=(5.5 * len(cluster_list), 4.2), squeeze=False)

    for col, cid in enumerate(cluster_list):
        ax = axes[0][col]
        mask = fit_ids == cid
        X_c, y_c = X_fit[mask], y_fit[mask]
        if len(X_c) < 2:
            ax.axis("off")
            continue

        years = sorted(X_c["Year"].unique(), key=lambda v: int(float(v)))
        template = X_c.iloc[[0]].copy()
        for c in NUMERIC_FEATURES:
            template[c] = X_c[c].median()
        for c in CATEGORICAL_FEATURES:
            template[c] = X_c[c].mode().iloc[0]
        X_grid = pd.concat([template] * len(years), ignore_index=True)
        X_grid["Year"] = years

        x_num = [int(float(v)) for v in years]
        actual = y_c.groupby(X_c["Year"]).mean().reindex(years)
        ax.scatter(X_c["Year"].map(lambda v: int(float(v))), y_c,
                   alpha=0.10, s=10, color="gray")
        ax.plot(x_num, actual.values, "k--", lw=2, label="actual mean")

        for name, pipeline in candidates.items():
            try:
                fitted = clone(pipeline).fit(X_c, y_c)
                ax.plot(x_num, fitted.predict(X_grid), lw=1.6, marker="o",
                        ms=3, label=name)
            except Exception:
                continue

        ax.set_title(f"Cluster {cid} (n={len(X_c)})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Predicted price")
        ax.ticklabel_format(style="plain", axis="y")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(f"Response vs Year — {split_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / f"response_Year_{Path(split_name).stem}.png", dpi=110)
    plt.close(fig)

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
        
        # K_Means_Clustering: Sweep to find best k
        print(f"\nSweeping k over {list(K_RANGE)} using silhouette score...")
        best_k, scores = sweep_k(X_train_scaled, K_RANGE)

        # K_Means_Clustering: get kmeans
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
        
        # KMeans Clustered Linear Regression model (Degree 3 polynomial)
        poly_model2 = ClusteredRegressor(make_polynomial_pipeline(degree=3), kmeans, preprocessor)
        poly_model2.fit(X_train, y_train)
        poly3_train = evaluate_model(poly_model2, X_train, y_train)
        poly3_val = evaluate_model(poly_model2, X_val, y_val)

        # KMeans Clustered Linear Regression model (Degree 4 polynomial)
        poly_model3 = ClusteredRegressor(make_polynomial_pipeline(degree=4), kmeans, preprocessor)
        poly_model3.fit(X_train, y_train)
        poly4_train = evaluate_model(poly_model3, X_train, y_train)
        poly4_val = evaluate_model(poly_model3, X_val, y_val)

        # KMeans Clustered Linear Regression model (Piecewise linear)
        piecewise_model = ClusteredRegressor(make_piecewise_pipeline({"Year Built": 2000}), kmeans, preprocessor)
        piecewise_model.fit(X_train, y_train)
        piecewise_train = evaluate_model(piecewise_model, X_train, y_train)
        piecewise_val = evaluate_model(piecewise_model, X_val, y_val)

        # KMeans Clustered Decision Tree model
        tree_model = ClusteredRegressor(make_tree_pipeline(), kmeans, preprocessor)
        tree_model.fit(X_train, y_train)
        tree_train = evaluate_model(tree_model, X_train, y_train)
        tree_val = evaluate_model(tree_model, X_val, y_val)

        # KMeans Clustered Periodic (Fourier) model
        periodic_model = ClusteredRegressor(make_periodic_pipeline({"Year": 18}), kmeans, preprocessor)
        periodic_model.fit(X_train, y_train)
        periodic_train = evaluate_model(periodic_model, X_train, y_train)
        periodic_val = evaluate_model(periodic_model, X_val, y_val)        
        
        candidates = {
            "linear": make_linear_regression_pipeline(),
            "poly_deg2": make_polynomial_pipeline(degree=2),
            "piecewise": make_piecewise_pipeline({"Year Built": 2000}),
            "tree": make_tree_pipeline(),
            "periodic_yr10": make_periodic_pipeline({"Year": 10}, n_harmonics=2),
            "periodic_yr7": make_periodic_pipeline({"Year": 7}),
            "periodic_yr18": make_periodic_pipeline({"Year": 18}),
        }

        model = BestPerClusterRegressor(candidates, kmeans, preprocessor)
        model.fit(X_train, y_train)   # validation never enters here
        plot_dir = data_dir / "cluster_plots"
        plot_dir.mkdir(exist_ok=True)        
        # plot_all_candidates(candidates, kmeans, preprocessor,
        #                             X_train, y_train, X_val, y_val,
        #                             f"{file_name}_val", plot_dir,
        #                             winners=model.cluster_choice_)
        # plot_all_candidates(candidates, kmeans, preprocessor,
        #                     X_train, y_train, X_train, y_train,
        #                     f"{file_name}_train", plot_dir,
                #                     winners=model.cluster_choice_)
        raw = pd.read_csv(data_dir / "BC_2004_to_2023.csv")
        print(raw.corr(numeric_only=True)["Market Price"].sort_values())
        print(raw.groupby("Neighborhood")["Market Price"].agg(["mean", "std", "count"]))
        print(raw.groupby("Year")["Market Price"].mean())                
        plot_response_curves(candidates, kmeans, preprocessor, X_train, y_train, "Square Footage (House)", f"{file_name}_train", plot_dir)  
        plot_year_response(candidates, kmeans, preprocessor, X_train, y_train, f"{file_name}_train", plot_dir)
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)     
        
        results.append(
                    {
                        "Split": file_name,
                        "Baseline Train MAPE": baseline_train["MAPE"],
                        "Baseline Val MAPE": baseline_val["MAPE"],
                        "Linear Train MAPE": linear_train["MAPE"],
                        "Linear Val MAPE": linear_val["MAPE"],
                        "Poly2 Train MAPE": poly_train["MAPE"],
                        "Poly2 Val MAPE": poly_val["MAPE"],
                        "Poly3 Train MAPE": poly3_train["MAPE"],
                        "Poly3 Val MAPE": poly3_val["MAPE"],
                        "Poly4 Train MAPE": poly4_train["MAPE"],
                        "Poly4 Val MAPE": poly4_val["MAPE"],
                        "Piecewise Train MAPE": piecewise_train["MAPE"],
                        "Piecewise Val MAPE": piecewise_val["MAPE"],
                        "Tree Train MAPE": tree_train["MAPE"],
                        "Tree Val MAPE": tree_val["MAPE"],
                        "BestPerCluster Train MAPE": train_metrics["MAPE"],
                        "BestPerCluster Val MAPE": val_metrics["MAPE"],
                        "Periodic Train MAPE": periodic_train["MAPE"],
                        "Periodic Val MAPE": periodic_val["MAPE"],
                        "Periodic Val RMSE": periodic_val["RMSE"],
                        "Periodic Val MAE": periodic_val["MAE"],
                        "Periodic Val R2": periodic_val["R2"],                        
                        "Baseline Val RMSE": baseline_val["RMSE"],
                        "Baseline Val MAE": baseline_val["MAE"],
                        "Baseline Val R2": baseline_val["R2"],
                        "Linear Val RMSE": linear_val["RMSE"],
                        "Linear Val MAE": linear_val["MAE"],
                        "Linear Val R2": linear_val["R2"],
                        "Poly2 Val RMSE": poly_val["RMSE"],
                        "Poly2 Val MAE": poly_val["MAE"],
                        "Poly2 Val R2": poly_val["R2"],
                        "Poly3 Val RMSE": poly3_val["RMSE"],
                        "Poly3 Val MAE": poly3_val["MAE"],
                        "Poly3 Val R2": poly3_val["R2"],
                        "Poly4 Val RMSE": poly4_val["RMSE"],
                        "Poly4 Val MAE": poly4_val["MAE"],
                        "Poly4 Val R2": poly4_val["R2"],
                        "Piecewise Val RMSE": piecewise_val["RMSE"],
                        "Piecewise Val MAE": piecewise_val["MAE"],
                        "Piecewise Val R2": piecewise_val["R2"],
                        "Tree Val RMSE": tree_val["RMSE"],
                        "Tree Val MAE": tree_val["MAE"],
                        "Tree Val R2": tree_val["R2"],
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

    print("\nLinear Regression Results")
    print(results_df.to_string(index=False, formatters={
        "Baseline Train MAPE": "{:.2f}%".format,
        "Baseline Val MAPE": "{:.2f}%".format,
        "Linear Train MAPE": "{:.2f}%".format,
        "Linear Val MAPE": "{:.2f}%".format,
        "Poly2 Train MAPE": "{:.2f}%".format,
        "Poly2 Val MAPE": "{:.2f}%".format,
        "Poly3 Train MAPE": "{:.2f}%".format,
        "Poly3 Val MAPE": "{:.2f}%".format,
        "Poly4 Train MAPE": "{:.2f}%".format,
        "Poly4 Val MAPE": "{:.2f}%".format,
        "Piecewise Train MAPE": "{:.2f}%".format,
        "Piecewise Val MAPE": "{:.2f}%".format,
        "Tree Train MAPE": "{:.2f}%".format,
        "Tree Val MAPE": "{:.2f}%".format,
        "BestPerCluster Train MAPE": "{:.2f}%".format,
        "BestPerCluster Val MAPE": "{:.2f}%".format,
        "Periodic Train MAPE": "{:.2f}%".format,
        "Periodic Val MAPE": "{:.2f}%".format,     
        "Baseline Val RMSE": "${:,.2f}".format,
        "Baseline Val MAE": "${:,.2f}".format,
        "Baseline Val R2": "{:.4f}".format,
        "Linear Val RMSE": "${:,.2f}".format,
        "Linear Val MAE": "${:,.2f}".format,
        "Linear Val R2": "{:.4f}".format,
        "Poly2 Val RMSE": "${:,.2f}".format,
        "Poly2 Val MAE": "${:,.2f}".format,
        "Poly2 Val R2": "{:.4f}".format,
        "Poly3 Val RMSE": "${:,.2f}".format,
        "Poly3 Val MAE": "${:,.2f}".format,
        "Poly3 Val R2": "{:.4f}".format,
        "Poly4 Val RMSE": "${:,.2f}".format,
        "Poly4 Val MAE": "${:,.2f}".format,
        "Poly4 Val R2": "{:.4f}".format,
        "Piecewise Val RMSE": "${:,.2f}".format,
        "Piecewise Val MAE": "${:,.2f}".format,
        "Piecewise Val R2": "{:.4f}".format,
        "Tree Val RMSE": "${:,.2f}".format,
        "Tree Val MAE": "${:,.2f}".format,
        "Tree Val R2": "{:.4f}".format,
        "BestPerCluster Val RMSE": "${:,.2f}".format,
        "BestPerCluster Val MAE": "${:,.2f}".format,
        "BestPerCluster Val R2": "{:.4f}".format,
        "Periodic Val RMSE": "${:,.2f}".format,
        "Periodic Val MAE": "${:,.2f}".format,
        "Periodic Val R2": "{:.4f}".format,           
    }))

    output_path = data_dir / "linear_regression_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved at: {output_path}")


if __name__ == "__main__":
    main()
