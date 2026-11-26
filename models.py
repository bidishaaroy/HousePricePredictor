import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

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

# ---------------------------------------------------------------------------
# Data cleaning (must be applied identically at train AND predict time)
# ---------------------------------------------------------------------------


def clean_data(df, drop_missing_target=False):
    """Preprocess data for training/validation/inference.

    Set drop_missing_target=True on training data to discard rows whose
    Market Price is missing or non-numeric. It stays False at inference
    time, where the target column isn't present at all.
    """
    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Missing").astype(str)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)

    if drop_missing_target and TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def make_encoder():
    """Create a one-hot encoder functional for different sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_cluster_preprocessor():
    """The scaler+encoder used to embed rows for KMeans clustering."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", make_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Candidate hypothesis pipelines
# ---------------------------------------------------------------------------


def make_linear_regression_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", make_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def make_polynomial_pipeline(degree=2, alpha=1.0):
    numeric_transform = Pipeline(
        [
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transform, NUMERIC_FEATURES),
            ("categorical", make_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


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
    return Pipeline(
        [
            ("hinge", PiecewiseHinge(breakpoints)),
            ("preprocess", preprocessor),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

# Trees split on raw thresholds, so scaling the numerics buys nothing --
# passthrough keeps the learned split points readable ("Year Built <= 1987"
# instead of "Year Built <= -0.42").
def make_tree_pipeline(random_state=42, **tree_kwargs):
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
                    **tree_kwargs,
                ),
            ),
        ]
    )


# Full grid: thorough but 240 combos x folds. Used when tune_tree(fast=False).
TREE_PARAM_GRID = {
    "regressor__max_depth": [3, 5, 8, 12, None],
    "regressor__min_samples_split": [2, 5, 10, 20],
    "regressor__min_samples_leaf": [1, 2, 5, 10],
    "regressor__max_features": [None, "sqrt", "log2"],
}

# Reduced grid: 20 combos. max_features is dropped because subsampling
# features on a *single* tree mostly just adds variance -- it earns its
# keep in a forest, not here.
TREE_PARAM_GRID_FAST = {
    "regressor__max_depth": [5, 8, 12, None],
    "regressor__min_samples_leaf": [1, 5, 10, 20, 50],
}

# Sensible fixed settings for per-cluster trees, where each cluster has too
# few rows to justify its own grid search.
TREE_DEFAULTS = {"max_depth": 8, "min_samples_leaf": 5}


def tune_tree(X_train, y_train, n_folds=5, fast=True, random_state=42, n_jobs=-1):
    """Select tree complexity by cross-validating on training data only.

    Returns the fitted GridSearchCV -- use .best_estimator_ for the model
    and .best_params_ / .best_score_ for reporting.
    """
    n_splits = min(n_folds, len(X_train))
    if n_splits < 2:
        raise ValueError("At least 2 training rows are required to tune a tree.")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    search = GridSearchCV(
        estimator=make_tree_pipeline(random_state=random_state),
        param_grid=TREE_PARAM_GRID_FAST if fast else TREE_PARAM_GRID,
        scoring="neg_mean_absolute_percentage_error",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


# ---------------------------------------------------------------------------
# Clustered estimators
# ---------------------------------------------------------------------------


class ClusteredRegressor(BaseEstimator, RegressorMixin):
    """Fits one clone of base_pipeline per KMeans cluster."""

    def __init__(self, base_pipeline, kmeans, cluster_preprocessor):
        self.base_pipeline = base_pipeline
        self.kmeans = kmeans
        self.cluster_preprocessor = cluster_preprocessor

    def fit(self, X, y):
        X_scaled = self.cluster_preprocessor.transform(X[FEATURES])
        cluster_ids = self.kmeans.predict(X_scaled)
        self.cluster_models_ = {}
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
            preds[mask] = self._model_for(cid).predict(X[mask])
        return preds

    def _model_for(self, cid):
        """Fall back to any fitted model if a cluster was empty at train time."""
        if cid in self.cluster_models_:
            return self.cluster_models_[cid]
        return next(iter(self.cluster_models_.values()))


class BestPerClusterRegressor(BaseEstimator, RegressorMixin):
    """For each cluster, uses k-fold CV on the training data alone to pick
    the best-performing candidate hypothesis, then refits that winner on
    the full training data for that cluster. Validation data is never
    touched during fit() -- only used later for the final reported metric."""

    def __init__(self, candidate_pipelines, kmeans, cluster_preprocessor,
                 n_folds=5, random_state=42):
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
            X_c = X[mask].reset_index(drop=True)
            y_c = y[mask].reset_index(drop=True)

            n_folds_c = min(self.n_folds, len(X_c))
            if n_folds_c < 2:
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
            preds[mask] = self._model_for(cid).predict(X[mask])
        return preds

    def _model_for(self, cid):
        """Fall back to any fitted model if a cluster was empty at train time."""
        if cid in self.cluster_models_:
            return self.cluster_models_[cid]
        return next(iter(self.cluster_models_.values()))