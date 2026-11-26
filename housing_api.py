from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from K_Means_Clustering import get_kmeans, sweep_k
from models import (
    FEATURES,
    TARGET,
    TREE_DEFAULTS,
    BestPerClusterRegressor,
    ClusteredRegressor,
    clean_data,
    make_cluster_preprocessor,
    make_linear_regression_pipeline,
    make_piecewise_pipeline,
    make_polynomial_pipeline,
    make_tree_pipeline,
    tune_tree,
)

K_RANGE = range(2, 8)
PIECEWISE_BREAKPOINTS = {"Year Built": 2000}


class HousingPriceAPI:
    """One training split in, every model family out."""

    def __init__(self, train_csv=None, train_df=None, k=None,
                 tune_trees=True, fast_tree_grid=True):
        if train_df is None:
            if train_csv is None:
                raise ValueError("Provide train_csv or train_df.")
            train_df = pd.read_csv(train_csv)

        train_df = clean_data(train_df, drop_missing_target=True)
        self.X_train = train_df[FEATURES]
        self.y_train = train_df[TARGET]
        self.tune_trees = tune_trees
        self.fast_tree_grid = fast_tree_grid

        self.cluster_preprocessor = make_cluster_preprocessor()
        X_scaled = self.cluster_preprocessor.fit_transform(self.X_train)
        if k is None:
            k, _scores = sweep_k(X_scaled, K_RANGE)
        self.k = k
        self.kmeans = get_kmeans(self.k, X_scaled)

        self._models = {}  
    def _build(self, name):
        """Construct an unfitted estimator for the given model family."""
        if name == "baseline":
            return DummyRegressor(strategy="mean")
        if name == "linear":
            return make_linear_regression_pipeline()
        if name == "polynomial":
            return ClusteredRegressor(
                make_polynomial_pipeline(degree=2),
                self.kmeans, self.cluster_preprocessor,
            )
        if name == "piecewise":
            return ClusteredRegressor(
                make_piecewise_pipeline(PIECEWISE_BREAKPOINTS),
                self.kmeans, self.cluster_preprocessor,
            )
        if name == "tree":
            return make_tree_pipeline(**TREE_DEFAULTS)
        if name == "tree_per_cluster":
            return ClusteredRegressor(
                make_tree_pipeline(**TREE_DEFAULTS),
                self.kmeans, self.cluster_preprocessor,
            )
        if name == "best_per_cluster":
            candidates = {
                "linear": make_linear_regression_pipeline(),
                "poly_deg2": make_polynomial_pipeline(degree=2),
                "piecewise": make_piecewise_pipeline(PIECEWISE_BREAKPOINTS),
            }
            return BestPerClusterRegressor(
                candidates, self.kmeans, self.cluster_preprocessor
            )
        raise ValueError(
            f"Unknown model '{name}'. Options: {self.available_models()}"
        )

    def _get(self, name):
        """Fit-on-first-use, then cache."""
        if name not in self._models:
            if self.X_train is None:
                raise RuntimeError(
                    f"'{name}' was not fitted before this API was saved, and "
                    f"the training data isn't in the bundle. Rebuild with "
                    f"save(..., keep_training_data=True), or retrain from CSV."
                )
            if name == "tree" and self.tune_trees:
                search = tune_tree(self.X_train, self.y_train, fast=self.fast_tree_grid)
                self._models[name] = search.best_estimator_
                self.tree_search_ = search
                self.tree_best_params_ = search.best_params_
                self.tree_cv_mape_ = -search.best_score_ * 100
            else:
                model = self._build(name)
                model.fit(self.X_train, self.y_train)
                self._models[name] = model
        return self._models[name]

    def available_models(self):
        return [
            "baseline",
            "linear",
            "polynomial",
            "piecewise",
            "tree",
            "tree_per_cluster",
            "best_per_cluster",
        ]

    @staticmethod
    def _as_frame(features):
        if isinstance(features, dict):
            return pd.DataFrame([features]), True
        if isinstance(features, pd.DataFrame):
            return features, False
        return pd.DataFrame(list(features)), False

    def predict(self, model_name, features):
        df, single = self._as_frame(features)

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        df = clean_data(df)

        preds = self._get(model_name).predict(df[FEATURES])
        preds = np.asarray(preds, dtype=float)
        return float(preds[0]) if single else preds

    def predict_with_baseline(self, features):
        return self.predict("baseline", features)

    def predict_with_linear(self, features):
        return self.predict("linear", features)

    def predict_with_polynomial(self, features):
        return self.predict("polynomial", features)

    def predict_with_piecewise(self, features):
        return self.predict("piecewise", features)

    def predict_with_tree(self, features):
        return self.predict("tree", features)

    def predict_with_tree_per_cluster(self, features):
        return self.predict("tree_per_cluster", features)

    def predict_with_best_per_cluster(self, features):
        return self.predict("best_per_cluster", features)
    
    def tree_info(self):
        model = self._get("tree")
        regressor = model.named_steps["regressor"]
        return {
            "best_params": getattr(self, "tree_best_params_", "not tuned"),
            "cv_mape": getattr(self, "tree_cv_mape_", None),
            "depth": regressor.get_depth(),
            "leaves": regressor.get_n_leaves(),
        }

    def tree_feature_importances(self, top_n=10):
        """Most important features for the tuned tree, highest first."""
        model = self._get("tree")
        names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["regressor"].feature_importances_
        order = np.argsort(importances)[::-1][:top_n]
        return [(str(names[i]), float(importances[i])) for i in order]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def fit_all(self):
        """Force-fit every model family (useful right before save())."""
        for name in self.available_models():
            self._get(name)
        return self

    def save(self, path, keep_training_data=False):
        self.fit_all()

        if keep_training_data:
            joblib.dump(self, path)
            return path

        X_train, y_train = self.X_train, self.y_train
        try:
            self.X_train, self.y_train = None, None
            joblib.dump(self, path)
        finally:
            self.X_train, self.y_train = X_train, y_train
        return path

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict) and "model" in obj:
            raise TypeError(
                f"{path} is a single-model bundle from train.py, not a "
                f"{cls.__name__}. Load it with predict.py, or build an API "
                f"bundle first:\n"
                f"    python housing_api.py --train-csv <split.csv> --out <out.joblib>"
            )
        raise TypeError(f"{path} does not contain a {cls.__name__}.")

    @classmethod
    def load_or_train(cls, path, train_csv=None, train_df=None, **kwargs):
        path = Path(path)
        if path.exists():
            return cls.load(path)
        api = cls(train_csv=train_csv, train_df=train_df, **kwargs)
        api.save(path)
        return api


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train every model family on one split and save it as a "
                    "joblib bundle that HousingPriceAPI.load() can read."
    )
    parser.add_argument("--train-csv", required=True, help="Training split CSV.")
    parser.add_argument("--out", default="housing_api.joblib", help="Output path.")
    parser.add_argument("--k", type=int, default=None,
                        help="Cluster count. Omit to sweep by silhouette score.")
    parser.add_argument("--no-tree-tuning", action="store_true",
                        help="Skip the decision tree grid search.")
    args = parser.parse_args()

    api = HousingPriceAPI(
        train_csv=args.train_csv, k=args.k, tune_trees=not args.no_tree_tuning
    )
    api.save(args.out)
    print(f"\nFitted: {', '.join(api.available_models())}")
    print(f"Clusters (k): {api.k}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    _cli()