# Made with the assistance of python code in CMPT 310 A2
# Credit to:
# Mohammad Soltanshah & the CMPT 310 teaching team

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import pandas as pd
import os
# -----------------------------
# Preprocessing utilities
# -----------------------------
def fit_preprocess(train_df, feature_plan):
    params = {"standard": {}, "onehot": {}}
    for col, how in feature_plan.items():
        if how == "standard":
            mu = train_df[col].mean()
            sigma = train_df[col].std(ddof=0)
            if sigma == 0:
                sigma = 1.0
            params["standard"][col] = (mu, sigma)
        elif how == "one-hot":
            params["onehot"][col] = sorted(train_df[col].dropna().unique())
    return params


def transform(df, feature_plan, params):
    X_parts = []
    for col, how in feature_plan.items():
        if how == "drop":
            continue

        if how == "standard":
            mu, sigma = params["standard"][col]
            x = (df[col] - mu) / sigma
            X_parts.append(x.to_numpy().reshape(-1, 1))

        elif how == "one-hot":
            cats = params["onehot"][col]
            onehot = np.zeros((len(df), len(cats)))
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            for i, v in enumerate(df[col]):
                if v in cat_to_idx:
                    onehot[i, cat_to_idx[v]] = 1.0
            X_parts.append(onehot)

    return np.hstack(X_parts)  # (n, d)

def add_intercept(X_row_major):
    # X: (n, d) -> (n, d+1)
    return np.hstack([np.ones((X_row_major.shape[0], 1)), X_row_major])


feature_plan = {
    "Neighborhood": "one-hot",
    "Year": "one-hot",
    "Property Type": "one-hot",
    "Bedrooms": "one-hot",
    "Bathrooms": "one-hot",
    "Year Built": "standard",
    "Renovation Year": "standard",
    "Garage Type": "one-hot",
    "Square Footage (House)": "standard",
    "Square Footage (Land)": "standard",
    "Basement": "one-hot",
    "Legal Units": "one-hot"
}


data_dir = "./Datasets/bc_last_20_years/split_datasets/"
excluded = []
# excluded = ["validation.csv"]

features = ["Neighborhood","Year","Property Type",
            "Bedrooms","Bathrooms","Year Built","Renovation Year",
            "Garage Type",	"Square Footage (House)",
            "Square Footage (Land)","Basement",	"Legal Units"]

output = ["Market Price"]
def get_feature_names(feature_plan, params):
    names = ["Intercept"]

    for col, how in feature_plan.items():
        if how == "standard":
            names.append(col)

        elif how == "one-hot":
            for cat in params["onehot"][col]:
                names.append(f"{col}_{cat}")

    return names
            
for split in os.listdir(data_dir):
    abs_path = os.path.join(data_dir, split)
    if os.path.isfile(abs_path) and split not in excluded and "standardized" not in split:
        
        df = pd.read_csv(abs_path)
        params = fit_preprocess(df, feature_plan)
        X = add_intercept(transform(df, feature_plan, params))
        out_name = os.path.splitext(split)[0] + "_standardized.csv"
        out_path = os.path.join(data_dir, out_name)
        cols = get_feature_names(feature_plan, params)
        pd.DataFrame(X, columns=cols).to_csv(out_path, index=False)

# dataframe = pd.read_csv()