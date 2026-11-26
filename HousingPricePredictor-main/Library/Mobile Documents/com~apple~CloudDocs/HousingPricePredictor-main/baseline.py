from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import os

data_dir = "./Datasets/bc_last_20_years/split_datasets/"
features = ["Neighborhood","Year","Property Type",
            "Bedrooms","Bathrooms","Year Built","Renovation Year",
            "Garage Type",  "Square Footage (House)",
            "Square Footage (Land)","Basement", "Legal Units"]
output = ["Market Price"]

dfvalidation = pd.read_csv(f"{data_dir}validation.csv")
X_test = dfvalidation[features]
Y_test = dfvalidation[output]

for split in sorted(os.listdir(data_dir)):
    # FIX: Skip results files, standardized files, or anything that isn't a base dataset
    if not split.startswith("BC_") or not split.endswith(".csv") or "standardized" in split:
        continue
        
    abs_path = os.path.join(data_dir, split)
    if os.path.isfile(abs_path):
        df = pd.read_csv(abs_path)
        X_train = df[features]
        Y_train = df[output]
        
        baseline_model  = DummyRegressor(strategy="mean")
        baseline_model.fit(X_train, Y_train)
        
        test_preds = baseline_model.predict(X_test)
        train_preds = baseline_model.predict(X_train)
        
        print(f" {split}, Train MAPE: {mean_absolute_percentage_error(Y_train, train_preds)*100:.2f}%, Val MAPE: {mean_absolute_percentage_error(Y_test, test_preds)*100:.2f}%")