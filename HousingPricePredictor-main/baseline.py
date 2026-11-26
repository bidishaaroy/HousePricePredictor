from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import pandas as pd
import os

data_dir = "./Datasets/bc_last_20_years/split_datasets/"
excluded = []
excluded = ["validation.csv"]
features = ["Neighborhood","Year","Property Type",
            "Bedrooms","Bathrooms","Year Built","Renovation Year",
            "Garage Type",	"Square Footage (House)",
            "Square Footage (Land)","Basement",	"Legal Units"]
output = ["Market Price"]
dfvalidation = pd.read_csv(f"{data_dir}validation.csv")
X_test = dfvalidation[features]
Y_test = dfvalidation[output]
for split in sorted(os.listdir(data_dir)):
    abs_path = os.path.join(data_dir, split)
    if os.path.isfile(abs_path):
        if(split not in excluded):
            df = pd.read_csv(abs_path)
            X_train = df[features]
            Y_train = df[output]
            # test_size = 0.2: 80% train, 20% test. random_state: seed
            # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=11)
            # X doesnt even matter (input features are irrelevant for the baseline model)
            baseline_model  = DummyRegressor(strategy="mean")
            baseline_model.fit(X_train, Y_train)
            test_preds = baseline_model.predict(X_test)
            train_preds = baseline_model.predict(X_train)
            
            # print(f"""  Split: {split}\n
            #             Training set: \n
            #             \t MSE: {mean_squared_error(Y_train, train_preds)}\n
            #             \t MAE: {mean_absolute_error(Y_train, train_preds)}\n
            #             \t MAPE: {mean_absolute_percentage_error(Y_train, train_preds)*100}%\n

            #             Test set: \n
            #             \t MSE: {mean_squared_error(Y_test, test_preds)}\n                        
            #             \t MAE: {mean_absolute_error(Y_test, test_preds)}\n
            #             \t MAPE: {mean_absolute_percentage_error(Y_test, test_preds)*100}%\n

            #       """)
            print(f""" {split}, {mean_absolute_percentage_error(Y_train, train_preds)*100:.2f}%,{mean_absolute_percentage_error(Y_test, test_preds)*100:.2f}%""")            

# dataframe = pd.read_csv()