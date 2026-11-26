import pandas as pd
import numpy as np
import math

def sample_from_distribution(series, n, rng):
    # probabilistic approach. using the mean seems too shallow/obliterates variance
    # with this approach the variance across the dataset remains the same
    # the entire dataset will have the same variance as the first dataset
    probs = series.value_counts(normalize=True) # probabilities and not counts
    return rng.choice(probs.index, size=n, p=probs.values)

data1 = pd.read_csv("BC2021to2023.csv", nrows=528)
data2 = pd.read_csv("BC2021to2023.csv", skiprows=529)
rng = np.random.default_rng(seed=12)
n = len(data2)
# Standardize the property type, season, garage type, square footage (House)
# print(data1)
# print(data2)
# Property type: Count the number of times each value shows up: 
# print(data1["Property Type"].value_counts())
# print(data1["Year Built"].value_counts())
# print(data1["Renovation Year"].value_counts())
# print(data1["Basement"].value_counts())
# print(data1["Legal Units"].value_counts())
# print(data1["Garage Type"].value_counts())

# to account for null entries
# was_renovated_prob = data1["Renovation Year"].notna().mean()
# was_renovated = rng.random(n) < was_renovated_prob
# renovation_years = sample_from_distribution(data1["Renovation Year"].dropna(), was_renovated.sum(), rng)
# data2["Property Type"] = sample_from_distribution(data1["Property Type"], n, rng)
# data2["Year Built"] = sample_from_distribution(data1["Year Built"], n, rng)
# data2["Renovation Year"]        = np.nan
# data2.loc[was_renovated, "Renovation Year"] = renovation_years
# data2["Basement"]               = sample_from_distribution(data1["Basement"], n, rng)
# data2["Legal Units"]            = sample_from_distribution(data1["Legal Units"], n, rng)
# data2["Garage Type"]            = sample_from_distribution(data1["Garage Type"], n, rng)
# data2["Square Footage (House)"] = rng.choice(data1["Square Footage (House)"], size=n, replace=True)
# data2["Square Footage (Land)"]  = rng.choice(data1["Square Footage (Land)"], size=n, replace=True)
# data1["Data Source"]            = "real"
# data2["Data Source"]            = "synthetic"
was_renovated_prob = data1["Renovation Year"].notna().mean()
was_renovated = rng.random(n) < was_renovated_prob
renovation_years = sample_from_distribution(data1["Renovation Year"].dropna(), was_renovated.sum(), rng)
data2["Property Type"] = np.nan
data2["Year Built"] = np.nan
data2["Renovation Year"]        = np.nan
# data2.loc[was_renovated, "Renovation Year"] = renovation_years
data2["Basement"]               = np.nan
data2["Legal Units"]            = np.nan 
data2["Garage Type"]            = np.nan 
data2["Square Footage (House)"] = np.nan
data2["Square Footage (Land)"]  = np.nan
data1["Data Source"]            = "real"
data2["Data Source"]            = "real"
# merge the two datasets
merged = {
    "Neighborhood" :            pd.concat([data1["Neighborhood"], data2["City"]], ignore_index=True),
    "Year Built":               pd.concat([data1["Year Built"], data2["Year Built"]], ignore_index=True),
    "Property Type":            pd.concat([data1["Year Built"], data2["Property Type"]], ignore_index=True),
    "Bedrooms":                 pd.concat([data1["Bedrooms"], data2["Number_Beds"]], ignore_index=True),
    "Bathrooms":                pd.concat([data1["Bathrooms"], data2["Number_Baths"]], ignore_index=True),
    "Renovation Year":          pd.concat([data1["Renovation Year"], data2["Renovation Year"]], ignore_index=True),
    "Garage Type":              pd.concat([data1["Garage Type"], data2["Garage Type"]], ignore_index=True),
    "Square Footage (House)":   pd.concat([data1["Square Footage (House)"], data2["Square Footage (House)"]], ignore_index=True),
    "Square Footage (Land)":    pd.concat([data1["Square Footage (Land)"], data2["Square Footage (Land)"]], ignore_index=True),
    "Basement":                 pd.concat([data1["Basement"], data2["Basement"]], ignore_index=True),
    "Legal Units":              pd.concat([data1["Legal Units"], data2["Legal Units"]], ignore_index=True),
    "Market Price":             pd.concat([data1["Market Price"], data2["Price"]], ignore_index=True),
    "Data Source":              pd.concat([data1["Data Source"], data2["Data Source"]], ignore_index=True)
}
# index false so there isnt the first column of indices
merged_dataframe = pd.DataFrame(merged)
merged_dataframe.to_csv("mergedblank.csv", index=False)

# print((data["Year"]))
# print(data["Year"][0:100].mean())
# print(data["Year"])