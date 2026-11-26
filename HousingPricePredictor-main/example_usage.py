from housing_api import HousingPriceAPI

api = HousingPriceAPI.load("housing_api.joblib")

listing = {
    "Neighborhood": "Metrotown", "Year": 2021, "Property Type": "House",
    "Bedrooms": 3, "Bathrooms": 2, "Year Built": 1995,
    "Renovation Year": 2015, "Garage Type": "Single",
    "Square Footage (House)": 2100, "Square Footage (Land)": 5000,
    "Basement": "Yes", "Legal Units": 1,
}

print(f"Baseline:          ${api.predict_with_baseline(listing):,.0f}")
print(f"Linear:            ${api.predict_with_linear(listing):,.0f}")
print(f"Polynomial:        ${api.predict_with_polynomial(listing):,.0f}")
print(f"Piecewise:         ${api.predict_with_piecewise(listing):,.0f}")
print(f"Tree:              ${api.predict_with_tree(listing):,.0f}")
print(f"Tree per cluster:  ${api.predict_with_tree_per_cluster(listing):,.0f}")
print(f"Best per cluster:  ${api.predict_with_best_per_cluster(listing):,.0f}")