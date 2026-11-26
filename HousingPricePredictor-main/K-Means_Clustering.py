import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

# 1. Define the exact same feature sets from linear_regression.py
FEATURES = [
    "Neighborhood", "Year", "Property Type", "Bedrooms", "Bathrooms",
    "Year Built", "Renovation Year", "Garage Type", "Square Footage (House)",
    "Square Footage (Land)", "Basement", "Legal Units"
]
TARGET = "Market Price"
CATEGORICAL_FEATURES = [
    "Neighborhood", "Year", "Property Type", "Bedrooms", "Bathrooms",
    "Garage Type", "Basement", "Legal Units"
]

NUMERIC_FEATURES = [
    "Year Built", "Renovation Year", "Square Footage (House)", "Square Footage (Land)"
]

# 2. Reuse the robust cleaning function
def clean_data(df):
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Missing").astype(str)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Impute missing numeric values with 0 (or median) to prevent KMeans from crashing
        df[col] = df[col].fillna(0) 
    return df

def main():
    # Set up your path (using the best performing split from Milestone 1)
    data_dir = Path("./Datasets/bc_last_20_years/split_datasets")
    train_file = data_dir / "BC_2017_to_2023.csv"
    val_file = data_dir / "validation.csv"
    
    # Load and clean
    train_df = clean_data(pd.read_csv(train_file))
    val_df = clean_data(pd.read_csv(val_file))
    
    X_train = train_df[FEATURES]
    X_val = val_df[FEATURES]

    # 3. Build the Preprocessor (No Data Leakage!)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    # Fit ONLY on training data, then transform both
    print("Fitting preprocessor on training data...")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)

    # 4. Execute K-Means Clustering
    # We will start with k=4 as a placeholder
    k_clusters = 4 
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
    
    print(f"Running K-Means with {k_clusters} clusters...")
    # Learn boundaries on Training data
    train_df['Cluster_ID'] = kmeans.fit_predict(X_train_scaled)
    
    # Assign validation data to the learned clusters
    val_df['Cluster_ID'] = kmeans.predict(X_val_scaled)

    print("\nTraining Set Cluster Distribution:")
    print(train_df['Cluster_ID'].value_counts())

# 5. Profile the Clusters (Human-Readable)
    print("\n--- Cluster Profiles ---")
    
    # Calculate averages for numeric features (including Price to see the value difference)
    numeric_profiles = train_df.groupby('Cluster_ID')[NUMERIC_FEATURES + [TARGET]].mean().round(0)
    
    # Calculate the most frequent value (mode) for categorical features
    categorical_profiles = train_df.groupby('Cluster_ID')[CATEGORICAL_FEATURES].agg(lambda x: x.mode()[0])
    
    # Merge them together
    cluster_profiles = pd.concat([numeric_profiles, categorical_profiles], axis=1)
    
    # Print each profile cleanly to the terminal
    for cluster_id in range(k_clusters):
        print(f"\n[ Cluster {cluster_id} ]")
        print(cluster_profiles.loc[cluster_id].to_string())

    train_df.to_csv(data_dir / "train_clustered.csv", index=False)
    val_df.to_csv(data_dir / "val_clustered.csv", index=False)
if __name__ == "__main__":
    main()
    # Save the clustered data for Phase 4
    