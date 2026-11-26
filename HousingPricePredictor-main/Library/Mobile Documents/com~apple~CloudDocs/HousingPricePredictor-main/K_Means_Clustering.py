import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Feature Definitions
FEATURES = [
    "Neighborhood", "Year", "Property Type", "Bedrooms", "Bathrooms",
    "Year Built", "Renovation Year", "Garage Type", "Square Footage (House)",
    "Square Footage (Land)", "Basement", "Legal Units"
]
TARGET = "Market Price"
CATEGORICAL_FEATURES = [
    "Year", "Property Type",
    "Garage Type", "Basement",
]
NUMERIC_FEATURES = [
    "Bedrooms", "Bathrooms", "Year Built", "Renovation Year",
    "Square Footage (House)", "Square Footage (Land)", "Legal Units"
]
K_RANGE = range(2, 8)

# 2. Data Cleaning
def clean_data(df):
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Missing").astype(str)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def sweep_k(X_train_scaled, k_range):
    """Preserved logic: Fit KMeans for each k, score with silhouette."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_train_scaled)
        score = silhouette_score(X_train_scaled, labels)
        scores[k] = score
        print(f"  k={k}: silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nBest k by silhouette score: {best_k} (score={scores[best_k]:.4f})")
    return best_k, scores

def get_kmeans(k, X_scaled):
    """Fit and return a KMeans model with k clusters on already-scaled features."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_scaled)
    return kmeans

def main():
    data_dir = Path("./Datasets/bc_last_20_years/split_datasets")
    train_file = data_dir / "BC_2004_to_2023.csv"
    val_file = data_dir / "validation.csv"

    print("Loading and cleaning datasets...")
    train_df = clean_data(pd.read_csv(train_file))
    val_df = clean_data(pd.read_csv(val_file))
    
    X_train = train_df[FEATURES]
    X_val = val_df[FEATURES]

    # 3. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    print("Fitting preprocessor on training data...")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)

    # 4. Final K-Means Execution (Hardcoded to the proven optimal k=2)
    k_clusters = 2
    print(f"\nExecuting final K-Means assignment with optimal k={k_clusters}...")
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
    train_df['Cluster_ID'] = kmeans.fit_predict(X_train_scaled)
    val_df['Cluster_ID'] = kmeans.predict(X_val_scaled)

    print("\nTraining Set Cluster Distribution:")
    print(train_df['Cluster_ID'].value_counts())

    # 5. Cluster Profiling
    print("\n--- Cluster Profiles ---")
    numeric_profiles = train_df.groupby('Cluster_ID')[NUMERIC_FEATURES + [TARGET]].mean().round(0)
    categorical_profiles = train_df.groupby('Cluster_ID')[CATEGORICAL_FEATURES].agg(lambda x: x.mode()[0])
    cluster_profiles = pd.concat([numeric_profiles, categorical_profiles], axis=1)

    for cluster_id in sorted(train_df["Cluster_ID"].unique()):
        print(f"\n[ Cluster {cluster_id} ]")
        print(cluster_profiles.loc[cluster_id].to_string())



if __name__ == "__main__":
    main()