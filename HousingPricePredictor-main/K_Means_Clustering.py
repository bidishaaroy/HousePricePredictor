import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define the exact same feature sets from linear_regression.py
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

K_RANGE = range(2, 8)  # candidate k values to sweep


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


def sweep_k(X_train_scaled, k_range):
    """Fit KMeans for each k in k_range, score with silhouette, and plot the results."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_train_scaled)
        score = silhouette_score(X_train_scaled, labels)
        scores[k] = score
        print(f"  k={k}: silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nBest k by silhouette score: {best_k} (score={scores[best_k]:.4f})")

    # Plot silhouette score vs. k
    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.plot(list(scores.keys()), list(scores.values()), marker="o")
    # ax.set_xlabel("k (number of clusters)")
    # ax.set_ylabel("Silhouette Score")
    # ax.set_title("Silhouette Score vs. k")
    # ax.axvline(best_k, color="red", linestyle="--", alpha=0.5, label=f"best k={best_k}")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    return best_k, scores

def get_kmeans(k, X_scaled):
    """Fit and return a KMeans model with k clusters on already-scaled features."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_scaled)
    return kmeans
    
    
def main():
    # Set up your path (using the best performing split from Milestone 1)
    data_dir = Path("./Datasets/bc_last_20_years/split_datasets")
    train_file = data_dir / "BC_2004_to_2023.csv"
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

    # 4. Sweep k using silhouette score to pick a principled cluster count
    print(f"\nSweeping k over {list(K_RANGE)} using silhouette score...")
    best_k, scores = sweep_k(X_train_scaled, K_RANGE)

    # Use the best k found (or override manually if you want to compare against your own judgment)
    k_clusters = best_k
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
    print(f"\nRunning final K-Means with k={k_clusters} clusters...")
    train_df['Cluster_ID'] = kmeans.fit_predict(X_train_scaled)
    val_df['Cluster_ID'] = kmeans.predict(X_val_scaled)

    print("\nTraining Set Cluster Distribution:")
    print(train_df['Cluster_ID'].value_counts())

    # 5. Profile the Clusters (Human-Readable)
    print("\n--- Cluster Profiles ---")
    numeric_profiles = train_df.groupby('Cluster_ID')[NUMERIC_FEATURES + [TARGET]].mean().round(0)
    categorical_profiles = train_df.groupby('Cluster_ID')[CATEGORICAL_FEATURES].agg(lambda x: x.mode()[0])
    cluster_profiles = pd.concat([numeric_profiles, categorical_profiles], axis=1)

    plot_features = ["Year Built", "Renovation Year", "Square Footage (House)",
                      "Square Footage (Land)", "Bedrooms", "Bathrooms", "Legal Units"]

    for cluster_id in sorted(train_df["Cluster_ID"].unique()):
        print(f"\n[ Cluster {cluster_id} ]")
        print(cluster_profiles.loc[cluster_id].to_string())

        subset = train_df[train_df["Cluster_ID"] == cluster_id]
        fig, axes = plt.subplots(1, len(plot_features), figsize=(20, 4))
        fig.suptitle(f"Cluster {cluster_id}")

        for ax, feat in zip(axes, plot_features):
            sns.regplot(
                data=subset, x=feat, y="Market Price", lowess=True,
                scatter_kws={"alpha": 0.3, "s": 10}, ax=ax
            )
            ax.set_xlabel(feat)
            ax.set_ylabel("Market Price")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()