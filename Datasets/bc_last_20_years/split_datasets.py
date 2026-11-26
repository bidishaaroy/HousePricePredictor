import os
import pandas as pd

SOURCE_FILE = "source.csv"
OUTPUT_FOLDER = "split_datasets"

VALIDATION_FRACTION = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(
    SOURCE_FILE,
    keep_default_na=False,
    na_values=[""]
)

# Fill missing renovation years with the year built
df["Renovation Year"] = df["Renovation Year"].fillna(df["Year Built"])

# Randomly select validation rows from the entire dataset
validation = df.sample(
    frac=VALIDATION_FRACTION,
    random_state=RANDOM_STATE
)

# Everything not selected for validation becomes training data
training = df.drop(validation.index)

# Shuffle both datasets so rows are not grouped by year or neighborhood
training = training.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

validation = validation.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# Save training set
training.to_csv(
    os.path.join(OUTPUT_FOLDER, "training.csv"),
    index=False
)

# Save validation set
validation.to_csv(
    os.path.join(OUTPUT_FOLDER, "validation.csv"),
    index=False
)

print(f"Saved training.csv ({len(training)} rows)")
print(f"Saved validation.csv ({len(validation)} rows)")