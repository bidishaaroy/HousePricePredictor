import pandas as pd

SOURCE_FILE = "source.csv"

df = pd.read_csv(
    SOURCE_FILE,
    keep_default_na=False,
    na_values=[""]
)

# Remove Season if it exists
df = df.drop(columns=["Season"], errors="ignore")

# Overwrite source.csv without the Season column
df.to_csv(SOURCE_FILE, index=False)

print(f"Updated {SOURCE_FILE}: Season column removed (if present).")