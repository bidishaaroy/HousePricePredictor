# This script is to collect all the options for the drop down menu for the html interface. 

from pathlib import Path
import json

import pandas as pd


SOURCE_FILE = Path(__file__).with_name("source.csv")
OUTPUT_FILE = Path(__file__).with_name("options.json")

# These fields have a small, fixed set of choices and work well as dropdowns.
DROPDOWN_FIELDS = [
    "Neighborhood",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Garage Type",
    "Basement",
    "Legal Units",
]


def sort_values(values):
    """Sort numbers numerically and text alphabetically."""
    if all(isinstance(value, (int, float)) for value in values):
        return sorted(values)

    return sorted(values, key=lambda value: str(value).casefold())


def build_options():
    # keep_default_na=False is important:
    # it preserves the literal Garage Type value "None" as text.
    data = pd.read_csv(SOURCE_FILE, keep_default_na=False)

    missing_columns = [
        field for field in DROPDOWN_FIELDS
        if field not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing expected columns: " + ", ".join(missing_columns)
        )

    options = {}

    for field in DROPDOWN_FIELDS:
        unique_values = data[field].unique().tolist()

        unique_values = [
            value for value in unique_values
            if value != ""
        ]

        cleaned_values = []

        for value in unique_values:
            if hasattr(value, "item"):
                value = value.item()

            cleaned_values.append(value)

        options[field] = sort_values(cleaned_values)

    with OUTPUT_FILE.open("w", encoding="utf-8") as json_file:
        json.dump(options, json_file, indent=2, ensure_ascii=False)

    print(f"Created {OUTPUT_FILE.name}")

    for field, values in options.items():
        print(f"{field}: {len(values)} options")


if __name__ == "__main__":
    build_options()
