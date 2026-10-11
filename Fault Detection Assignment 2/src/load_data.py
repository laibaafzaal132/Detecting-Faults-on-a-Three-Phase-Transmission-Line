"""Task 1.1 - the data loader: read detect_dataset.csv and show shape, columns and first rows."""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "detect_dataset.csv")


def load_data(path=DEFAULT_CSV):
    """Return the raw DataFrame exactly as stored in the CSV (no cleaning yet)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Put detect_dataset.csv in data/.")
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())
