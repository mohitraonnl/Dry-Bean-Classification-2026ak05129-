
"""Download the UCI Dry Bean dataset and save it as data/dry_bean_full.csv."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

def main():
    dataset = fetch_ucirepo(id=602)  # UCI Dry Bean
    features = dataset.data.features
    targets = dataset.data.targets
    df = pd.concat([features, targets], axis=1)
    Path("data").mkdir(exist_ok=True)
    output = Path("data/dry_bean_full.csv")
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows and {len(df.columns)} columns to {output}")

if __name__ == "__main__":
    main()
