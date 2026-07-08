from __future__ import annotations

import pandas as pd

from nvda_rl.data_downloader import download_nvda
from nvda_rl.features import add_market_features, fit_unsupervised_train_test


def main() -> None:
    """
    Download NVDA data and rebuild processed feature/regime CSV files.

    Run this script before the notebooks when you want to refresh the dataset.
    The EDA notebook only loads saved files, so downloading stays separate from
    analysis.
    """
    prices = download_nvda(output_path="data/raw/nvda_daily.csv")
    features = add_market_features(prices)
    features.to_csv("data/processed/nvda_features.csv", index=False)
    train_enriched, test_enriched, _, _ = fit_unsupervised_train_test(features, split_date="2021-01-01", n_clusters=3)
    enriched = pd.concat([train_enriched, test_enriched], ignore_index=True).sort_values("date").reset_index(drop=True)
    enriched.to_csv("data/processed/nvda_regimes.csv", index=False)
    print(f"Saved raw rows: {len(prices):,}")
    print(f"Saved feature rows: {len(features):,}")
    print(f"Saved regime rows: {len(enriched):,}")


if __name__ == "__main__":
    main()
