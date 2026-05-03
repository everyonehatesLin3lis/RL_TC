from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prices(path: str | Path = "data/raw/nvda_daily.csv") -> pd.DataFrame:
    """
    Load the saved NVDA daily price CSV from disk.

    Parameters
    ----------
    path:
        CSV file created by `download_nvda`.

    Returns
    -------
    pd.DataFrame
        Date-sorted price data ready for feature engineering.
    """
    data = pd.read_csv(path, parse_dates=["date"])
    return data.sort_values("date").reset_index(drop=True)
