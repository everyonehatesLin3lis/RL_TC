from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_nvda(
    start: str = "2010-01-01",
    end: str | None = None,
    output_path: str | Path = "data/raw/nvda_daily.csv",
) -> pd.DataFrame:
    """
    Download daily NVDA prices from Yahoo Finance and save them as a CSV file.

    Parameters
    ----------
    start:
        First calendar date requested from Yahoo Finance.
    end:
        End date requested from Yahoo Finance. When omitted, tomorrow is used so
        today's completed market data is included when available.
    output_path:
        File path where the raw downloaded CSV should be stored.

    Returns
    -------
    pd.DataFrame
        Cleaned daily OHLCV price data with lower-case column names.
    """
    if end is None:
        end = (date.today() + timedelta(days=1)).isoformat()

    data = yf.download("NVDA", start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError("No NVDA data was downloaded. Check the network connection or date range.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns = [str(col).lower().replace(" ", "_") for col in data.columns]
    data["date"] = pd.to_datetime(data["date"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data
