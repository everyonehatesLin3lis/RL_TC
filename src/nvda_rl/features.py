from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "return",
    "volatility_20d",
    "momentum_5d",
    "momentum_20d",
    "volume_z_20d",
    "drawdown",
    "rsi_14",
    "macd",
    "macd_signal",
    "sma_20_gap",
    "sma_50_gap",
    "atr_14_pct",
]


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).

    RSI compares recent average gains with recent average losses. Values above
    70 often suggest a stock is stretched upward, while values below 30 often
    suggest it is stretched downward.
    """
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def calculate_macd(
    prices: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD, MACD signal, and MACD histogram.

    MACD compares a fast trend with a slow trend. Positive MACD means shorter
    trend is above longer trend, while the signal line helps show turning points.
    """
    fast_ema = prices.ewm(span=fast_span, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_span, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_atr_percent(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range as a percentage of adjusted close.

    ATR is a volatility measure based on the daily high-low range and gaps from
    the prior close. Dividing by price makes it comparable across years.
    """
    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window).mean()
    return atr / data["adj_close"]


def add_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create market features used for EDA, clustering, and RL state construction.

    The function turns raw price data into return, risk, trend, momentum,
    volume, drawdown, RSI, MACD, moving-average, and ATR features. These are
    easier for models to learn from than raw price level alone.
    """
    data = prices.copy().sort_values("date").reset_index(drop=True)
    data["return"] = data["adj_close"].pct_change()
    data["log_return"] = np.log(data["adj_close"]).diff()
    data["volatility_20d"] = data["return"].rolling(20).std()
    data["momentum_5d"] = data["adj_close"].pct_change(5)
    data["momentum_20d"] = data["adj_close"].pct_change(20)

    volume_mean = data["volume"].rolling(20).mean()
    volume_std = data["volume"].rolling(20).std()
    data["volume_z_20d"] = (data["volume"] - volume_mean) / volume_std

    data["rolling_max"] = data["adj_close"].cummax()
    data["drawdown"] = data["adj_close"] / data["rolling_max"] - 1.0

    data["rsi_14"] = calculate_rsi(data["adj_close"], window=14)
    data["macd"], data["macd_signal"], data["macd_hist"] = calculate_macd(data["adj_close"])
    data["sma_20"] = data["adj_close"].rolling(20).mean()
    data["sma_50"] = data["adj_close"].rolling(50).mean()
    data["sma_20_gap"] = data["adj_close"] / data["sma_20"] - 1.0
    data["sma_50_gap"] = data["adj_close"] / data["sma_50"] - 1.0
    data["atr_14_pct"] = calculate_atr_percent(data, window=14)

    return data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def fit_unsupervised_features(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Pipeline, Pipeline]:
    """
    Fit k-means regimes and PCA embeddings on market features.

    K-means groups similar days into discrete regimes. PCA compresses the same
    feature matrix into two numeric components that summarize broad market
    structure for later RL state features.
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    X = data[feature_columns]
    regime_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=25, random_state=random_state)),
        ]
    )
    regimes = regime_model.fit_predict(X)

    pca_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=random_state)),
        ]
    )
    components = pca_model.fit_transform(X)

    enriched = data.copy()
    enriched["regime"] = regimes
    enriched["pca_1"] = components[:, 0]
    enriched["pca_2"] = components[:, 1]
    return enriched, regime_model, pca_model


def describe_regimes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the economic behavior of each discovered regime.

    The output helps label clusters by showing whether each regime has higher
    returns, higher volatility, stronger momentum, deeper drawdowns, or more
    stretched RSI values.
    """
    summary = (
        data.groupby("regime")
        .agg(
            days=("date", "count"),
            avg_return=("return", "mean"),
            avg_volatility=("volatility_20d", "mean"),
            avg_momentum_20d=("momentum_20d", "mean"),
            avg_drawdown=("drawdown", "mean"),
            avg_rsi=("rsi_14", "mean"),
            avg_atr_pct=("atr_14_pct", "mean"),
        )
        .sort_values("avg_return")
    )
    return summary


def discretize_state_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous market features into bins for tabular Q-learning.

    Q-learning stores values in a table, so each continuous feature is converted
    into a small integer bucket. The regime label is already discrete.
    """
    states = data.copy()
    states["return_bin"] = pd.qcut(states["return"], q=5, labels=False, duplicates="drop")
    states["vol_bin"] = pd.qcut(states["volatility_20d"], q=5, labels=False, duplicates="drop")
    states["momentum_bin"] = pd.qcut(states["momentum_20d"], q=5, labels=False, duplicates="drop")
    states["rsi_bin"] = pd.qcut(states["rsi_14"], q=5, labels=False, duplicates="drop")
    states["macd_bin"] = pd.qcut(states["macd_hist"], q=5, labels=False, duplicates="drop")
    states["pca_1_bin"] = pd.qcut(states["pca_1"], q=5, labels=False, duplicates="drop")
    state_cols = ["regime", "return_bin", "vol_bin", "momentum_bin", "rsi_bin", "macd_bin", "pca_1_bin"]
    states[state_cols] = states[state_cols].astype(int)
    return states

