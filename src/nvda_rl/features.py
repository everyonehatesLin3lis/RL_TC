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

CLUSTER_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column != "return"]

STATE_BIN_COLUMNS = {
    "vol_bin": "volatility_20d",
    "momentum_bin": "momentum_20d",
    "rsi_bin": "rsi_14",
    "macd_bin": "macd_hist",
    "pca_1_bin": "pca_1",
}


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
    data["next_day_return"] = data["return"].shift(-1)
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
    structure for later RL state features. By default, current-day return is
    excluded from clustering to avoid creating mechanically return-sorted
    regimes.
    """
    if feature_columns is None:
        feature_columns = CLUSTER_FEATURE_COLUMNS

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


def transform_unsupervised_features(
    data: pd.DataFrame,
    regime_model: Pipeline,
    pca_model: Pipeline,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply fitted regime and PCA models to new chronological data.

    This is used for the test period so cluster centers and PCA components come
    only from the training period.
    """
    if feature_columns is None:
        feature_columns = CLUSTER_FEATURE_COLUMNS

    X = data[feature_columns]
    enriched = data.copy()
    enriched["regime"] = regime_model.predict(X)
    components = pca_model.transform(X)
    enriched["pca_1"] = components[:, 0]
    enriched["pca_2"] = components[:, 1]
    return enriched


def fit_unsupervised_train_test(
    data: pd.DataFrame,
    split_date: str = "2021-01-01",
    feature_columns: list[str] | None = None,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, Pipeline]:
    """
    Chronologically split data, fit unsupervised models on train, transform test.

    This avoids test-period leakage in StandardScaler, K-means, and PCA.
    """
    data = data.sort_values("date").reset_index(drop=True)
    train = data[data["date"] < split_date].reset_index(drop=True)
    test = data[data["date"] >= split_date].reset_index(drop=True)
    train_enriched, regime_model, pca_model = fit_unsupervised_features(
        train,
        feature_columns=feature_columns,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    test_enriched = transform_unsupervised_features(
        test,
        regime_model=regime_model,
        pca_model=pca_model,
        feature_columns=feature_columns,
    )
    return train_enriched, test_enriched, regime_model, pca_model


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
            avg_next_day_return=("next_day_return", "mean"),
            avg_volatility=("volatility_20d", "mean"),
            avg_momentum_20d=("momentum_20d", "mean"),
            avg_drawdown=("drawdown", "mean"),
            avg_rsi=("rsi_14", "mean"),
            avg_atr_pct=("atr_14_pct", "mean"),
        )
        .sort_values("avg_next_day_return")
    )
    return summary


def fit_state_bins(train: pd.DataFrame, q: int = 5) -> dict[str, np.ndarray]:
    """
    Learn fixed state-bin boundaries from training data only.

    The returned edges can be applied to validation or test data with
    `apply_state_bins`, avoiding quantile leakage from future periods.
    """
    bin_edges = {}
    for output_column, source_column in STATE_BIN_COLUMNS.items():
        _, edges = pd.qcut(train[source_column], q=q, retbins=True, duplicates="drop")
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        bin_edges[output_column] = np.unique(edges)
    return bin_edges


def apply_state_bins(data: pd.DataFrame, bin_edges: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Apply fixed training-period bin boundaries to a dataset.
    """
    states = data.copy()
    for output_column, edges in bin_edges.items():
        source_column = STATE_BIN_COLUMNS[output_column]
        states[output_column] = pd.cut(states[source_column], bins=edges, labels=False, include_lowest=True)
    states[list(bin_edges)] = states[list(bin_edges)].astype(int)
    return states


def discretize_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    q: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """
    Fit Q-learning bins on training data and apply them to train and test.
    """
    bin_edges = fit_state_bins(train, q=q)
    return apply_state_bins(train, bin_edges), apply_state_bins(test, bin_edges), bin_edges


def discretize_state_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous market features into bins for tabular Q-learning.

    Q-learning stores values in a table, so each continuous feature is converted
    into a small integer bucket. This helper fits bins on the supplied data and
    is kept for quick demos. For train/test experiments, use
    `discretize_train_test` to avoid leakage.
    """
    return apply_state_bins(data, fit_state_bins(data))
