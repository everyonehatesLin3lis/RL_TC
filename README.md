# NVDA Market Regimes and RL Trading Agent

This project combines unsupervised learning and reinforcement learning on daily NVIDIA (`NVDA`) stock data from 2010 onward.

The unsupervised part discovers market regimes with k-means and PCA. The RL part compares Q-learning, a lightweight PPO-style policy, random trading, and buy-and-hold.

## How To Run

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Refresh the data first:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\download_data.py
```

The RL notebook trains models, saves them in `models/`, reloads them, and only then evaluates the test-period returns.

Saved artifacts include:

- `models/q_learning.pkl`
- `models/q_regime_only.pkl`
- `models/q_no_unsup.pkl`
- `models/ppo_scaler.pkl`
- `models/ppo_60s.pkl`
- `models/ppo_300s.pkl`
- `models/ppo_600s.pkl`

Then start Jupyter:

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Run notebooks in order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_unsupervised_regimes.ipynb`
3. `notebooks/03_rl_trading_agent.ipynb`

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_unsupervised_regimes.ipynb
|   `-- 03_rl_trading_agent.ipynb
|-- reports/
|   `-- figures/
|-- scripts/
|   |-- download_data.py
|   |-- create_notebooks.py
|   `-- generate_readme_figures.py
|-- src/
|   `-- nvda_rl/
|       |-- data_downloader/
|       |-- features.py
|       |-- env.py
|       |-- agents.py
|       |-- ppo.py
|       `-- evaluation.py
`-- requirements.txt
```

## Source Code Explained

The notebooks show the analysis step by step, but the reusable project logic is kept in `src/nvda_rl/`. This makes the work easier to repeat: the same feature engineering, environment rules, agents, and metrics can be used from notebooks or scripts without copying code.

### `data_downloader/downloader.py` and `data_downloader/loader.py`

These files handle the raw NVDA price data.

`download_nvda()` downloads daily OHLCV data from Yahoo Finance. OHLCV means open, high, low, close, adjusted close, and volume. Adjusted close is important because it accounts for stock splits and dividends, so it is better for long historical return calculations than the raw close price.

`load_prices()` loads the saved CSV and sorts it by date. This is simple, but important, because time-series work must be in the correct chronological order.

Why this choice was used: Yahoo Finance through `yfinance` is easy to reproduce and gives enough daily data for a learning project. The limitation is that this is not institutional-grade market data, so the results should be treated as educational rather than production trading evidence.

### `features.py`

This is one of the most important files. It turns raw price data into model features that describe return, risk, trend, momentum, volume pressure, and drawdown.

Important features:

- `return`: daily percentage change in adjusted close. This is the main profit/loss input.
- `volatility_20d`: rolling 20-day standard deviation of returns. It measures recent risk.
- `momentum_5d` and `momentum_20d`: recent price movement over short and medium windows. These help detect trend strength.
- `volume_z_20d`: how unusual today's volume is compared with the last 20 days.
- `drawdown`: how far price is below its previous running high. This measures downside pain, not just upside growth.
- `rsi_14`: Relative Strength Index. Higher values often mean strong upward pressure; very high values can also mean the stock is stretched.
- `macd`, `macd_signal`, `macd_hist`: trend-following indicators comparing faster and slower moving averages.
- `sma_20_gap` and `sma_50_gap`: distance from 20-day and 50-day moving averages. These show whether price is above or below recent trend.
- `atr_14_pct`: Average True Range divided by price. This measures daily range risk in percentage terms.

The file also creates unsupervised regimes:

- `KMeans` groups market days into 3 clusters. A cluster is not given a label first; the model discovers groups from the features.
- `StandardScaler` is used before clustering because features have different units. For example, RSI is around 0-100, while returns are usually small decimals.
- `PCA` compresses the feature set into two components. This helps summarize the broad market state and gives the RL agent extra state information.

To reduce data leakage, `fit_unsupervised_train_test()` splits chronologically first. The scaler, K-means, and PCA models are fitted only before `2021-01-01`; the test period is assigned regimes with `.predict()` and PCA `.transform()`.

Current-day return is not used as a clustering feature. Regimes are interpreted with `next_day_return`, volatility, momentum, RSI, ATR, and drawdown. This makes the regime analysis less mechanical because clusters are not directly built from the same return that is later used to describe them.

Why 3 regimes: it keeps the interpretation simple. The current regimes are best read as different market states by momentum, volatility, drawdown, and RSI rather than as guaranteed positive/negative return labels. More clusters might fit history more closely, but they would be harder to explain and easier to overfit.

### `env.py`

This file defines the trading simulation. The agent can choose only three actions:

- `-1`: short NVDA
- `0`: stay flat, meaning no position
- `1`: long NVDA

The reward formula is:

```text
r_net_t = a_(t-1) * return_t - cost * |a_t - a_(t-1)|
```

This means the agent observes end-of-day features, chooses the next position, earns the next daily return from its previous position, then pays a transaction cost if it changes position. This prevents the model from looking unrealistically good by trading constantly for free.

There are two environment classes:

- `TradingEnvironment`: used by tabular Q-learning. It expects discrete state bins.
- `GymTradingEnvironment`: a Gymnasium-style environment for continuous observations. It is useful for PPO-style policy learning.

Why previous action is included: transaction cost depends on whether the agent changes position. Without knowing the previous action, the model cannot properly learn that switching from short to long is more expensive than staying long.

### `agents.py`

This file contains the Q-learning agent.

Q-learning is a good baseline here because it is interpretable. It stores a Q-value table where each state-action pair gets a learned score. A state is made from binned market features, and the action is short, flat, or long.

Important settings:

- `alpha`: learning rate. Higher values update the table faster but can be noisier.
- `gamma`: future reward discount. A value near 1 means the agent cares about future returns, not only today.
- `epsilon`: exploration rate. During training, the agent sometimes tries random actions so it can discover better choices.
- `epsilon_decay`: slowly reduces random exploration as training continues.

Why feature binning is needed: tabular Q-learning cannot directly handle continuous values like RSI `63.42` or volatility `0.027`. The project learns quantile bin boundaries from the training period only, then applies the same fixed boundaries to the test period.

The Q-learning state was intentionally reduced to avoid a very sparse Q-table. The main comparison uses:

- baseline: `momentum_bin + vol_bin + previous_action`
- regime only: `regime + momentum_bin + vol_bin + previous_action`
- regime and PCA: `regime + momentum_bin + vol_bin + pca_1_bin + previous_action`

`previous_action` is included because transaction costs depend on whether the agent changes position. Tie handling is also randomized, so unseen states no longer automatically choose short just because `-1` is the first action.

Result meaning: if Q-learning performs poorly, it does not automatically mean regimes are useless. It may mean the discretized state table is too simple, the action space is too limited, transaction costs are too high for the learned behavior, or the bullish test period strongly favors buy-and-hold.

### `ppo.py`

This file contains a lightweight PPO-style policy agent implemented with NumPy.

PPO stands for Proximal Policy Optimization. The main idea is to improve a policy gradually while avoiding updates that are too large. In this project, the policy is linear and outputs probabilities for short, flat, and long.

Important parts:

- `PPO_OBSERVATION_COLUMNS`: the continuous features used by the policy, including risk, momentum, regime, and PCA components. Current-day return is not passed as an observation feature.
- `scale_train_test()`: scales features using only the training period. This avoids leaking test-period information into training.
- `LinearPPOAgent`: learns action probabilities from market features.
- `train_ppo_for_seconds()`: trains for fixed time budgets like 60, 300, or 600 seconds.
- `timed_ppo_comparison()`: checks whether longer training improves out-of-sample performance.

Why a lightweight PPO-style model was chosen: it is easier to inspect and run than a deep neural network PPO implementation, while still showing the key idea of policy learning from continuous features. The tradeoff is that a linear policy is limited and may not capture complex market behavior.

Result meaning: in the executed notebook, longer PPO training did not solve the trading problem. That is still a useful result because it shows that more training time does not automatically create a profitable policy. The model may be too simple, the features may not predict enough, or the market period may favor passive holding.

### `evaluation.py`

This file converts actions into strategy results and metrics.

Main outputs:

- `strategy_frame()`: converts daily actions into net returns and an equity curve after transaction costs.
- `buy_and_hold_frame()`: creates the passive benchmark. This is important because NVDA had very strong long-term upside.
- `random_policy_actions()`: creates a random baseline. A trained model should beat random trading to be meaningful.
- `max_drawdown()`: worst loss from a previous equity peak.
- `performance_metrics()`: returns cumulative return, average daily return, annualized return, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, hit ratio, and turnover.

Metric meanings:

- `cumulative_return`: total growth over the test period.
- `annualized_return`: CAGR estimated from the equity curve and elapsed trading years.
- `annualized_volatility`: yearly-scaled return variation, used as a risk measure.
- `sharpe_ratio`: return per unit of volatility.
- `sortino_ratio`: return per unit of downside volatility.
- `max_drawdown`: the largest peak-to-trough fall. Lower drawdown is usually better.
- `hit_ratio`: share of days with positive net return.
- `turnover`: how much the strategy changes position. Higher turnover usually means more trading cost.

Why these metrics were chosen: return alone is not enough. A strategy can make money but have huge drawdowns or excessive trading. These metrics show both performance and risk.

## Key EDA Graphics

![NVDA price and drawdown](reports/figures/readme_price_drawdown.png)

After feature warmup, NVDA increased from about `$0.41` to about `$198.45`. The same period still had a max drawdown near `-66%`, so the project should not evaluate only upside.

![Feature correlations](reports/figures/readme_correlations.png)

The strongest direct relationship with daily return is 5-day momentum. ATR/range risk is slightly negative versus same-day return, which supports separating momentum features from risk features.

## Regime Graphics

![Regime chart](reports/figures/readme_regimes.png)

The k-means regimes are economically interpretable as different combinations of momentum, volatility, drawdown, and RSI. Because current-day return is excluded from clustering, the regimes should be treated as market-state features rather than guaranteed return labels.

## RL Graphic

![Equity curves](reports/figures/readme_equity_curves.png)

Buy-and-hold is a difficult benchmark for NVDA because the test period is strongly bullish. The RL strategies are still useful for learning because they show how transaction costs, overtrading, and regime features affect policy behavior.

In the executed notebook, all test results come from saved-and-reloaded models. Buy-and-hold remains a difficult benchmark because the test period is strongly bullish. The important RL comparison is the ablation table: technical-feature Q-learning versus Q-learning with regimes versus Q-learning with regimes plus PCA.

## Reward Definition

```text
r_net_t = a_(t-1) * return_t - cost * |a_t - a_(t-1)|
```

The agent observes features at the end of day `t`, chooses the position for the next step, earns return from the previous position, and pays a cost when changing position. The Q-learning state includes previous position so the cost is not hidden from the agent.

## Review Talking Points

- The central question is whether unsupervised regime and PCA features improve out-of-sample RL behavior compared with technical features alone.
- Regime/PCA models are fitted on train only, then applied to test to avoid leakage.
- Q-learning is interpretable but needs feature binning and a small enough state space.
- PPO here is a PPO-style clipped policy-gradient model, not a full canonical PPO implementation with critic, GAE, and minibatch optimization.
- This is a learning project, not a production trading strategy. A real strategy would need walk-forward testing, slippage, short-borrow assumptions, transaction-cost sensitivity, and multi-seed robustness checks.
