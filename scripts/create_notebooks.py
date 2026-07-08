from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def md(text: str):
    """
    Create a markdown notebook cell with clean indentation.

    The generator uses triple-quoted strings, so dedenting keeps notebooks free
    from accidental leading spaces.
    """
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    """
    Create a code notebook cell with clean indentation.

    The returned cell is ready to execute with nbconvert or Jupyter.
    """
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(name: str, cells: list) -> None:
    """
    Write one notebook file into the notebooks folder.

    The notebook metadata points to the standard Python kernel used by the
    local `.venv`.
    """
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_DIR / name)


setup_cell = code(
    """
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent
    if not (PROJECT_ROOT / "src").exists():
        PROJECT_ROOT = Path.cwd().parent
    sys.path.append(str(PROJECT_ROOT / "src"))
    """
)


eda_cells = [
    md(
        """
        # 01 - NVDA EDA

        Goal: understand price, return, risk, and feature relationships before modeling.
        """
    ),
    setup_cell,
    code(
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        from nvda_rl.data_downloader import load_prices
        from nvda_rl.features import add_market_features

        sns.set_theme(style="whitegrid", context="notebook")
        """
    ),
    md(
        """
        ## Load saved data

        This notebook does not download data. Run `scripts/download_data.py` first when the CSV needs refreshing.
        """
    ),
    code(
        """
        raw_path = PROJECT_ROOT / "data" / "raw" / "nvda_daily.csv"
        prices = load_prices(raw_path)
        print(f"Rows: {len(prices):,}")
        print(f"Range: {prices['date'].min().date()} to {prices['date'].max().date()}")
        prices.head()
        """
    ),
    md(
        """
        ## Build features

        `add_market_features()` creates returns, volatility, momentum, volume z-score, drawdown, RSI, MACD, moving-average gaps, and ATR.
        """
    ),
    code(
        """
        data = add_market_features(prices)
        data.to_csv(PROJECT_ROOT / "data" / "processed" / "nvda_features.csv", index=False)

        feature_preview = [
            "date", "adj_close", "return", "volatility_20d", "momentum_20d",
            "rsi_14", "macd_hist", "sma_50_gap", "atr_14_pct", "drawdown"
        ]
        data[feature_preview].head()
        """
    ),
    md(
        """
        RSI shows stretched up/down moves, MACD shows trend change, and ATR shows daily range risk.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(data["date"], data["adj_close"], color="tab:green")
        axes[0].set_title("NVDA adjusted close")
        axes[0].set_ylabel("Price")

        axes[1].fill_between(data["date"], data["drawdown"], 0, color="tab:red", alpha=0.35)
        axes[1].set_title("Drawdown from running high")
        axes[1].set_ylabel("Drawdown")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        After feature warmup, NVDA grew from about $0.41 to $198.45. That huge gain still came with a max drawdown near -66%, so risk control matters.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(data["return"], bins=80, kde=True, ax=axes[0], color="tab:blue")
        axes[0].set_title("Daily returns")
        sns.histplot(data["volatility_20d"], bins=60, kde=True, ax=axes[1], color="tab:orange")
        axes[1].set_title("20-day volatility")
        sns.histplot(data["rsi_14"], bins=60, kde=True, ax=axes[2], color="tab:purple")
        axes[2].set_title("RSI")
        plt.tight_layout()
        plt.show()

        data[["return", "volatility_20d", "momentum_20d", "rsi_14", "atr_14_pct", "drawdown"]].describe().T
        """
    ),
    md(
        """
        Average daily return is about 0.19%, while daily volatility is about 2.87%. Single-day moves ranged from about -18.8% to +29.8%, so the tails are meaningful.
        """
    ),
    md(
        """
        ## Correlations

        Correlations show which features move together and which may add new information.
        """
    ),
    code(
        """
        corr_cols = [
            "return", "volatility_20d", "momentum_5d", "momentum_20d",
            "volume_z_20d", "drawdown", "rsi_14", "macd_hist",
            "sma_20_gap", "sma_50_gap", "atr_14_pct"
        ]
        corr = data[corr_cols].corr()

        plt.figure(figsize=(11, 8))
        sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.5)
        plt.title("Feature correlation heatmap")
        plt.tight_layout()
        plt.show()

        corr["return"].sort_values(ascending=False)
        """
    ),
    md(
        """
        The strongest return relationship is with 5-day momentum at about 0.44 correlation. ATR is slightly negative, which suggests high range-risk days are not automatically good return days.
        """
    ),
    code(
        """
        monthly = data.set_index("date")["return"].resample("M").agg(["mean", "std"])
        monthly.columns = ["monthly_avg_daily_return", "monthly_daily_volatility"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        monthly["monthly_avg_daily_return"].plot(ax=axes[0], color="tab:blue")
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_title("Monthly average daily return")
        monthly["monthly_daily_volatility"].plot(ax=axes[1], color="tab:red")
        axes[1].set_title("Monthly daily volatility")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        Volatility clearly clusters in certain periods. A trading rule trained on calm periods may behave poorly in stress periods.
        """
    ),
]


unsup_cells = [
    md(
        """
        # 02 - Unsupervised Regimes

        Goal: group similar market days without using trading labels.
        """
    ),
    setup_cell,
    code(
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.metrics import silhouette_score

        from nvda_rl.features import CLUSTER_FEATURE_COLUMNS, describe_regimes, fit_unsupervised_train_test

        sns.set_theme(style="whitegrid", context="notebook")
        data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nvda_features.csv", parse_dates=["date"])
        """
    ),
    md(
        """
        ## How k-means works

        K-means tries to place days into groups where days inside one group look similar.
        """
    ),
    md(
        """
        It starts with cluster centers, assigns each day to the closest center, then moves centers and repeats.
        """
    ),
    md(
        """
        We scale features first because RSI, returns, and volume z-score use different units.
        """
    ),
    code(
        """
        split_date = "2021-01-01"
        train_enriched, test_enriched, regime_model, pca_model = fit_unsupervised_train_test(
            data,
            split_date=split_date,
            n_clusters=3,
        )
        enriched = pd.concat([train_enriched, test_enriched], ignore_index=True).sort_values("date").reset_index(drop=True)
        scaled_features = regime_model.named_steps["scaler"].transform(train_enriched[CLUSTER_FEATURE_COLUMNS])
        score = silhouette_score(scaled_features, train_enriched["regime"])
        print(f"Silhouette score: {score:.3f}")

        regime_summary = describe_regimes(enriched)
        regime_summary
        """
    ),
    md(
        """
        The models are fitted only on the training period before 2021. Test rows are transformed with the saved scaler, K-means centers, and PCA components.

        Regimes are interpreted with next-day return, not the same return used to create features. This avoids mechanically labeling clusters by today's return.
        """
    ),
    code(
        """
        regime_counts = enriched["regime"].value_counts().sort_index()
        regime_counts.plot(kind="bar", figsize=(7, 4), color="tab:blue")
        plt.title("Number of days in each regime")
        plt.xlabel("Regime")
        plt.ylabel("Days")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        Regime 0 has the most days, so most trading days are normal positive days. Regime 1 is less frequent but has much stronger average return.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        axes[0].plot(enriched["date"], enriched["adj_close"], color="black", linewidth=1)
        axes[0].scatter(enriched["date"], enriched["adj_close"], c=enriched["regime"], cmap="Set2", s=10, alpha=0.8)
        axes[0].set_title("NVDA price colored by regime")
        axes[0].set_ylabel("Adjusted close")

        axes[1].scatter(enriched["date"], enriched["return"], c=enriched["regime"], cmap="Set2", s=10, alpha=0.7)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Daily returns colored by regime")
        axes[1].set_ylabel("Daily return")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        The stress regime appears around large price disruptions and drawdowns. This is useful because the RL agent can treat those days differently.
        """
    ),
    md(
        """
        ## How PCA works

        PCA rotates the feature space into new axes that keep as much variation as possible.
        """
    ),
    md(
        """
        We use two PCA columns as compact market-state signals for RL.
        """
    ),
    code(
        """
        explained = pca_model.named_steps["pca"].explained_variance_ratio_
        print(f"PC1 explained variance: {explained[0]:.2%}")
        print(f"PC2 explained variance: {explained[1]:.2%}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=enriched, x="pca_1", y="pca_2", hue="regime", palette="Set2", s=25, alpha=0.75)
        plt.title("PCA market-state map")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        The PCA chart shows that regimes occupy different areas of the feature map, so the clusters are not just random labels.
        """
    ),
    code(
        """
        transition = pd.crosstab(
            enriched["regime"].shift(1).rename("previous"),
            enriched["regime"].rename("current"),
            normalize="index",
        )
        transition
        """
    ),
    md(
        """
        The transition table shows whether regimes persist. Persistence matters because a regime that lasts more than one day is easier to trade.
        """
    ),
    code(
        """
        output_path = PROJECT_ROOT / "data" / "processed" / "nvda_regimes.csv"
        enriched.to_csv(output_path, index=False)
        enriched[["date", "return", "volatility_20d", "rsi_14", "regime", "pca_1", "pca_2"]].head()
        """
    ),
    md(
        """
        Output:

        `nvda_regimes.csv` is the bridge from unsupervised learning to the RL notebooks. The file keeps the chronological train/test boundary honest: unsupervised models learn from train and only transform test.
        """
    ),
]


rl_cells = [
    md(
        """
        # 03 - RL Trading Agents

        Goal: compare Q-learning, PPO, random policy, and buy-and-hold.
        """
    ),
    setup_cell,
    code(
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from nvda_rl.agents import QLearningAgent
        from nvda_rl.env import TradingEnvironment
        from nvda_rl.evaluation import (
            buy_and_hold_frame,
            performance_metrics,
            random_policy_actions,
            strategy_frame,
        )
        from nvda_rl.features import discretize_train_test
        from nvda_rl.ppo import (
            PPO_OBSERVATION_COLUMNS,
            LinearPPOAgent,
            load_scaler,
            predict_ppo_actions,
            save_scaler,
            scale_train_test,
            train_ppo_for_seconds,
        )

        sns.set_theme(style="whitegrid", context="notebook")
        enriched = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nvda_regimes.csv", parse_dates=["date"])
        MODEL_DIR = PROJECT_ROOT / "models"
        MODEL_DIR.mkdir(exist_ok=True)
        """
    ),
    md(
        """
        ## Environment idea

        The environment walks through NVDA one day at a time.
        """
    ),
    md(
        """
        Action choices are `-1` short, `0` flat, and `1` long.
        """
    ),
    md(
        """
        Reward is previous position times today's return minus transaction cost for changing position.
        """
    ),
    code(
        """
        split_date = "2021-01-01"
        train_raw = enriched[enriched["date"] < split_date].reset_index(drop=True)
        test_raw = enriched[enriched["date"] >= split_date].reset_index(drop=True)
        train, test, bin_edges = discretize_train_test(train_raw, test_raw)

        baseline_state_columns = ["momentum_bin", "vol_bin"]
        regime_state_columns = ["regime", "momentum_bin", "vol_bin"]
        regime_pca_state_columns = ["regime", "momentum_bin", "vol_bin", "pca_1_bin"]
        transaction_cost = 0.001

        print(f"Train rows: {len(train):,}")
        print(f"Test rows: {len(test):,}")
        train[regime_pca_state_columns].head()
        """
    ),
    md(
        """
        ## Q-learning idea

        Q-learning stores a table of expected rewards for every state-action pair.
        """
    ),
    md(
        """
        Because it uses a table, continuous features are converted into bins first.
        """
    ),
    code(
        """
        train_env = TradingEnvironment(train, state_columns=regime_pca_state_columns, transaction_cost=transaction_cost)
        q_agent = QLearningAgent(alpha=0.08, gamma=0.95, epsilon=0.30, epsilon_decay=0.992, min_epsilon=0.03)
        q_rewards = q_agent.train(train_env, episodes=500)

        regime_env = TradingEnvironment(train, state_columns=regime_state_columns, transaction_cost=transaction_cost)
        regime_agent = QLearningAgent(alpha=0.08, gamma=0.95, epsilon=0.30, epsilon_decay=0.992, min_epsilon=0.03)
        regime_rewards = regime_agent.train(regime_env, episodes=500)

        no_unsup_env = TradingEnvironment(train, state_columns=baseline_state_columns, transaction_cost=transaction_cost)
        no_unsup_agent = QLearningAgent(alpha=0.08, gamma=0.95, epsilon=0.30, epsilon_decay=0.992, min_epsilon=0.03)
        no_unsup_rewards = no_unsup_agent.train(no_unsup_env, episodes=500)

        q_agent.save(MODEL_DIR / "q_learning.pkl")
        regime_agent.save(MODEL_DIR / "q_regime_only.pkl")
        no_unsup_agent.save(MODEL_DIR / "q_no_unsup.pkl")

        pd.Series(q_rewards).rolling(20).mean().plot(figsize=(10, 4), color="tab:blue")
        plt.title("Q-learning training reward")
        plt.xlabel("Episode")
        plt.ylabel("20-episode rolling reward")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        The trained Q-learning models are saved before testing. This keeps training and evaluation separate.
        """
    ),
    md(
        """
        ## Load saved Q-learning models

        From this point, test results use models loaded from `models/`, not the in-memory training objects.
        """
    ),
    code(
        """
        loaded_q_agent = QLearningAgent.load(MODEL_DIR / "q_learning.pkl")
        loaded_regime_agent = QLearningAgent.load(MODEL_DIR / "q_regime_only.pkl")
        loaded_no_unsup_agent = QLearningAgent.load(MODEL_DIR / "q_no_unsup.pkl")

        q_actions = loaded_q_agent.predict(TradingEnvironment(test, regime_pca_state_columns, transaction_cost))
        regime_actions = loaded_regime_agent.predict(TradingEnvironment(test, regime_state_columns, transaction_cost))
        no_unsup_actions = loaded_no_unsup_agent.predict(TradingEnvironment(test, baseline_state_columns, transaction_cost))
        random_actions = random_policy_actions(len(test), random_state=7)

        q_results = strategy_frame(test, q_actions, transaction_cost, "q_learning")
        regime_results = strategy_frame(test, regime_actions, transaction_cost, "q_regime")
        no_unsup_results = strategy_frame(test, no_unsup_actions, transaction_cost, "q_no_unsup")
        random_results = strategy_frame(test, random_actions, transaction_cost, "random")
        hold_results = buy_and_hold_frame(test)

        comparison = (
            q_results[["date", "q_learning_return", "q_learning_equity", "action"]]
            .merge(regime_results[["date", "q_regime_return", "q_regime_equity"]], on="date")
            .merge(no_unsup_results[["date", "q_no_unsup_return", "q_no_unsup_equity"]], on="date")
            .merge(random_results[["date", "random_return", "random_equity"]], on="date")
            .merge(hold_results[["date", "buy_hold_return", "buy_hold_equity"]], on="date")
        )
        """
    ),
    md(
        """
        ## PPO idea

        PPO is a policy-gradient RL method. This project uses a lightweight linear PPO-style policy so it runs without heavy deep-learning installs.
        """
    ),
    md(
        """
        Here PPO receives scaled volatility, momentum, RSI, MACD, ATR, regime, and PCA features directly. Current-day return is not included as an observation feature.
        """
    ),
    code(
        """
        ppo_train, ppo_test, ppo_scaler = scale_train_test(train, test, PPO_OBSERVATION_COLUMNS)
        save_scaler(ppo_scaler, MODEL_DIR / "ppo_scaler.pkl")
        loaded_ppo_scaler = load_scaler(MODEL_DIR / "ppo_scaler.pkl")

        # For a full project run use [60, 300, 600].
        # For a quick classroom/demo run, change this to smaller values like [10, 30, 60].
        ppo_budgets_seconds = [60, 300, 600]

        ppo_rows = []
        ppo_actions = {}
        for seconds in ppo_budgets_seconds:
            model, episodes, elapsed, rewards = train_ppo_for_seconds(
                ppo_train,
                PPO_OBSERVATION_COLUMNS,
                seconds=seconds,
                transaction_cost=transaction_cost,
            )
            model_path = MODEL_DIR / f"ppo_{seconds}s.pkl"
            model.save(model_path)

            loaded_model = LinearPPOAgent.load(model_path)
            actions = predict_ppo_actions(loaded_model, ppo_test, PPO_OBSERVATION_COLUMNS, transaction_cost)
            ppo_actions[seconds] = actions
            result = strategy_frame(test, actions, transaction_cost, "ppo")
            row_metrics = performance_metrics(result["ppo_return"], result["ppo_equity"], result["action"])
            ppo_rows.append({
                "seconds": seconds,
                "elapsed_seconds": elapsed,
                "episodes": episodes,
                "last_training_reward": rewards[-1] if rewards else np.nan,
                "model_path": str(model_path),
                **row_metrics,
            })

        ppo_metrics = pd.DataFrame(ppo_rows)
        ppo_metrics
        """
    ),
    md(
        """
        Each PPO-style model is saved and then loaded back before testing. The `model_path` column shows which saved file produced each row.
        """
    ),
    code(
        """
        metric_rows = {
            "q_learning": performance_metrics(comparison["q_learning_return"], comparison["q_learning_equity"], q_results["action"]),
            "q_regime_only": performance_metrics(comparison["q_regime_return"], comparison["q_regime_equity"], regime_results["action"]),
            "q_no_unsup": performance_metrics(comparison["q_no_unsup_return"], comparison["q_no_unsup_equity"], no_unsup_results["action"]),
            "buy_hold": performance_metrics(comparison["buy_hold_return"], comparison["buy_hold_equity"], pd.Series(np.ones(len(comparison)))),
            "random": performance_metrics(comparison["random_return"], comparison["random_equity"], pd.Series(random_actions)),
        }

        for seconds, actions in ppo_actions.items():
            result = strategy_frame(test, actions, transaction_cost, f"ppo_{seconds}s")
            metric_rows[f"ppo_{seconds}s"] = performance_metrics(
                result[f"ppo_{seconds}s_return"],
                result[f"ppo_{seconds}s_equity"],
                result["action"],
            )

        metrics = pd.DataFrame(metric_rows).T
        metrics[["cumulative_return", "annualized_return", "sharpe_ratio", "max_drawdown", "hit_ratio", "turnover"]]
        """
    ),
    md(
        """
        Buy-and-hold strongly wins in this test period. The best PPO-style saved model still lost about 77%, while buy-and-hold gained more than 1,400%.
        """
    ),
    code(
        """
        plt.figure(figsize=(12, 5))
        plt.plot(comparison["date"], comparison["q_learning_equity"], label="Q-learning")
        plt.plot(comparison["date"], comparison["q_regime_equity"], label="Q-learning regime only", alpha=0.8)
        plt.plot(comparison["date"], comparison["q_no_unsup_equity"], label="Q-learning no regimes/PCA", alpha=0.8)
        plt.plot(comparison["date"], comparison["buy_hold_equity"], label="Buy and hold")
        plt.plot(comparison["date"], comparison["random_equity"], label="Random", alpha=0.7)

        for seconds, actions in ppo_actions.items():
            ppo_result = strategy_frame(test, actions, transaction_cost, f"ppo_{seconds}s")
            plt.plot(ppo_result["date"], ppo_result[f"ppo_{seconds}s_equity"], label=f"PPO {seconds}s", alpha=0.85)

        plt.title("Out-of-sample equity curves")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        The equity curves show that the RL agents missed most of NVDA's bull-market upside. PPO traded less than random, but lower turnover did not make it profitable.
        """
    ),
    code(
        """
        action_by_regime = (
            test.assign(action=q_actions)
            .groupby("regime")["action"]
            .value_counts(normalize=True)
            .rename("share")
            .reset_index()
            .pivot(index="regime", columns="action", values="share")
            .fillna(0)
        )
        action_by_regime
        """
    ),
    md(
        """
        Q-learning shorted most often in every regime. It shorted least in the strong momentum regime, but the policy was still too bearish overall.
        """
    ),
    md(
        """
        Final note:

        This is a learning project. Real trading would need stronger validation, slippage modeling, and walk-forward testing.
        """
    ),
]


write_notebook("01_eda.ipynb", eda_cells)
write_notebook("02_unsupervised_regimes.ipynb", unsup_cells)
write_notebook("03_rl_trading_agent.ipynb", rl_cells)
