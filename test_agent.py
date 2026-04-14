import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation

from data_pipeline import build_technical_indicators, download_stock_data, train_test_split
from trading_env import ForexTradingEnv


def run_backtest(model, vec_env):
    reset_out = vec_env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, _ = reset_out
    else:
        obs = reset_out
    equity_curve = []
    trade_events = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        equity_curve.append(float(info.get("equity_usd", 0.0)))

        trade_info = info.get("last_trade_info")
        if isinstance(trade_info, dict):
            trade_events.append(trade_info)

        if done:
            break

    return equity_curve, trade_events


def calculate_metrics(equity_curve: list[float], trade_events: list[dict]) -> dict:
    series = pd.Series(equity_curve)
    returns = series.pct_change().dropna()
    sharpe_ratio = (
        returns.mean() / returns.std() * np.sqrt(252)
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )

    # ForexTradingEnv emits "CLOSE" events (not "SELL")
    close_events = [trade for trade in trade_events if trade.get("event") == "CLOSE"]
    wins = [trade for trade in close_events if trade.get("profit_pct", 0.0) > 0]
    win_rate = float(len(wins)) / len(close_events) * 100 if close_events else 0.0

    losses = [-trade.get("profit_pct", 0.0) for trade in close_events if trade.get("profit_pct", 0.0) < 0]
    profit_factor = (
        float(sum(t.get("profit_pct", 0.0) for t in close_events) / max(sum(losses), 1e-9))
        if losses else float("inf")
    )

    total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100 if len(series) > 1 else 0.0
    peak = series.cummax()
    max_drawdown = ((series - peak) / peak).min() * 100

    return {
        "total_return_pct": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "win_rate_pct": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(max_drawdown),
        "trade_count": len(close_events),
    }


def create_env(
    df,
    feature_cols,
    window_size: int = 14,
    allow_short: bool = False,
    commission_pct: float = 0.001,
    drawdown_penalty: float = 4.0,
):
    def make_env():
        env = ForexTradingEnv(
            df=df,
            feature_columns=feature_cols,
            window_size=window_size,
            allow_short=allow_short,
            commission_pct=commission_pct,
            drawdown_penalty=drawdown_penalty,
            random_start=False,
        )
        return FlattenObservation(env)

    return DummyVecEnv([make_env])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a trained stock trading agent.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="US stock ticker symbol")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--model-path", type=str, default="models/ppo_aapl_agent.zip", help="Path to the saved PPO model")
    parser.add_argument("--scaler-path", type=str, default="models/preprocessor_aapl.pkl", help="Path to the saved preprocessor")
    return parser.parse_args()


def plot_equity_curve(equity_curve: list[float], trade_events: list[dict], title: str):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Agent Equity")

    # ForexTradingEnv emits "OPEN" and "CLOSE" events (not "BUY"/"SELL")
    opens = [trade for trade in trade_events if trade.get("event") == "OPEN"]
    closes = [trade for trade in trade_events if trade.get("event") == "CLOSE"]

    for trade in opens:
        step = trade["step"]
        if step < len(equity_curve):
            plt.scatter(step, equity_curve[step], marker="^", color="green",
                        label="Open" if trade == opens[0] else "")
    for trade in closes:
        step = trade["step"]
        if step < len(equity_curve):
            plt.scatter(step, equity_curve[step], marker="v", color="red",
                        label="Close" if trade == closes[0] else "")

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)
    scaler_data = joblib.load(args.scaler_path)
    feature_cols = scaler_data["feature_cols"]
    scaler = scaler_data["scaler"]  # FIX: was never loaded before

    df = download_stock_data(args.ticker, args.start, args.end)
    df, _ = build_technical_indicators(df)
    _, test_df = train_test_split(df, split_ratio=0.8)

    # FIX: apply the scaler to features before creating the env.
    # The model was trained on scaled observations — feeding raw data
    # produces a completely different input distribution → random-looking output.
    test_df = test_df.copy()
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].astype(float).values)

    env = create_env(test_df, feature_cols, window_size=14, allow_short=False)
    equity_curve, trade_events = run_backtest(model, env)
    metrics = calculate_metrics(equity_curve, trade_events)

    start_price = test_df.iloc[0]["Close"]
    end_price = test_df.iloc[-1]["Close"]
    buy_hold_pct = float((end_price / start_price - 1) * 100)

    print("Backtest results")
    print(metrics)
    print(f"Buy & Hold return: {buy_hold_pct:.2f}%")

    out_csv = "stock_trade_history.csv"
    if trade_events:
        pd.DataFrame(trade_events).to_csv(out_csv, index=False)
        print(f"Saved trade history to {out_csv}")

    plot_equity_curve(equity_curve, trade_events, f"Agent Equity Curve ({args.ticker})")


if __name__ == "__main__":
    main()