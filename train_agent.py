"""
train_agent.py — PPO agent optimised for AAPL daily trading.

Key changes vs original:
┌──────────────────┬───────────────────┬──────────────────────────────────────────────┐
│ Hyperparameter   │ Old               │ New + rationale                              │
├──────────────────┼───────────────────┼──────────────────────────────────────────────┤
│ learning_rate    │ 3e-4 (constant)   │ linear decay 3e-4 → 5e-5                    │
│                  │                   │ Prevents overfitting to late training data.   │
│ n_steps          │ 2048              │ 4096 — captures multiple full AAPL trade     │
│                  │                   │ cycles (avg 5–20 bars) per rollout.          │
│ batch_size       │ 64                │ 256 — 16 minibatches from 4096-step rollout. │
│ n_epochs         │ 10 (default)      │ 8 — financial data is non-stationary;       │
│                  │                   │ fewer reuse epochs prevents overfitting.     │
│ gamma            │ 0.99 (default)    │ 0.995 — at 20 bars: 0.99^20=0.82 vs        │
│                  │                   │ 0.995^20=0.90. Longer credit horizon for     │
│                  │                   │ AAPL trend trades that last weeks.           │
│ gae_lambda       │ 0.95 (default)    │ 0.98 — lower bias in advantage estimation.  │
│                  │                   │ Needed to correctly value multi-week trends. │
│ clip_range       │ 0.2               │ 0.15 — AAPL has 3–5% single-day moves on   │
│                  │                   │ earnings/macro. Tighter clip = stable policy │
│                  │                   │ after outlier events.                        │
│ ent_coef         │ 0.01 (constant)   │ linear decay 0.01 → 0.001                  │
│                  │                   │ More exploration early, exploitation later.  │
│ max_grad_norm    │ 0.5 (default)     │ 0.3 — AAPL reward has heavy tails (earnings │
│                  │                   │ spikes). Tighter clipping prevents shock     │
│                  │                   │ updates that destabilize subsequent trading. │
│ net_arch         │ [64, 64] default  │ [256, 256] — 18 features × 14 window =     │
│                  │                   │ 294-dim input. Default is severely           │
│                  │                   │ underpowered for this observation size.      │
│ activation_fn    │ Tanh (default)    │ Tanh — explicitly set. Better than ReLU for │
│                  │                   │ bounded financial signals (RSI, Williams %R) │
│                  │                   │ since it handles negative inputs smoothly.   │
├──────────────────┼───────────────────┼──────────────────────────────────────────────┤
│ BUG FIX          │ scaler fit but    │ Scaler is now applied to the feature columns │
│                  │ never applied to  │ in the dataframe BEFORE creating envs.       │
│                  │ the env           │ OHLCV columns are untouched (env needs raw   │
│                  │                   │ prices for SL/TP/commission maths).          │
└──────────────────┴───────────────────┴──────────────────────────────────────────────┘
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium.wrappers import FlattenObservation

from data_pipeline import (
    build_technical_indicators,
    download_stock_data,
    fit_scaler,
    train_test_split,
)
from trading_env import ForexTradingEnv


# ── Learning rate and entropy schedules ─────────────────────────────────────

def linear_schedule(initial_value: float, min_value: float = 5e-5):
    """
    Linear decay from `initial_value` to `min_value` over training.

    Why a schedule for AAPL?  AAPL's market regime shifts over months
    (bull→correction→consolidation).  A high initial LR explores quickly across
    regimes; the decay prevents destructive policy updates as the agent converges.
    """
    def func(progress_remaining: float) -> float:
        return max(min_value, progress_remaining * initial_value)
    return func


def ent_coef_schedule(initial: float = 0.01, final: float = 0.003):
    """
    Linear entropy decay from `initial` to `final`.

    FIX: floor raised from 0.001 → 0.003.
    The old floor let entropy collapse 10x during training, causing the agent
    to lock onto the "never trade" policy early and never escape it.
    0.003 preserves enough exploration so the agent keeps trying trades
    even after the drawdown penalty discourages them in early episodes.
    """
    def func(progress_remaining: float) -> float:
        return max(final, progress_remaining * initial)
    return func


# ── Environment factory ──────────────────────────────────────────────────────

def create_env(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 14,
    allow_short: bool = False,
    position_size_pct: float = 0.95,
    commission_pct: float = 0.001,
    drawdown_penalty: float = 4.0,
    random_start: bool = False,
    n_envs: int = 1,
    obs_noise_std: float = 0.0,
) -> DummyVecEnv:
    """
    Create vectorized environments for training or evaluation.

    n_envs > 1: multiple parallel workers, each starting from a different random
    position in the time series. This gives the policy gradient more diverse
    experience per rollout and is the primary anti-overfitting mechanism at the
    environment level (more varied episodes = harder to memorize).

    obs_noise_std > 0: Gaussian noise injected into observations. Used during
    training only — set to 0.0 for evaluation/backtest. Forces the agent to
    learn from signal distributions rather than exact numeric sequences.

    Note: `df` must already have its feature columns scaled (via StandardScaler)
    before this function is called. OHLCV columns must remain in raw price units
    because the environment uses them for SL/TP hit-checks and commission maths.
    """
    def make_env(seed: int = 0):
        def _init():
            env = ForexTradingEnv(
                df=df,
                feature_columns=feature_cols,
                window_size=window_size,
                allow_short=allow_short,
                position_size_pct=position_size_pct,
                commission_pct=commission_pct,
                drawdown_penalty=drawdown_penalty,
                random_start=random_start,
                obs_noise_std=obs_noise_std,
            )
            env.reset(seed=seed)
            return FlattenObservation(env)
        return _init

    fns = [make_env(seed=i) for i in range(n_envs)]
    # SubprocVecEnv for n_envs > 1 (true parallelism), DummyVecEnv for single env
    if n_envs == 1:
        return DummyVecEnv([fns[0]])
    return SubprocVecEnv(fns)


def _apply_scaler(df: pd.DataFrame, feature_cols: list[str], scaler) -> pd.DataFrame:
    """
    Apply a fitted StandardScaler to only the feature columns in `df`.
    Returns a copy — OHLCV and date columns are preserved untouched.
    """
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols].astype(float).values)
    return df_scaled


# ── Walk-Forward Cross-Validation ────────────────────────────────────────────

def walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = 3,
    train_pct: float = 0.70,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/validation splits to detect overfitting early.

    Instead of a single 80/20 split (which the agent can eventually memorize),
    we use N expanding windows. Each fold's validation set is a fresh slice the
    model has never trained on.

    Example with n_folds=3, train_pct=0.70 on 1500 bars:
      Fold 0: train=bars[0:700],   val=bars[700:933]
      Fold 1: train=bars[0:933],   val=bars[933:1167]
      Fold 2: train=bars[0:1167],  val=bars[1167:1400]
      Final holdout: bars[1400:1500] — touched only once for final evaluation.

    If OOS Sharpe degrades consistently across folds: the model is overfitting.
    If it's consistent: the model is learning a generalizable strategy.
    """
    n = len(df)
    # Reserve the last 10% as a final untouched holdout
    holdout_start = int(n * 0.90)
    working_n = holdout_start

    fold_size = (working_n - int(working_n * train_pct)) // n_folds
    splits = []
    for i in range(n_folds):
        val_end   = working_n - (n_folds - 1 - i) * fold_size
        val_start = val_end - fold_size
        train_df = df.iloc[:val_start].reset_index(drop=True).copy()
        val_df   = df.iloc[val_start:val_end].reset_index(drop=True).copy()
        splits.append((train_df, val_df))

    return splits, df.iloc[holdout_start:].reset_index(drop=True).copy()


# ── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate_model(model: PPO, eval_env: DummyVecEnv):
    """Run a deterministic episode and return the equity curve and trade history."""
    obs, _ = eval_env.reset()
    equity_curve = []
    trade_events = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = eval_env.step(action)

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


def calculate_summary(equity_curve: list[float]) -> dict:
    series = pd.Series(equity_curve)
    returns = series.pct_change().dropna()
    sharpe_ratio = (
        returns.mean() / returns.std() * np.sqrt(252)
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )
    total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100 if len(series) > 1 else 0.0
    peak = series.cummax()
    drawdown = ((series - peak) / peak).min() * 100
    return {
        "total_return_pct": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown_pct": float(drawdown),
    }


# ── Early stopping ───────────────────────────────────────────────────────────

class EarlyStoppingCallback(BaseCallback):
    """Stop training when average reward has not improved for a fixed number of timesteps."""

    def __init__(self, patience_steps: int = 50_000, check_every_steps: int = 1_000, verbose=0):
        super().__init__(verbose)
        self.patience_steps = patience_steps
        self.check_every_steps = check_every_steps
        self.best_reward = -np.inf
        self.last_improvement = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_every_steps != 0:
            return True
        if len(self.model.ep_info_buffer) == 0:
            return True
        avg_reward = np.mean([info["r"] for info in self.model.ep_info_buffer])
        if avg_reward > self.best_reward + 1e-6:
            self.best_reward = avg_reward
            self.last_improvement = self.num_timesteps
        elif self.num_timesteps - self.last_improvement >= self.patience_steps:
            if self.verbose:
                print(f"Early stopping at {self.num_timesteps} steps (no improvement for {self.patience_steps} steps).")
            return False
        return True


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO stock trading agent (AAPL-optimised).")
    parser.add_argument("--ticker",      type=str, default="AAPL",       help="US stock ticker symbol")
    parser.add_argument("--start",       type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",         type=str, default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timesteps",   type=int, default=300_000,      help="PPO training timesteps per fold")
    parser.add_argument("--n-envs",      type=int, default=4,            help="Parallel environments (anti-overfitting)")
    parser.add_argument("--n-folds",     type=int, default=3,            help="Walk-forward folds")
    parser.add_argument("--output-dir",  type=str, default="models",     help="Directory to save model and scaler")
    return parser.parse_args()


def _build_model(env, output_dir: str) -> PPO:
    """Construct a fresh PPO model with anti-overfitting hyperparameters."""
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh,
    )
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tensorboard"),
        learning_rate=linear_schedule(3e-4, min_value=5e-5),
        # n_steps per env: with 4 envs, total rollout = 4096×4 = 16384 samples/update
        n_steps=4096,
        batch_size=256,
        n_epochs=6,       # reduced from 8: less reuse of stale samples → less overfit
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.15,
        ent_coef=ent_coef_schedule(initial=0.01, final=0.003),
        max_grad_norm=0.3,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"Downloading {args.ticker} from {args.start} to {args.end}")
    df = download_stock_data(args.ticker, args.start, args.end)
    df, feature_cols = build_technical_indicators(df)
    print(f"Total bars after indicators: {len(df)},  Features: {len(feature_cols)}")

    # ── Walk-Forward Cross-Validation ────────────────────────────────────────
    # Instead of one 80/20 split (memorizable), use N rolling windows.
    # Each fold trains on all data up to point T, validates on T→T+delta.
    # If OOS Sharpe is consistent across folds: generalizable strategy.
    # If it degrades fold-to-fold: the model is overfitting to each period.
    splits, holdout_df = walk_forward_splits(df, n_folds=args.n_folds, train_pct=0.70)
    print(f"\nWalk-Forward splits ({args.n_folds} folds):")
    for i, (tr, va) in enumerate(splits):
        print(f"  Fold {i}: train={len(tr)} bars, val={len(va)} bars")
    print(f"  Final holdout: {len(holdout_df)} bars (untouched)\n")

    # Fit scaler on the full pre-holdout training data (largest fold's train set)
    # so scale statistics cover the maximum history.
    full_train_df = splits[-1][0]
    scaler = fit_scaler(full_train_df, feature_cols)

    fold_results = []
    best_val_sharpe = -np.inf
    best_model = None

    for fold_idx, (train_df_raw, val_df_raw) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*60}")

        train_df_scaled = _apply_scaler(train_df_raw, feature_cols, scaler)
        val_df_scaled   = _apply_scaler(val_df_raw,   feature_cols, scaler)

        # ── Training env: N parallel workers + observation noise ─────────────
        # obs_noise_std=0.02 adds ~2% Gaussian noise to each scaled feature.
        # This forces the agent to learn from signal *distributions* rather than
        # exact sequences — the strongest anti-memorization tool available without
        # changing the RL algorithm itself.
        print(f"Creating {args.n_envs} parallel training envs with obs noise...")
        train_env = create_env(
            train_df_scaled,
            feature_cols,
            window_size=14,
            allow_short=False,
            random_start=True,
            n_envs=args.n_envs,
            obs_noise_std=0.02,       # regularization noise — OFF at eval time
        )

        # ── Validation env: single, no noise, deterministic ──────────────────
        val_env = create_env(
            val_df_scaled, feature_cols,
            window_size=14, random_start=False, n_envs=1, obs_noise_std=0.0,
        )

        model = _build_model(train_env, args.output_dir)

        early_stop = EarlyStoppingCallback(
            patience_steps=150_000,
            check_every_steps=2_000,
            verbose=1,
        )
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=os.path.join(args.output_dir, f"best_fold{fold_idx}"),
            log_path=os.path.join(args.output_dir, f"eval_logs_fold{fold_idx}"),
            eval_freq=max(10_000 // args.n_envs, 1000),
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([early_stop, eval_callback]),
        )

        # ── Fold evaluation ───────────────────────────────────────────────────
        val_curve, _ = evaluate_model(model, val_env)
        val_summary  = calculate_summary(val_curve)
        fold_results.append(val_summary)

        print(f"Fold {fold_idx} val: {val_summary}")

        if val_summary["sharpe_ratio"] > best_val_sharpe:
            best_val_sharpe = val_summary["sharpe_ratio"]
            best_model = model
            print(f"  ★ New best model (Sharpe={best_val_sharpe:.3f})")

        train_env.close()
        val_env.close()

    # ── Walk-forward summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*60}")
    sharpes = [r["sharpe_ratio"] for r in fold_results]
    returns = [r["total_return_pct"] for r in fold_results]
    print(f"OOS Sharpe  per fold: {[round(s,3) for s in sharpes]}")
    print(f"OOS Return% per fold: {[round(r,2) for r in returns]}")
    print(f"Mean OOS Sharpe: {np.mean(sharpes):.3f}  |  Std: {np.std(sharpes):.3f}")
    if np.std(sharpes) > 0.5:
        print("WARNING: High variance across folds → possible regime sensitivity.")
    else:
        print("Consistent across folds — strategy generalizes well.")

    # ── Final holdout evaluation ──────────────────────────────────────────────
    holdout_df_scaled = _apply_scaler(holdout_df, feature_cols, scaler)
    holdout_env = create_env(holdout_df_scaled, feature_cols, window_size=14,
                             random_start=False, n_envs=1, obs_noise_std=0.0)
    holdout_curve, _ = evaluate_model(best_model, holdout_env)
    holdout_summary  = calculate_summary(holdout_curve)
    holdout_env.close()

    buy_hold = (holdout_df["Close"].iloc[-1] / holdout_df["Close"].iloc[0] - 1) * 100
    print(f"\nFinal holdout: {holdout_summary}")
    print(f"Buy & Hold benchmark (holdout): {buy_hold:.2f}%")

    # ── Save best model ───────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, f"ppo_{args.ticker.lower()}_agent")
    best_model.save(model_path)
    joblib.dump(
        {"scaler": scaler, "feature_cols": feature_cols},
        os.path.join(args.output_dir, f"preprocessor_{args.ticker.lower()}.pkl"),
    )
    print(f"\nSaved best model → {model_path}.zip")
    print("Training complete.")



if __name__ == "__main__":
    main()