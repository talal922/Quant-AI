"""
app.py — Unified Quant AI Trading Dashboard
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation

from data_pipeline import (
    build_technical_indicators,
    download_stock_data,
    fit_scaler,
    train_test_split,
)
from trading_env import ForexTradingEnv

# ── Import Live Trading module (must NOT call set_page_config) ────────────────
import Live_Trading as _lt

# ── 1. Page Configuration & CSS (Must be at the top) ─────────────────────────
st.set_page_config(
    page_title="Quant AI Trading System",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

try:
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── 2. Navigation State Management ───────────────────────────────────────────
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

def change_page(page_name):
    st.session_state.current_page = page_name

# ── 3. Core RL Classes & Helper Functions ────────────────────────────────────
class StreamlitCallback(BaseCallback):
    def __init__(self, total_timesteps: int, progress_bar, status_text, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.rewards_history = []
        self.timesteps_history = []

    def _on_step(self) -> bool:
        if self.num_timesteps % 500 == 0:
            progress = min(1.0, self.num_timesteps / self.total_timesteps)
            self.progress_bar.progress(progress)
            self.status_text.text(f"⏳ Training: {self.num_timesteps}/{self.total_timesteps} steps")
            if len(self.model.ep_info_buffer) > 0:
                avg_reward = np.mean([info['r'] for info in self.model.ep_info_buffer])
                self.rewards_history.append(avg_reward)
                self.timesteps_history.append(self.num_timesteps)
        return True

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience_steps: int = 50000, check_every_steps: int = 1000, verbose=0):
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

        avg_reward = np.mean([info['r'] for info in self.model.ep_info_buffer])
        if avg_reward > self.best_reward + 1e-6:
            self.best_reward = avg_reward
            self.last_improvement = self.num_timesteps
        elif self.num_timesteps - self.last_improvement >= self.patience_steps:
            return False
        return True

@st.cache_data
def load_stock_data(ticker: str, start_date, end_date, interval: str = "1d"):
    df = download_stock_data(ticker, start_date.isoformat(), end_date.isoformat(), interval=interval)
    df, feature_cols = build_technical_indicators(df)
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df, feature_cols

def create_env(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler,
    window_size: int,
    allow_short: bool,
    position_size_pct: float = 0.95,
    commission_pct: float = 0.001,
    drawdown_penalty: float = 1.5,
    random_start: bool = False,
):
    def make_env():
        env = ForexTradingEnv(
            df=df,
            feature_columns=feature_cols,
            window_size=window_size,
            allow_short=allow_short,
            position_size_pct=position_size_pct,
            commission_pct=commission_pct,
            drawdown_penalty=drawdown_penalty,
            random_start=random_start,
        )
        return FlattenObservation(env)

    return DummyVecEnv([make_env])

def evaluate_model(model: PPO, vec_env: DummyVecEnv):
    reset_out = vec_env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, _ = reset_out
    else:
        obs = reset_out
    equity_curve = []
    trade_events = []
    trade_returns = []

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
            if trade_info.get("event") in {"CLOSE", "SELL"}:
                if "profit_pct" in trade_info:
                    trade_returns.append(float(trade_info["profit_pct"]))
                elif "net_profit_pct" in trade_info:
                    trade_returns.append(float(trade_info["net_profit_pct"]))
                elif "profit_usd" in trade_info and "entry_investment" in trade_info:
                    entry_investment = float(trade_info.get("entry_investment", 10000.0))
                    trade_returns.append(float(trade_info["profit_usd"]) / max(entry_investment, 1e-9))
                elif "net_profit_usd" in trade_info:
                    trade_returns.append(float(trade_info["net_profit_usd"]) / 10000.0)

        if done:
            break

    if trade_returns and max(abs(r) for r in trade_returns) < 0.001:
        trade_returns = [float(r) * 100.0 for r in trade_returns]

    return equity_curve, trade_events, trade_returns

def summarize_equity_curve(equity_curve: list[float]):
    series = pd.Series(equity_curve)
    returns = series.pct_change().dropna()
    total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100 if len(series) > 1 else 0.0
    sharpe_ratio = (
        returns.mean() / returns.std() * np.sqrt(252)
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )
    max_drawdown = ((series - series.cummax()) / series.cummax()).min() * 100
    return total_return, sharpe_ratio, max_drawdown

def calculate_sortino_ratio(equity_curve: list[float]) -> float:
    returns = pd.Series(equity_curve).pct_change().dropna()
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(252))

def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    if abs(max_drawdown) < 1e-9:
        return float("inf") if total_return > 0 else 0.0
    return float(total_return / abs(max_drawdown))

def evaluate_segment_metrics(
    model: PPO,
    segment_df: pd.DataFrame,
    feature_cols: list[str],
    scaler,
    window_size: int,
    allow_short: bool,
    position_size_pct: float,
    commission_pct: float,
    drawdown_penalty: float,
) -> tuple[float, float, float]:
    env = create_env(
        segment_df,
        feature_cols,
        scaler,
        window_size=window_size,
        allow_short=allow_short,
        position_size_pct=position_size_pct,
        commission_pct=commission_pct,
        drawdown_penalty=drawdown_penalty,
        random_start=False,
    )
    equity_curve, _, _ = evaluate_model(model, env)
    return summarize_equity_curve(equity_curve)

def run_monte_carlo_stress_test(trade_returns: list[float], n_sims: int = 100, seed: int = 42) -> tuple[float, float, list[float]]:
    if trade_returns and isinstance(trade_returns[0], dict):
        extracted_returns = []
        for trade in trade_returns:
            if "profit_pct" in trade:
                extracted_returns.append(float(trade["profit_pct"]))
            elif "net_profit_pct" in trade:
                extracted_returns.append(float(trade["net_profit_pct"]))
            elif "profit_usd" in trade and "entry_investment" in trade:
                entry_investment = float(trade.get("entry_investment", 10000.0))
                extracted_returns.append(float(trade["profit_usd"]) / max(entry_investment, 1e-9))
            elif "net_pips" in trade:
                extracted_returns.append(float(trade["net_pips"]) / 10000.0)
            elif "net_profit_usd" in trade:
                extracted_returns.append(float(trade["net_profit_usd"]) / 10000.0)
        trade_returns = extracted_returns

    trade_returns = [float(r) for r in trade_returns if not np.isnan(float(r))]
    if len(trade_returns) == 0:
        return 0.0, 0.0, []

    rng = np.random.default_rng(seed)
    final_equities = []
    drawdowns = []
    for _ in range(n_sims):
        permuted = rng.choice(trade_returns, size=len(trade_returns), replace=True)
        equity = 1.0
        equity_curve = [equity]
        for r in permuted:
            equity *= 1.0 + r
            equity_curve.append(equity)
        final_equities.append(equity)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = np.min((equity_curve - peak) / peak) * 100 if len(peak) > 0 else 0.0
        drawdowns.append(abs(drawdown))

    profit_probability = float(sum(1 for eq in final_equities if eq > 1.0) / len(final_equities) * 100)
    expected_drawdown = float(np.mean(drawdowns))
    return profit_probability, expected_drawdown, final_equities


# ══════════════════════════════════════════════════════════════════════════════
# ── 4. View: HOME (Landing Page) ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.current_page == "Home":
    st.markdown("""
    <div class="home-hero">
        <div class="home-hero-badge">INSTITUTIONAL GRADE</div>
        <h1 class="home-hero-title">Quant AI<br><span class="home-hero-accent">Trading System</span></h1>
        <p class="home-hero-subtitle">
            PPO Reinforcement Learning · Live Paper Trading · Bloomberg-Style Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="nav-card nav-card--train">
            <div class="nav-card__icon">🧠</div>
            <div class="nav-card__label">PAGE 1</div>
            <div class="nav-card__title">Train Agent</div>
            <div class="nav-card__desc">
                Configure hyperparameters, train a PPO agent on any US stock,
                run out-of-sample backtests, Monte Carlo stress tests, and
                download the trained model for deployment.
            </div>
            <ul class="nav-card__features">
                <li>✦ PPO training with early stopping</li>
                <li>✦ In-sample vs OOS equity curves</li>
                <li>✦ Sharpe / Sortino / Calmar ratios</li>
                <li>✦ Monte Carlo stress testing</li>
                <li>✦ Segment consistency check</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="nav-card nav-card--live">
            <div class="nav-card__icon">⚡</div>
            <div class="nav-card__label">PAGE 2</div>
            <div class="nav-card__title">Live Trading</div>
            <div class="nav-card__desc">
                Connect your Alpaca Paper Trading account, upload your trained
                model and scaler, and let the AI agent analyse live market data
                to generate BUY / SELL / HOLD signals in real-time.
            </div>
            <ul class="nav-card__features">
                <li>✦ Alpaca Paper Trading integration</li>
                <li>✦ Real-time price &amp; VIX Fear Index</li>
                <li>✦ AI signal with glow effects</li>
                <li>✦ Account balance &amp; positions</li>
                <li>✦ One-click order execution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown('<div class="status-chip status-chip--green">● System Online</div>', unsafe_allow_html=True)
    s2.markdown('<div class="status-chip status-chip--blue">● SB3 PPO Ready</div>', unsafe_allow_html=True)
    s3.markdown('<div class="status-chip status-chip--amber">● Paper Trading Mode</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, btn_col1, btn_col2, _ = st.columns([1, 2, 2, 1])

    with btn_col1:
        if st.button("🧠 GO TO TRAINING DASHBOARD", type="primary", use_container_width=True):
            change_page("Train Agent")
            st.rerun()

    with btn_col2:
        if st.button("⚡ GO TO LIVE TRADING", type="primary", use_container_width=True):
            change_page("Live_Trading.py")
            st.rerun()

    st.markdown("""
    <div style="margin-top:2.5rem;padding:1rem 1.5rem;background:rgba(0,170,255,0.04);
         border:1px solid rgba(0,170,255,0.12);border-radius:10px;
         font-family:'DM Sans',sans-serif;font-size:0.83rem;color:#4A5E7A;line-height:1.7;">
        <strong style="color:#6B80A0;">⚠ Disclaimer:</strong>
        This dashboard is for <strong style="color:#8A9BB5;">educational and paper-trading purposes only</strong>.
        Past performance is not indicative of future results. Never risk capital you cannot afford to lose.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── 5. View: TRAIN AGENT ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.current_page == "Train Agent":

    if st.button("⬅️ Back to Home", key="back_btn"):
        change_page("Home")
        st.rerun()

    st.title("Quantitative Stock Trading AI")

    if 'model_history' not in st.session_state:
        st.session_state.model_history = []
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None

    with st.sidebar:
        st.markdown(
            '<p style="font-size:2rem;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;color:#4A5E7A;margin-bottom:0.35rem;">Market Data</p>',
            unsafe_allow_html=True,
        )
        ticker = st.text_input("Ticker", value="AAPL").upper()
        start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("today"))
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

        st.markdown(
            '<hr style="border:none;border-top:1px solid #1F2D42;margin:1rem 0;">'
            '<p style="font-size:1rem;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;color:#4A5E7A;margin-bottom:0.35rem;">Model Parameters</p>',
            unsafe_allow_html=True,
        )
        timesteps = st.number_input("Timesteps", min_value=10000, max_value=500000, value=100000, step=10000)
        position_pct = st.slider("Position Size (%)", min_value=10, max_value=100, value=95, step=5) / 100.0
        commission_pct = st.slider("Commission (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100.0
        drawdown_penalty = st.slider("Drawdown Penalty", min_value=0.0, max_value=10.0, value=1.5, step=0.5)
        window_size = st.slider("Window Size", min_value=10, max_value=60, value=14, step=1)
        allow_short = st.checkbox("Allow Short Positions", value=False)

    st.subheader(f"Market Overview: {ticker}")
    if st.button("Fetch Price Data", type="primary", use_container_width=True):
        st.rerun()

    try:
        df, feature_cols = load_stock_data(ticker, start_date, end_date, interval)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        df, feature_cols = None, None

    if df is not None and not df.empty:
        st.write(f"Loaded {len(df)} bars for {ticker}.")
        st.line_chart(df.set_index(df.columns[0])[['Close']])

        with st.expander("View sample data"):
            st.dataframe(df.head())

        train_df, test_df = train_test_split(df, split_ratio=0.8)
        st.markdown("---")
        st.header("AI Training & Backtest")

        if st.button("🚀 Train Agent", type="primary", use_container_width=True):
            scaler = fit_scaler(train_df, feature_cols)
            
            # --- التعديل 1: تطبيق الـ Scaler قبل إنشاء البيئات ---
            train_df_scaled = train_df.copy()
            train_df_scaled[feature_cols] = scaler.transform(train_df_scaled[feature_cols].astype(float).values)
            
            test_df_scaled = test_df.copy()
            test_df_scaled[feature_cols] = scaler.transform(test_df_scaled[feature_cols].astype(float).values)
            # ----------------------------------------------------

            train_env = create_env(
                train_df_scaled,
                feature_cols,
                scaler,
                window_size=window_size,
                allow_short=allow_short,
                position_size_pct=position_pct,
                commission_pct=commission_pct,
                drawdown_penalty=drawdown_penalty,
                random_start=True,
            )
            from stable_baselines3.common.utils import set_random_seed
            set_random_seed(42)

            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=0,
                device="cpu", # --- التعديل 2: إضافة cpu لمنع التحذير ---
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                ent_coef=0.01,
            )

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            callback = StreamlitCallback(total_timesteps=timesteps, progress_bar=progress_bar, status_text=status_text)
            early_stop = EarlyStoppingCallback(patience_steps=50000, check_every_steps=1000, verbose=0)
            eval_env = create_env(
                test_df_scaled,
                feature_cols,
                scaler,
                window_size=window_size,
                allow_short=allow_short,
                position_size_pct=position_pct,
                commission_pct=commission_pct,
                drawdown_penalty=drawdown_penalty,
                random_start=False,
            )
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path="./best_model/",
                eval_freq=5000,
                deterministic=True,
                verbose=0,
            )

            with st.spinner("Training PPO agent..."):
                model.learn(total_timesteps=timesteps, callback=CallbackList([callback, early_stop, eval_callback]))
                progress_bar.progress(1.0)
                status_text.text("Training completed.")

            if callback.rewards_history:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(callback.timesteps_history, callback.rewards_history, label="Average reward")
                ax.set_xlabel("Training steps")
                ax.set_ylabel("Reward")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            train_eval_env = create_env(
                train_df_scaled,
                feature_cols,
                scaler,
                window_size=window_size,
                allow_short=allow_short,
                position_size_pct=position_pct,
                commission_pct=commission_pct,
                drawdown_penalty=drawdown_penalty,
                random_start=False,
            )
            test_eval_env = create_env(
                test_df_scaled,
                feature_cols,
                scaler,
                window_size=window_size,
                allow_short=allow_short,
                position_size_pct=position_pct,
                commission_pct=commission_pct,
                drawdown_penalty=drawdown_penalty,
                random_start=False,
            )

            train_curve, _, _ = evaluate_model(model, train_eval_env)
            test_curve, trades, trade_returns = evaluate_model(model, test_eval_env)

            train_return, train_sharpe, train_dd = summarize_equity_curve(train_curve)
            test_return, test_sharpe, test_dd = summarize_equity_curve(test_curve)
            buy_hold_return = ((test_df['Close'].iloc[-1] / test_df['Close'].iloc[0]) - 1) * 100

            st.session_state.training_results = {
                "ticker": ticker,
                "train_curve": train_curve,
                "test_curve": test_curve,
                "trades": trades,
                "trade_returns": trade_returns,
                "test_return": test_return,
                "test_sharpe": test_sharpe,
                "test_dd": test_dd,
                "train_return": train_return,
                "buy_hold_return": buy_hold_return,
                "trained_model_bytes": None,
                "trades_csv": None,
            }

            st.markdown("### Performance Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("OOS Return", f"{test_return:.2f}%")
            col2.metric("OOS Sharpe", f"{test_sharpe:.2f}")
            col3.metric("Buy & Hold", f"{buy_hold_return:.2f}%")

            st.markdown("### Equity Curves")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(train_curve, label="In-sample")
            ax.plot(test_curve, label="Out-of-sample")
            ax.set_ylabel("Equity ($)")
            ax.set_xlabel("Step")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            trade_returns = trade_returns if isinstance(trade_returns, list) else []
            trade_returns = [float(r) for r in trade_returns if not np.isnan(float(r))]
            if trade_returns:
                profit_probability, expected_drawdown, equity_simulations = run_monte_carlo_stress_test(trade_returns, n_sims=100)
                profit_probability = float(np.nan_to_num(profit_probability, nan=0.0))
                expected_drawdown = float(np.nan_to_num(expected_drawdown, nan=0.0))

                st.markdown("### Stress Test Result")
                st.markdown(f"**Success Probability:** {profit_probability:.1f}%")
                st.markdown(f"**Expected Drawdown:** {expected_drawdown:.1f}%")

                fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
                bins = 10 if len(set(equity_simulations)) > 1 else 1
                ax_mc.hist(equity_simulations, bins=bins, color="#4c78a8", edgecolor="#ffffff")
                ax_mc.set_xlim(min(equity_simulations) * 0.95, max(equity_simulations) * 1.05)
                ax_mc.set_title("Monte Carlo Final Equity Distribution")
                ax_mc.set_xlabel("Final Equity Multiplier")
                ax_mc.set_ylabel("Frequency")
                st.pyplot(fig_mc)

                if trades:
                    win_trades = [r for r in trade_returns if r > 0]
                    loss_trades = [r for r in trade_returns if r <= 0]
                    total_wins = len(win_trades)
                    total_losses = len(loss_trades)
                    total_trades = len(trade_returns)
                    win_rate_pct = float(total_wins / total_trades * 100) if total_trades > 0 else 0.0
                    gross_profit = sum(win_trades)
                    gross_loss = sum(-r for r in loss_trades)
                    profit_factor = float(gross_profit / max(gross_loss, 1e-9)) if gross_loss > 0 else float("inf")
                    avg_win_pct = float(np.mean(win_trades) * 100) if win_trades else 0.0
                    avg_loss_pct = float(np.mean(loss_trades) * 100) if loss_trades else 0.0

                    st.markdown("### Trade Evaluation Metrics")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    metric_col1.metric("Win Rate", f"{win_rate_pct:.2f}%")
                    metric_col2.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "inf")
                    metric_col3.metric("Avg Win %", f"{avg_win_pct:.2f}%")
                    metric_col4.metric("Avg Loss %", f"{avg_loss_pct:.2f}%")

                    fig_dd, ax_dd = plt.subplots(figsize=(10, 3))
                    equity_array = np.array(test_curve, dtype=float)
                    peak = np.maximum.accumulate(equity_array)
                    drawdown_curve = (equity_array - peak) / np.maximum(peak, 1e-9) * 100
                    ax_dd.fill_between(range(len(drawdown_curve)), drawdown_curve, color="#d62728", alpha=0.3)
                    ax_dd.plot(drawdown_curve, color="#d62728", linewidth=1.5)
                    ax_dd.set_title("Underwater Plot (Drawdown over Time)")
                    ax_dd.set_xlabel("Step")
                    ax_dd.set_ylabel("Drawdown (%)")
                    ax_dd.grid(True, alpha=0.3)
                    st.pyplot(fig_dd)

                    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                    ax_hist.hist(trade_returns, bins=20, color="#4c78a8", edgecolor="#ffffff")
                    ax_hist.axvline(0.0, color="red", linewidth=2)
                    ax_hist.set_title("Trade Return Distribution")
                    ax_hist.set_xlabel("Trade Return")
                    ax_hist.set_ylabel("Frequency")
                    st.pyplot(fig_hist)

                    accuracy_df = pd.DataFrame(
                        {
                            "Metric": ["Total Wins", "Total Losses", "Average Win %", "Average Loss %"],
                            "Value": [
                                total_wins,
                                total_losses,
                                f"{avg_win_pct:.2f}%",
                                f"{avg_loss_pct:.2f}%",
                            ],
                        }
                    )
                    st.markdown("### Trading Accuracy")
                    st.table(accuracy_df)
            else:
                st.markdown("### Stress Test Result")
                st.markdown("No trade-return data available for stress testing.")
                equity_simulations = []

            # --- التعديل 3: استخدام test_df_scaled بدلاً من test_df لتقييم الفترات ---
            segment_size = len(test_df_scaled) // 3
            early = test_df_scaled.iloc[:segment_size].reset_index(drop=True)
            mid = test_df_scaled.iloc[segment_size:2 * segment_size].reset_index(drop=True)
            late = test_df_scaled.iloc[2 * segment_size:].reset_index(drop=True)

            consistency_rows = []
            for name, segment_df in [("Early", early), ("Mid", mid), ("Late", late)]:
                if len(segment_df) == 0:
                    consistency_rows.append(
                        {
                            "Segment": name,
                            "Return (%)": 0.0,
                            "Sharpe Ratio": 0.0,
                            "Max Drawdown (%)": 0.0,
                        }
                    )
                    continue

                seg_return, seg_sharpe, seg_dd = evaluate_segment_metrics(
                    model,
                    segment_df,
                    feature_cols,
                    scaler,
                    window_size,
                    allow_short,
                    position_pct,
                    commission_pct,
                    drawdown_penalty,
                )
                seg_return = float(np.nan_to_num(seg_return, nan=0.0))
                seg_sharpe = float(np.nan_to_num(seg_sharpe, nan=0.0))
                seg_dd = float(np.nan_to_num(seg_dd, nan=0.0))
                consistency_rows.append(
                    {
                        "Segment": name,
                        "Return (%)": round(seg_return, 2),
                        "Sharpe Ratio": round(seg_sharpe, 2),
                        "Max Drawdown (%)": round(seg_dd, 2),
                    }
                )

            consistency_df = pd.DataFrame(consistency_rows)
            st.markdown("### Consistency Check")
            st.dataframe(consistency_df, use_container_width=True)

            segment_returns = np.array(consistency_df["Return (%)"].astype(float).tolist(), dtype=float)
            consistency_label = "CONSISTENT" if np.ptp(np.nan_to_num(segment_returns, nan=0.0)) < 25 else "VOLATILE/LUCKY"
            st.markdown(f"**Model Status:** {consistency_label}")

            sortino = calculate_sortino_ratio(test_curve)
            calmar = calculate_calmar_ratio(test_return, test_dd)
            st.markdown("### Risk Metrics Dashboard")
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Sortino Ratio", f"{sortino:.2f}")
            col_r2.metric("Calmar Ratio", f"{calmar:.2f}")

            st.session_state.model_history.append(
                {
                    "Ticker": ticker,
                    "Timesteps": timesteps,
                    "OOS Return (%)": round(test_return, 2),
                    "Sharpe": round(test_sharpe, 2),
                    "Max Drawdown (%)": round(test_dd, 2),
                    "Buy & Hold (%)": round(buy_hold_return, 2),
                }
            )
            if trades:
                trades_df = pd.DataFrame(trades)
                st.markdown("### Trades")
                st.dataframe(trades_df, use_container_width=True)

            model.save("trained_agent.zip")
            with open("./best_model/best_model.zip", "rb") as f:
                trained_model_bytes = f.read()

            import joblib
            import io
            scaler_buffer = io.BytesIO()
            joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, scaler_buffer)

            st.session_state.training_results["trained_model_bytes"] = trained_model_bytes
            st.session_state.training_results["scaler_bytes"] = scaler_buffer.getvalue()

            if trades:
                import io
                st.session_state.training_results["trades_csv"] = pd.DataFrame(trades).to_csv(index=False).encode('utf-8')

        if st.session_state.training_results is not None:
            res = st.session_state.training_results
            st.markdown("---")
            st.markdown("### Download Results")

            dl_col1, dl_col2, dl_col3 = st.columns(3)

            if res.get("trained_model_bytes"):
                dl_col1.download_button(
                    "⬇️ Download Trained Model",
                    res["trained_model_bytes"],
                    file_name="best_model.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

            if res.get("scaler_bytes"):
                dl_col2.download_button(
                    "⬇️ Download Scaler (.pkl)",
                    res["scaler_bytes"],
                    file_name="trained_scaler.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )

            if res.get("trades_csv"):
                dl_col3.download_button(
                    "⬇️ Download Trades",
                    res["trades_csv"],
                    file_name=f"{res.get('ticker', 'AAPL')}_trades.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        if st.session_state.model_history:
            st.markdown("---")
            st.subheader("Leaderboard")
            st.dataframe(pd.DataFrame(st.session_state.model_history), use_container_width=True)
            # ══════════════════════════════════════════════════════════════════
            # ── OOA TESTING ENTRY POINT ──
            # ══════════════════════════════════════════════════════════════════
            if st.session_state.training_results is not None and \
               st.session_state.training_results.get("trained_model_bytes") and \
               st.session_state.training_results.get("scaler_bytes"):

                st.markdown("---")
                st.markdown(
                    "### 🔬 Out-of-Asset Evaluation",
                    help="Test the model on a completely different stock "
                         "without any retraining.",
                )
                st.caption(
                    "The model trained above will be loaded in **inference mode** "
                    "and run on any ticker you choose below. No weights are updated."
                )
                if st.button(
                    "🧪 Test Model on New Data",
                    type="primary",
                    use_container_width=True,
                    key="goto_ooa_btn",
                ):
                    change_page("OOA_Testing")
                    st.rerun()
    else:
        st.warning("No data available for the selected ticker/date range. Adjust your inputs and try again.")


# ══════════════════════════════════════════════════════════════════════════════
# ── 6. View: LIVE TRADING
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.current_page == "Live_Trading.py":
    # Delegate entirely to the Live_Trading module.
    # CSS is already loaded above — the module does NOT call set_page_config().
    _lt.render_page(change_page)

# ══════════════════════════════════════════════════════════════════════════════
# ── 7. View: OUT-OF-ASSET TESTING ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.current_page == "OOA_Testing":
    import io
    import tempfile
    import joblib
    import plotly.graph_objects as go
    from data_loader import load_and_prepare_oos_data

    # ── Back navigation ───────────────────────────────────────────────────────
    if st.button("⬅️ Back to Training Dashboard", key="ooa_back_btn"):
        change_page("Train Agent")
        st.rerun()

    st.title("🔬 Out-of-Asset Model Evaluation")
    st.caption(
        "The model is loaded in **pure inference mode** — no weights are updated. "
        "Results show generalisation to a completely unseen asset."
    )
    st.markdown("---")

    # ── Guard and Model Selection ────────────────────────────────────────────
    res = st.session_state.get("training_results") or {}
    session_model_bytes  = res.get("trained_model_bytes")
    session_scaler_bytes = res.get("scaler_bytes")

    st.markdown("#### 📂 Model Selection")
    
    # خيار دائم لرفع مودل خارجي
    use_external = st.checkbox(
        "(Upload External Model)", 
        value=(not session_model_bytes), 
        help=" if you have a trained model and scaler saved locally, you can upload them here. "
    )

    model_bytes = session_model_bytes
    scaler_bytes = session_scaler_bytes

    if use_external:
        up_col1, up_col2 = st.columns(2)
        with up_col1:
            uploaded_model = st.file_uploader(
                "Upload Trained Model (.zip)",
                type=["zip"],
                key="ooa_model_upload",
            )
        with up_col2:
            uploaded_scaler = st.file_uploader(
                "Upload Scaler (.pkl)",
                type=["pkl"],
                key="ooa_scaler_upload",
            )

        if uploaded_model is not None and uploaded_scaler is not None:
            model_bytes = uploaded_model.read()
            scaler_bytes = uploaded_scaler.read()
            st.success("    files uploaded successfully. Configure the evaluation asset below.")
        else:
            st.warning("⬆ Please upload both files to proceed.")
            st.stop()
            
    elif not model_bytes or not scaler_bytes:
        st.warning("⬆ Please upload both files to proceed.")
        st.stop()

        st.markdown("---")

        # If still missing after upload widgets, stop here gracefully
        if not model_bytes or not scaler_bytes:
            st.warning("⬆ Upload both files above to proceed with evaluation.")
            st.stop()

        # Persist into session state so the rest of the page can use them
        if "training_results" not in st.session_state or st.session_state.training_results is None:
            st.session_state.training_results = {}
        st.session_state.training_results["trained_model_bytes"] = model_bytes
        st.session_state.training_results["scaler_bytes"] = scaler_bytes
        st.success(" Files uploaded successfully. Configure the evaluation asset below.")
        st.markdown("---")

    # ── Load model from bytes via temp file (SB3 requires a path) ────────────
    @st.cache_resource(show_spinner="Loading model weights…")
    def _load_model_from_bytes(model_bytes: bytes) -> PPO:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name
        return PPO.load(tmp_path)

    @st.cache_resource(show_spinner="Loading scaler…")
    def _load_scaler_from_bytes(scaler_bytes: bytes):
        return joblib.load(io.BytesIO(scaler_bytes))

    model         = _load_model_from_bytes(model_bytes)
    scaler_data   = _load_scaler_from_bytes(scaler_bytes)
    ooa_scaler    = scaler_data["scaler"]
    ooa_feat_cols = scaler_data["feature_cols"]

    trained_on = res.get("ticker") or "Uploaded Model"
    st.success(
        f" Model loaded — originally trained on **{trained_on}**. "
        f"Using **{len(ooa_feat_cols)}** feature columns."
    )

    # ── User inputs for the new (unseen) asset ────────────────────────────────
    st.subheader("Configure Evaluation Asset")
    col_a, col_b, col_c, col_d = st.columns([2, 1.5, 1.5, 1])

    with col_a:
        ooa_ticker = st.text_input(
            "New Ticker Symbol",
            value="MSFT",
            help="Any US stock different from the training ticker.",
        ).upper()
    with col_b:
        ooa_start = st.date_input(
            "Start Date",
            pd.to_datetime("2023-01-01"),
            key="ooa_start",
        )
    with col_c:
        ooa_end = st.date_input(
            "End Date",
            pd.to_datetime("today"),
            key="ooa_end",
        )
    with col_d:
        ooa_window = st.number_input(
            "Window Size",
            min_value=5,
            max_value=60,
            value=14,
            step=1,
            key="ooa_window",
        )

    col_e, col_f = st.columns(2)
    with col_e:
        ooa_commission = (
            st.slider(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key="ooa_commission",
            )
            / 100.0
        )
    with col_f:
        ooa_allow_short = st.checkbox(
            "Allow Short Positions",
            value=False,
            key="ooa_short",
        )

    st.markdown("---")
    run_eval = st.button(
        f"▶️ Run Inference on {ooa_ticker}",
        type="primary",
        use_container_width=True,
        key="ooa_run_btn",
    )

    if run_eval:
        # ── 1. Fetch & prepare data ───────────────────────────────────────────
        with st.spinner(f"Fetching data for {ooa_ticker}…"):
            try:
                ooa_df, _ = load_and_prepare_oos_data(
                    ticker=ooa_ticker,
                    start=ooa_start.isoformat(),
                    end=ooa_end.isoformat(),
                    scaler=ooa_scaler,
                    feature_cols=ooa_feat_cols,
                )
            except (ValueError, KeyError) as exc:
                st.error(f"Data error: {exc}")
                st.stop()

        st.info(
            f" {ooa_ticker}: **{len(ooa_df)}** bars loaded "
            f"({ooa_start} → {ooa_end})"
        )

        # ── 2. Build env & run inference (deterministic=True, no training) ────
        with st.spinner("Running model in inference mode…"):
            ooa_env = create_env(
                ooa_df,
                ooa_feat_cols,
                ooa_scaler,
                window_size=int(ooa_window),
                allow_short=ooa_allow_short,
                position_size_pct=0.95,
                commission_pct=ooa_commission,
                drawdown_penalty=1.5,
                random_start=False,
            )
            ooa_equity, ooa_trades, ooa_returns = evaluate_model(model, ooa_env)

        # ── 3. Performance metrics ────────────────────────────────────────────
        ooa_total_return, ooa_sharpe, ooa_max_dd = summarize_equity_curve(ooa_equity)
        ooa_sortino  = calculate_sortino_ratio(ooa_equity)
        ooa_calmar   = calculate_calmar_ratio(ooa_total_return, ooa_max_dd)
        bh_return    = (
            (ooa_df["Close"].iloc[-1] / ooa_df["Close"].iloc[0] - 1) * 100
            if "Close" in ooa_df.columns and len(ooa_df) > 1
            else 0.0
        )

        close_trades  = [t for t in ooa_trades if t.get("event") in {"CLOSE", "SELL"}]
        wins          = [t for t in close_trades if t.get("profit_pct", 0.0) > 0]
        win_rate      = len(wins) / len(close_trades) * 100 if close_trades else 0.0
        gross_profit  = sum(t.get("profit_pct", 0.0) for t in wins)
        gross_loss    = sum(-t.get("profit_pct", 0.0) for t in close_trades if t.get("profit_pct", 0.0) < 0)
        profit_factor = gross_profit / max(gross_loss, 1e-9) if gross_loss > 0 else float("inf")

        # ── 4. Summary metric cards ───────────────────────────────────────────
        st.markdown(f"##  Results — {ooa_ticker} (Unseen Asset)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Return",  f"{ooa_total_return:.2f}%",
                   delta=f"B&H: {bh_return:.2f}%")
        mc2.metric("Sharpe Ratio",  f"{ooa_sharpe:.2f}")
        mc3.metric("Max Drawdown",  f"{ooa_max_dd:.2f}%")
        mc4.metric("Trade Count",   str(len(close_trades)))

        mc5, mc6, mc7, mc8 = st.columns(4)
        mc5.metric("Win Rate",      f"{win_rate:.1f}%")
        mc6.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")
        mc7.metric("Sortino Ratio", f"{ooa_sortino:.2f}")
        mc8.metric("Calmar Ratio",  f"{ooa_calmar:.2f}")

        st.markdown("---")

        # ── 5. Equity curve with Buy / Close signals (Plotly) ─────────────────
        st.markdown("### Equity Curve & Trade Signals")

        opens_steps  = [t["step"] for t in ooa_trades if t.get("event") == "OPEN"  and t.get("step", 0) < len(ooa_equity)]
        closes_steps = [t["step"] for t in ooa_trades if t.get("event") in {"CLOSE", "SELL"} and t.get("step", 0) < len(ooa_equity)]

        fig_eq = go.Figure()

        # Equity line
        fig_eq.add_trace(go.Scatter(
            x=list(range(len(ooa_equity))),
            y=ooa_equity,
            mode="lines",
            name="Agent Equity",
            line=dict(color="#00aaff", width=2),
            hovertemplate="Step %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
        ))

        # Buy signals
        if opens_steps:
            open_profits = [
                next((t.get("profit_pct", 0.0) for t in ooa_trades
                      if t.get("step") == s and t.get("event") == "OPEN"), 0.0)
                for s in opens_steps
            ]
            fig_eq.add_trace(go.Scatter(
                x=opens_steps,
                y=[ooa_equity[s] for s in opens_steps],
                mode="markers",
                name="Open (Buy)",
                marker=dict(
                    symbol="triangle-up",
                    color="#2ecc71",
                    size=12,
                    line=dict(color="#ffffff", width=1),
                ),
                customdata=open_profits,
                hovertemplate=(
                    "BUY at step %{x}<br>"
                    "Equity: $%{y:,.2f}<extra></extra>"
                ),
            ))

        # Sell / Close signals
        if closes_steps:
            close_profits = [
                next((t.get("profit_pct", 0.0) for t in ooa_trades
                      if t.get("step") == s and t.get("event") in {"CLOSE", "SELL"}), 0.0)
                for s in closes_steps
            ]
            fig_eq.add_trace(go.Scatter(
                x=closes_steps,
                y=[ooa_equity[s] for s in closes_steps],
                mode="markers",
                name="Close (Sell)",
                marker=dict(
                    symbol="triangle-down",
                    color="#e74c3c",
                    size=12,
                    line=dict(color="#ffffff", width=1),
                ),
                customdata=close_profits,
                hovertemplate=(
                    "SELL at step %{x}<br>"
                    "Equity: $%{y:,.2f}<br>"
                    "Trade P&L: %{customdata:.2f}%<extra></extra>"
                ),
            ))

        fig_eq.update_layout(
            title=dict(
                text=f"Out-of-Asset Equity Curve — {ooa_ticker}  (Model trained on {trained_on})",
                font=dict(size=15),
            ),
            xaxis_title="Step",
            yaxis_title="Equity ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            height=420,
            margin=dict(l=40, r=20, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── 6. Drawdown plot (Plotly) ─────────────────────────────────────────
        st.markdown("### Drawdown Over Time")
        eq_arr = np.array(ooa_equity, dtype=float)
        peak   = np.maximum.accumulate(eq_arr)
        dd_arr = (eq_arr - peak) / np.maximum(peak, 1e-9) * 100
        steps  = list(range(len(dd_arr)))

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=steps,
            y=dd_arr.tolist(),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.18)",
            line=dict(color="#d62728", width=1.5),
            name="Drawdown %",
            hovertemplate="Step %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ))
        fig_dd.update_layout(
            title="Underwater Plot (Drawdown %)",
            xaxis_title="Step",
            yaxis_title="Drawdown (%)",
            height=260,
            margin=dict(l=40, r=20, t=45, b=40),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── 7. Trade return histogram (Plotly) ────────────────────────────────
        if ooa_returns:
            st.markdown("### Trade Return Distribution")

            win_count  = sum(1 for r in ooa_returns if r > 0)
            loss_count = len(ooa_returns) - win_count

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=ooa_returns,
                nbinsx=25,
                marker=dict(
                    color="#4c78a8",
                    line=dict(color="#1a2a3a", width=0.8),
                ),
                name="Trade Returns",
                hovertemplate="Return: %{x:.3f}<br>Count: %{y}<extra></extra>",
            ))
            fig_hist.add_vline(
                x=0.0,
                line=dict(color="#e74c3c", width=2, dash="dash"),
                annotation_text="Break-even",
                annotation_position="top right",
                annotation_font=dict(color="#e74c3c", size=11),
            )
            fig_hist.update_layout(
                title=dict(
                    text=(
                        f"Trade Returns — {ooa_ticker} &nbsp;&nbsp;"
                        f"<span style='color:#2ecc71'>▲ {win_count} wins</span>"
                        f" &nbsp; "
                        f"<span style='color:#e74c3c'>▼ {loss_count} losses</span>"
                    ),
                    font=dict(size=14),
                ),
                xaxis_title="Trade Return",
                yaxis_title="Frequency",
                height=340,
                margin=dict(l=40, r=20, t=55, b=40),
                bargap=0.05,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── 8. Trade log table ────────────────────────────────────────────────
        if ooa_trades:
            st.markdown("### Trade Log")
            trades_df = pd.DataFrame(ooa_trades)
            st.dataframe(trades_df, use_container_width=True)

            ooa_csv = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download OOA Trade Log",
                ooa_csv,
                file_name=f"{ooa_ticker}_ooa_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # ── 9. vs. training-asset comparison callout ──────────────────────────
        st.markdown("---")
        training_return = res.get("test_return", None)
        training_sharpe = res.get("test_sharpe", None)
        if training_return is not None:
            st.markdown("###  Generalisation Comparison")
            cmp1, cmp2, cmp3 = st.columns(3)
            cmp1.metric(
                f"Return — {trained_on} (OOS)",
                f"{training_return:.2f}%",
                delta=f"{ooa_total_return - training_return:+.2f}% on {ooa_ticker}",
            )
            cmp2.metric(
                f"Sharpe — {trained_on} (OOS)",
                f"{training_sharpe:.2f}",
                delta=f"{ooa_sharpe - training_sharpe:+.2f} on {ooa_ticker}",
            )
            cmp3.metric(
                "Generalisation",
                " Generalises" if ooa_total_return > 0 else " Struggles",
            )