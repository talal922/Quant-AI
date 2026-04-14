"""
Live_Trading.py  —  Pure module (no standalone Streamlit entry-point).
Called exclusively through app.py's navigation router via render_page().

DO NOT add st.set_page_config() or any top-level st.* calls here.
"""

import tempfile
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import PPO

from data_pipeline import build_technical_indicators, download_stock_data


# ═══════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS  (safe — no Streamlit calls)
# ═══════════════════════════════════════════════════════════════════

WINDOW_SIZE   = 14
ACTION_LABELS = {0: "HOLD", 1: "BUY", 2: "SELL"}
ACTION_ICONS  = {0: "⏸", 1: "▲", 2: "▼"}


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (pure Python — no Streamlit calls)
# ═══════════════════════════════════════════════════════════════════

def load_model_from_bytes(model_bytes: bytes) -> PPO:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name
    return PPO.load(tmp_path)


def load_scaler_from_bytes(scaler_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp.write(scaler_bytes)
        tmp_path = tmp.name
    return joblib.load(tmp_path)


def get_alpaca_client(api_key: str, api_secret: str, paper: bool = True):
    try:
        import alpaca_trade_api as tradeapi
        base_url = (
            "https://paper-api.alpaca.markets/v2"
            if paper
            else "https://api.alpaca.markets/v2"
        )
        return tradeapi.REST(api_key, api_secret, base_url, api_version="v2")
    except ImportError:
        return None
    except Exception:
        return None


def fetch_account_info(api) -> dict | None:
    if api is None:
        return None
    try:
        acc = api.get_account()
        return {
            "portfolio_value": float(acc.portfolio_value),
            "cash":            float(acc.cash),
            "buying_power":    float(acc.buying_power),
            "equity":          float(acc.equity),
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_positions(api, ticker: str) -> dict | None:
    if api is None:
        return None
    try:
        pos = api.get_position(ticker)
        return {
            "qty":             float(pos.qty),
            "avg_price":       float(pos.avg_entry_price),
            "current_price":   float(pos.current_price),
            "unrealized_pl":   float(pos.unrealized_pl),
            "unrealized_plpc": float(pos.unrealized_plpc) * 100,
        }
    except Exception:
        return {
            "qty":             0.0,
            "avg_price":       0.0,
            "current_price":   0.0,
            "unrealized_pl":   0.0,
            "unrealized_plpc": 0.0,
        }


def build_observation(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    last_rows = df.iloc[-window_size:][feature_cols].astype(float).values
    if len(last_rows) < window_size:
        pad = np.zeros((window_size - len(last_rows), len(feature_cols)))
        last_rows = np.vstack([pad, last_rows])
    scaled = scaler.transform(last_rows).astype(np.float32)
    return scaled.flatten().reshape(1, -1)


def submit_order(api, ticker: str, qty: float, side: str) -> str:
    if api is None:
        return "❌ Alpaca client not initialised."
    try:
        order = api.submit_order(
            symbol=ticker, qty=qty, side=side, type="market", time_in_force="gtc"
        )
        return f"✅ Order submitted — ID: `{order.id}` | {side.upper()} {qty} {ticker}"
    except Exception as e:
        return f"❌ Order failed: {e}"


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISER
# ═══════════════════════════════════════════════════════════════════

def _init_session_state() -> None:
    for key, val in {
        "lt_analysis_done": False,
        "lt_action":        None,
        "lt_action_val":    None,
        "lt_price":         None,
        "lt_vix":           None,
        "lt_account":       None,
        "lt_positions":     None,
        "lt_df":            None,
        "lt_feature_cols":  None,
        "lt_obs":           None,
        "lt_last_update":   None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR  (called inside render_page)
# ═══════════════════════════════════════════════════════════════════

def _render_sidebar() -> tuple:
    """
    Renders the Live Trading sidebar — styled identically to the Train Agent sidebar.
    Returns: (api_key, api_secret, paper_mode, ticker, trade_qty, model_file, scaler_file)
    """
    with st.sidebar:
        # ── Section 1: API Credentials ──────────────────────────────────
        st.markdown(
            '<p style="font-size:1.5rem;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;color:#4A5E7A;margin-bottom:0.35rem;">API Config</p>',
            unsafe_allow_html=True,
        )
        api_key    = st.text_input("API Key",    type="password", placeholder="PK...",  key="lt_api_key")
        api_secret = st.text_input("API Secret", type="password", placeholder="FnU...", key="lt_api_secret")
        paper_mode = st.checkbox("Paper Trading Mode", value=True, key="lt_paper_mode")

        # ── Section 2: Ticker ────────────────────────────────────────────
        st.markdown(
            '<hr style="border:none;border-top:1px solid #1F2D42;margin:1rem 0;">'
            '<p style="font-size:1rem;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;color:#4A5E7A;margin-bottom:0.35rem;">Ticker</p>',
            unsafe_allow_html=True,
        )
        ticker    = st.text_input("Symbol", value="AAPL", key="lt_ticker").upper()
        trade_qty = st.number_input(
            "Trade Quantity (shares)",
            min_value=1, max_value=1000, value=10, step=1, key="lt_qty",
        )

        # ── Section 3: AI Model ──────────────────────────────────────────
        st.markdown(
            '<hr style="border:none;border-top:1px solid #1F2D42;margin:1rem 0;">'
            '<p style="font-size:1rem;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;color:#4A5E7A;margin-bottom:0.35rem;">AI Model</p>',
            unsafe_allow_html=True,
        )
        model_file  = st.file_uploader("Model (.zip)",  type=["zip"], key="lt_model_file")
        scaler_file = st.file_uploader("Scaler (.pkl)", type=["pkl"], key="lt_scaler_file")

        # ── Status chips ─────────────────────────────────────────────────
        st.markdown(
            '<hr style="border:none;border-top:1px solid #1F2D42;margin:1rem 0;">',
            unsafe_allow_html=True,
        )
        if api_key and api_secret:
            st.markdown(
                '<div class="sidebar-status sidebar-status--green">🟢 API Credentials Entered</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sidebar-status sidebar-status--amber">⚪ Enter API Credentials</div>',
                unsafe_allow_html=True,
            )

        if model_file and scaler_file:
            st.markdown(
                '<div class="sidebar-status sidebar-status--green">🟢 Model &amp; Scaler Loaded</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sidebar-status sidebar-status--amber">🟡 Upload Model + Scaler</div>',
                unsafe_allow_html=True,
            )

    return api_key, api_secret, paper_mode, ticker, trade_qty, model_file, scaler_file


# ═══════════════════════════════════════════════════════════════════
# MAIN PAGE RENDERER  —  entry point called from app.py
# ═══════════════════════════════════════════════════════════════════

def render_page(change_page_fn) -> None:
    """
    Render the full Live Trading page.

    Parameters
    ----------
    change_page_fn : callable
        The ``change_page(name)`` function defined in app.py.
        Used so the ← Back button can navigate back to Home.
    """
    _init_session_state()

    # ── ← Back to Home (matches Train Agent pattern exactly) ─────────────────
    if st.button("⬅️ Back to Home", key="back_live_btn"):
        change_page_fn("Home")
        st.rerun()

    # ── Page header ───────────────────────────────────────────────────────────
    st.title("⚡ Live Trading Dashboard")
    st.caption("AI-Powered Signal Engine · Alpaca Paper Trading Integration")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    api_key, api_secret, paper_mode, ticker, trade_qty, model_file, scaler_file = (
        _render_sidebar()
    )

    # ── Pre-flight banners ────────────────────────────────────────────────────
    if not (model_file and scaler_file):
        st.markdown(
            """
            <div class="preflight-banner">
                <span class="preflight-icon">📂</span>
                <span>
                    Upload your trained <strong>model (.zip)</strong> and
                    <strong>scaler (.pkl)</strong> to get started.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not (api_key and api_secret):
        st.markdown(
            """
            <div class="preflight-banner preflight-banner--blue">
                <span class="preflight-icon">🔑</span>
                <span>
                    Enter your <strong>Alpaca Paper Trading API keys.</strong>
                    Get your keys at <em>app.alpaca.markets</em>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Action buttons ────────────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns([2, 1], gap="medium")
    run_analysis = btn_col1.button(
        "🔍 Run AI Analysis Now", type="primary", use_container_width=True
    )

    execute_trade = False
    if st.session_state.lt_analysis_done and st.session_state.lt_action_val in (1, 2):
        execute_trade = btn_col2.button(
            f"🚀 Execute {ACTION_LABELS[st.session_state.lt_action_val]} Order",
            use_container_width=True,
        )

    # ── Run analysis logic ────────────────────────────────────────────────────
    if run_analysis:
        if not (model_file and scaler_file):
            st.error("⛔ Please upload both the model (.zip) and scaler (.pkl) files first.")
            st.stop()

        with st.spinner("🔄 Fetching live market data and running AI analysis..."):
            try:
                model        = load_model_from_bytes(model_file.getvalue())
                scaler_data  = load_scaler_from_bytes(scaler_file.getvalue())
                scaler       = scaler_data["scaler"]
                feature_cols = scaler_data["feature_cols"]

                end_dt   = datetime.now().strftime("%Y-%m-%d")
                start_dt = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
                df_raw   = download_stock_data(ticker, start_dt, end_dt)
                df, _    = build_technical_indicators(df_raw)
                df[feature_cols] = df[feature_cols].fillna(0.0)

                latest_close = float(df["Close"].iloc[-1])
                prev_close   = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_close
                daily_chg    = (latest_close / prev_close - 1) * 100
                latest_vix   = (
                    float(df["VIX_Close"].iloc[-1]) if "VIX_Close" in df.columns else 0.0
                )

                obs        = build_observation(df, feature_cols, scaler, WINDOW_SIZE)
                action, _  = model.predict(obs, deterministic=True)
                action_val = int(action[0]) if hasattr(action, "__len__") else int(action)

                api      = (
                    get_alpaca_client(api_key, api_secret, paper=paper_mode)
                    if (api_key and api_secret)
                    else None
                )
                account  = fetch_account_info(api)
                position = fetch_positions(api, ticker)

                st.session_state.lt_analysis_done = True
                st.session_state.lt_action_val    = action_val
                st.session_state.lt_action        = ACTION_LABELS[action_val]
                st.session_state.lt_price         = {"close": latest_close, "chg": daily_chg}
                st.session_state.lt_vix           = latest_vix
                st.session_state.lt_account       = account
                st.session_state.lt_positions     = position
                st.session_state.lt_df            = df
                st.session_state.lt_feature_cols  = feature_cols
                st.session_state.lt_obs           = obs
                st.session_state.lt_last_update   = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            except Exception as exc:
                st.error(f"❌ Analysis failed: {exc}")
                st.stop()

        st.rerun()

    # ── Execute trade logic ───────────────────────────────────────────────────
    if execute_trade:
        api = get_alpaca_client(api_key, api_secret, paper=paper_mode)
        if api is None:
            st.markdown(
                """
                <div class="preflight-banner">
                    <span class="preflight-icon">❌</span>
                    <span>
                        Could not connect to Alpaca.
                        Check your API credentials in the sidebar.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            action_val  = st.session_state.lt_action_val
            current_qty = (
                st.session_state.lt_positions.get("qty", 0.0)
                if st.session_state.lt_positions
                else 0.0
            )

            with st.spinner("Submitting order to Alpaca..."):
                if action_val == 1:
                    msg = (
                        submit_order(api, ticker, trade_qty, "buy")
                        if current_qty == 0
                        else f"⚠️ Already holding {current_qty} shares."
                    )
                elif action_val == 2:
                    msg = (
                        submit_order(api, ticker, current_qty, "sell")
                        if current_qty > 0
                        else "⚠️ No position to sell."
                    )
                else:
                    msg = "ℹ️ HOLD — no order submitted."

            st.markdown(
                f'<div class="order-result">{msg}</div>',
                unsafe_allow_html=True,
            )

            time.sleep(1)
            st.session_state.lt_account   = fetch_account_info(api)
            st.session_state.lt_positions = fetch_positions(api, ticker)

    # ═════════════════════════════════════════════════════════════════
    # DASHBOARD  (rendered only after a successful analysis run)
    # ═════════════════════════════════════════════════════════════════

    if st.session_state.lt_analysis_done:
        action_val  = st.session_state.lt_action_val
        action_icon = ACTION_ICONS[action_val]
        last_update = st.session_state.lt_last_update or ""

        # ── Live header bar ───────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="live-header">
                <div class="live-header__left">
                    <span class="live-dot"></span>
                    <span class="live-label">Live</span>
                    <span>AI Signal Engine &mdash; {ticker}</span>
                </div>
                <div class="live-header__right">Last updated: {last_update} UTC</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── AI Signal Banner ──────────────────────────────────────────────────
        if action_val == 1:          # BUY
            chip_class = "status-chip--green"
            chip_label = "BUY SIGNAL"
        elif action_val == 2:        # SELL
            chip_class = "status-chip--blue"
            chip_label = "SELL SIGNAL"
        else:                        # HOLD
            chip_class = "status-chip--amber"
            chip_label = "HOLD SIGNAL"

        vix_val   = st.session_state.lt_vix
        price_val = st.session_state.lt_price["close"]
        chg_val   = st.session_state.lt_price["chg"]

        st.markdown(
            f"""
            <div class="signal-banner">
                <div class="signal-banner__left">
                    <div class="signal-icon">{action_icon}</div>
                    <div>
                        <div class="signal-label">
                            <span class="status-chip {chip_class}">{chip_label}</span>
                        </div>
                        <div class="signal-sub">
                            PPO AGENT &middot; DETERMINISTIC MODE &middot; {ticker}
                        </div>
                    </div>
                </div>
                <div class="signal-banner__right">
                    <div class="signal-vix-badge">VIX {vix_val:.1f}</div>
                    <div class="signal-meta">
                        PRICE ${price_val:.2f} &nbsp;|&nbsp; DAILY {chg_val:+.2f}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Market Intelligence ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Market Intelligence")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Current Price",
            f"${st.session_state.lt_price['close']:.2f}",
            f"{st.session_state.lt_price['chg']:+.2f}%",
        )
        col2.metric("VIX Fear Index", f"{st.session_state.lt_vix:.1f}")
        if st.session_state.lt_df is not None:
            week_chg = float(
                (
                    st.session_state.lt_df["Close"].iloc[-1]
                    / st.session_state.lt_df["Close"].iloc[-5]
                    - 1
                )
                * 100
            )
            col3.metric("5-Day Trend", f"{week_chg:+.2f}%")

        # ── Alpaca Account & Positions ────────────────────────────────────────
        st.markdown("---")
        st.subheader("💼 Alpaca Account & Positions")

        acc = st.session_state.lt_account
        pos = st.session_state.lt_positions

        if acc and "error" not in acc:
            a1, a2, a3 = st.columns(3)
            a1.metric("Portfolio Value", f"${acc['portfolio_value']:,.2f}")
            a2.metric("Buying Power",    f"${acc['buying_power']:,.2f}")

            if pos and pos.get("qty", 0) > 0:
                a3.metric(
                    f"Shares of {ticker}",
                    f"{pos['qty']:.0f}",
                    f"{pos['unrealized_plpc']:+.2f}%",
                )
            else:
                a3.metric(f"Shares of {ticker}", "0")
                st.markdown(
                    f"""
                    <div class="position-empty">
                        <span>📭</span>
                        <span>
                            No open position for <strong>{ticker}</strong>.
                            The AI signal will guide your next entry.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="no-api-card">
                    <span>🔌</span>
                    <span>
                        No Alpaca API connection. Enter your keys in the sidebar
                        to view live account data.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Technical Analysis Chart ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("📈 Technical Analysis & Charts")

        if st.session_state.lt_df is not None:
            chart_df = st.session_state.lt_df.tail(90).copy()
            if "Date" in chart_df.columns:
                chart_df.index = pd.to_datetime(chart_df["Date"])
            st.line_chart(chart_df[["Close"]])

    else:
        # ── Empty state (before first analysis run) ───────────────────────────
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state__icon">🤖</div>
                <div class="empty-state__title">AI Agent Standby</div>
                <div class="empty-state__sub">
                    Upload your trained <strong>model (.zip)</strong> and
                    <strong>scaler (.pkl)</strong> from the sidebar, then press
                    <strong>Run AI Analysis Now</strong> to receive your first
                    live trading signal.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )