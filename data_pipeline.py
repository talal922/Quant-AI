import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def download_stock_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data and VIX index for a US stock ticker using yfinance."""
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    if start_ts >= end_ts:
        raise ValueError("Start date must be before end date.")

    end_inclusive = end_ts + pd.Timedelta(days=1)
    
    # 1. Download stock data
    df = yf.download(
        ticker,
        start=start_ts.strftime("%Y-%m-%d"),
        end=end_inclusive.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(
            f"No data downloaded for ticker={ticker} from {start_ts.date()} to {end_ts.date()}."
        )

    # 2. Download VIX data (fear index)
    vix = yf.download(
        "^VIX",
        start=start_ts.strftime("%Y-%m-%d"),
        end=end_inclusive.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
    )
    
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 3. Merge VIX with stock data
    df['VIX_Close'] = vix['Close']
    df = df.dropna()
    
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close']] = df[['Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close']].astype(float)
    df.reset_index(inplace=True)
    return df


def _ensure_series(value, name: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 1:
            return value.iloc[:, 0]
        raise ValueError(
            f"Expected a single-series result for {name}, got DataFrame with shape {value.shape}"
        )
    if not isinstance(value, pd.Series):
        raise TypeError(f"Expected pandas.Series for {name}, got {type(value)}")
    return value


def _flatten_cols(df: pd.DataFrame):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(map(str, col)).strip("_") for col in df.columns
        ]


def build_technical_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the full feature set for AAPL RL trading.

    Changes vs original:
      - ma_20_slope / ma_50_slope replaced by normalised _pct variants (tanh-scaled,
        price-invariant — critical as AAPL moves from $50 to $200+).
      - 7 new indicators: ADX-14, Williams %R-14, OBV momentum (z-score), ROC-5,
        ROC-20, Stochastic %K, and 200-day MA distance.
      - All new features are bounded in [-1, 1] or [0, 1] — no raw price-scale values.
    """
    df = df.copy()

    # Flatten any MultiIndex columns from yfinance / pandas_ta
    df.columns = [
        "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]

    # VIX_Close was added in download_stock_data
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_ohlcv = required_columns - set(df.columns)
    if missing_ohlcv:
        raise ValueError(
            f"Input DataFrame is missing required OHLCV columns: {sorted(missing_ohlcv)}"
        )

    close_series = df["Close"].squeeze()
    if not isinstance(close_series, pd.Series):
        close_series = pd.Series(close_series, index=df.index, name="Close")

    # ── Existing core indicators ─────────────────────────────────────────────
    df["rsi_14"] = _ensure_series(ta.rsi(close_series, length=14), "rsi_14")
    df["atr_14"] = _ensure_series(ta.atr(df["High"], df["Low"], close_series, length=14), "atr_14")
    df["ma_20"] = _ensure_series(ta.sma(close_series, length=20), "ma_20")
    df["ma_50"] = _ensure_series(ta.sma(close_series, length=50), "ma_50")

    bbands = ta.bbands(close_series, length=20, std=2)
    if bbands is None or bbands.empty:
        bbands = pd.DataFrame(
            index=df.index,
            columns=["BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0"],
            dtype=float,
        )
    else:
        _flatten_cols(bbands)
        bbands = bbands.reindex(df.index)

    df["bb_upper"] = _ensure_series(bbands.get("BBU_20_2.0", pd.Series(np.nan, index=df.index)), "bb_upper")
    df["bb_mid"]   = _ensure_series(bbands.get("BBM_20_2.0", pd.Series(np.nan, index=df.index)), "bb_mid")
    df["bb_lower"] = _ensure_series(bbands.get("BBL_20_2.0", pd.Series(np.nan, index=df.index)), "bb_lower")

    macd = ta.macd(close_series, fast=12, slow=26, signal=9)
    if macd is None or macd.empty:
        df["macd_hist"] = np.nan
    else:
        _flatten_cols(macd)
        df["macd_hist"] = _ensure_series(
            macd.get("MACDh_12_26_9", pd.Series(np.nan, index=df.index)), "macd_hist"
        )

    df["price_pct"]    = df["Close"].pct_change()
    df["volume_pct"]   = df["Volume"].pct_change()
    df["volume_force"] = df["price_pct"] * df["volume_pct"]

    # NEW: Intraday range ratio — (High-Low)/Close, normalized by ATR.
    # Captures whether the current bar is a wide-range (high uncertainty) or
    # narrow-range (low uncertainty / compression) bar. Critical for AAPL because
    # narrow-range compressions often precede explosive breakouts.
    # Bounded ≈ [0, ~3] in normal conditions; StandardScaler handles the rest.
    df["high_low_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)

    # NEW: Overnight gap — (Open - prev_Close)/prev_Close.
    # AAPL frequently gaps on earnings/macro news. The gap direction and size
    # is a strong regime signal: large gap-ups tend to continue intraday while
    # gap-downs often see fade attempts. tanh(/0.03) maps ±3% gap → ±1.
    df["gap_pct"] = np.tanh(
        (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + 1e-9) / 0.03
    )

    if {"bb_upper", "bb_lower", "bb_mid"}.issubset(df.columns):
        df["volatility_z"] = (df["Close"] - df["bb_mid"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    else:
        df["volatility_z"] = 0.5

    df["atr_14_pct"]       = df["atr_14"] / (df["Close"] + 1e-9)
    df["close_ma20_diff"]  = (df["Close"] - df["ma_20"]) / (df["ma_20"] + 1e-9)
    df["close_ma50_diff"]  = (df["Close"] - df["ma_50"]) / (df["ma_50"] + 1e-9)
    df["ma_spread"]        = (df["ma_20"] - df["ma_50"]) / (df["ma_50"] + 1e-9)
    df["ma_spread_slope"]  = df["ma_spread"].diff()

    # ── FIX: Normalised MA slopes (price-invariant) ──────────────────────────
    # Old `ma_20.diff()` is in raw $ terms — meaningless across AAPL's price history.
    # New: (slope / close_price) * scale → tanh → bounded [-1, 1].
    # Why *200? At a "steep" AAPL day (+0.5% MA slope), 0.005*200=1.0, tanh≈0.76.
    # At a flat day (0.05% slope), 0.0005*200=0.1, tanh≈0.10. Good dynamic range.
    df["ma_20_slope_pct"] = np.tanh(df["ma_20"].diff() / (df["Close"] + 1e-9) * 200.0)
    df["ma_50_slope_pct"] = np.tanh(df["ma_50"].diff() / (df["Close"] + 1e-9) * 200.0)

    # ── NEW 1: ADX (Average Directional Index, 14) ───────────────────────────
    # Why AAPL: AAPL spends ~40% of its time in strong directional trends (ADX>25).
    # The agent needs to know whether it's in a trending or ranging regime to decide
    # between holding through pullbacks vs. cutting positions quickly.
    # Normalised to [0, 1] by dividing by 100.
    adx_df = ta.adx(df["High"], df["Low"], close_series, length=14)
    if adx_df is not None and not adx_df.empty:
        _flatten_cols(adx_df)
        adx_col = next((c for c in adx_df.columns if "ADX" in c and "DM" not in c), None)
        df["adx_14_norm"] = (adx_df[adx_col] / 100.0) if adx_col else 0.5
    else:
        df["adx_14_norm"] = 0.5

    # ── NEW 2: Williams %R (14) ───────────────────────────────────────────────
    # Why AAPL: %R excels at identifying short-term exhaustion on AAPL's intratrend
    # pullbacks. Drops to −80/−90 on healthy dips; stays near 0 at overbought peaks.
    # Already bounded [−100, 0] — divide by 100 to get [−1, 0].
    willr_series = ta.willr(df["High"], df["Low"], close_series, length=14)
    df["willr_14_norm"] = _ensure_series(willr_series, "willr_14") / 100.0

    # ── NEW 3: OBV Momentum (z-score of 5-bar OBV change) ───────────────────
    # Why AAPL: Institutional flows dominate AAPL's moves. OBV leads price by
    # 1–3 bars on accumulation/distribution phases. Raw OBV is cumulative and
    # scale-arbitrary; z-scoring the 5-bar change gives a stationary signal.
    obv_series = _ensure_series(ta.obv(close_series, df["Volume"]), "obv")
    obv_5bar   = obv_series.diff(5)
    obv_std    = obv_series.rolling(window=20).std()
    df["obv_slope"] = obv_5bar / (obv_std + 1e-9)
    # FIX: clip to [-5, 5] to prevent extreme outliers during quiet periods
    # (near-zero std in denominator blows up the z-score unboundedly).
    df["obv_slope"] = df["obv_slope"].clip(-5.0, 5.0)

    # ── NEW 4 & 5: Rate of Change — short and medium momentum ────────────────
    # Why AAPL: ROC-5 captures the ~1-week momentum persistence typical of AAPL
    # after a gap-up/down. ROC-20 captures the ~1-month trend persistence that
    # overlaps with AAPL's inter-earnings directional phases.
    # tanh(/10) converts the % values to [-1, 1]: at 10% move → tanh(1)=0.76.
    roc_5_raw  = _ensure_series(ta.roc(close_series, length=5),  "roc_5")
    roc_20_raw = _ensure_series(ta.roc(close_series, length=20), "roc_20")
    df["roc_5_tanh"]  = np.tanh(roc_5_raw  / 10.0)
    df["roc_20_tanh"] = np.tanh(roc_20_raw / 10.0)

    # ── NEW 6: Stochastic %K (14, 3, 3) ─────────────────────────────────────
    # Why AAPL: Stochastic confirms RSI reversals and catches divergences where
    # RSI-14 lags. Extremely effective on AAPL's weekly swing highs/lows.
    # Normalised to [0, 1] by dividing by 100.
    stoch_df = ta.stoch(df["High"], df["Low"], close_series, k=14, d=3, smooth_k=3)
    if stoch_df is not None and not stoch_df.empty:
        _flatten_cols(stoch_df)
        stoch_k_col = next(
            (c for c in stoch_df.columns if "STOCHk" in c or "stochk" in c.lower()), None
        )
        df["stoch_k_norm"] = (stoch_df[stoch_k_col] / 100.0) if stoch_k_col else 0.5
    else:
        df["stoch_k_norm"] = 0.5

    # ── NEW 7: 200-day MA distance ────────────────────────────────────────────
    # Why AAPL: The 200-day MA is the single most-watched institutional support level
    # for AAPL. Sitting 10%+ above the 200MA signals an extended bull phase.
    # Below it signals a bear phase. This is a major regime feature the env was missing.
    # NaN for first 199 bars → filled with 0.0 ("at the MA") later — do NOT add
    # to dropna_subset so we keep the full training set.
    df["ma_200"]          = _ensure_series(ta.sma(close_series, length=200), "ma_200")
    df["close_ma200_diff"] = (df["Close"] - df["ma_200"]) / (df["ma_200"] + 1e-9)

    # ── NEW 8: VIX Fear Index (Real Market Sentiment) ─────────────────────────
    # Why AAPL: Instead of fake news sentiment, we use the real VIX index. 
    # VIX > 30 means panic, VIX < 15 means complacency. 
    # tanh normalizes it cleanly to [-1.0, 1.0] centered around 20.
    if "VIX_Close" in df.columns:
        df["vix_sentiment"] = np.tanh((df["VIX_Close"] - 20) / 10.0)
    else:
        # Fallback if VIX isn't available for some reason
        df["vix_sentiment"] = 0.0

    # ── Drop NaN rows from indicators with lookback periods ──────────────────
    dropna_subset = [
        # Existing
        "rsi_14", "atr_14", "ma_20", "ma_50", "macd_hist",
        "price_pct", "volume_pct", "volume_force",
        "high_low_pct", "gap_pct",
        "atr_14_pct", "close_ma20_diff", "close_ma50_diff",
        "ma_spread", "ma_spread_slope",
        # Fixed (replaced ma_20_slope, ma_50_slope)
        "ma_20_slope_pct", "ma_50_slope_pct",
        # New (all require their own lookbacks; exclude close_ma200_diff — see above)
        "adx_14_norm", "willr_14_norm", "obv_slope",
        "roc_5_tanh", "roc_20_tanh", "stoch_k_norm",
    ]
    df.dropna(subset=dropna_subset, inplace=True)

    if df.empty:
        raise ValueError(
            "Not enough historical bars after indicator calculation. "
            "Please choose a longer date range or check the input data."
        )

    # ── Final feature columns (18 features; 14 window × 21 obs after state concat) ──
    feature_cols = [
        # ── Core momentum & mean-reversion ──
        "rsi_14",           # 0–100, not normalized here (StandardScaler handles it)
        "atr_14_pct",       # dimensionless volatility ratio
        "volatility_z",     # Bollinger Band z-score
        "macd_hist",        # raw MACD histogram (scaler normalizes)
        # ── Price action (NEW) ──
        "high_low_pct",     # intraday range / close — compression vs expansion signal
        "gap_pct",          # tanh-normalized overnight gap — earnings/macro regime
        # ── Volume ──
        "volume_force",     # single-bar price×volume interaction
        "obv_slope",        # z-score of 5-bar OBV momentum
        # ── Trend / regime ──
        "adx_14_norm",      # [0,1] trend strength
        "ma_20_slope_pct",  # tanh-normalized, price-invariant
        "ma_50_slope_pct",  # tanh-normalized, price-invariant
        "close_ma20_diff",  # distance from 20MA (relative)
        "close_ma50_diff",  # distance from 50MA (relative)
        "close_ma200_diff", # distance from 200MA — regime signal
        "ma_spread",        # 20/50 MA spread (relative)
        "ma_spread_slope",  # rate of spread change
        # ── Oscillators ──
        "willr_14_norm",    # [-1,0] Williams %R
        "stoch_k_norm",     # [0,1] Stochastic %K
        # ── Momentum ──
        "roc_5_tanh",       # [-1,1] 5-bar ROC
        "roc_20_tanh",      # [-1,1] 20-bar ROC
        "vix_sentiment",    # [-1.0, 1.0] VIX fear index
    ]

    # Fill any residual NaN (e.g. close_ma200_diff for first 199 rows)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected indicator columns: {missing_cols}. "
            f"Available columns: {sorted(df.columns)}"
        )

    return df, feature_cols


def fit_scaler(df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    """Fit a StandardScaler on the training indicators to normalize observations."""
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].astype(float).values)
    return scaler


def transform_features(df: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> np.ndarray:
    """Transform indicator columns with the fitted scaler."""
    return scaler.transform(df[feature_cols].astype(float).values).astype(np.float32)


def train_test_split(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and test sequentially."""
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df