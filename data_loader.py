"""
data_loader.py  —  Out-of-Asset (OOA) data fetching helper.

Fetches OHLCV data and builds the same technical indicator set
used during training, then applies the caller-supplied scaler so
the observations match the distribution the model was trained on.

Usage (from app.py OOA page):
    from data_loader import load_and_prepare_oos_data
    df_scaled, feature_cols = load_and_prepare_oos_data(
        ticker="MSFT",
        start="2023-01-01",
        end="2024-12-31",
        scaler=scaler_object,
        feature_cols=feature_cols_list,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline import build_technical_indicators, download_stock_data


def load_and_prepare_oos_data(
    ticker: str,
    start: str,
    end: str,
    scaler,
    feature_cols: list[str],
    interval: str = "1d",
) -> tuple[pd.DataFrame, list[str]]:
    """Download new ticker data, build indicators, apply trained scaler.

    Parameters
    ----------
    ticker      : Stock symbol to evaluate (e.g. "MSFT", "TSLA").
    start / end : Date strings in "YYYY-MM-DD" format.
    scaler      : The fitted sklearn scaler saved alongside the model.
    feature_cols: Column names the scaler was fitted on (from training).
    interval    : yfinance interval string (default "1d").

    Returns
    -------
    df_scaled   : DataFrame with feature columns normalised by the
                  trained scaler — ready to pass into ForexTradingEnv.
    feature_cols: Echoed back for convenience.

    Raises
    ------
    ValueError  : If the downloaded data is empty or too short.
    KeyError    : If expected feature columns are missing after
                  building indicators (mismatch with training pipeline).
    """
    # ── 1. Download raw OHLCV ─────────────────────────────────────────────────
    df = download_stock_data(ticker, start, end, interval=interval)

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for '{ticker}' between {start} and {end}. "
            "Check the ticker symbol and date range."
        )

    # ── 2. Build the same technical features as the training pipeline ─────────
    df, derived_cols = build_technical_indicators(df)
    df[derived_cols] = df[derived_cols].fillna(0.0)

    # ── 3. Guard: make sure every feature the scaler knows exists ────────────
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Feature mismatch — the following columns built during training "
            f"are absent from the new ticker's data: {missing}. "
            "Ensure data_pipeline.build_technical_indicators() is deterministic."
        )

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} rows available for '{ticker}'. "
            "Need at least 30 bars to run a meaningful evaluation."
        )

    # ── 4. Apply the trained scaler (inference only — no re-fitting) ──────────
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(
        df_scaled[feature_cols].astype(float).values
    )

    return df_scaled, feature_cols