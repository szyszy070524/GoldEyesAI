from __future__ import annotations

import pandas as pd

from analysis.indicators import (
    add_bollinger_bands,
    add_exponential_moving_averages,
    add_macd,
    add_moving_averages,
    add_rsi,
)


def build_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame["return"] = frame["close"].pct_change()
    frame["range"] = frame["high"] - frame["low"]
    frame["body"] = frame["close"] - frame["open"]

    frame = add_moving_averages(frame, config["ma_windows"])
    frame = add_exponential_moving_averages(frame, config["ema_windows"])
    frame = add_rsi(frame, config["rsi_period"])
    frame = add_macd(frame, config["macd_fast"], config["macd_slow"], config["macd_signal"])
    frame = add_bollinger_bands(frame, config["bollinger_window"], config["bollinger_std"])

    for lag in range(1, config["lag_days"] + 1):
        frame[f"return_lag_{lag}"] = frame["return"].shift(lag)
        frame[f"close_lag_{lag}"] = frame["close"].shift(lag)

    frame["rolling_mean_5"] = frame["return"].rolling(5).mean()
    frame["rolling_vol_5"] = frame["return"].rolling(5).std()

    return frame


def add_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["future_close"] = frame["close"].shift(-horizon)
    frame["target"] = (frame["future_close"] > frame["close"]).astype(int)
    return frame


def finalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna().reset_index(drop=True)
