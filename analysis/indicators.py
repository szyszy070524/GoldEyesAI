from __future__ import annotations

import pandas as pd


def add_moving_averages(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for window in windows:
        frame[f"ma_{window}"] = frame["close"].rolling(window).mean()
    return frame


def add_exponential_moving_averages(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for window in windows:
        frame[f"ema_{window}"] = frame["close"].ewm(span=window, adjust=False).mean()
    return frame


def add_rsi(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = frame["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    frame["rsi"] = 100 - (100 / (1 + rs))
    return frame


def add_macd(frame: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = frame["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = frame["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    frame["macd"] = macd
    frame["macd_signal"] = macd.ewm(span=signal, adjust=False).mean()
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
    return frame


def add_bollinger_bands(frame: pd.DataFrame, window: int, std: float) -> pd.DataFrame:
    rolling_mean = frame["close"].rolling(window).mean()
    rolling_std = frame["close"].rolling(window).std()
    frame["bollinger_upper"] = rolling_mean + std * rolling_std
    frame["bollinger_lower"] = rolling_mean - std * rolling_std
    return frame
