from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import io

import pandas as pd
import requests


@dataclass
class FetchConfig:
    url: str
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    timeout: int = 20


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def fetch_stooq(config: FetchConfig) -> pd.DataFrame:
    response = requests.get(config.url, timeout=config.timeout)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    frame = _normalize_columns(frame)

    if config.start_date:
        frame = frame[frame["date"] >= config.start_date]
    if config.end_date:
        frame = frame[frame["date"] <= config.end_date]

    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def fetch_recent_days(url: str, days: int) -> pd.DataFrame:
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days * 2)
    frame = fetch_stooq(FetchConfig(url=url, start_date=start_date, end_date=end_date))
    return frame.tail(days)
