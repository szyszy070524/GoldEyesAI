from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_close_trend(frame: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "close_trend.png"
    plt.figure(figsize=(10, 4))
    plt.plot(frame["date"], frame["close"], label="Close")
    plt.title("Gold Close Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_return_distribution(frame: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "return_distribution.png"
    plt.figure(figsize=(6, 4))
    frame["return"].hist(bins=30)
    plt.title("Daily Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
