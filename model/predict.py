from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class PredictionResult:
    horizon: int
    probability_up: float
    probability_down: float


def load_model(model_dir: Path) -> object:
    metadata_path = model_dir / "metrics.json"
    metadata = pd.read_json(metadata_path, typ="series")
    model_path = Path(metadata["model_path"])
    return joblib.load(model_path)


def predict_probabilities(model: object, features: pd.DataFrame) -> float:
    prob_up = float(model.predict_proba(features)[:, 1][0])
    return prob_up


def format_prediction(horizon: int, prob_up: float) -> PredictionResult:
    return PredictionResult(
        horizon=horizon,
        probability_up=prob_up,
        probability_down=1 - prob_up,
    )
