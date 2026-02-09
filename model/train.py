from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class TrainConfig:
    model_dir: Path
    random_state: int
    train_window: int
    test_window: int
    min_train_size: int


def _walk_forward_splits(data: pd.DataFrame, train_window: int, test_window: int) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
    total = len(data)
    start = 0
    while start + train_window + test_window <= total:
        train_slice = data.iloc[start : start + train_window]
        test_slice = data.iloc[start + train_window : start + train_window + test_window]
        yield train_slice, test_slice
        start += test_window


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = None
    return metrics


def train_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    config: TrainConfig,
) -> dict:
    config.model_dir.mkdir(parents=True, exist_ok=True)

    X = dataset[feature_columns]
    y = dataset["target"].values

    models = {
        "logistic": LogisticRegression(max_iter=1000, random_state=config.random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=config.random_state,
            class_weight="balanced",
        ),
    }

    scores: dict[str, list[dict]] = {name: [] for name in models}

    for train_slice, test_slice in _walk_forward_splits(dataset, config.train_window, config.test_window):
        if len(train_slice) < config.min_train_size:
            continue
        X_train = train_slice[feature_columns]
        y_train = train_slice["target"].values
        X_test = test_slice[feature_columns]
        y_test = test_slice["target"].values

        for name, model in models.items():
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
            scores[name].append(_evaluate(y_test, preds, probs))

    summary = {
        name: {
            "accuracy": float(np.nanmean([m["accuracy"] for m in metrics])),
            "auc": float(np.nanmean([m["auc"] for m in metrics if m["auc"] is not None]))
            if any(m["auc"] is not None for m in metrics)
            else None,
        }
        for name, metrics in scores.items()
    }

    best_model_name = max(summary, key=lambda key: summary[key]["accuracy"])
    best_model = models[best_model_name]
    best_model.fit(X, y)

    model_path = config.model_dir / f"{best_model_name}.joblib"
    joblib.dump(best_model, model_path)

    metadata = {
        "best_model": best_model_name,
        "model_path": str(model_path),
        "metrics": summary,
    }
    metadata_path = config.model_dir / "metrics.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata
