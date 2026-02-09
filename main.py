from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from analysis.feature_engineering import add_targets, build_features, finalize_dataset
from crawler.fetch_price import fetch_recent_days
from model.predict import format_prediction, load_model, predict_probabilities
from model.train import TrainConfig, train_models
from visualization.charts import plot_close_trend, plot_return_distribution


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    settings = load_settings()
    data_days = settings["project"]["data_days"]
    url = settings["data_source"]["stooq_url"]

    raw_frame = fetch_recent_days(url, data_days)
    raw_csv = Path(settings["storage"]["raw_csv"])
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(raw_csv, index=False)

    feature_frame = build_features(raw_frame, settings["features"])

    horizons = settings["project"]["horizons"]
    predictions = []

    processed_csv = Path(settings["storage"]["processed_csv"])
    processed_csv.parent.mkdir(parents=True, exist_ok=True)

    for horizon in horizons:
        labeled = add_targets(feature_frame, horizon)
        dataset = finalize_dataset(labeled)
        dataset.to_csv(processed_csv, index=False)

        feature_columns = [
            column
            for column in dataset.columns
            if column
            not in {
                "date",
                "future_close",
                "target",
            }
        ]

        model_dir = Path(settings["storage"]["model_dir"]) / f"horizon_{horizon}"
        train_config = TrainConfig(
            model_dir=model_dir,
            random_state=settings["model"]["random_state"],
            train_window=settings["model"]["train_window"],
            test_window=settings["model"]["test_window"],
            min_train_size=settings["model"]["min_train_size"],
        )
        train_models(dataset, feature_columns, train_config)
        model = load_model(model_dir)

        latest_features = dataset[feature_columns].tail(1)
        prob_up = predict_probabilities(model, latest_features)
        predictions.append(format_prediction(horizon, prob_up))

    output_dir = Path(settings["visualization"]["output_dir"])
    plot_close_trend(feature_frame, output_dir)
    plot_return_distribution(feature_frame, output_dir)

    results = [prediction.__dict__ for prediction in predictions]
    result_path = output_dir / "predictions.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Prediction results saved to:", result_path)


if __name__ == "__main__":
    main()
