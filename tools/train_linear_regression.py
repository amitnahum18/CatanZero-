"""Fit a fixed-alpha linear ridge baseline using an existing NumPy environment.

Run with host `python`, after the project venv exports numeric data. No raw
simulator records or diagnostic labels are imported into the fitting process.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def load(root, split, size):
    with gzip.open(root / f"{split}.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if any(set(row) != {"x", "y"} or len(row["x"]) != size for row in rows):
        raise ValueError("invalid numeric X/y schema")
    x = np.asarray([r["x"] for r in rows], dtype=float)
    y = np.asarray([r["y"] for r in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError("nonfinite features or invalid target")
    return x, y


def metrics(actual, predicted):
    residual = actual - predicted
    denominator = np.sum((actual - actual.mean()) ** 2)
    return {"mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r2": float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    train_x, train_y = load(root, "train", schema["input_columns"])
    if len(train_y) == 0:
        raise ValueError("no training rows")
    # Alpha is fixed in advance. No validation/test-based hyperparameter search.
    alpha = 1.0
    mean_x, mean_y = train_x.mean(axis=0), train_y.mean()
    centered_x, centered_y = train_x - mean_x, train_y - mean_y
    weights = np.linalg.solve(centered_x.T @ centered_x + alpha * np.eye(train_x.shape[1]),
                              centered_x.T @ centered_y)
    intercept = float(mean_y - mean_x @ weights)
    model = {"model_type": "linear_ridge_regression", "alpha": alpha,
             "feature_version": schema["version"], "feature_names": schema["feature_names"],
             "input_columns": len(weights), "output_columns": 1,
             "trained_on": "train", "train_rows": len(train_y),
             "coefficients": weights.tolist(), "intercept": intercept,
             "normalizer": "normalizer.json", "prediction_clipping": [0, 1],
             "target": "weighted_score", "probability_calibration": False}
    with (root / "linear_model.json").open("x", encoding="utf-8") as stream:
        json.dump(model, stream, indent=2, allow_nan=False)
    evaluations = {}
    for split in ("train", "validation", "test"):
        x, y = (train_x, train_y) if split == "train" else load(root, split, len(weights))
        predicted_raw = x @ weights + intercept
        predicted = np.clip(predicted_raw, 0, 1)
        evaluations[split] = {"rows": len(y), "model": metrics(y, predicted),
                              "raw_model": metrics(y, predicted_raw),
                              "constant_train_mean_baseline": metrics(y, np.full(len(y), mean_y)),
                              "predictions_clipped": int(((predicted_raw < 0) | (predicted_raw > 1)).sum())}
    report = {"model_type": model["model_type"], "alpha_fixed_before_evaluation": alpha,
              "input_columns": len(weights), "output_columns": 1,
              "constant_baseline_value_fitted_on_train": float(mean_y), "splits": evaluations,
              "scope": "label prediction only; no measured game strength or calibrated confidence"}
    with (root / "model_metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
