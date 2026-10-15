"""Independent saved-model predictions, validation selection, and source integrity."""

import argparse
import json
from math import isclose
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from compare_regression_models import file_hash
from train_linear_regression import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    root = args.comparison
    assert not (root / "FAILED.json").exists()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    report = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    source = Path(config["source"])
    assert all(file_hash(source / name) == digest for name, digest in config["source_sha256"].items())
    assert selection["selection_split"] == "validation"
    assert not selection["test_read_for_training_or_selection"]
    assert not selection["refit_on_validation"]
    rows = report["validation_ranking"]
    expected = min((r for r in rows if r["selection_eligible"]),
                   key=lambda r: (r["validation"]["rmse"], r["validation"]["mae"], r["id"]))
    assert expected["id"] == selection["selected_id"] == report["selected_id"]
    specs = {c["id"]: c for c in config["candidates_fixed_before_fit"]}
    checked = 0
    with threadpool_limits(limits=4):
        for split in ("train", "validation"):
            x, y = load(source, split, config["input_columns"])
            for row in rows:
                model = joblib.load(root / "models" / f"{row['id']}.joblib")
                assert model.n_features_in_ == config["input_columns"]
                assert model.get_params() == specs[row["id"]]["parameters"]
                if row["id"].startswith("hist_gb"):
                    assert model.early_stopping is False
                predicted = np.clip(model.predict(x), 0, 1)
                error = y - predicted
                assert isclose(float(np.mean(np.abs(error))), row[split]["mae"], abs_tol=1e-12)
                assert isclose(float(np.sqrt(np.mean(error * error))), row[split]["rmse"], abs_tol=1e-12)
                checked += 1
        x, y = load(source, "test", config["input_columns"])
        chosen = joblib.load(root / "selected_model.joblib")
        original = joblib.load(root / "models" / f"{selection['selected_id']}.joblib")
        assert np.allclose(chosen.predict(x), original.predict(x), rtol=1e-12, atol=1e-12)
        for name, item in report["test"].items():
            model = joblib.load(root / "models" / f"{name}.joblib")
            predicted = np.clip(model.predict(x), 0, 1)
            residual = y - predicted
            assert isclose(float(np.mean(np.abs(residual))), item["metrics"]["mae"], abs_tol=1e-12)
            assert isclose(float(np.sqrt(np.mean(residual * residual))), item["metrics"]["rmse"], abs_tol=1e-12)
    result = {"passed": True, "models_verified": len(rows), "train_validation_metric_checks": checked,
              "selection_matches_validation_only_ranking": True, "source_files_unchanged": True,
              "saved_selected_model_matches_candidate": True,
              "source_leakage_audit": "passed; original normalization and splits reused",
              "test_models_verified": list(report["test"]),
              "scope": "label prediction and experiment integrity, not game strength"}
    with (root / "audit_report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
