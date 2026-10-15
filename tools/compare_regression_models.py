"""Small, predefined classical-regression comparison; select on validation only.

Uses the installed host Python's sklearn/NumPy. All model fitting uses train;
the existing test split is evaluated for the frozen winner and original Ridge.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from threadpoolctl import threadpool_limits

from train_linear_regression import load, metrics


def candidates():
    models = [(f"ridge_{alpha}", "Ridge", Ridge(alpha=alpha)) for alpha in (1, 10, 100, 1000)]
    models += [
        ("lasso", "Lasso", Lasso(alpha=0.001, max_iter=10000, tol=0.0001)),
        ("elastic_net", "ElasticNet", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, tol=0.0001)),
        ("random_forest", "Random Forest", RandomForestRegressor(
            n_estimators=160, max_depth=10, min_samples_leaf=20,
            max_features=0.75, n_jobs=4, random_state=42)),
        ("extra_trees", "Extra Trees", ExtraTreesRegressor(
            n_estimators=160, max_depth=10, min_samples_leaf=20,
            max_features=0.75, n_jobs=4, random_state=42)),
        ("hist_gb_depth3", "Gradient Boosting", HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_depth=3, max_leaf_nodes=8,
            min_samples_leaf=50, l2_regularization=10.0, early_stopping=False, random_state=42)),
        ("hist_gb_depth5", "Gradient Boosting", HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=5, max_leaf_nodes=24,
            min_samples_leaf=50, l2_regularization=10.0, early_stopping=False, random_state=42)),
    ]
    return models


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(estimator, x, y):
    started = perf_counter()
    raw = estimator.predict(x)
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite model prediction")
    clipped = np.clip(raw, 0, 1)
    return {"rows": len(y), **metrics(y, clipped),
            "predictions_clipped": int(((raw < 0) | (raw > 1)).sum()),
            "prediction_seconds": perf_counter() - started}


def game_metrics(root, split, actual, predicted):
    metadata = [json.loads(line) for line in (root / f"{split}.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metadata) == len(actual)
    groups = {}
    for i, meta in enumerate(metadata):
        groups.setdefault((meta["source"], meta["game_index"]), []).append(i)
    games = [{"source": source, "game_index": index, "rows": len(indices),
              **metrics(actual[indices], predicted[indices])}
             for (source, index), indices in groups.items()]
    return {"games": games, "game_count": len(games),
            "mean_per_game_mae": float(np.mean([g["mae"] for g in games])),
            "mean_per_game_rmse": float(np.mean([g["rmse"] for g in games]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    if (root / "FAILED.json").exists():
        parser.error("source is marked FAILED")
    audit = json.loads((root / "leakage_audit.json").read_text(encoding="utf-8"))
    if not audit["passed"] or audit["cross_split_feature_overlap"]:
        parser.error("source leakage audit did not pass")
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    normalizer = json.loads((root / "normalizer.json").read_text(encoding="utf-8"))
    if normalizer["fit_split"] != "train" or normalizer["feature_names"] != schema["feature_names"]:
        parser.error("invalid source normalizer")
    train_x, train_y = load(root, "train", schema["input_columns"])
    val_x, val_y = load(root, "validation", schema["input_columns"])
    if normalizer["fit_rows"] != len(train_y):
        parser.error("normalizer train row count mismatch")
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    specs = candidates()
    hashed_files = [f"{split}{suffix}" for split in ("train", "validation", "test")
                    for suffix in (".jsonl.gz", ".metadata.jsonl")]
    hashed_files += ["normalizer.json", "feature_schema.json", "leakage_audit.json"]
    hashes = {name: file_hash(root / name) for name in hashed_files}
    write_new(output / "config.json", {
        "version": "validation_selected_classical_regression_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "source": str(root),
        "input_columns": schema["input_columns"], "output_columns": 1,
        "train_rows": len(train_y), "validation_rows": len(val_y),
        "selection_metric": "validation RMSE after fixed [0,1] clipping",
        "tie_break": "validation MAE, then model ID",
        "candidates_fixed_before_fit": [{"id": name, "family": family, "parameters": estimator.get_params()}
                                        for name, family, estimator in specs],
        "source_sha256": hashes, "environment": {"numpy": np.__version__, "sklearn": sklearn.__version__,
                                                 "joblib": joblib.__version__},
        "limitations": ["weak heuristic/outcome labels, not action correctness or calibrated confidence",
                        "only five validation and five test games; rows within games are correlated",
                        "existing test has been reported before; this is exploratory confirmation, not a new sealed test",
                        "label-source heuristic versions differ; no real game strength measured"],
    })
    started = perf_counter()
    trained, results = {}, []
    try:
        with threadpool_limits(limits=4), (output / "validation_results.jsonl").open("x", encoding="utf-8") as stream:
            for name, family, estimator in specs:
                fit_started = perf_counter()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(train_x, train_y)
                fit_seconds = perf_counter() - fit_started
                row = {"id": name, "family": family, "fit_seconds": fit_seconds,
                       "train": score(estimator, train_x, train_y),
                       "validation": score(estimator, val_x, val_y),
                       "warnings": [str(w.message) for w in captured],
                       "selection_eligible": not any(issubclass(w.category, ConvergenceWarning) for w in captured)}
                artifact = output / "models" / f"{name}.joblib"
                joblib.dump(estimator, artifact, compress=3)
                reloaded = joblib.load(artifact)
                assert np.allclose(estimator.predict(val_x[:8]), reloaded.predict(val_x[:8]), rtol=1e-12, atol=1e-12)
                trained[name] = estimator
                results.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print(f"{name}: validation RMSE={row['validation']['rmse']:.6f} MAE={row['validation']['mae']:.6f} fit={fit_seconds:.2f}s", flush=True)
            eligible = [r for r in results if r["selection_eligible"]]
            selected = min(eligible, key=lambda r: (r["validation"]["rmse"], r["validation"]["mae"], r["id"]))
            # Persist the selection BEFORE reading test labels or scoring test.
            write_new(output / "selection.json", {
                "selected_id": selected["id"], "family": selected["family"],
                "selection_split": "validation", "selection_metric": "rmse",
                "validation_metrics": selected["validation"],
                "test_read_for_training_or_selection": False,
                "refit_on_validation": False, "train_rows": len(train_y),
            })
            shutil.copyfile(output / "models" / f"{selected['id']}.joblib", output / "selected_model.joblib")
            shutil.copyfile(root / "normalizer.json", output / "normalizer.json")
            shutil.copyfile(root / "feature_schema.json", output / "feature_schema.json")
            test_x, test_y = load(root, "test", schema["input_columns"])
            final_test = {}
            for name in dict.fromkeys((selected["id"], "ridge_1")):
                estimator = trained[name]
                final_test[name] = {"metrics": score(estimator, test_x, test_y),
                                    "per_game": game_metrics(root, "test", test_y,
                                                              np.clip(estimator.predict(test_x), 0, 1))}
            baseline = metrics(test_y, np.full(len(test_y), train_y.mean()))
            # Cross-check sklearn Ridge against the existing custom solver.
            original = json.loads((root / "linear_model.json").read_text(encoding="utf-8"))
            expected = test_x @ np.asarray(original["coefficients"]) + original["intercept"]
            assert np.allclose(trained["ridge_1"].predict(test_x), expected, rtol=1e-8, atol=1e-8)
        assert hashes == {name: file_hash(root / name) for name in hashed_files}
        rankings = sorted(results, key=lambda r: (r["validation"]["rmse"], r["validation"]["mae"], r["id"]))
        report = {"selected_id": selected["id"], "selected_family": selected["family"],
                  "models_tried": len(results), "ranking_metric": "validation RMSE",
                  "input_columns": schema["input_columns"], "output_columns": 1,
                  "split_rows": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
                  "validation_ranking": rankings, "test": final_test,
                  "constant_train_mean_test_baseline": baseline,
                  "source_data_unchanged": True, "saved_model_roundtrips_verified": len(results),
                  "original_ridge_predictions_reproduced": True,
                  "elapsed_seconds": perf_counter() - started,
                  "scope": "predicts weak labels; not a game win-rate evaluation"}
        write_new(output / "comparison.json", report)
        lines = ["# Classical model comparison", "", "Selection uses validation RMSE only. Lower is better.", "",
                 "| Model | Validation MAE | Validation RMSE | Train RMSE | Fit seconds |",
                 "|---|---:|---:|---:|---:|"]
        lines += [f"| {r['id']} | {r['validation']['mae']:.6f} | {r['validation']['rmse']:.6f} | {r['train']['rmse']:.6f} | {r['fit_seconds']:.2f} |" for r in rankings]
        lines += ["", f"Selected: **{selected['id']}**. No validation refit.", "",
                  "| Frozen test model | Test MAE | Test RMSE | Test R2 |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {name} | {item['metrics']['mae']:.6f} | {item['metrics']['rmse']:.6f} | {item['metrics']['r2']:.6f} |" for name, item in final_test.items()]
        lines += [f"| Constant train mean | {baseline['mae']:.6f} | {baseline['rmse']:.6f} | {baseline['r2']:.6f} |", "",
                  "All source data hashes are unchanged. All 10 saved estimators were reloaded and predictions checked.", "",
                  "Only five held-out games per split. Existing test results were reported earlier; this is an exploratory comparison.",
                  "Target remains a heuristic/outcome blend, not expert correctness or calibrated confidence. No game-playing strength measured."]
        (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"selected": selected["id"], "test": final_test,
                          "constant_baseline": baseline, "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    except Exception as error:
        write_new(output / "FAILED.json", {"error": str(error), "use_for_training": False})
        raise


if __name__ == "__main__":
    main()
