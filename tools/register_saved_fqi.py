"""Register already-trained FQI weights with their frozen normalizer; no fit."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from catanzero.portable_trees import PortableTrees


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, nargs="?", default=REPO / "runs/fqi-pilot-500-v1")
    args = parser.parse_args()
    root = args.study.resolve()
    tracking = json.loads((root / "tracking.json").read_text())
    report = json.loads((root / "report.json").read_text())
    config = json.loads((root / "config.json").read_text())
    normalizer = json.loads((root / "normalizer.json").read_text())
    model_path = root / "model.joblib"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_hash == report["model_sha256"]
    assert normalizer["fit_split"] == "train"
    model = joblib.load(model_path)
    assert type(model).__name__ == "ExtraTreesRegressor" and model.n_features_in_ == 164
    assert len(model.estimators_) == 48 and model.max_depth == 8
    # Deterministic accumulation during inference; fitted trees are unchanged.
    model.set_params(n_jobs=1)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(normalizer["offset"], dtype=np.float64)
    scaler.scale_ = np.asarray(normalizer["scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = model.n_features_in_
    scaler.n_samples_seen_ = normalizer["fit_rows"]
    assert np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all()
    # Preserve training's float64 arithmetic before sklearn converts to float32.
    pipeline = Pipeline([("float64", FunctionTransformer(np.asarray, kw_args={"dtype": np.float64})),
                         ("normalization", scaler), ("extra_trees", model)])
    samples = []
    for index in (0, 1, 2, 3, 400, 401, 402, 403):
        with gzip.open(root / f"collection/game_{index:04d}/experience.pkl.gz", "rb") as stream:
            episode = pickle.load(stream)
        rows = [np.asarray(s["features"], dtype=np.float64).reshape(s["candidate_count"], 164)
                for s in episode["states"]]
        if rows:
            samples.append(np.concatenate(rows)[::3][:64])
    raw = np.concatenate(samples)
    normalized = ((raw-scaler.mean_)/scaler.scale_).astype(np.float32)
    expected = model.predict(normalized)
    assert np.array_equal(pipeline.predict(raw), expected)
    assert np.array_equal(pipeline.predict(raw.astype(np.float32)), expected)
    portable = PortableTrees.load(root / "model.json", root / "normalizer.json")
    portable_error = float(np.max(np.abs(expected-np.asarray([portable.predict_normalized(x) for x in normalized]))))
    assert portable_error < 1e-10
    mlflow.set_tracking_uri(tracking["tracking_uri"])
    client = MlflowClient()
    previous_registration = root / "registration.json"
    if previous_registration.exists():
        previous = json.loads(previous_registration.read_text())
        assert previous["model_sha256"] == model_hash
        restored = mlflow.sklearn.load_model(previous["registry_uri"])
        assert np.array_equal(restored.predict(raw), expected)
        print(json.dumps(previous, indent=2), flush=True)
        return
    mlflow.set_experiment(experiment_id=tracking["experiment_id"])
    requirements = [line.strip() for line in (REPO / "requirements-mlflow.txt").read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
    with mlflow.start_run(run_name="register_completed_FQI_with_normalization") as active:
        run_id = active.info.run_id
        mlflow.set_tags({"source_training_run_id": tracking["run_id"], "phase": "save_existing_model_only",
                         "game_strength_evaluated": "false", "model_retrained": "false",
                         "input_contract": "164 raw predecision features; normalization included"})
        mlflow.log_params({"model_type": "ExtraTreesRegressor", "algorithm": "Fitted Q Iteration",
                           "n_estimators": model.n_estimators, "max_depth": model.max_depth,
                           "min_samples_leaf": model.min_samples_leaf,
                           "fqi_iterations": config["iterations"], "training_transitions": report["training_transitions"],
                           "validation_transitions": report["validation_transitions"], "test_games_run": 0,
                           "model_sha256": model_hash, "serialization_format": "cloudpickle",
                           "normalization_included": True})
        # This is our own fitted model, verified against its training SHA256.
        # Explicit format avoids the skops default that rejected sklearn Trees.
        info = mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle",
                                       input_example=raw[:3], signature=infer_signature(raw, expected),
                                       pip_requirements=requirements)
        name = "CatanZero_ExtraTrees_FQI"
        version = mlflow.register_model(info.model_uri, name)
        registry_uri = f"models:/{name}/{version.version}"
        restored = mlflow.sklearn.load_model(registry_uri)
        reload_error = float(np.max(np.abs(restored.predict(raw)-expected)))
        assert reload_error == 0
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash
        client.set_registered_model_alias(name, "pilot", str(version.version))
        registration = {"saved": True, "registered_model": name, "version": str(version.version), "alias": "pilot",
                        "registry_uri": registry_uri, "logged_model_uri": info.model_uri,
                        "registration_run_id": run_id, "source_training_run_id": tracking["run_id"],
                        "tracking_uri": tracking["tracking_uri"], "model_sha256": model_hash,
                        "normalization_included": True, "input_columns": 164, "input_dtype": "float64",
                        "verification_rows": len(raw), "reload_max_abs_error": reload_error,
                        "portable_max_abs_error": portable_error, "model_retrained": False,
                        "game_evaluation_completed": False}
        write(previous_registration, registration)
        for file in (model_path, root / "model.json", root / "normalizer.json", root / "config.json",
                     root / "training_history.json", root / "experience_audit.json", root / "report.json", previous_registration):
            mlflow.log_artifact(str(file), "saved_model")
        mlflow.log_metrics({"reload_max_abs_error": reload_error, "portable_max_abs_error": portable_error,
                            "verification_rows": len(raw)})
        client.set_tag(tracking["run_id"], "model_registration_recovered_by_run", run_id)
        client.set_tag(tracking["run_id"], "registered_model_version", f"{name}/{version.version}")
    assert client.get_run(run_id).info.status == "FINISHED"
    report.update(sklearn_registry_model_logged=True, registration=registration)
    write(root / "report.json", report)
    client.log_artifact(run_id, str(root / "report.json"), "saved_model")
    print(json.dumps(registration, indent=2), flush=True)


if __name__ == "__main__":
    main()
