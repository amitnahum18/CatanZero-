"""Short live SARSA continuation with new features and paired frozen evaluation."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
from scipy.stats import binomtest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from train_online_sarsa import CentralSARSA, play_phase, write, REPO
from refit_linear_returns import FrozenLinear
from catanzero.online_linear import LinearActionValues
from catanzero.board_exclusions import normalization_source_board_hashes


def expanded_normalizer(source, dataset):
    old = json.loads((source / "normalizer.json").read_text())
    new = json.loads((dataset / "normalizer.json").read_text())
    count = len(old["feature_names"])
    assert new["feature_names"][:count] == old["feature_names"]
    for key in ("offset", "scale", "train_mean", "train_std"):
        if key in old and key in new:
            new[key][:count] = old[key]
    new["normalization_source_blocks"] = [
        {"start": 0, "stop": count, "source": str(source / "normalizer.json"), "fit_rows": old["fit_rows"]},
        {"start": count, "stop": len(new["feature_names"]), "source": str(dataset / "normalizer.json"), "fit_rows": new["fit_rows"]}]
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--eval-games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.games, args.eval_games) < 4 or args.games % 4 or args.eval_games % 4 or args.workers < 1:
        parser.error("balanced positive game counts and workers required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = REPO / "runs/online-sarsa-low-exploration-500-v3"
    dataset = REPO / "runs/regression-build-capacity-v3"
    norm = expanded_normalizer(source, dataset)
    write(root / "normalizer.json", norm)
    shutil.copyfile(dataset / "feature_schema.json", root / "feature_schema.json")
    old_norm = json.loads((source / "normalizer.json").read_text())
    old_estimator = joblib.load(source / "model.joblib")
    model = CentralSARSA(root, norm, expand_from=source)
    assert not model.model.coef_[164:].any()
    raw = np.asarray(norm["offset"])+np.random.default_rng(42).normal(size=(64, len(norm["offset"])))*norm["scale"]
    np.testing.assert_allclose(model.model.predict(model.normalize(raw)),
                               old_estimator.predict(FrozenLinear(old_estimator, old_norm, 0).normalize(raw[:, :164])), atol=1e-12, rtol=0)
    manifest = json.loads((REPO / "runs/mlflow-label-study-v2/manifest.json").read_text())
    excluded = list(manifest["source_board_hashes"])
    excluded += normalization_source_board_hashes(dataset)
    excluded += json.loads((REPO / "runs/fqi-pilot-500-v1/experience_audit.json").read_text())["collection"]["board_hashes"]
    ancestor = source
    while ancestor:
        for phase in ("initial_evaluation", "live_training", "final_evaluation"):
            excluded.extend(json.loads(s)["initial_map_hash"] for s in (ancestor / phase / "results.jsonl").read_text().splitlines())
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        ancestor = Path(parent) if parent else None
    for name in ("linear-monte-carlo-refit-v2", "gbrl-mixed-fast-200-v1"):
        for phase in ("initial_evaluation", "live_training", "final_evaluation"):
            path = REPO / "runs" / name / phase / "results.jsonl"
            if path.exists():
                excluded.extend(json.loads(s)["initial_map_hash"] for s in path.read_text().splitlines())
    path = REPO / "runs/regression-search-depth-1-vs-2-v1/results.jsonl"
    excluded.extend(json.loads(s)["initial_map_hash"] for s in path.read_text().splitlines())
    config = {"algorithm": "live mini-batch semi-gradient SARSA", "model": "SGDRegressor", "input_columns": len(norm["offset"]),
              "trainable_parameters": len(norm["offset"])+1, "resume_from": str(source), "feature_expansion_from": 164,
              "new_weights_initially_zero": True, "old_normalization_preserved": True,
              "new_normalization_source": str(dataset), "source_model_sha256": hashlib.sha256((source / "model.joblib").read_bytes()).hexdigest(),
              "training_games": args.games, "evaluation_games": args.eval_games, "workers": args.workers,
              "forward_simulation": False, "reward_unchanged": True, "gamma": .999, "vp_weight": .05,
              "epsilon_start": .1, "epsilon_end": .05, "evaluation_seed": 17000000, "training_seed": 18000000,
              "opponent": "three frozen F_static", "selection": "fixed final model; no automatic promotion or extra training"}
    write(root / "config.json", config)
    client = mlflow.MlflowClient(tracking_uri=manifest["tracking_uri"])
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    mlflow.set_experiment("CatanZero_Online_Linear_SARSA")
    opponents = {"opponent_policy": "F_static", "max_actions": 2000, "max_seconds": 90, "trace_compresslevel": 1}
    started = perf_counter()
    with mlflow.start_run(run_name="feature_expansion_v3_100_live_games") as run:
        write(root / "tracking.json", {"run_id": run.info.run_id, "tracking_uri": manifest["tracking_uri"]})
        mlflow.log_params({"algorithm": "SARSA", "input_columns": config["input_columns"], "train_games": args.games,
                           "eval_games": args.eval_games, "new_weights_initially_zero": True, "forward_simulation": False,
                           "opponent": "F_static", "feature_version": norm["feature_version"]})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_play")
        print("STAGE: original 164-feature frozen model, fresh paired test", flush=True)
        before_rows, before = play_phase(root, "initial_evaluation", FrozenLinear(old_estimator, old_norm, 0),
                                         args.eval_games, args.workers, training=False, base_seed=config["evaluation_seed"],
                                         excluded=excluded, opponent_config=opponents)
        print("STAGE: live SARSA, 264 inputs, fresh training games", flush=True)
        train_started = perf_counter()
        training_rows, training = play_phase(root, "live_training", model, args.games, args.workers, training=True,
                                             base_seed=config["training_seed"], excluded=excluded+[r["initial_map_hash"] for r in before_rows],
                                             exploration_start=.1, exploration_end=.05, opponent_config=opponents)
        training_seconds = perf_counter()-train_started
        model.trace.close()
        write(root / "training_history.json", model.history)
        write(root / "linear_model.json", model.document())
        joblib.dump(model.model, root / "model.joblib", compress=3)
        frozen_weights = model.model.coef_.copy()
        print("STAGE: frozen expanded model, same test boards and seats", flush=True)
        after_rows, after = play_phase(root, "final_evaluation", model, args.eval_games, args.workers, training=False,
                                       base_seed=config["evaluation_seed"], excluded=excluded+[r["initial_map_hash"] for r in training_rows],
                                       opponent_config=opponents)
        assert np.array_equal(model.model.coef_, frozen_weights)
        plus = minus = 0
        for a, b in zip(before_rows, after_rows):
            assert (a["seed"], a["initial_map_hash"], a["learner_seat"]) == (b["seed"], b["initial_map_hash"], b["learner_seat"])
            if a["termination"] == b["termination"] == "victory":
                plus += b["learner_won"] and not a["learner_won"]
                minus += a["learner_won"] and not b["learner_won"]
        report = {"before": before, "training": training, "after": after, "central_updates": model.version,
                  "training_seconds": training_seconds, "training_transitions": model.transitions,
                  "updates_with_weight_change": sum(h["weight_change_l2"] > 0 for h in model.history),
                  "new_feature_weight_norm": float(np.linalg.norm(model.model.coef_[164:])),
                  "paired_improvements": int(plus), "paired_regressions": int(minus),
                  "paired_two_sided_p": float(binomtest(plus, plus+minus, .5).pvalue) if plus+minus else 1.,
                  "limitations": ["32 paired test games is a small pilot", "features plus additional training changed together",
                                  "capped games unknown; missing final transitions discarded", "road VP projections are estimates",
                                  "no independent full engine/update replay audit for this new run"]}
        write(root / "report.json", report)
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = np.asarray(norm["offset"]), np.asarray(norm["scale"])
        scaler.var_, scaler.n_features_in_ = scaler.scale_**2, len(norm["offset"])
        scaler.n_samples_seen_ = np.asarray([norm["normalization_source_blocks"][0]["fit_rows"]]*164+[norm["fit_rows"]]*(len(norm["offset"])-164))
        pipeline = Pipeline([("float64", FunctionTransformer(np.asarray, kw_args={"dtype": np.float64})), ("normalization", scaler),
                             ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -4, "a_max": 4})), ("sarsa", model.model)])
        expected = model.model.predict(model.normalize(raw))
        portable = LinearActionValues(model.document(), norm)
        np.testing.assert_allclose([portable.predict(x.tolist()) for x in raw], expected, atol=1e-12, rtol=0)
        requirements = [s for s in (REPO / "requirements-mlflow.txt").read_text().splitlines() if s and not s.startswith("#")]
        info = mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle",
                                      input_example=raw[:2], signature=infer_signature(raw, expected), pip_requirements=requirements)
        restored = mlflow.sklearn.load_model(info.model_uri)
        error = float(np.max(np.abs(restored.predict(raw)-expected)))
        assert error < 1e-12
        registered = mlflow.register_model(info.model_uri, "CatanZero_Online_Linear_SARSA")
        client.set_registered_model_alias("CatanZero_Online_Linear_SARSA", "feature_pilot", str(registered.version))
        write(root / "registration.json", {"registered_model": "CatanZero_Online_Linear_SARSA", "version": str(registered.version),
                                           "alias": "feature_pilot", "run_id": run.info.run_id, "model_uri": info.model_uri,
                                           "input_columns": len(norm["offset"]), "reload_max_abs_error": error,
                                           "normalization_and_clipping_included": True, "promoted_over_original": False})
        for filename in ("tools/train_feature_sarsa.py", "tools/train_online_sarsa.py", "src/catanzero/online_env.py", "src/catanzero/features.py", "src/catanzero/strategic_features.py", "src/catanzero/build_capacity.py"):
            dest = root / "implementation_snapshot" / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / filename, dest)
        (root / "README.md").write_text(
            "# Live SARSA with strategic and build-capacity features\n\n"
            "Original 164-input linear SARSA vs 264-input continuation, without forward search. Old coefficients, intercept, optimizer state and old normalization retained at initialization; new coefficients initialized at zero. "
            "Additional-feature normalization fitted only on recorded training rows. 100 new real training games with epsilon .10 to .05; reward/discount unchanged.\n\n"
            f"Original: {before['learner_wins']}/{args.eval_games}, {before['unfinished_games']} unfinished. "
            f"Expanded: {after['learner_wins']}/{args.eval_games}, {after['unfinished_games']} unfinished. "
            f"Paired p={report['paired_two_sided_p']:.4g}. Same fresh boards and seats versus three F_static. No automatic promotion.\n\n"
            "Feature expansion and additional training are both part of this change; this pilot cannot isolate their separate effects. "
            "No independent full engine/update replay audit of the new run has been performed.\n", encoding="utf-8")
        mlflow.log_metrics({"before_wins": before["learner_wins"], "after_wins": after["learner_wins"],
                            "training_seconds": training_seconds, "updates": model.version, "new_feature_weight_norm": report["new_feature_weight_norm"],
                            "paired_p": report["paired_two_sided_p"]})
        for path in root.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path), "feature_sarsa")
        print("RESULT: "+json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
