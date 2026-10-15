"""Fast offline Monte Carlo return refit, followed by a frozen paired pilot."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import gzip
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from train_online_sarsa import CentralSARSA, play_phase, write, REPO


def complete_returns(rows):
    """Return targets only for a complete, continuous, single-player trajectory."""
    rows = sorted(rows, key=lambda s: s["action_index"])
    if not rows or not rows[-1]["terminal"]:
        raise ValueError("Monte Carlo labels require a terminal outcome")
    for index, row in enumerate(rows):
        if row["terminal"] != (index == len(rows)-1):
            raise ValueError("early terminal")
        if row["terminal"] and row["discount"] != 0:
            raise ValueError("terminal must have zero continuation")
        if not 0 <= row["discount"] <= 1 or not np.isfinite(row["reward"]):
            raise ValueError("invalid reward or discount")
        if index+1 < len(rows):
            following = rows[index+1]
            if (row["next_action_index"] != following["action_index"] or
                    row["actor"] != following["actor"] or
                    row["game_index"] != following["game_index"] or
                    row["next_selected_index"] != following["selected_index"] or
                    not np.array_equal(row["next_x"], following["x"])):
                raise ValueError("trajectory discontinuity")
    targets = np.empty(len(rows))
    value = 0.
    for index in range(len(rows)-1, -1, -1):
        value = rows[index]["reward"] + rows[index]["discount"]*value
        targets[index] = value
    return rows, targets


class FrozenLinear:
    normalize = CentralSARSA.normalize
    choose = CentralSARSA.choose

    def __init__(self, estimator, normalizer, version):
        self.model, self.normalizer, self.version = estimator, normalizer, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=REPO / "runs/online-sarsa-low-exploration-500-v3")
    parser.add_argument("--eval-games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.eval_games < 4 or args.eval_games % 4 or args.workers < 1:
        parser.error("positive workers; evaluation games divisible by four")
    started = perf_counter()
    root, source = args.output.resolve(), args.source.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256((source / "model.joblib").read_bytes()).hexdigest()
    normalizer = json.loads((source / "normalizer.json").read_text())
    write(root / "normalizer.json", normalizer)
    old_model = joblib.load(source / "model.joblib")
    results = {r["game_index"]: r for r in map(json.loads, (source / "live_training/results.jsonl").read_text().splitlines())}
    grouped = {}
    with gzip.open(source / "updates.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            for sample in json.loads(line)["samples"]:
                assert sample["phase"] == "live_training"
                grouped.setdefault(sample["game_index"], []).append(sample)
    assert set(grouped) == set(results)
    x_list, y_list, group_list, weight_list, label_keys = [], [], [], [], []
    for game, rows in sorted(grouped.items()):
        assert results[game]["termination"] == "victory"
        rows, targets = complete_returns(rows)
        assert len(rows) == results[game]["learner_decisions"]
        x_list.extend(r["x"] for r in rows)
        y_list.extend(targets)
        group_list.extend([game]*len(rows))
        weight_list.extend([1/len(rows)]*len(rows))
        label_keys.extend((game, r["action_index"]) for r in rows)
    assert len(set(label_keys)) == len(label_keys)
    x, y, groups, weights = map(np.asarray, (x_list, y_list, group_list, weight_list))
    assert x.shape == (len(y), 164) and np.isfinite(x).all() and np.isfinite(y).all()
    games = np.asarray(sorted(grouped))
    train_games, valid_games = train_test_split(games, test_size=.2, random_state=42,
                                               stratify=[results[g]["learner_won"] for g in games])
    train, valid = np.isin(groups, train_games), np.isin(groups, valid_games)
    assert set(train_games).isdisjoint(valid_games)
    protocol = {"method": "offline Monte Carlo action-value regression with Ridge",
                "source": str(source), "source_model_sha256": source_hash,
                "training_games_available": len(games), "training_rows": len(y),
                "selection_train_games": train_games.tolist(), "selection_validation_games": valid_games.tolist(),
                "regularization_grid": [.001, .01, .1],
                "selection_metric": "held-out whole-game weighted return MSE; never evaluation wins",
                "sample_weight": "1 / number of learner decisions in each game",
                "targets": "G[t] = reward[t] + discount[t] * G[t+1], terminal continuation zero",
                "reward_and_discount_changed": False, "normalization_refitted": False,
                "forward_simulation": False, "fresh_training_games": 0,
                "evaluation_seed": 15000000, "evaluation_games_per_model": args.eval_games,
                "opponent": "three frozen F_static", "workers": args.workers,
                "promotion_rule": "no automatic promotion from this small pilot"}
    write(root / "config.json", protocol)
    fit_started = perf_counter()
    choices = []
    for strength in protocol["regularization_grid"]:
        estimator = Ridge(alpha=strength*len(train_games), solver="cholesky")
        estimator.fit(x[train], y[train], sample_weight=weights[train])
        mse = float(np.average((estimator.predict(x[valid])-y[valid])**2, weights=weights[valid]))
        choices.append({"regularization": strength, "validation_return_mse": mse})
    selected = min(choices, key=lambda c: c["validation_return_mse"])
    estimator = Ridge(alpha=selected["regularization"]*len(games), solver="cholesky")
    estimator.fit(x, y, sample_weight=weights)
    fit_seconds = perf_counter()-fit_started
    document = {"algorithm": "offline_Monte_Carlo_Ridge", "weights": estimator.coef_.tolist(),
                "intercept": float(estimator.intercept_), "feature_clip": 4., "input_columns": 164, "model_version": 1}
    write(root / "linear_model.json", document)
    joblib.dump(estimator, root / "model.joblib", compress=3)
    np.savez_compressed(root / "return_labels.npz", game_index=groups, action_index=np.asarray(label_keys)[:, 1], target=y)
    fit_report = {"choices": choices, "selected_regularization": selected["regularization"],
                  "baseline_validation_return_mse": float(np.average((old_model.predict(x[valid])-y[valid])**2, weights=weights[valid])),
                  "fit_seconds": fit_seconds,
                  "read_and_fit_seconds": perf_counter()-started,
                  "weight_change_l2": float(np.linalg.norm(estimator.coef_-old_model.coef_)),
                  "intercept_change": float(estimator.intercept_-old_model.intercept_[0]),
                  "complete_game_count": len(games), "unique_transition_count": len(y),
                  "return_target_min": float(y.min()), "return_target_max": float(y.max())}
    write(root / "fit_report.json", fit_report)
    print("REFIT COMPLETE: " + json.dumps(fit_report), flush=True)
    # Keep fresh test maps disjoint from all data used to train the source and normalizer.
    manifest = json.loads((REPO / "runs/mlflow-label-study-v2/manifest.json").read_text())
    excluded = list(manifest["source_board_hashes"])
    excluded += json.loads((REPO / "runs/fqi-pilot-500-v1/experience_audit.json").read_text())["collection"]["board_hashes"]
    ancestor = source
    while ancestor:
        for phase in ("initial_evaluation", "live_training", "final_evaluation"):
            excluded.extend(json.loads(line)["initial_map_hash"] for line in (ancestor / phase / "results.jsonl").read_text().splitlines())
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        ancestor = Path(parent) if parent else None
    excluded = sorted(set(excluded))
    write(root / "excluded_board_hashes.json", excluded)
    opponents = {"opponent_policy": "F_static", "max_actions": 2000, "max_seconds": 90, "trace_compresslevel": 1}
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    client = mlflow.MlflowClient()
    experiment_name = "CatanZero_Linear_Return_Refit"
    experiment = client.get_experiment_by_name(experiment_name)
    eid = experiment.experiment_id if experiment else client.create_experiment(experiment_name, artifact_location=(REPO / "runs/mlflow/artifacts").as_uri())
    mlflow.set_experiment(experiment_id=eid)
    with mlflow.start_run(run_name="offline_MC_Ridge_fast_paired_pilot") as run:
        write(root / "tracking.json", {"run_id": run.info.run_id, "tracking_uri": manifest["tracking_uri"]})
        mlflow.log_params({"method": protocol["method"], "source": str(source), "training_rows": len(y),
                           "regularization": selected["regularization"], "eval_games": args.eval_games,
                           "fresh_training_games": 0, "forward_simulation": False})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_test")
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = np.asarray(normalizer["offset"]), np.asarray(normalizer["scale"])
        scaler.var_, scaler.n_features_in_, scaler.n_samples_seen_ = scaler.scale_**2, 164, normalizer["fit_rows"]
        pipeline = Pipeline([("float64", FunctionTransformer(np.asarray, kw_args={"dtype": np.float64})),
                             ("normalization", scaler), ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -4, "a_max": 4})),
                             ("ridge", estimator)])
        raw = scaler.mean_+np.random.default_rng(42).normal(size=(64, 164))*scaler.scale_
        expected = estimator.predict(FrozenLinear(estimator, normalizer, 1).normalize(raw))
        np.testing.assert_allclose(pipeline.predict(raw), expected, atol=1e-12, rtol=0)
        from catanzero.online_linear import LinearActionValues
        portable = LinearActionValues(document, normalizer)
        np.testing.assert_allclose([portable.predict(row.tolist()) for row in raw], expected, atol=1e-12, rtol=0)
        requirements = [line for line in (REPO / "requirements-mlflow.txt").read_text().splitlines() if line and not line.startswith("#")]
        info = mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle", signature=infer_signature(raw, expected),
                                      input_example=raw[:2], pip_requirements=requirements)
        restored = mlflow.sklearn.load_model(info.model_uri)
        error = float(np.max(np.abs(restored.predict(raw)-expected)))
        assert error < 1e-12
        version = mlflow.register_model(info.model_uri, "CatanZero_Linear_MC_Ridge")
        write(root / "registration.json", {"run_id": run.info.run_id, "registered_model": "CatanZero_Linear_MC_Ridge",
                                           "version": str(version.version), "model_uri": info.model_uri,
                                           "reload_max_abs_error": error, "normalization_and_clipping_included": True,
                                           "promoted": False})
        frozen_hash = hashlib.sha256((root / "model.joblib").read_bytes()).hexdigest()
        print("STAGE: original frozen linear, 32 fresh games against F_static", flush=True)
        before_rows, before = play_phase(root, "initial_evaluation", FrozenLinear(old_model, normalizer, 0),
                                         args.eval_games, args.workers, training=False, base_seed=protocol["evaluation_seed"],
                                         excluded=excluded, opponent_config=opponents)
        print("STAGE: refitted frozen linear, same boards and seats", flush=True)
        after_rows, after = play_phase(root, "final_evaluation", FrozenLinear(estimator, normalizer, 1),
                                       args.eval_games, args.workers, training=False, base_seed=protocol["evaluation_seed"],
                                       excluded=excluded, opponent_config=opponents, learner_policy="Linear_MC_Ridge")
        positive = negative = 0
        for a, b in zip(before_rows, after_rows):
            assert (a["game_index"], a["seed"], a["initial_map_hash"], a["learner_seat"]) == (b["game_index"], b["seed"], b["initial_map_hash"], b["learner_seat"])
            assert a["initial_map_hash"] not in excluded
            if a["termination"] == b["termination"] == "victory":
                positive += b["learner_won"] and not a["learner_won"]
                negative += a["learner_won"] and not b["learner_won"]
        assert source_hash == hashlib.sha256((source / "model.joblib").read_bytes()).hexdigest()
        assert frozen_hash == hashlib.sha256((root / "model.joblib").read_bytes()).hexdigest()
        report = {"fit": fit_report, "before": before, "after": after,
                  "paired_improvements": int(positive), "paired_regressions": int(negative),
                  "paired_two_sided_p": float(binomtest(positive, positive+negative, .5).pvalue) if positive+negative else 1.,
                  "source_unchanged": True, "no_updates_during_evaluation": True,
                  "model_saved_before_testing": True, "elapsed_seconds": perf_counter()-started,
                  "limitations": ["small pilot; no automatic promotion", "offline returns describe the changing exploratory source policy, not optimal actions",
                                  "return prediction accuracy does not guarantee improved action selection", "spatial feature compression unchanged",
                                  "capped matches unknown, not counted as terminal losses", "no independent full engine replay of this new pilot"]}
        write(root / "report.json", report)
        (root / "README.md").write_text(
            f"# Fast linear Monte Carlo return refit\n\nOffline refit on {len(games)} recorded completed training games ({len(y)} unique decisions). "
            "Targets use the original rewards and discounts to terminal outcome. Ridge regularization selected on separate whole games, then refitted on all recorded training games. "
            "No new training games, teacher imitation, or forward search.\n\n"
            f"Original: {before['learner_wins']}/{args.eval_games} wins, {before['unfinished_games']} unfinished. "
            f"Refitted: {after['learner_wins']}/{args.eval_games} wins, {after['unfinished_games']} unfinished. "
            f"Same initial maps and seats; paired p={report['paired_two_sided_p']:.4g}. No automatic promotion.\n\n"
            "The source policy varied during collection. Monte Carlo return regression fits its outcomes; it does not establish optimal action values or calibrated win probabilities. "
            "Held-out return MSE and actual gameplay strength are separate measures. Full replay audit has not been performed for these new evaluation games.\n",
            encoding="utf-8")
        for filename in ("tools/refit_linear_returns.py", "tools/train_online_sarsa.py", "src/catanzero/online_env.py", "src/catanzero/online_linear.py"):
            destination = root / "implementation_snapshot" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / filename, destination)
        mlflow.log_metrics({"before_wins": before["learner_wins"], "after_wins": after["learner_wins"],
                            "fit_seconds": fit_seconds, "weight_change_l2": fit_report["weight_change_l2"],
                            "paired_p": report["paired_two_sided_p"]})
        for path in root.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path), "return_refit")
        print("RESULT: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
