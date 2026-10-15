"""MLflow study: frozen mixed-label GB versus same-parameter terminal-only GB.

Run with .ml-venv Python. Simulation remains in the pinned engine .venv.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from math import comb
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import sklearn
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from train_linear_regression import load, metrics
from compare_regression_models import file_hash

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from catanzero.portable_trees import PortableTrees

EXPERIMENT_NAME = "CatanZero_Label_Target_Study"
REGISTRY_NAME = "CatanZero_GradientBoosting_LabelStudy"


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def canonical_hash(value):
    # Match the pinned simulator's tournament hashing convention.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def labels_from_sources(dataset):
    report = json.loads((dataset / "data_report.json").read_text(encoding="utf-8"))
    lookup, board_hashes = {}, set()
    for entry in report["sources"]:
        root = Path(entry["source"])
        for line in (root / "maps.jsonl").read_text(encoding="utf-8").splitlines():
            board_hashes.add(canonical_hash(json.loads(line)["map"]))
        results = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        for result in results:
            with gzip.open(root / result["moves_file"], "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if not row["train_eligible"]:
                        continue
                    key = (root.name, row["game_index"], row["action_index"])
                    if key in lookup:
                        raise ValueError("duplicate source label")
                    targets = row["targets"]
                    outcome = targets["terminal_outcome"]
                    if outcome not in (0, 1) or outcome != float(row["inputs"]["observation"]["actor"] == result["winner"]):
                        raise ValueError("terminal outcome mismatch")
                    lookup[key] = {"heuristic": targets["heuristic_score"], "outcome": outcome,
                                   "mixed": targets["weighted_score"], "split": result["split"]}
    labels = {}
    for split in ("train", "validation", "test"):
        metadata = [json.loads(line) for line in (dataset / f"{split}.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
        labels[split] = [lookup[(m["source"], m["game_index"], m["action_index"])] for m in metadata]
        assert all(label["split"] == split for label in labels[split])
    return labels, sorted(board_hashes)


def export_model(estimator, schema, policy):
    trees = []
    for stage in estimator._predictors:
        if len(stage) != 1:
            raise ValueError("portable format supports one-output regression")
        nodes = stage[0].nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("categorical splits unsupported by portable format")
        trees.append([{"leaf": bool(n["is_leaf"]), "feature": int(n["feature_idx"]),
                       "threshold": float(n["num_threshold"]), "left": int(n["left"]),
                       "right": int(n["right"]), "value": float(n["value"])} for n in nodes])
    return {"version": "numeric_hist_gb_portable_v1", "policy": policy,
            "feature_version": schema["version"], "feature_names": schema["feature_names"],
            "input_columns": schema["input_columns"], "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": trees, "ranking": "argmax raw prediction; canonical action-order tie break"}


def play_job(job):
    manifest, output, board, leg, seed, seconds = job
    command = [str(REPO / ".venv/Scripts/python.exe"), "-u", "-m", "catanzero.learned_match", str(manifest),
               "--seed", str(seed), "--board-index", str(board), "--leg", str(leg),
               "--output", str(output), "--max-seconds", str(seconds)]
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    with (output / "worker.log").open("x", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(command, cwd=REPO, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=seconds + 20)
            status = completed.returncode
        except subprocess.TimeoutExpired:
            status = None
    path = output / "result.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        checkpoint = output / "checkpoint.json"
        result = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {
            "seed": seed, "board_index": board, "leg": leg, "actions": 0,
        }
        result.update(termination="hard_time_limit" if status is None else "worker_error",
                      winner=None, winner_policy=None, error=None if status is None else "worker exited without a result")
    result.update(worker_exit_code=status, match_directory=str(output))
    return result


def summarize(results, direct=False):
    completed = [r for r in results if r["termination"] == "victory"]
    wins = Counter(r["winner_policy"] for r in completed)
    n, unknown = len(results), len(results) - len(completed)
    groups = {}
    for policy in (("A_mixed", "B_terminal") if direct else ("A_mixed", "B_terminal", "F")):
        groups[policy] = {"wins": wins[policy], "win_rate_completed": wins[policy] / len(completed) if completed else None,
                          "all_scheduled_win_bounds": [wins[policy] / n, (wins[policy] + unknown) / n] if n else None}
    differences = []
    for board in sorted({r["board_index"] for r in results}):
        pair = [r for r in results if r["board_index"] == board]
        if len(pair) == 2 and all(r["termination"] == "victory" for r in pair):
            differences.append(sum(r["winner_policy"] == "A_mixed" for r in pair)
                               - sum(r["winner_policy"] == "B_terminal" for r in pair))
    plus, minus = sum(d > 0 for d in differences), sum(d < 0 for d in differences)
    discordant = plus + minus
    two_sided = min(1.0, 2 * sum(comb(discordant, k) for k in range(min(plus, minus) + 1)) / 2 ** discordant) if discordant else 1.0
    return {"games": n, "completed_games": len(completed), "unfinished_games": unknown,
            "termination_counts": dict(Counter(r["termination"] for r in results)), "groups": groups,
            "seed_blocks_completed": len(differences), "paired_seed_win_differences_A_minus_B": differences,
            "paired_sign_test_two_sided_p": two_sided,
            "model_win_share_excluding_F_wins": wins["A_mixed"] / (wins["A_mixed"] + wins["B_terminal"]) if wins["A_mixed"] + wins["B_terminal"] else None,
            "note": ("Fresh four-player 2A-vs-2B boards with swapped seats; exploratory pilot; unfinished games censored." if direct else
                     "Fresh four-player A,B,F,F boards with swapped seats; pilot only. F can win; unfinished games censored.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "runs/regression-competitive-v2")
    parser.add_argument("--existing-model", type=Path, default=REPO / "runs/model-comparison-v1/selected_model.joblib")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.boards < 4 or args.boards % 4 or args.workers < 1 or args.max_seconds <= 0:
        parser.error("boards must be a positive multiple of four; positive workers/time")
    output, dataset = args.output.resolve(), args.dataset.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "matches").mkdir()
    schema = json.loads((dataset / "feature_schema.json").read_text(encoding="utf-8"))
    normalizer = json.loads((dataset / "normalizer.json").read_text(encoding="utf-8"))
    leakage = json.loads((dataset / "leakage_audit.json").read_text(encoding="utf-8"))
    if not leakage["passed"] or normalizer["fit_split"] != "train":
        raise ValueError("source leakage audit or normalizer failed")
    labels, board_hashes = labels_from_sources(dataset)
    data = {s: load(dataset, s, schema["input_columns"]) for s in labels}
    for split, (x, y) in data.items():
        assert len(y) == len(labels[split])
        assert np.allclose(y, [l["mixed"] for l in labels[split]], atol=1e-12, rtol=0)
    baseline = joblib.load(args.existing_model)
    if type(baseline).__name__ != "HistGradientBoostingRegressor" or baseline.max_depth != 3:
        raise ValueError("expected existing selected depth-3 histogram GB regressor")
    model_b = clone(baseline)
    started = perf_counter()
    with threadpool_limits(limits=4):
        model_b.fit(data["train"][0], np.asarray([l["outcome"] for l in labels["train"]]))
    if model_b.get_params() != baseline.get_params():
        raise ValueError("training architecture/parameters differ")
    shutil.copyfile(dataset / "normalizer.json", output / "models/normalizer.json")
    shutil.copyfile(dataset / "feature_schema.json", output / "models/feature_schema.json")
    source_files = [f"{s}{suffix}" for s in data for suffix in (".jsonl.gz", ".metadata.jsonl")]
    source_files += ["feature_schema.json", "normalizer.json", "leakage_audit.json"]
    hashes = {name: file_hash(dataset / name) for name in source_files}
    config = {"created_at": datetime.now(timezone.utc).isoformat(), "hypothesis": "effect of removing heuristic supervision",
              "model_type": type(baseline).__name__, "model_parameters": baseline.get_params(),
              "dataset": str(dataset), "input_columns": schema["input_columns"], "output_columns": 1,
              "split_rows": {s: len(v[1]) for s, v in data.items()}, "source_hashes": hashes,
              "A_mixed_weights": {"heuristic": 0.5, "terminal_outcome": 0.5},
              "B_terminal_weights": {"heuristic": 0.0, "terminal_outcome": 1.0},
              "game_protocol": {"boards": args.boards, "legs_per_board": 2, "base_seed": args.seed,
                                "lineup": ["A_mixed", "B_terminal", "F", "F"], "seat_swaps": True,
                                "max_seconds": args.max_seconds, "max_actions": 3000, "max_turns": 1000,
                                "workers": args.workers, "policy_selection": "argmax raw prediction",
                                "hidden_information": "all players act on detached public-count belief; learned encoder whitelists visible inputs"},
              "environment": {"mlflow": mlflow.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
              "limitations": ["binary terminal labels remain noisy action supervision",
                              "label-scale errors between different targets are not directly comparable",
                              "existing held-out label test has previously been reported",
                              "four-player study, two F fillers; seed-paired pilot, not a definitive strength ranking"]}
    write_json(output / "config.json", config)
    tracking_root = REPO / "runs/mlflow"
    tracking_root.mkdir(exist_ok=True)
    artifact_root = tracking_root / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    tracking_uri = f"sqlite:///{(tracking_root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root.as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    model_records, label_metrics = {}, {}
    try:
        for policy, estimator, weights, alias in (
            ("A_mixed", baseline, config["A_mixed_weights"], "mixed_50_50"),
            ("B_terminal", model_b, config["B_terminal_weights"], "terminal_only")):
            joblib.dump(estimator, output / "models" / f"{policy}.joblib", compress=3)
            portable_doc = export_model(estimator, schema, policy)
            portable_path = output / "models" / f"{policy}.json"
            write_json(portable_path, portable_doc)
            portable = PortableTrees(portable_doc, normalizer)
            label_metrics[policy] = {}
            with threadpool_limits(limits=4):
                for split, (x, y_mixed) in data.items():
                    raw = estimator.predict(x)
                    exported = np.asarray([portable.predict_normalized(v) for v in x])
                    assert np.allclose(raw, exported, rtol=1e-12, atol=1e-12)
                    y = y_mixed if policy == "A_mixed" else np.asarray([l["outcome"] for l in labels[split]])
                    label_metrics[policy][split] = {"own_target": metrics(y, np.clip(raw, 0, 1)),
                                                  "terminal_reference": metrics(np.asarray([l["outcome"] for l in labels[split]]), np.clip(raw, 0, 1)),
                                                  "portable_max_abs_error": float(np.max(np.abs(raw - exported)))}
            with mlflow.start_run(run_name=policy) as active:
                run_id = active.info.run_id
                mlflow.log_params({"label_heuristic_weight": weights["heuristic"],
                                   "label_outcome_weight": weights["terminal_outcome"],
                                   "train_rows": len(data["train"][1]), "input_columns": schema["input_columns"],
                                   "max_depth": estimator.max_depth, "max_iter": estimator.max_iter,
                                   "learning_rate": estimator.learning_rate, "l2_regularization": estimator.l2_regularization,
                                   "min_samples_leaf": estimator.min_samples_leaf, "fit_split": "train"})
                mlflow.set_tags({"policy": policy, "study": "label_target", "source_output": str(output),
                                 "normalizer_fit_split": "train", "game_policy": "argmax_raw_prediction"})
                mlflow.log_artifact(output / "config.json", "study")
                mlflow.log_artifact(portable_path, "portable")
                mlflow.log_artifact(output / "models/normalizer.json", "portable")
                mlflow.log_artifact(output / "models/feature_schema.json", "portable")
                mlflow.log_artifact(dataset / "leakage_audit.json", "study")
                for split, values in label_metrics[policy].items():
                    mlflow.log_metrics({f"{split}_{metric}": value for metric, value in values["own_target"].items() if value is not None})
                model_info = mlflow.sklearn.log_model(
                    estimator, name="model", serialization_format="cloudpickle",
                    signature=infer_signature(data["train"][0][:3], estimator.predict(data["train"][0][:3])),
                    input_example=data["train"][0][:3],
                    pip_requirements=[f"scikit-learn=={sklearn.__version__}", f"numpy=={np.__version__}", f"cloudpickle=={__import__('cloudpickle').__version__}"],
                    metadata={"label_weights": weights, "input_contract": "164 already-normalized predecision numeric features"})
                version = mlflow.register_model(model_info.model_uri, REGISTRY_NAME)
                client.set_registered_model_alias(REGISTRY_NAME, alias, str(version.version))
                reload_path = output / "mlflow_restored" / policy
                reload_path.mkdir(parents=True)
                sklearn_reload = reload_path / "sklearn"
                sklearn_reload.mkdir()
                restored = mlflow.sklearn.load_model(model_info.model_uri, dst_path=str(sklearn_reload))
                assert np.allclose(restored.predict(data["validation"][0]), estimator.predict(data["validation"][0]), rtol=1e-12, atol=1e-12)
                downloaded = Path(client.download_artifacts(run_id, "portable", dst_path=str(reload_path)))
                downloaded_model = downloaded / f"{policy}.json"
                assert file_hash(downloaded_model) == file_hash(portable_path)
                model_records[policy] = {"run_id": run_id, "model_uri": model_info.model_uri,
                                         "registered_model": REGISTRY_NAME, "model_version": str(version.version),
                                         "alias": alias, "label_weights": weights,
                                         "portable_path": str(downloaded_model), "portable_sha256": file_hash(downloaded_model),
                                         "mlflow_reload_verified": True}
            print(f"Saved and restored {policy} in MLflow run={run_id} version={version.version}", flush=True)
        write_json(output / "label_metrics.json", label_metrics)
        manifest = {**model_records, "normalizer_path": str(output / "models/normalizer.json"),
                    "source_board_hashes": board_hashes, "tracking_uri": tracking_uri,
                    "experiment_id": experiment_id, "models_saved_before_gameplay": True}
        write_json(output / "manifest.json", manifest)
        jobs = []
        for board in range(args.boards):
            for leg in (0, 1):
                directory = output / "matches" / f"board_{board:04d}_leg_{leg}"
                directory.mkdir()
                jobs.append((output / "manifest.json", directory, board, leg, args.seed + board, args.max_seconds))
        results = []
        with mlflow.start_run(run_name="fresh_paired_games") as evaluation_run:
            evaluation_id = evaluation_run.info.run_id
            mlflow.log_params({"boards": args.boards, "scheduled_games": len(jobs), "base_seed": args.seed,
                               "A_run_id": model_records["A_mixed"]["run_id"], "B_run_id": model_records["B_terminal"]["run_id"],
                               "lineup": "A_mixed,B_terminal,F,F", "seat_swaps": True})
            with ThreadPoolExecutor(max_workers=args.workers) as pool, (output / "results.jsonl").open("x", encoding="utf-8") as stream:
                futures = [pool.submit(play_job, job) for job in jobs]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    stream.flush()
                    print(f"[{len(results)}/{len(jobs)}] board={result['board_index']} leg={result['leg']} {result['termination']} winner={result.get('winner_policy')} actions={result['actions']}", flush=True)
            summary = summarize(results)
            summary.update(evaluation_run_id=evaluation_id, tracking_uri=tracking_uri,
                           experiment_id=experiment_id, elapsed_seconds=perf_counter() - started,
                           model_records=model_records)
            write_json(output / "summary.json", summary)
            mlflow.log_metrics({"completed_games": summary["completed_games"], "unfinished_games": summary["unfinished_games"],
                                "paired_seed_sign_p": summary["paired_sign_test_two_sided_p"],
                                **{f"{p}_wins": g["wins"] for p, g in summary["groups"].items()}})
            mlflow.log_artifacts(str(output / "matches"), "matches")
            for file in ("summary.json", "results.jsonl", "config.json", "manifest.json", "label_metrics.json"):
                mlflow.log_artifact(output / file, "study")
            for policy, record in model_records.items():
                client.set_tag(record["run_id"], "evaluation_run_id", evaluation_id)
                client.log_metric(record["run_id"], "fresh_game_wins", summary["groups"][policy]["wins"])
                if summary["groups"][policy]["win_rate_completed"] is not None:
                    client.log_metric(record["run_id"], "fresh_game_win_rate", summary["groups"][policy]["win_rate_completed"])
        assert hashes == {name: file_hash(dataset / name) for name in source_files}
        write_json(output / "tracking.json", {"tracking_uri": tracking_uri, "experiment_id": experiment_id,
                                               "evaluation_run_id": evaluation_id, "model_runs": model_records,
                                               "source_data_unchanged": True})
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        write_json(output / "FAILED.json", {"error": str(error), "use_for_training": False})
        raise


if __name__ == "__main__":
    main()
