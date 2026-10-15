"""Eight live environments, one incremental SGD/SARSA model, frozen evals."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from catanzero.online_linear import LinearActionValues, shaped_transition


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class CentralSARSA:
    def __init__(self, root, normalizer, batch_size=32, resume_from=None, expand_from=None, eta0=None):
        if eta0 is not None and (not np.isfinite(eta0) or eta0 <= 0):
            raise ValueError('eta0 must be finite and positive')
        self.root, self.normalizer, self.batch_size = root, normalizer, batch_size
        self.input_columns = len(normalizer["offset"])
        if resume_from is not None and expand_from is not None:
            raise ValueError("Choose resume or feature expansion")
        self.model = SGDRegressor(loss="squared_error", penalty="l2", alpha=.0001,
                                  learning_rate="constant", eta0=.05, fit_intercept=True,
                                  shuffle=False, average=False, random_state=42)
        self.model.partial_fit(np.zeros((1, self.input_columns)), np.zeros(1))
        assert not self.model.coef_.any() and not self.model.intercept_.any()
        def compatible(saved):
            if not isinstance(saved, SGDRegressor):
                return False
            expected, existing = self.model.get_params(), saved.get_params()
            if eta0 is not None:
                expected['eta0'] = existing['eta0']
            return existing == expected
        self.source_version = None
        if expand_from is not None:
            old_norm = json.loads((expand_from / "normalizer.json").read_text())
            count = len(old_norm["offset"])
            if count >= self.input_columns or any(normalizer[k][:count] != old_norm[k] for k in ("offset", "scale", "feature_names")):
                raise ValueError("Expansion requires an identical old feature prefix and normalization")
            saved = joblib.load(expand_from / "model.joblib")
            document = json.loads((expand_from / "linear_model.json").read_text())
            if not compatible(saved):
                raise ValueError("Expansion requires the same SGDRegressor settings")
            if not np.array_equal(saved.coef_, document["weights"]) or saved.intercept_[0] != document["intercept"]:
                raise ValueError("Source estimator and portable weights disagree")
            saved.coef_ = np.pad(saved.coef_, (0, self.input_columns-count))
            saved.n_features_in_ = self.input_columns
            self.model = saved
            self.source_version = document["model_version"]
        if resume_from is not None:
            saved_norm = json.loads((resume_from / "normalizer.json").read_text())
            if saved_norm != normalizer:
                raise ValueError("Continuation requires identical frozen normalization")
            saved = joblib.load(resume_from / "model.joblib")
            document = json.loads((resume_from / "linear_model.json").read_text())
            if not compatible(saved):
                raise ValueError("Continuation requires the same SGDRegressor settings")
            if not np.array_equal(saved.coef_, document["weights"]) or saved.intercept_[0] != document["intercept"]:
                raise ValueError("Saved estimator and portable weights disagree")
            self.model = saved
            self.source_version = document["model_version"]
        # Validate the saved estimator against the established configuration
        # first; an explicit learning-rate override preserves its learned
        # coefficients and optimizer counter, and changes only eta0.
        if eta0 is not None:
            self.model.set_params(eta0=float(eta0))
        self.version, self.transitions, self.queue = 0, 0, []
        self.history = []
        (root / "snapshots").mkdir()
        self.trace = gzip.open(root / "updates.jsonl.gz", "wt", encoding="utf-8")
        self.save_snapshot()
        joblib.dump(self.model, root / "initial_model.joblib", compress=3)

    def document(self):
        return {"algorithm": "online_linear_SARSA", "weights": self.model.coef_.tolist(),
                "intercept": float(self.model.intercept_[0]), "feature_clip": 4.0,
                "model_version": self.version, "input_columns": self.input_columns,
                "feature_version": self.normalizer.get("feature_version")}

    def save_snapshot(self):
        write(self.root / "snapshots" / f"version_{self.version:06d}.json", self.document())

    def normalize(self, x):
        return np.clip((np.asarray(x, dtype=np.float64)-self.normalizer["offset"])/self.normalizer["scale"], -4, 4)

    def choose(self, response, rng, epsilon):
        x = self.normalize(response["features"])
        scores = self.model.predict(x)
        explored = len(x) > 1 and rng.random() < epsilon
        if explored:
            index = rng.randrange(len(x))
        else:
            index = rng.choice(np.flatnonzero(scores == scores.max()).tolist())
        return {"index": index, "scores": scores.tolist(), "x": x[index],
                "version": self.version, "explored": explored}

    def add(self, x, following_x, reward, discount, metadata):
        self.queue.append((x, following_x, reward, discount, metadata))

    def flush(self, force=False):
        if not self.queue or (len(self.queue) < self.batch_size and not force):
            return
        pending, self.queue = self.queue, []
        x = np.asarray([row[0] for row in pending])
        next_x = np.asarray([row[1] for row in pending])
        rewards = np.asarray([row[2] for row in pending])
        discounts = np.asarray([row[3] for row in pending])
        # SARSA: value of the next action actually committed to execution,
        # never max over hypothetical next actions, and never a simulator step.
        targets = rewards + discounts * self.model.predict(next_x)
        previous_prediction = self.model.predict(x)
        previous_weights = self.model.coef_.copy()
        previous_intercept = float(self.model.intercept_[0])
        # Normalize step magnitude for varying state vector lengths.
        sample_weight = 1/(1+np.sum(x*x, axis=1))
        self.model.partial_fit(x, targets, sample_weight=sample_weight)
        assert np.isfinite(self.model.coef_).all() and np.isfinite(self.model.intercept_).all()
        self.version += 1
        self.transitions += len(pending)
        change = float(np.linalg.norm(np.r_[self.model.coef_-previous_weights,
                                            self.model.intercept_[0]-previous_intercept]))
        row = {"version": self.version, "transitions": len(pending), "cumulative_transitions": self.transitions,
               "weight_change_l2": change, "td_rmse_before_update": float(np.sqrt(np.mean((targets-previous_prediction)**2))),
               "samples": [{**item[4], "x": item[0].tolist(), "next_x": item[1].tolist(),
                            "reward": item[2], "discount": item[3], "target": float(target)}
                           for item, target in zip(pending, targets)]}
        self.trace.write(json.dumps(row, allow_nan=False) + "\n")
        self.trace.flush()
        self.history.append({k: v for k, v in row.items() if k != "samples"})
        self.save_snapshot()


class Worker:
    def __init__(self, config, log):
        self.log = log.open("w", encoding="utf-8")
        self.process = subprocess.Popen([str(REPO / ".venv/Scripts/python.exe"), "-u", "-m",
                                         "catanzero.online_env", str(config)], cwd=REPO,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, encoding="utf-8", bufsize=1,
                                        env={**os.environ, "PYTHONHASHSEED": "0"})

    def rpc(self, request):
        self.process.stdin.write(json.dumps(request, allow_nan=False)+"\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(f"environment worker exited ({self.process.poll()}); see {self.log.name}")
        return json.loads(response)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"op":"close"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=15)
        if self.process.returncode:
            raise RuntimeError(f"worker failed: {self.log.name}")
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def exploration_rate(index, games, training, start=.3, end=.1):
    if not 0 <= end <= start <= 1:
        raise ValueError("Exploration must satisfy 0 <= end <= start <= 1")
    return start + (end-start)*index/max(games-1, 1) if training else 0.0


def play_phase(root, name, model, games, workers, *, training, base_seed, excluded,
               exploration_start=.3, exploration_end=.1, learner_policy="Linear_SARSA",
               opponent_config=None, gamma=.999, vp_weight=.05, gap_weight=0.0):
    directory = root / name
    directory.mkdir()
    config = {"base_seed": base_seed, "max_actions": 2000, "max_seconds": 120,
              "excluded_board_hashes": excluded, "learner_policy": learner_policy, "training": training,
              "exploration_start": exploration_start if training else 0.,
              "exploration_end": exploration_end if training else 0., "gamma_per_actual_action": gamma,
              "vp_potential_weight": vp_weight, "relative_gap_potential_weight": gap_weight}
    config["feature_version"] = model.normalizer.get("feature_version", "predecision_relative_spatial_linear_v1")
    if opponent_config:
        allowed = {"opponent_policy", "opponent_model", "opponent_normalizer", "max_actions", "max_seconds", "trace_compresslevel"}
        if set(opponent_config)-allowed:
            raise ValueError("Invalid opponent configuration keys")
        config.update(opponent_config)
    write(directory / "config.json", config)
    started, results, completed, next_index = perf_counter(), [], 0, 0
    worker_list = [Worker(directory / "config.json", directory / f"worker_{i}.log")
                   for i in range(min(workers, games))]
    slots = {}
    discarded = 0

    def start_game(worker, index):
        response = worker.rpc({"op": "reset", "index": index, "output": str(directory / f"game_{index:04d}")})
        assert response["kind"] == "decision"
        rng = random.Random((base_seed+index)*1000003+71)
        epsilon = exploration_rate(index, games, training, exploration_start, exploration_end)
        return {"worker": worker, "response": response, "rng": rng, "epsilon": epsilon,
                "plan": model.choose(response, rng, epsilon)}

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool, (directory / "results.jsonl").open("w", encoding="utf-8") as result_stream:
            initial = list(pool.map(lambda pair: start_game(*pair), [(w, i) for i, w in enumerate(worker_list)]))
            slots = dict(enumerate(initial))
            next_index = len(initial)
            while slots:
                keys = sorted(slots)
                calls = []
                for key in keys:
                    slot, plan = slots[key], slots[key]["plan"]
                    calls.append((slot["worker"], {"op": "step", "selected_index": plan["index"],
                                  "model_version": plan["version"], "scores": plan["scores"], "explored": plan["explored"]}))
                responses = list(pool.map(lambda pair: pair[0].rpc(pair[1]), calls))
                for key, response in zip(keys, responses):
                    slot, plan = slots[key], slots[key]["plan"]
                    terminal = response["kind"] == "finished" and response["termination"] == "victory"
                    next_plan = model.choose(response, slot["rng"], slot["epsilon"]) if response["kind"] == "decision" else None
                    if training and (terminal or next_plan is not None):
                        reward, discount = shaped_transition(slot["response"], response, terminal=terminal,
                                                             won=response.get("learner_won"), gamma=gamma, vp_weight=vp_weight, gap_weight=gap_weight)
                        model.add(plan["x"], next_plan["x"] if next_plan else np.zeros(len(model.normalizer["offset"])), reward, discount,
                                  {"phase": name, "game_index": response["game_index"], "actor": response["learner_color"],
                                   "action_index": slot["response"]["action_index"], "next_action_index": response["action_index"],
                                   "selected_index": plan["index"], "next_selected_index": next_plan["index"] if next_plan else None,
                                   "terminal": terminal, "action_model_version": plan["version"],
                                   "next_action_model_version": next_plan["version"] if next_plan else None})
                    elif training:
                        discarded += 1
                    if response["kind"] == "finished":
                        if training and hasattr(model, "finish_game"):
                            model.finish_game(response["game_index"], censored=not terminal)
                        results.append(response)
                        result_stream.write(json.dumps(response)+"\n")
                        result_stream.flush()
                        completed += 1
                        if completed % 50 == 0 or completed == games:
                            wins = sum(r["learner_won"] is True for r in results)
                            print(f"{name}: {completed}/{games}, actual {learner_policy} wins={wins}, updates={model.version}, elapsed={perf_counter()-started:.1f}s", flush=True)
                        if next_index < games:
                            slots[key] = start_game(slot["worker"], next_index)
                            next_index += 1
                        else:
                            del slots[key]
                    else:
                        slot.update(response=response, plan=next_plan)
                if training:
                    model.flush()
            if training:
                model.flush(force=True)
    finally:
        for worker in worker_list:
            worker.close()
    complete = [r for r in results if r["termination"] == "victory"]
    wins = sum(r["learner_won"] is True for r in complete)
    summary = {"phase": name, "learner_policy": learner_policy, "weights_updated": training,
               "scheduled_games": games, "completed_games": len(complete), "unfinished_games": games-len(complete),
               "learner_wins": wins, "win_rate_completed": wins/len(complete) if complete else None,
               "all_scheduled_win_bounds": [wins/games, (wins+games-len(complete))/games],
               "actions": sum(r["actions"] for r in results), "discarded_truncated_tail_transitions": discarded,
               "elapsed_seconds": perf_counter()-started, "forbidden_engine_calls": sum(r["forbidden_engine_calls"] for r in results)}
    write(directory / "summary.json", summary)
    return sorted(results, key=lambda r: r["game_index"]), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--training-base-seed", type=int, default=1000000)
    parser.add_argument("--evaluation-base-seed", type=int, default=1200000)
    parser.add_argument("--exploration-start", type=float, default=.3)
    parser.add_argument("--exploration-end", type=float, default=.1)
    args = parser.parse_args()
    if args.games < 4 or args.games % 4 or args.eval_games < 4 or args.eval_games % 4 or args.workers < 1:
        parser.error("positive workers and game counts divisible by four required")
    if not 0 <= args.exploration_end <= args.exploration_start <= 1:
        parser.error("exploration must satisfy 0 <= end <= start <= 1")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    resume_from = args.resume_from.resolve() if args.resume_from else None
    source_normalizer = (resume_from if resume_from else REPO / "runs/fqi-pilot-500-v1") / "normalizer.json"
    normalizer = json.loads(source_normalizer.read_text())
    write(root / "normalizer.json", normalizer)
    old_manifest = json.loads((REPO / "runs/mlflow-label-study-v2/manifest.json").read_text())
    old_boards = old_manifest["source_board_hashes"] + json.loads((REPO / "runs/fqi-pilot-500-v1/experience_audit.json").read_text())["collection"]["board_hashes"]
    if resume_from:
        # Exclude every ancestor's training and evaluation boards from this run.
        ancestor, visited = resume_from, set()
        while ancestor is not None:
            if str(ancestor) in visited:
                raise ValueError("Cyclic continuation ancestry")
            visited.add(str(ancestor))
            for phase in ("initial_evaluation", "live_training", "final_evaluation"):
                old_boards.extend(json.loads(line)["initial_map_hash"] for line in
                                  (ancestor / phase / "results.jsonl").read_text().splitlines())
            parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
            ancestor = Path(parent) if parent else None
    config = {"algorithm": "online mini-batch semi-gradient SARSA", "model_type": "SGDRegressor",
              "input_columns": 164, "trainable_parameters": 165,
              "initial_weights": "saved trained model" if resume_from else "all zero",
              "resume_from": str(resume_from) if resume_from else None,
              "resume_estimator_sha256": hashlib.sha256((resume_from / "model.joblib").read_bytes()).hexdigest() if resume_from else None,
              "training_games": args.games, "evaluation_games_per_snapshot": args.eval_games,
              "environment_processes": args.workers, "central_parameter_updater_processes": 1,
              "batch_transitions": 32,
              "exploration": f"epsilon from {args.exploration_start} to {args.exploration_end} by training game index; zero during frozen evaluation",
              "exploration_start": args.exploration_start, "exploration_end": args.exploration_end,
              "improvement_rule": "positive paired win difference and two-sided paired p < .05, with all evaluation games completed",
              "reward": "win=1, loss=0; .05*known own VP potential shaping; terminal potential zero",
              "gamma_per_actual_action": .999, "feature_clip": 4.0,
              "normalization_source": str(source_normalizer), "normalization_refitted": False,
              "normalization_source_sha256": hashlib.sha256(source_normalizer.read_bytes()).hexdigest(),
              "sample_weight": "1/(1+sum(normalized_features**2)); bounded step magnitude",
              "training_base_seed": args.training_base_seed, "evaluation_base_seed": args.evaluation_base_seed,
              "selection": "fixed final model; initial/final evaluated on same fresh boards; no tuning on evaluation outcomes",
              "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in (
                  "src/catanzero/online_linear.py", "src/catanzero/online_env.py", "tools/train_online_sarsa.py")}}
    write(root / "config.json", config)
    model = CentralSARSA(root, normalizer, resume_from=resume_from)
    config["source_model_version"] = model.source_version
    config["initial_optimizer_t"] = float(model.model.t_)
    config["estimator_parameters"] = model.model.get_params()
    write(root / "config.json", config)
    mlflow.set_tracking_uri(old_manifest["tracking_uri"])
    client = MlflowClient()
    name = "CatanZero_Online_Linear_SARSA"
    experiment = client.get_experiment_by_name(name)
    eid = experiment.experiment_id if experiment else client.create_experiment(name, artifact_location=(REPO / "runs/mlflow/artifacts").as_uri())
    mlflow.set_experiment(experiment_id=eid)
    with mlflow.start_run(run_name=f"{'continued' if resume_from else 'zero_start'}_live_training_{args.games}_games") as run:
        run_id = run.info.run_id
        write(root / "tracking.json", {"run_id": run_id, "experiment_id": eid, "tracking_uri": old_manifest["tracking_uri"]})
        mlflow.log_params({"algorithm": "SARSA", "model": "SGDRegressor", "training_games": args.games,
                           "evaluation_games": args.eval_games, "workers": args.workers, "updater_processes": 1,
                           "batch_transitions": 32, "initial_weights": config["initial_weights"], "forward_simulation": False,
                           "exploration_start": args.exploration_start, "exploration_end": args.exploration_end,
                           "resume_from": str(resume_from) if resume_from else "none"})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_play")
        # Initial model evaluation is independent of training; no fit/update.
        initial_version = model.version
        print("STAGE: frozen initial evaluation", flush=True)
        before_results, before = play_phase(root, "initial_evaluation", model, args.eval_games, args.workers,
                                           training=False, base_seed=args.evaluation_base_seed, excluded=old_boards)
        assert model.version == initial_version
        train_excluded = old_boards + [r["initial_map_hash"] for r in before_results]
        print("STAGE: live training, weights updated from fresh gameplay", flush=True)
        train_results, train = play_phase(root, "live_training", model, args.games, args.workers,
                                         training=True, base_seed=args.training_base_seed, excluded=train_excluded,
                                         exploration_start=args.exploration_start, exploration_end=args.exploration_end)
        model.trace.close()
        write(root / "training_history.json", model.history)
        write(root / "linear_model.json", model.document())
        joblib.dump(model.model, root / "model.joblib", compress=3)
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = np.asarray(normalizer["offset"]), np.asarray(normalizer["scale"])
        scaler.var_, scaler.n_features_in_, scaler.n_samples_seen_ = scaler.scale_**2, 164, normalizer["fit_rows"]
        pipeline = Pipeline([("float64", FunctionTransformer(np.asarray, kw_args={"dtype": np.float64})),
                             ("normalization", scaler), ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -4, "a_max": 4})),
                             ("sarsa", model.model)])
        rng = np.random.default_rng(42)
        sample = scaler.mean_ + rng.normal(size=(512, 164))*scaler.scale_
        expected = model.model.predict(model.normalize(sample))
        portable = LinearActionValues(model.document(), normalizer)
        portable_error = float(np.max(np.abs(expected-[portable.predict(x.tolist()) for x in sample])))
        assert portable_error < 1e-10
        requirements = [line for line in (REPO / "requirements-mlflow.txt").read_text().splitlines() if line and not line.startswith("#")]
        info = mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle",
                                      input_example=sample[:3], signature=infer_signature(sample, expected),
                                      pip_requirements=requirements)
        version = mlflow.register_model(info.model_uri, "CatanZero_Online_Linear_SARSA")
        restored = mlflow.sklearn.load_model(info.model_uri)
        reload_error = float(np.max(np.abs(restored.predict(sample)-expected)))
        assert reload_error < 1e-12
        alias = "continued" if resume_from else "pilot"
        client.set_registered_model_alias("CatanZero_Online_Linear_SARSA", alias, str(version.version))
        write(root / "registration.json", {"registered_model": "CatanZero_Online_Linear_SARSA", "version": str(version.version),
                                          "alias": alias, "model_uri": info.model_uri, "run_id": run_id,
                                          "normalization_and_clipping_included": True, "reload_max_abs_error": reload_error,
                                          "portable_max_abs_error": portable_error})
        frozen_weights, frozen_version = model.model.coef_.copy(), model.version
        print("STAGE: saved model; frozen final evaluation", flush=True)
        final_results, after = play_phase(root, "final_evaluation", model, args.eval_games, args.workers,
                                         training=False, base_seed=args.evaluation_base_seed,
                                         excluded=old_boards+[r["initial_map_hash"] for r in train_results])
        assert model.version == frozen_version and np.array_equal(model.model.coef_, frozen_weights)
        paired = []
        for a, b in zip(before_results, final_results):
            assert a["seed"] == b["seed"] and a["initial_map_hash"] == b["initial_map_hash"]
            if a["termination"] == b["termination"] == "victory":
                paired.append(int(b["learner_won"])-int(a["learner_won"]))
        from scipy.stats import binomtest
        positive, negative = paired.count(1), paired.count(-1)
        p = binomtest(positive, positive+negative, .5).pvalue if positive+negative else 1.0
        improvement = (positive > negative and p < .05 and before["unfinished_games"] == 0 and after["unfinished_games"] == 0)
        report = {"run_id": run_id, "initial_evaluation": before, "live_training": train, "final_evaluation": after,
                  "central_updates": model.version, "fresh_training_transitions": model.transitions,
                  "updates_with_nonzero_parameter_change": sum(h["weight_change_l2"] > 0 for h in model.history),
                  "initial_parameters_all_zero": not bool(resume_from),
                  "continued_from": str(resume_from) if resume_from else None,
                  "final_weight_norm": float(np.linalg.norm(model.model.coef_)),
                  "paired_improvements": positive, "paired_regressions": negative, "paired_two_sided_p": float(p),
                  "improvement_detected_by_precommitted_rule": bool(improvement),
                  "model_saved_before_final_evaluation": True, "portable_max_abs_error": portable_error,
                  "reload_max_abs_error": reload_error,
                  "limitations": ["linear compressed spatial features may limit strength",
                                  "frozen normalization comes from prior training-only data",
                                  "fresh mini-batches with actual committed next actions, interleaved vector environments",
                                  "training win counts reflect changing policy; only frozen evaluations measure snapshot strength",
                                  "truncated games unknown, not losses; dropped missing final transitions"]}
        write(root / "report.json", report)
        mlflow.log_metrics({"initial_eval_wins": before["learner_wins"], "final_eval_wins": after["learner_wins"],
                            "live_training_wins": train["learner_wins"], "central_updates": model.version,
                            "fresh_training_transitions": model.transitions, "final_eval_completed": after["completed_games"],
                            "portable_max_abs_error": portable_error, "reload_max_abs_error": reload_error})
        for file in root.iterdir():
            if file.is_file():
                mlflow.log_artifact(str(file), "online_sarsa")
        for phase in ("initial_evaluation", "live_training", "final_evaluation"):
            mlflow.log_artifact(str(root / phase / "summary.json"), phase)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
