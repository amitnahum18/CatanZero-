"""Short live SARSA pilot with incremental GBRL trees and parallel real games."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
from gbrl.models import GBTModel
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy.stats import binomtest

from train_online_sarsa import CentralSARSA, REPO, play_phase, write

TREE_STRUCT = {"max_depth": 4, "n_bins": 32, "min_data_in_leaf": 8,
               "par_th": 2, "grow_policy": "oblivious"}
OPTIMIZER = {"algo": "SGD", "lr": .05, "start_idx": 0, "stop_idx": 1}
PARAMS = {"split_score_func": "L2", "generator_type": "Quantile"}


def new_tree_model():
    return GBTModel(input_dim=164, output_dim=1, tree_struct=TREE_STRUCT.copy(),
                    optimizers=OPTIMIZER.copy(), params=PARAMS.copy(), device="cpu")


def tree_predict(model, x):
    return model(np.ascontiguousarray(x, dtype=np.float32), requires_grad=False, tensor=False).reshape(-1)


class GBRLSARSA:
    def __init__(self, root, normalizer, batch_size=128, resume_from=None, checkpoint_every=1):
        self.root, self.normalizer, self.batch_size = root, normalizer, batch_size
        self.model = new_tree_model() if resume_from is None else GBTModel.load_learner(str(resume_from / "model"), device="cpu")
        if resume_from is not None and json.loads((resume_from / "normalizer.json").read_text()) != normalizer:
            raise ValueError("Frozen normalization must match the source tree model")
        self.initial_trees = self.model.get_iteration()
        self.checkpoint_every = checkpoint_every
        self.version, self.transitions, self.queue, self.history = 0, 0, [], []
        (root / "snapshots").mkdir()
        self.trace = gzip.open(root / "updates.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
        self.save_snapshot()

    normalize = CentralSARSA.normalize
    add = CentralSARSA.add

    def predict(self, x):
        return tree_predict(self.model, x)

    def choose(self, response, rng, epsilon):
        x = self.normalize(response["features"]).astype(np.float32)
        scores = self.predict(x)
        assert np.isfinite(scores).all()
        explored = len(x) > 1 and rng.random() < epsilon
        index = rng.randrange(len(x)) if explored else rng.choice(np.flatnonzero(scores == scores.max()).tolist())
        return {"index": index, "scores": scores.tolist(), "x": x[index],
                "version": self.version, "explored": explored}

    def save_snapshot(self):
        self.model.save_learner(str(self.root / "snapshots" / f"version_{self.version:06d}"))

    def flush(self, force=False):
        if not self.queue or (len(self.queue) < self.batch_size and not force):
            return
        pending, self.queue = self.queue, []
        x = np.asarray([s[0] for s in pending], dtype=np.float32)
        following = np.asarray([s[1] for s in pending], dtype=np.float32)
        rewards, discounts = np.asarray([s[2] for s in pending]), np.asarray([s[3] for s in pending])
        targets = rewards + discounts*self.predict(following)
        predictions = self.predict(x)
        # Fresh on-policy SARSA targets; one tree fits per-sample TD gradients.
        # No max-next-action target, replay buffer, or engine lookahead.
        grads = np.clip(predictions-targets, -1., 1.).astype(np.float32).reshape(-1, 1)
        started = perf_counter()
        self.model.step(x, grads)
        elapsed = perf_counter()-started
        after = self.predict(x)
        assert np.isfinite(after).all() and self.model.get_iteration() == self.initial_trees+self.version+1
        self.version += 1
        self.transitions += len(pending)
        row = {"version": self.version, "transitions": len(pending), "cumulative_transitions": self.transitions,
               "prediction_change_l2": float(np.linalg.norm(after-predictions)),
               "td_rmse_before_update": float(np.sqrt(np.mean((targets-predictions)**2))),
               "boosting_seconds": elapsed,
               "samples": [{**s[4], "x": s[0].tolist(), "next_x": s[1].tolist(),
                            "reward": s[2], "discount": s[3], "target": float(target), "gradient": float(g)}
                           for s, target, g in zip(pending, targets, grads[:, 0])]}
        self.trace.write(json.dumps(row, allow_nan=False)+"\n")
        self.trace.flush()
        self.history.append({k: v for k, v in row.items() if k != "samples"})
        if force or self.version % self.checkpoint_every == 0:
            self.save_snapshot()


class GBRLActionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from gbrl.models import GBTModel
        self.trees = GBTModel.load_learner(context.artifacts["trees"], device="cpu")
        self.normalizer = json.loads(Path(context.artifacts["normalizer"]).read_text())

    def predict(self, context, model_input, params=None):
        x = np.asarray(model_input, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 164 or not np.isfinite(x).all():
            raise ValueError("Expected finite raw state-action vectors with 164 columns")
        normalized = np.clip((x-self.normalizer["offset"])/self.normalizer["scale"], -4, 4)
        return tree_predict(self.trees, normalized)


def board_hashes(root, phases):
    return [json.loads(line)["initial_map_hash"] for phase in phases
            for line in (root / phase / "results.jsonl").read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.games, args.eval_games) < 4 or args.games % 4 or args.eval_games % 4 or args.workers < 1:
        parser.error("game counts must be positive multiples of four")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reference = REPO / "runs/online-sarsa-low-exploration-500-v3"
    normalizer = json.loads((reference / "normalizer.json").read_text())
    write(root / "normalizer.json", normalizer)
    manifest = json.loads((REPO / "runs/mlflow-label-study-v2/manifest.json").read_text())
    excluded = manifest["source_board_hashes"] + json.loads((REPO / "runs/fqi-pilot-500-v1/experience_audit.json").read_text())["collection"]["board_hashes"]
    ancestor = reference
    while ancestor:
        excluded.extend(board_hashes(ancestor, ("initial_evaluation", "live_training", "final_evaluation")))
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        ancestor = Path(parent) if parent else None
    config = {"algorithm": "online mini-batch semi-gradient SARSA with GBRL", "input_columns": 164,
              "initial_model": "zero bias, zero trees; no imitation or pretrained weights",
              "training_games": args.games, "evaluation_games_per_model": args.eval_games,
              "workers": args.workers, "updater_processes": 1, "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
              "tree_struct": TREE_STRUCT, "optimizer": OPTIMIZER, "gbrl_params": PARAMS,
              "batch_transitions": 128, "gradient_clip": 1., "exploration_start": .1, "exploration_end": .05,
              "gamma_per_actual_action": .999, "vp_potential_weight": .05,
              "reward": "win=1/loss=0 plus discounted own-known-VP potential; terminal potential zero",
              "frozen_normalization": True, "feature_clip": 4., "gbrl_version": "1.1.9",
              "reference": str(reference), "reference_sha256": hashlib.sha256((reference / "model.joblib").read_bytes()).hexdigest(),
              "training_base_seed": 4000000, "evaluation_base_seed": 4200000, "benchmark_base_seed": 4400000,
              "selection": "fixed final trees after 200 games; no evaluation-based tuning",
              "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in
                                ("tools/train_gbrl_sarsa.py", "tools/train_online_sarsa.py", "src/catanzero/online_env.py", "src/catanzero/online_linear.py")}}
    write(root / "config.json", config)
    baseline_dir = root / "linear_reference"
    baseline_dir.mkdir()
    linear = CentralSARSA(baseline_dir, normalizer, resume_from=reference)
    model = GBRLSARSA(root, normalizer)
    client = MlflowClient(tracking_uri=manifest["tracking_uri"])
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    name = "CatanZero_GBRL_SARSA"
    experiment = client.get_experiment_by_name(name)
    eid = experiment.experiment_id if experiment else client.create_experiment(name, artifact_location=(REPO / "runs/mlflow/artifacts").as_uri())
    mlflow.set_experiment(experiment_id=eid)
    with mlflow.start_run(run_name=f"fast_live_tree_SARSA_{args.games}") as run:
        write(root / "tracking.json", {"run_id": run.info.run_id, "tracking_uri": manifest["tracking_uri"], "experiment_id": eid})
        mlflow.log_params({"algorithm": "SARSA", "model": "GBRL", "train_games": args.games,
                           "eval_games": args.eval_games, "workers": args.workers, "depth": 4, "bins": 32,
                           "batch_transitions": 128, "exploration_start": .1, "exploration_end": .05,
                           "forward_simulation": False})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_play")
        print("STAGE: 8-game parallelism benchmark, fixed linear policy", flush=True)
        one_results, one = play_phase(root, "benchmark_1_worker", linear, 8, 1, training=False,
                                     base_seed=4400000, excluded=excluded)
        many_results, many = play_phase(root, "benchmark_8_workers", linear, 8, args.workers, training=False,
                                       base_seed=4400000, excluded=excluded)
        for a, b in zip(one_results, many_results):
            assert all(a[k] == b[k] for k in ("initial_map_hash", "seed", "winner", "actions", "learner_won"))
        speedup = one["elapsed_seconds"]/many["elapsed_seconds"]
        write(root / "parallel_benchmark.json", {"games_per_setting": 8, "one_worker": one,
                                                "parallel_workers": many, "speedup": speedup,
                                                "same_game_results_verified": True})
        print(f"BENCHMARK: one={one['elapsed_seconds']:.2f}s, parallel={many['elapsed_seconds']:.2f}s, speedup={speedup:.2f}x", flush=True)
        excluded.extend(r["initial_map_hash"] for r in one_results)
        print("STAGE: frozen linear reference evaluation", flush=True)
        before_results, before = play_phase(root, "linear_evaluation", linear, args.eval_games, args.workers,
                                           training=False, base_seed=4200000, excluded=excluded)
        linear.trace.close()
        train_excluded = excluded + [r["initial_map_hash"] for r in before_results]
        print("STAGE: live GBRL SARSA training", flush=True)
        train_results, train = play_phase(root, "live_training", model, args.games, args.workers,
                                         training=True, base_seed=4000000, excluded=train_excluded,
                                         exploration_start=.1, exploration_end=.05, learner_policy="GBRL_SARSA")
        model.trace.close()
        model.model.save_learner(str(root / "model"))
        write(root / "training_history.json", model.history)
        raw = np.asarray(normalizer["offset"])+np.random.default_rng(42).normal(size=(64, 164))*normalizer["scale"]
        expected = model.predict(model.normalize(raw))
        restored = GBTModel.load_learner(str(root / "model"), device="cpu")
        assert np.array_equal(tree_predict(restored, model.normalize(raw)), expected)
        requirements = [line for line in (REPO / "requirements-mlflow.txt").read_text().splitlines() if line and not line.startswith("#")]
        requirements += ["gbrl==1.1.9", "torch==2.13.0"]
        info = mlflow.pyfunc.log_model(name="model", python_model=GBRLActionModel(),
                                      artifacts={"trees": str(root / "model.gbrl_model"), "normalizer": str(root / "normalizer.json")},
                                      pip_requirements=requirements, signature=infer_signature(raw, expected), input_example=raw[:2])
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        reload_error = float(np.max(np.abs(loaded.predict(raw)-expected)))
        assert reload_error == 0
        version = mlflow.register_model(info.model_uri, name)
        client.set_registered_model_alias(name, "pilot", str(version.version))
        write(root / "registration.json", {"registered_model": name, "version": str(version.version), "alias": "pilot",
                                          "run_id": run.info.run_id, "model_uri": info.model_uri,
                                          "normalization_and_clipping_included": True, "reload_max_abs_error": reload_error})
        frozen_version = model.version
        print("STAGE: saved GBRL model; frozen comparison on same 100 boards", flush=True)
        final_results, after = play_phase(root, "tree_evaluation", model, args.eval_games, args.workers,
                                         training=False, base_seed=4200000,
                                         excluded=excluded+[r["initial_map_hash"] for r in train_results], learner_policy="GBRL_SARSA")
        assert model.version == frozen_version
        paired = []
        for a, b in zip(before_results, final_results):
            assert a["initial_map_hash"] == b["initial_map_hash"] and a["seed"] == b["seed"]
            if a["termination"] == b["termination"] == "victory":
                paired.append(int(b["learner_won"])-int(a["learner_won"]))
        positive, negative = paired.count(1), paired.count(-1)
        p = float(binomtest(positive, positive+negative).pvalue) if positive+negative else 1.
        report = {"linear_evaluation": before, "live_training": train, "tree_evaluation": after,
                  "central_updates": model.version, "fresh_training_transitions": model.transitions,
                  "updates_with_prediction_change": sum(h["prediction_change_l2"] > 0 for h in model.history),
                  "boosting_seconds": sum(h["boosting_seconds"] for h in model.history),
                  "parallel_speedup": speedup, "paired_tree_only_wins": positive, "paired_linear_only_wins": negative,
                  "paired_two_sided_p": p, "model_saved_before_evaluation": True, "reload_max_abs_error": reload_error,
                  "limitations": ["short pilot, not equal lifetime training budget", "100 test games are a coarse estimate",
                                  "compressed features unchanged", "tree model starts at zero; linear reference has 1500 training games",
                                  "benchmark uses only 8 fixed-policy games; not a proof of scaling at larger budgets"]}
        write(root / "report.json", report)
        mlflow.log_metrics({"linear_eval_win_rate": before["win_rate_completed"], "tree_eval_win_rate": after["win_rate_completed"],
                            "updates": model.version, "fresh_transitions": model.transitions, "parallel_speedup": speedup,
                            "boosting_seconds": report["boosting_seconds"], "paired_two_sided_p": p})
        for file in root.iterdir():
            if file.is_file():
                mlflow.log_artifact(str(file), "pilot")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
