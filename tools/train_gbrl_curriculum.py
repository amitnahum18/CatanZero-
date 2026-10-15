"""Continue real tree SARSA against frozen linear bots; gated switch to F_static."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
from scipy.stats import binomtest

from train_gbrl_sarsa import GBRLSARSA, GBRLActionModel, board_hashes, tree_predict, REPO, TREE_STRUCT, OPTIMIZER, PARAMS
from train_online_sarsa import play_phase, write


def linear_opponents(reference):
    return {"opponent_policy": "Linear_SARSA", "opponent_model": str(reference / "linear_model.json"),
            "opponent_normalizer": str(reference / "normalizer.json"), "max_actions": 4000, "max_seconds": 180}


def victory_gate(summary, threshold=.01):
    n, wins = summary["completed_games"], summary["learner_wins"]
    p = float(binomtest(wins, n, .25, alternative="greater").pvalue) if n else 1.
    return {"passed": summary["unfinished_games"] == 0 and wins/n > .25 and p < threshold if n else False,
            "win_rate": wins/n if n else None, "p_one_sided_above_25_percent": p, "threshold": threshold,
            "opponent": "three frozen Linear_SARSA v3", "theoretical_equal_strength_rate": .25}


def save_and_register(root, model, normalizer, client, run_id):
    model.model.save_learner(str(root / "model"))
    model.save_snapshot()
    write(root / "training_history.json", model.history)
    raw = np.asarray(normalizer["offset"])+np.random.default_rng(42).normal(size=(64, 164))*normalizer["scale"]
    expected = model.predict(model.normalize(raw))
    requirements = [line for line in (REPO / "requirements-mlflow.txt").read_text().splitlines() if line and not line.startswith("#")]
    info = mlflow.pyfunc.log_model(name="model", python_model=GBRLActionModel(),
                                  artifacts={"trees": str(root / "model.gbrl_model"), "normalizer": str(root / "normalizer.json")},
                                  pip_requirements=requirements+["gbrl==1.1.9", "torch==2.13.0"],
                                  signature=infer_signature(raw, expected), input_example=raw[:2])
    restored = mlflow.pyfunc.load_model(info.model_uri)
    error = float(np.max(np.abs(restored.predict(raw)-expected)))
    assert error == 0
    version = mlflow.register_model(info.model_uri, "CatanZero_GBRL_SARSA")
    client.set_registered_model_alias("CatanZero_GBRL_SARSA", "curriculum", str(version.version))
    write(root / "registration.json", {"registered_model": "CatanZero_GBRL_SARSA", "version": str(version.version),
                                      "alias": "curriculum", "run_id": run_id, "model_uri": info.model_uri,
                                      "normalization_and_clipping_included": True, "reload_max_abs_error": error})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()
    if min(args.games, args.eval_games) < 4 or args.games % 4 or args.eval_games % 4 or min(args.workers, args.batches) < 1:
        parser.error("positive counts; game counts divisible by four required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reference = REPO / "runs/online-sarsa-low-exploration-500-v3"
    source = REPO / "runs/gbrl-sarsa-fast-200-v1"
    normalizer = json.loads((source / "normalizer.json").read_text())
    manifest = json.loads((REPO / "runs/mlflow-label-study-v2/manifest.json").read_text())
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    client = MlflowClient()
    mlflow.set_experiment("CatanZero_GBRL_SARSA")
    excluded = manifest["source_board_hashes"] + json.loads((REPO / "runs/fqi-pilot-500-v1/experience_audit.json").read_text())["collection"]["board_hashes"]
    ancestor = reference
    while ancestor:
        excluded.extend(board_hashes(ancestor, ("initial_evaluation", "live_training", "final_evaluation")))
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        ancestor = Path(parent) if parent else None
    excluded.extend(board_hashes(source, ("benchmark_1_worker", "linear_evaluation", "live_training", "tree_evaluation")))
    overall = {"reference": str(reference), "initial_tree_source": str(source), "batch_budget": args.batches,
               "training_games_per_batch": args.games, "evaluation_games_per_batch": args.eval_games,
               "screening_threshold": .05/args.batches, "confirmation_threshold": .05,
               "rule": "frozen tree wins above 25% vs three frozen linear bots, one-sided exact p below screening threshold; then independent confirmation p < .05; no censored matches",
               "batches": [], "switched_to_F_static": False}
    write(root / "protocol.json", overall)
    started = perf_counter()
    for batch in range(1, args.batches+1):
        directory = root / f"batch_{batch:02d}"
        directory.mkdir()
        write(directory / "normalizer.json", normalizer)
        opponents = linear_opponents(reference)
        model = GBRLSARSA(directory, normalizer, resume_from=source, checkpoint_every=10)
        config = {"algorithm": "online GBRL semi-gradient SARSA", "curriculum": True,
                  "reference": str(reference), "resume_from": str(source), "initial_tree_count": model.initial_trees,
                  "reference_sha256": hashlib.sha256((reference / "model.joblib").read_bytes()).hexdigest(),
                  "resume_tree_sha256": hashlib.sha256((source / "model.gbrl_model").read_bytes()).hexdigest(),
                  "workers": args.workers, "updater_processes": 1, "tree_struct": TREE_STRUCT, "optimizer": OPTIMIZER,
                  "gbrl_params": PARAMS, "gbrl_version": "1.1.9", "batch_transitions": 128, "gradient_clip": 1.,
                  "checkpoint_every": 10, "exploration_start": .1, "exploration_end": .05,
                  "gamma_per_actual_action": .999, "vp_potential_weight": .05,
                  "opponents": opponents, "training_games": args.games, "evaluation_games": args.eval_games,
                  "training_base_seed": 5000000+batch*10000, "evaluation_base_seed": 6000000+batch*10000,
                  "gate": overall["rule"], "screening_threshold": overall["screening_threshold"],
                  "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in
                                    ("tools/train_gbrl_curriculum.py", "tools/train_gbrl_sarsa.py", "tools/train_online_sarsa.py",
                                     "src/catanzero/online_env.py", "src/catanzero/online_linear.py")}}
        write(directory / "config.json", config)
        source_snapshot = directory / "implementation_snapshot"
        for name in config["source_sha256"]:
            destination = source_snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, destination)
        with mlflow.start_run(run_name=f"curriculum_linear_batch_{batch}_{args.games}") as run:
            write(directory / "tracking.json", {"run_id": run.info.run_id, "tracking_uri": manifest["tracking_uri"]})
            mlflow.log_params({"algorithm": "SARSA", "model": "GBRL", "opponent": "Linear_SARSA_v3",
                               "train_games": args.games, "eval_games": args.eval_games, "workers": args.workers,
                               "initial_tree_count": model.initial_trees, "forward_simulation": False})
            mlflow.log_artifact(str(directory / "config.json"), "protocol_before_play")
            if batch == 1:
                print("STAGE: 8-game frozen tree smoke test against three frozen linear bots", flush=True)
                smoke_rows, smoke = play_phase(directory, "initial_evaluation", model, 8, args.workers,
                                              training=False, base_seed=5900000, excluded=excluded,
                                              learner_policy="GBRL_SARSA", opponent_config=opponents)
                excluded.extend(r["initial_map_hash"] for r in smoke_rows)
                print(f"SMOKE: completed={smoke['completed_games']}/8, wins={smoke['learner_wins']}, seconds={smoke['elapsed_seconds']:.1f}", flush=True)
            print(f"STAGE: curriculum batch {batch}, live trees versus frozen linear bots", flush=True)
            training_rows, training = play_phase(directory, "live_training", model, args.games, args.workers,
                                                 training=True, base_seed=config["training_base_seed"], excluded=excluded,
                                                 exploration_start=.1, exploration_end=.05, learner_policy="GBRL_SARSA",
                                                 opponent_config=opponents)
            model.trace.close()
            excluded.extend(r["initial_map_hash"] for r in training_rows)
            save_and_register(directory, model, normalizer, client, run.info.run_id)
            frozen = model.version
            print(f"STAGE: curriculum batch {batch}, frozen evaluation versus linear bots", flush=True)
            test_rows, test = play_phase(directory, "final_evaluation", model, args.eval_games, args.workers,
                                        training=False, base_seed=config["evaluation_base_seed"], excluded=excluded,
                                        learner_policy="GBRL_SARSA", opponent_config=opponents)
            assert model.version == frozen
            excluded.extend(r["initial_map_hash"] for r in test_rows)
            gate = victory_gate(test, overall["screening_threshold"])
            report = {"live_training": training, "final_evaluation": test, "gate": gate,
                      "central_updates": model.version, "initial_tree_count": model.initial_trees,
                      "total_tree_count": model.model.get_iteration(), "fresh_training_transitions": model.transitions,
                      "updates_with_prediction_change": sum(h["prediction_change_l2"] > 0 for h in model.history),
                      "boosting_seconds": sum(h["boosting_seconds"] for h in model.history),
                      "confirmation": None, "switched_to_F_static": False}
            if batch == 1:
                report["initial_evaluation"] = smoke
            if gate["passed"]:
                print("STAGE: independent frozen confirmation against linear bots", flush=True)
                confirm_rows, confirm = play_phase(directory, "confirmation", model, args.eval_games, args.workers,
                                                   training=False, base_seed=7000000+batch*10000, excluded=excluded,
                                                   learner_policy="GBRL_SARSA", opponent_config=opponents)
                excluded.extend(r["initial_map_hash"] for r in confirm_rows)
                report["confirmation"] = {"summary": confirm, "gate": victory_gate(confirm, .05)}
                if report["confirmation"]["gate"]["passed"]:
                    overall["switched_to_F_static"] = report["switched_to_F_static"] = True
                    print("STAGE: linear opponent surpassed and confirmed; train versus F_static", flush=True)
                    f_directory = directory / "F_stage"
                    f_directory.mkdir()
                    write(f_directory / "normalizer.json", normalizer)
                    f_model = GBRLSARSA(f_directory, normalizer, resume_from=directory, checkpoint_every=10)
                    write(f_directory / "config.json", {**config, "resume_from": str(directory),
                                                        "opponents": {"opponent_policy": "F_static"},
                                                        "initial_tree_count": f_model.initial_trees})
                    with mlflow.start_run(run_name="confirmed_curriculum_F_static_stage", nested=True) as f_run:
                        write(f_directory / "tracking.json", {"run_id": f_run.info.run_id, "tracking_uri": manifest["tracking_uri"]})
                        f_rows, f_train = play_phase(f_directory, "live_training", f_model, args.games, args.workers,
                                                    training=True, base_seed=8000000+batch*10000, excluded=excluded,
                                                    exploration_start=.1, exploration_end=.05, learner_policy="GBRL_SARSA")
                        excluded.extend(r["initial_map_hash"] for r in f_rows)
                        f_model.trace.close()
                        save_and_register(f_directory, f_model, normalizer, client, f_run.info.run_id)
                        _, f_test = play_phase(f_directory, "final_evaluation", f_model, args.eval_games, args.workers,
                                               training=False, base_seed=9000000+batch*10000, excluded=excluded, learner_policy="GBRL_SARSA")
                        write(f_directory / "report.json", {"live_training": f_train, "final_evaluation": f_test,
                                                            "central_updates": f_model.version, "initial_tree_count": f_model.initial_trees,
                                                            "fresh_training_transitions": f_model.transitions})
                        for file in f_directory.iterdir():
                            if file.is_file():
                                mlflow.log_artifact(str(file), "curriculum_F_stage")
                    report["F_training"], report["F_evaluation"] = f_train, f_test
            write(directory / "report.json", report)
            mlflow.log_metrics({"training_wins": training["learner_wins"], "frozen_eval_win_rate": test["win_rate_completed"],
                                "gate_p": gate["p_one_sided_above_25_percent"], "new_tree_updates": model.version,
                                "fresh_transitions": model.transitions})
            for file in directory.iterdir():
                if file.is_file():
                    mlflow.log_artifact(str(file), "curriculum")
            mlflow.log_artifacts(str(source_snapshot), "implementation_snapshot")
            print(json.dumps(report, indent=2), flush=True)
        overall["batches"].append({"directory": str(directory), "report": report})
        overall["elapsed_seconds"] = perf_counter()-started
        write(root / "progress.json", overall)
        source = directory
        if overall["switched_to_F_static"]:
            break
    print("CURRICULUM COMPLETE: " + json.dumps({"batches": len(overall["batches"]), "switched_to_F_static": overall["switched_to_F_static"]}), flush=True)


if __name__ == "__main__":
    main()
