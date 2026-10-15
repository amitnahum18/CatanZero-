"""Short mixed-opponent pilot with real 16-step tree SARSA."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from train_gbrl_sarsa import GBRLSARSA, GBRLActionModel, board_hashes, REPO
from train_gbrl_curriculum import save_and_register
from train_online_sarsa import play_phase, write


class MultiStepSARSA(GBRLSARSA):
    def __init__(self, *args, n_steps=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps, self.buffers, self.raw_transitions = n_steps, {}, 0
        self.censored_short_returns = 0

    def emit(self, window, *, censored=False):
        steps = list(window)[:self.n_steps]
        reward, discount = 0., 1.
        for row in steps:
            reward += discount*row[2]
            discount *= row[3]
        first, last = steps[0], steps[-1]
        metadata = {**first[4], "next_action_index": last[4]["next_action_index"],
                    "next_selected_index": last[4]["next_selected_index"],
                    "next_action_model_version": last[4]["next_action_model_version"],
                    "terminal": last[4]["terminal"], "return_steps": len(steps),
                    "censored_short_return": censored,
                    "steps": [{**row[4], "reward": row[2], "discount": row[3]} for row in steps]}
        self.queue.append((first[0], last[1], reward, discount, metadata))
        self.censored_short_returns += int(censored)
        window.popleft()

    def add(self, x, following_x, reward, discount, metadata):
        key = metadata["game_index"]
        window = self.buffers.setdefault(key, deque())
        if window and window[-1][4]["actor"] != metadata["actor"]:
            raise ValueError("n-step window crossed players")
        window.append((x, following_x, reward, discount, metadata))
        self.raw_transitions += 1
        if len(window) >= self.n_steps:
            self.emit(window)
        if metadata["terminal"]:
            while window:
                self.emit(window)

    def finish_game(self, game_index, *, censored):
        window = self.buffers.pop(game_index, deque())
        # Bootstrap at the last known action actually executed, not at the
        # unknown cutoff. No terminal win/loss label is invented for a cap.
        while window:
            if not censored:
                raise ValueError("terminal n-step tail was not drained")
            self.emit(window, censored=True)


def split_summary(results):
    summary = {}
    for opponent in ("Linear_SARSA", "F_static"):
        rows = [r for r in results if r["opponent_policy"] == opponent]
        wins = sum(r["learner_won"] is True for r in rows)
        complete = sum(r["termination"] == "victory" for r in rows)
        summary[opponent] = {"scheduled_games": len(rows), "completed_games": complete,
                             "unfinished_games": len(rows)-complete, "learner_wins": wins,
                             "all_scheduled_win_bounds": [wins/len(rows), (wins+len(rows)-complete)/len(rows)]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.games, args.eval_games) < 8 or args.games % 8 or args.eval_games % 8 or args.workers < 1:
        parser.error("positive game counts divisible by eight required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = REPO / "runs/gbrl-linear-curriculum-v1/batch_01"
    reference = REPO / "runs/online-sarsa-low-exploration-500-v3"
    normalizer = json.loads((source / "normalizer.json").read_text())
    write(root / "normalizer.json", normalizer)
    model = MultiStepSARSA(root, normalizer, resume_from=source, checkpoint_every=10)
    opponents = {"opponent_policy": "Mixed_linear_F_static", "opponent_model": str(reference / "linear_model.json"),
                 "opponent_normalizer": str(reference / "normalizer.json"), "max_actions": 1500,
                 "max_seconds": 90, "trace_compresslevel": 1}
    config = {"algorithm": "online 16-step GBRL SARSA", "curriculum": True, "mixed": True,
              "reference": str(reference), "resume_from": str(source), "initial_tree_count": model.initial_trees,
              "reference_sha256": hashlib.sha256((reference / "model.joblib").read_bytes()).hexdigest(),
              "resume_tree_sha256": hashlib.sha256((source / "model.gbrl_model").read_bytes()).hexdigest(),
              "gbrl_version": "1.1.9", "updater_processes": 1, "workers": args.workers,
              "training_games": args.games, "evaluation_games_per_snapshot": args.eval_games,
              "n_steps": 16, "gamma_per_actual_action": .9999, "vp_potential_weight": .05,
              "exploration_start": .1, "exploration_end": .05, "batch_transitions": 128,
              "opponents": opponents, "mixture": "index%8 <4: three frozen linear bots; otherwise three F_static; balanced seats in both groups",
              "selection": "fixed final weights after 200 games; same boards, seats, opponents and caps before/after; no outcome-based tuning",
              "experiment_changes": ["opponent mixture", "16-step returns", "gamma .9999", "1500-action cap", "lossless faster log compression"],
              "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in
                                ("tools/train_gbrl_mixed.py", "tools/train_gbrl_sarsa.py", "tools/train_online_sarsa.py",
                                 "src/catanzero/online_env.py", "src/catanzero/online_linear.py")}}
    write(root / "config.json", config)
    for name in config["source_sha256"]:
        destination = root / "implementation_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    manifest = json.loads((REPO / "runs/mlflow-label-study-v2/manifest.json").read_text())
    excluded = manifest["source_board_hashes"] + json.loads((REPO / "runs/fqi-pilot-500-v1/experience_audit.json").read_text())["collection"]["board_hashes"]
    ancestor = reference
    while ancestor:
        excluded.extend(board_hashes(ancestor, ("initial_evaluation", "live_training", "final_evaluation")))
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        ancestor = Path(parent) if parent else None
    excluded.extend(board_hashes(REPO / "runs/gbrl-sarsa-fast-200-v1", ("benchmark_1_worker", "linear_evaluation", "live_training", "tree_evaluation")))
    excluded.extend(board_hashes(source, ("initial_evaluation", "live_training", "final_evaluation")))
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    mlflow.set_experiment("CatanZero_GBRL_SARSA")
    client = MlflowClient()
    # Store serving methods by value, so loading doesn't need local tools/*.py.
    cloudpickle.register_pickle_by_value(sys.modules[GBRLActionModel.__module__])
    with mlflow.start_run(run_name="short_mixed_16_step_SARSA_200") as run:
        write(root / "tracking.json", {"run_id": run.info.run_id, "tracking_uri": manifest["tracking_uri"]})
        mlflow.log_params({"algorithm": "16-step SARSA", "model": "GBRL", "train_games": args.games,
                           "eval_games": args.eval_games, "workers": args.workers, "gamma": .9999,
                           "opponent_mixture": "50% linear / 50% F_static", "max_actions": 1500, "forward_simulation": False})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_play")
        print("STAGE: frozen mixed-opponent baseline, 64 games", flush=True)
        before_rows, before = play_phase(root, "initial_evaluation", model, args.eval_games, args.workers,
                                         training=False, base_seed=11000000, excluded=excluded,
                                         learner_policy="GBRL_SARSA", opponent_config=opponents)
        assert model.version == 0
        train_excluded = excluded+[r["initial_map_hash"] for r in before_rows]
        print("STAGE: 200 live mixed-opponent games, 16-step returns", flush=True)
        training_rows, training = play_phase(root, "live_training", model, args.games, args.workers,
                                            training=True, base_seed=12000000, excluded=train_excluded,
                                            exploration_start=.1, exploration_end=.05, learner_policy="GBRL_SARSA",
                                            opponent_config=opponents, gamma=.9999)
        assert not model.buffers and model.raw_transitions == model.transitions
        model.trace.close()
        save_and_register(root, model, normalizer, client, run.info.run_id)
        frozen = model.version
        print("STAGE: frozen final mixed-opponent test, same 64 boards", flush=True)
        after_rows, after = play_phase(root, "final_evaluation", model, args.eval_games, args.workers,
                                       training=False, base_seed=11000000,
                                       excluded=excluded+[r["initial_map_hash"] for r in training_rows],
                                       learner_policy="GBRL_SARSA", opponent_config=opponents)
        assert model.version == frozen
        for a, b in zip(before_rows, after_rows):
            assert all(a[k] == b[k] for k in ("seed", "initial_map_hash", "learner_seat", "opponent_policy"))
        report = {"initial_evaluation": before, "live_training": training, "final_evaluation": after,
                  "opponent_breakdown": {"before": split_summary(before_rows), "training": split_summary(training_rows), "after": split_summary(after_rows)},
                  "central_updates": model.version, "initial_tree_count": model.initial_trees,
                  "total_tree_count": model.model.get_iteration(), "fresh_training_transitions": model.transitions,
                  "raw_one_step_transitions": model.raw_transitions, "n_step_returns": model.transitions,
                  "censored_short_bootstrapped_returns": model.censored_short_returns,
                  "updates_with_prediction_change": sum(h["prediction_change_l2"] > 0 for h in model.history),
                  "boosting_seconds": sum(h["boosting_seconds"] for h in model.history),
                  "limitations": ["short pilot and several changes together; cannot attribute effect to one change",
                                  "capped matches unknown, never assigned terminal losses", "overlapping n-step windows are intentional",
                                  "64 test games give a coarse estimate; 32 per opponent group"]}
        write(root / "report.json", report)
        mlflow.log_metrics({"initial_observed_wins": before["learner_wins"], "final_observed_wins": after["learner_wins"],
                            "new_tree_updates": model.version, "n_step_returns": model.transitions,
                            "train_seconds": training["elapsed_seconds"], "final_unknown_matches": after["unfinished_games"]})
        for file in root.iterdir():
            if file.is_file():
                mlflow.log_artifact(str(file), "mixed_pilot")
        mlflow.log_artifacts(str(root / "implementation_snapshot"), "implementation_snapshot")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
