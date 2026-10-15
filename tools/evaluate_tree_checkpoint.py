"""Frozen checkpoint comparison, with conservative bounds for capped games."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mlflow
import numpy as np
from scipy.stats import binomtest
from train_gbrl_sarsa import GBRLSARSA, board_hashes, REPO
from train_gbrl_curriculum import linear_opponents
from train_online_sarsa import play_phase, write


def conservative_gate(summary, threshold=.05):
    n, wins = summary["scheduled_games"], summary["learner_wins"]
    p = float(binomtest(wins, n, .25, alternative="greater").pvalue)
    return {"passed": wins/n > .25 and p < threshold, "threshold": threshold,
            "p_one_sided": p, "scheduled_game_win_bounds": summary["all_scheduled_win_bounds"],
            "rule": "wins / all scheduled games above 25%, including capped games as non-wins for this conservative test only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6100000)
    parser.add_argument("--threshold", type=float, default=.025)
    args = parser.parse_args()
    root, source = args.output.resolve(), args.source.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reference = REPO / "runs/online-sarsa-low-exploration-500-v3"
    normalizer = json.loads((source / "normalizer.json").read_text())
    write(root / "normalizer.json", normalizer)
    model = GBRLSARSA(root, normalizer, resume_from=source)
    opponents = linear_opponents(reference)
    config = {"algorithm": "frozen GBRL checkpoint evaluation", "curriculum": True, "evaluation_only": True,
              "reference": str(reference), "resume_from": str(source), "initial_tree_count": model.initial_trees,
              "reference_sha256": hashlib.sha256((reference / "model.joblib").read_bytes()).hexdigest(),
              "resume_tree_sha256": hashlib.sha256((source / "model.gbrl_model").read_bytes()).hexdigest(),
              "gbrl_version": "1.1.9", "updater_processes": 0, "workers": args.workers,
              "opponents": opponents, "seed": args.seed, "evaluation_games": args.games,
              "gate_threshold": args.threshold, "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in
                                                                  ("tools/evaluate_tree_checkpoint.py", "tools/train_gbrl_sarsa.py", "tools/train_online_sarsa.py", "src/catanzero/online_env.py", "src/catanzero/online_linear.py")}}
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
    # Include completed training boards of the current curriculum, but allow
    # its planned evaluation boards for a paired checkpoint comparison.
    curriculum = REPO / "runs/gbrl-linear-curriculum-v1"
    for directory in sorted(curriculum.glob("batch_*")):
        for phase in ("initial_evaluation", "live_training"):
            path = directory / phase / "results.jsonl"
            if path.exists():
                excluded.extend(json.loads(s)["initial_map_hash"] for s in path.read_text().splitlines())
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    mlflow.set_experiment("CatanZero_GBRL_SARSA")
    with mlflow.start_run(run_name="frozen_tree_checkpoint_comparison") as run:
        write(root / "tracking.json", {"run_id": run.info.run_id, "tracking_uri": manifest["tracking_uri"]})
        mlflow.log_params({"source": str(source), "weights_updated": False, "eval_games": args.games,
                           "opponent": "Linear_SARSA_v3", "workers": args.workers})
        mlflow.log_artifact(str(root / "config.json"), "protocol_before_play")
        print("STAGE: frozen opening trees versus frozen linear bots", flush=True)
        results, summary = play_phase(root, "final_evaluation", model, args.games, args.workers,
                                      training=False, base_seed=args.seed, excluded=excluded,
                                      learner_policy="GBRL_SARSA", opponent_config=opponents)
        model.trace.close()
        assert model.version == 0 and model.model.get_iteration() == model.initial_trees
        model.model.save_learner(str(root / "model"))
        gate = conservative_gate(summary, args.threshold)
        report = {"final_evaluation": summary, "gate": gate, "weights_updated": False,
                  "initial_tree_count": model.initial_trees, "central_updates": 0,
                  "fresh_training_transitions": 0, "updates_with_prediction_change": 0,
                  "source": str(source), "comparison": "opening checkpoint vs later checkpoint on same evaluation seeds"}
        write(root / "report.json", report)
        mlflow.log_metrics({"scheduled_win_rate_lower_bound": summary["all_scheduled_win_bounds"][0], "gate_p": gate["p_one_sided"]})
        for file in root.iterdir():
            if file.is_file():
                mlflow.log_artifact(str(file), "checkpoint_evaluation")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
