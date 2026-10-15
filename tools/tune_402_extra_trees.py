"""Tune the frozen 402 policy's ExtraTrees correction on held-out games vs F."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from optimize_tree_policy_vs_f import TreePolicy
from train_residual_tree_fqi import REPO, play_phase, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


MODEL_ROOT = REPO / "runs/tree-policy-trade-aware-v3"
TRAIN_ROOT = REPO / "runs/tree-policy-native-f-max-v1"
SPECS = [(128, 16), (128, 20)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=78_800_000)
    parser.add_argument("--output", type=Path, default=REPO / "runs/402-extra-trees-tune-v1")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.workers < 1:
        parser.error("games must be even and >=2; workers must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    policy = json.loads((MODEL_ROOT / "best_model/policy.json").read_text(encoding="utf-8"))
    teacher = lgb.Booster(model_file=str(MODEL_ROOT / "best_model/ranker.model"))
    normalizer = json.loads((MODEL_ROOT / "normalizer.json").read_text(encoding="utf-8"))
    original_mc = joblib.load(MODEL_ROOT / "best_model/monte_carlo.joblib")
    excluded, _ = exclusions(REPO / "runs/online-sarsa-features-100-v4")

    train = np.load(TRAIN_ROOT / "MC_round_2.npz")
    x, y, weight = train["x"], train["target"], train["weight"]
    config = {
        "base_model": "frozen 402 LightGBM + ExtraTrees, mix 0.2",
        "opponent": "F",
        "players": 2,
        "validation_games_per_candidate": args.games,
        "workers": args.workers,
        "seed": args.seed,
        "shared_maps_and_seats": True,
        "training_source": str(TRAIN_ROOT / "MC_round_2.npz"),
        "training_rows": int(len(y)),
        "training_games": 200,
        "evaluation_games_used_for_training": False,
        "candidate_specs": [{"n_estimators": n, "max_depth": d, "min_samples_leaf": 16,
                             "max_features": 0.75} for n, d in SPECS],
        "selection_is_validation_only": True,
    }
    write(root / "config.json", config)
    candidates = [("baseline-64-d10", original_mc, 402)]
    for n_trees, depth in SPECS:
        print(f"FIT: ExtraTrees n={n_trees}, depth={depth}, rows={len(y)}", flush=True)
        model = ExtraTreesRegressor(
            n_estimators=n_trees, max_depth=depth, min_samples_leaf=16,
            max_features=0.75, n_jobs=4, random_state=2028,
        )
        model.fit(x, y, sample_weight=weight)
        model.n_jobs = 1
        name = f"et-{n_trees}-d{depth}"
        joblib.dump(model, root / f"{name}.joblib", compress=3)
        candidates.append((name, model, 500 + len(candidates)))

    started = perf_counter()
    outcomes = {}
    result_summaries = {}
    for name, mc, version in candidates:
        model = TreePolicy(teacher, normalizer, policy["scale"], mc, policy["mix"], version)
        model.policy_name = f"TreePolicy402_{name}"
        print(f"STAGE: {name}, {args.games} games vs F", flush=True)
        rows, summary, _ = play_phase(
            root, name, model, args.games, args.workers, training=False,
            base_seed=args.seed, excluded=excluded, opponent_policy="F", max_seconds=600,
        )
        outcomes[name] = {r["game_index"]: r["learner_won"] if r["termination"] == "victory" else None
                          for r in rows}
        result_summaries[name] = {
            "completed": summary["completed"], "unfinished": summary["unfinished"],
            "wins": summary["wins"], "losses": summary["completed"] - summary["wins"],
            "win_rate": summary["wins"] / summary["completed"] if summary["completed"] else None,
            "wilson_95": wilson_interval(summary["wins"], summary["completed"])
            if summary["completed"] else None,
            "model_version": version,
        }
        write(root / "report.json", {**config, "results": result_summaries,
                                      "elapsed_seconds": perf_counter() - started})
        print(f"RESULT: {name}: {summary['wins']}/{summary['completed']}", flush=True)

    baseline = outcomes[candidates[0][0]]
    paired = {}
    for name, _, _ in candidates:
        comparable = [(baseline[i], outcomes[name][i]) for i in baseline
                      if baseline[i] is not None and outcomes[name].get(i) is not None]
        paired[name] = {
            "complete_pairs_vs_baseline": len(comparable),
            "candidate_wins_where_baseline_lost": sum(a is False and b is True for a, b in comparable),
            "candidate_losses_where_baseline_won": sum(a is True and b is False for a, b in comparable),
            "same_outcome": sum(a == b for a, b in comparable),
        }
    report = {**config, "results": result_summaries, "paired_vs_baseline": paired,
              "elapsed_seconds": perf_counter() - started}
    write(root / "report.json", report)
    print("FINAL: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
