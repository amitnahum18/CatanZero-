"""Compare tree regressors as the RL correction head on frozen 402 vs native F."""
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
from sklearn.ensemble import RandomForestRegressor

from optimize_tree_policy_vs_f import TreePolicy
from train_residual_tree_fqi import REPO, play_phase, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


MODEL_ROOT = REPO / "runs/tree-policy-trade-aware-v3"
RL_ROOT = REPO / "runs/402-mc-rl-depth16-v1"


def pack(summary, version):
    n = summary["completed"]
    return {
        "completed": n,
        "unfinished": summary["unfinished"],
        "wins": summary["wins"],
        "losses": n - summary["wins"],
        "win_rate": summary["wins"] / n if n else None,
        "wilson_95": wilson_interval(summary["wins"], n) if n else None,
        "version": version,
    }


def paired(reference, candidate):
    pairs = [(reference[i], candidate.get(i)) for i in reference
             if reference[i] is not None and candidate.get(i) is not None]
    return {
        "complete_pairs": len(pairs),
        "candidate_wins_where_reference_lost": sum(a is False and b is True for a, b in pairs),
        "candidate_losses_where_reference_won": sum(a is True and b is False for a, b in pairs),
        "same_outcome": sum(a == b for a, b in pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=79_300_000)
    parser.add_argument("--output", type=Path, default=REPO / "runs/402-tree-families-v1")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.workers < 1:
        parser.error("games must be even and >=2; workers must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    policy = json.loads((MODEL_ROOT / "best_model/policy.json").read_text(encoding="utf-8"))
    teacher = lgb.Booster(model_file=str(MODEL_ROOT / "best_model/ranker.model"))
    normalizer = json.loads((MODEL_ROOT / "normalizer.json").read_text(encoding="utf-8"))
    starting_mc = joblib.load(MODEL_ROOT / "best_model/monte_carlo.joblib")
    round1_mc = joblib.load(RL_ROOT / "candidate_depth16_round_1/monte_carlo.joblib")
    round2_mc = joblib.load(RL_ROOT / "candidate_depth16_round_2/monte_carlo.joblib")
    training = np.load(RL_ROOT / "candidate_depth16_round_2/training_targets.npz")
    x, y, weight = training["x"], training["target"], training["weight"]
    excluded, _ = exclusions(REPO / "runs/online-sarsa-features-100-v4")

    print(f"FIT: new tree methods on {len(y)} return targets", flush=True)
    rf = RandomForestRegressor(
        n_estimators=256, max_depth=16, min_samples_leaf=16,
        max_features=0.75, n_jobs=4, random_state=2030,
    )
    rf.fit(x, y, sample_weight=weight)
    rf.n_jobs = 1
    rf_path = root / "random_forest_256_d16.joblib"
    joblib.dump(rf, rf_path, compress=3)

    lgbm = lgb.LGBMRegressor(
        objective="regression", n_estimators=350, learning_rate=0.04,
        num_leaves=31, max_depth=10, min_child_samples=40,
        reg_lambda=1.0, colsample_bytree=0.75, subsample=0.8,
        subsample_freq=1, n_jobs=4, random_state=2030,
        verbosity=-1, deterministic=True, force_col_wise=True,
    )
    lgbm.fit(x, y, sample_weight=weight)
    lgbm.n_jobs = 1
    lgbm_path = root / "lightgbm_regressor_350_d10.txt"
    lgbm.booster_.save_model(str(lgbm_path))

    specs = [
        ("402-original", starting_mc, 402),
        ("402-et16-round1", round1_mc, 701),
        ("402-et16-round2", round2_mc, 702),
        ("402-randomforest-256-d16", rf, 803),
        ("402-lightgbm-regressor-350-d10", lgbm, 804),
    ]
    config = {
        "purpose": "screen tree-family alternatives for the 402 RL correction head",
        "opponent": "F",
        "players": 2,
        "games_per_candidate": args.games,
        "workers": args.workers,
        "seed": args.seed,
        "same_maps_and_balanced_seats": True,
        "frozen_ranker": str(MODEL_ROOT / "best_model/ranker.model"),
        "correction_mix": policy["mix"],
        "training_rows": int(len(y)),
        "training_source": str(RL_ROOT / "candidate_depth16_round_2/training_targets.npz"),
        "test_1000_and_final_200_excluded_from_training": True,
        "new_model_specs": {
            "RandomForestRegressor": {"n_estimators": 256, "max_depth": 16,
                                      "min_samples_leaf": 16, "max_features": 0.75},
            "LGBMRegressor": {"n_estimators": 350, "max_depth": 10, "num_leaves": 31,
                              "min_child_samples": 40, "reg_lambda": 1.0},
        },
        "warning": "screening validation only; any selected candidate needs a fresh larger paired test",
    }
    write(root / "config.json", config)
    started = perf_counter()
    outcomes, results = {}, {}
    for name, correction, version in specs:
        model = TreePolicy(teacher, normalizer, policy["scale"], correction, policy["mix"], version)
        model.policy_name = name
        print(f"STAGE: {name}, {args.games} paired games vs F", flush=True)
        rows, summary, _ = play_phase(
            root, name, model, args.games, args.workers, training=False,
            base_seed=args.seed, excluded=excluded, opponent_policy="F", max_seconds=600,
        )
        outcomes[name] = {r["game_index"]: r["learner_won"]
                          if r["termination"] == "victory" else None for r in rows}
        results[name] = pack(summary, version)
        write(root / "report.json", {**config, "results": results,
                                      "elapsed_seconds": perf_counter() - started})
        print(f"RESULT: {name}: {summary['wins']}/{summary['completed']}", flush=True)

    reference = outcomes["402-original"]
    report = {
        **config,
        "results": results,
        "paired_vs_402_original": {name: paired(reference, outcome)
                                   for name, outcome in outcomes.items()},
        "best_by_validation_rate": max(results, key=lambda name: results[name]["win_rate"] or -1),
        "elapsed_seconds": perf_counter() - started,
    }
    write(root / "report.json", report)
    print("FINAL: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
