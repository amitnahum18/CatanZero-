"""Continue frozen 402 with depth-16 Monte Carlo ExtraTrees policy updates vs F."""
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

from optimize_tree_policy_vs_f import TreePolicy, monte_carlo_data
from train_residual_tree_fqi import REPO, play_phase, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


MODEL_ROOT = REPO / "runs/tree-policy-trade-aware-v3"
DATA_ROOT = REPO / "runs/tree-policy-native-f-max-v1"


def summary_row(result, version):
    games = result["completed"]
    return {
        "completed": games,
        "unfinished": result["unfinished"],
        "wins": result["wins"],
        "losses": games - result["wins"],
        "win_rate": result["wins"] / games if games else None,
        "wilson_95": wilson_interval(result["wins"], games) if games else None,
        "model_version": version,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--games-per-round", type=int, default=100)
    parser.add_argument("--validation-games", type=int, default=100)
    parser.add_argument("--test-games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=79_000_000)
    parser.add_argument("--output", type=Path, default=REPO / "runs/402-mc-rl-depth16-v1")
    args = parser.parse_args()
    if args.rounds < 1 or args.workers < 1 or not 0 <= args.epsilon <= 1:
        parser.error("rounds/workers positive and epsilon in [0,1] required")
    if any(n < 2 or n % 2 for n in (args.games_per_round, args.validation_games, args.test_games)):
        parser.error("game counts must be even and >=2")

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    policy_info = json.loads((MODEL_ROOT / "best_model/policy.json").read_text(encoding="utf-8"))
    normalizer = json.loads((MODEL_ROOT / "normalizer.json").read_text(encoding="utf-8"))
    teacher = lgb.Booster(model_file=str(MODEL_ROOT / "best_model/ranker.model"))
    starting_forest = joblib.load(MODEL_ROOT / "best_model/monte_carlo.joblib")
    old_data = np.load(DATA_ROOT / "MC_round_2.npz")
    base_x, base_y, base_weight = old_data["x"], old_data["target"], old_data["weight"]
    excluded, _ = exclusions(REPO / "runs/online-sarsa-features-100-v4")

    best = TreePolicy(teacher, normalizer, policy_info["scale"], starting_forest,
                      policy_info["mix"], policy_info["version"])
    best.policy_name = "TreePolicy402_RL_depth16_candidate"
    validation_seed = args.seed
    training_seeds = [args.seed + 100_000 + 10_000 * i for i in range(args.rounds)]
    test_seed = args.seed + 900_000
    config = {
        "algorithm": "Monte Carlo policy iteration on frozen 402 with real games vs native F",
        "base_policy": "402 LightGBM ranker frozen; ExtraTrees correction retrained each round",
        "players": 2,
        "opponent": "F",
        "reward_targets": "win=+1/loss=-1 at terminal, minus state potential (.05 own VP + .02 public VP gap)",
        "extra_trees": {"n_estimators": 128, "max_depth": 16, "min_samples_leaf": 16,
                         "max_features": 0.75, "mix": policy_info["mix"]},
        "warm_start_training_rows": int(len(base_y)),
        "warm_start_games": 200,
        "new_rl_games_per_round": args.games_per_round,
        "rounds": args.rounds,
        "epsilon": args.epsilon,
        "validation_games": args.validation_games,
        "final_test_games_per_policy": args.test_games,
        "validation_seed": validation_seed,
        "training_seeds": training_seeds,
        "test_seed": test_seed,
        "workers": args.workers,
        "evaluation_games_used_for_training": False,
        "selection": "highest wins on fixed validation set; ties keep earlier policy; final test used once",
        "excluded_board_hashes": excluded,
    }
    write(root / "config.json", config)
    started = perf_counter()
    np.savez_compressed(root / "warm_start.npz", x=base_x, target=base_y, weight=base_weight)

    print(f"STAGE: baseline 402 validation, {args.validation_games} games", flush=True)
    val_rows, val_summary, _ = play_phase(
        root, "validation_baseline", best, args.validation_games, args.workers,
        training=False, base_seed=validation_seed, excluded=excluded,
        opponent_policy="F", max_seconds=600,
    )
    validation_boards = [r["initial_map_hash"] for r in val_rows]
    best_wins = val_summary["wins"]
    best_model = best
    history = [{"version": best.version, "kind": "baseline_402",
                "validation": summary_row(val_summary, best.version)}]
    all_training_rows = []
    phases = []
    validation_results = {"baseline_402": summary_row(val_summary, best.version)}

    for round_index, seed in enumerate(training_seeds, start=1):
        phase = f"rl_training_round_{round_index}"
        print(f"STAGE: RL rollout {round_index}/{args.rounds}, {args.games_per_round} fresh games", flush=True)
        rows, train_summary, _ = play_phase(
            root, phase, best_model, args.games_per_round, args.workers,
            training=True, base_seed=seed,
            excluded=excluded + validation_boards + [r["initial_map_hash"] for r in all_training_rows],
            epsilon=args.epsilon, opponent_policy="F", max_seconds=600,
        )
        if train_summary["unfinished"]:
            raise RuntimeError(f"RL round {round_index} had unfinished games; refusing to label censored tails")
        all_training_rows.extend(rows)
        phases.append(phase)
        new_x, new_y, new_weight, ids = monte_carlo_data(root, phases)
        if not len(new_y):
            raise RuntimeError(f"RL round {round_index} produced no complete return targets")
        x = np.concatenate([base_x, new_x])
        y = np.concatenate([base_y, new_y])
        sample_weight = np.concatenate([base_weight, new_weight])

        print(f"FIT: depth-16 ExtraTrees on {len(y)} targets ({len(new_y)} from fresh RL games)", flush=True)
        forest = ExtraTreesRegressor(
            n_estimators=128, max_depth=16, min_samples_leaf=16,
            max_features=0.75, n_jobs=4, random_state=2028 + round_index,
        )
        forest.fit(x, y, sample_weight=sample_weight)
        forest.n_jobs = 1
        candidate_version = 700 + round_index
        candidate = TreePolicy(best.teacher, normalizer, best.scale, forest, policy_info["mix"],
                               candidate_version)
        candidate.policy_name = f"TreePolicy402_RL_depth16_v{round_index}"
        candidate_dir = root / f"candidate_depth16_round_{round_index}"
        candidate.save(candidate_dir)
        joblib.dump(forest, candidate_dir / "monte_carlo.joblib", compress=3)
        np.savez_compressed(candidate_dir / "training_targets.npz", x=x, target=y, weight=sample_weight)
        (candidate_dir / "training_target_sources.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ids), encoding="utf-8")

        print(f"STAGE: validate depth-16 round {round_index} on paired boards", flush=True)
        _, candidate_validation, _ = play_phase(
            root, f"validation_depth16_round_{round_index}", candidate, args.validation_games,
            args.workers, training=False, base_seed=validation_seed,
            excluded=excluded + [r["initial_map_hash"] for r in all_training_rows],
            opponent_policy="F", max_seconds=600,
        )
        row = {"version": candidate_version, "kind": "RL_ExtraTrees_depth16",
               "round": round_index, "fresh_training_games": len(all_training_rows),
               "new_return_targets": int(len(new_y)), "total_training_rows": int(len(y)),
               "validation": summary_row(candidate_validation, candidate_version)}
        history.append(row)
        validation_results[f"depth16_round_{round_index}"] = row["validation"]
        write(root / "selection_history.json", history)
        print(f"VALIDATION: 402={best_wins}/{args.validation_games}; depth16="
              f"{candidate_validation['wins']}/{args.validation_games}", flush=True)
        if not candidate_validation["unfinished"] and candidate_validation["wins"] > best_wins:
            best_model, best_wins = candidate, candidate_validation["wins"]

    best_model.save(root / "selected_candidate")
    joblib.dump(best_model.mc, root / "selected_candidate/monte_carlo.joblib", compress=3)
    write(root / "selection.json", {
        "selected_version": best_model.version,
        "selected_policy": best_model.policy_name,
        "validation_wins": best_wins,
        "validation_games": args.validation_games,
        "selection_used_final_test": False,
    })

    test_excluded = excluded + validation_boards + [r["initial_map_hash"] for r in all_training_rows]
    test_baseline = TreePolicy(teacher, normalizer, policy_info["scale"], starting_forest,
                               policy_info["mix"], policy_info["version"])
    print(f"STAGE: final paired test, {args.test_games} games per policy", flush=True)
    before_rows, before, _ = play_phase(
        root, "final_test_baseline_402", test_baseline, args.test_games, args.workers,
        training=False, base_seed=test_seed, excluded=test_excluded, opponent_policy="F", max_seconds=600,
    )
    after_rows, after, _ = play_phase(
        root, "final_test_selected", best_model, args.test_games, args.workers,
        training=False, base_seed=test_seed, excluded=test_excluded, opponent_policy="F", max_seconds=600,
    )
    for before_row, after_row in zip(before_rows, after_rows):
        for field in ("seed", "learner_seat", "learner_color", "initial_map_hash", "initial_stream_hash"):
            if before_row[field] != after_row[field]:
                raise AssertionError(f"unpaired test at field {field}")
    pairs = [(a["learner_won"], b["learner_won"]) for a, b in zip(before_rows, after_rows)
             if a["termination"] == b["termination"] == "victory"]
    gained = sum(not old and new for old, new in pairs)
    lost = sum(old and not new for old, new in pairs)
    report = {
        **config,
        "training_games_completed": len(all_training_rows),
        "training_wins_exploratory": sum(r["learner_won"] is True for r in all_training_rows),
        "training_returns_added": len(new_y),
        "baseline_validation": summary_row(val_summary, policy_info["version"]),
        "validation_results": validation_results,
        "selected_version": best_model.version,
        "final_test_baseline_402": summary_row(before, policy_info["version"]),
        "final_test_selected": summary_row(after, best_model.version),
        "paired_complete": len(pairs),
        "paired_gained_wins": gained,
        "paired_lost_wins": lost,
        "elapsed_seconds": perf_counter() - started,
        "limitations": [
            "validation-selected policy may overfit its fixed 100-game selection set",
            "final paired test is a pilot, not a guarantee of 60% strength",
            "two-player base simulator against F; no claim for four-player performance",
        ],
    }
    write(root / "report.json", report)
    print("FINAL: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
