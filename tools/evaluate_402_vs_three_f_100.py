"""Evaluate frozen TreePolicy 402 against three native F bots in 100 games."""
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

from optimize_tree_policy_vs_f import TreePolicy, REPO
from train_online_sarsa import play_phase, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


PARENT = REPO / "runs/tree-policy-trade-aware-v3"
ROOT = REPO / "runs/402-vs-three-native-f-500-v1"
GAMES = 500
WORKERS = 16
BASE_SEED = 70_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=GAMES)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args()
    if ROOT.exists():
        raise SystemExit(f"Output already exists: {ROOT}")
    ROOT.mkdir(parents=True)
    started = perf_counter()
    policy_doc = json.loads((PARENT / "best_model/policy.json").read_text())
    normalizer = json.loads((PARENT / "normalizer.json").read_text())
    model = TreePolicy(
        lgb.Booster(model_file=str(PARENT / "best_model/ranker.model")),
        normalizer,
        policy_doc["scale"],
        joblib.load(PARENT / "best_model/monte_carlo.joblib"),
        policy_doc["mix"],
        402,
    )
    excluded, _ = exclusions(REPO / "runs/online-sarsa-features-100-v4")
    config = {
        "model": "CatanZero_TreePolicy_NativeF version 3 / policy 402",
        "opponents": "three native unmodified F bots",
        "players": 4,
        "games": args.games,
        "workers": args.workers,
        "base_seed": args.seed,
        "learner_seats": f"balanced, {args.games // 4} games per seat plus remainder seats",
        "weights_updated": False,
        "exploration": 0.0,
        "forward_simulation": "native F opponents only; learned policy is frozen",
        "excluded_board_hashes": excluded,
    }
    write(ROOT / "config.json", config)
    print(f"STAGE: frozen 402 versus three native F bots, {args.games} four-player games", flush=True)
    rows, summary = play_phase(
        ROOT,
        "evaluation",
        model,
        args.games,
        args.workers,
        training=False,
        base_seed=args.seed,
        excluded=excluded,
        learner_policy="Tree_Ranking_MonteCarlo_402",
        opponent_config={
            "opponent_policy": "F",
            "max_actions": 2000,
            "max_seconds": 180,
            "trace_compresslevel": 1,
        },
    )
    assert len(rows) == args.games
    assert all(row["players"] == 4 and row["opponent_policy"] == "F" for row in rows)
    seat_counts = [sum(row["learner_seat"] == seat for row in rows) for seat in range(4)]
    assert max(seat_counts) - min(seat_counts) <= 1
    wins = summary["learner_wins"]
    report = {
        "model": "402",
        "lineup": "one frozen 402 versus three native F",
        "players": 4,
        "scheduled_games": args.games,
        "completed_games": summary["completed_games"],
        "unfinished_games": summary["unfinished_games"],
        "wins": wins,
        "win_rate_completed": summary["win_rate_completed"],
        "win_rate_all_scheduled": wins / args.games,
        "wilson_95_completed": wilson_interval(wins, summary["completed_games"])
        if summary["completed_games"] else None,
        "seat_wins": {
            str(seat): sum(row["learner_seat"] == seat and row["learner_won"] is True for row in rows)
            for seat in range(4)
        },
        "seat_counts": {str(seat): count for seat, count in enumerate(seat_counts)},
        "mean_actions": sum(row["actions"] for row in rows) / args.games,
        "elapsed_seconds": perf_counter() - started,
        "weights_updated": False,
    }
    write(ROOT / "report.json", report)
    print("RESULT: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
