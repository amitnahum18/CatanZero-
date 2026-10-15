"""Evaluate frozen TreePolicy 402 head-to-head against native AB:2."""
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
from train_residual_tree_fqi import play_phase
from train_three_game_sarsa import exclusions
from train_online_sarsa import write
from catanzero.duel_protocol import wilson_interval


PARENT = REPO / "runs/tree-policy-trade-aware-v3"
DEFAULT_ROOT = REPO / "runs/402-vs-ab2-1v1-100-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=76_000_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.games < 1 or args.workers < 1:
        parser.error("games and workers must be positive")
    root = args.output.resolve()
    if root.exists():
        raise SystemExit(f"Output already exists: {root}")
    root.mkdir(parents=True)

    policy_doc = json.loads((PARENT / "best_model/policy.json").read_text(encoding="utf-8"))
    normalizer = json.loads((PARENT / "normalizer.json").read_text(encoding="utf-8"))
    model = TreePolicy(
        lgb.Booster(model_file=str(PARENT / "best_model/ranker.model")),
        normalizer,
        policy_doc["scale"],
        joblib.load(PARENT / "best_model/monte_carlo.joblib"),
        policy_doc["mix"],
        402,
    )
    model.policy_name = "Tree_Ranking_MonteCarlo_402"
    excluded, _ = exclusions(REPO / "runs/online-sarsa-features-100-v4")
    config = {
        "model": "frozen TreePolicy 402 (LightGBM ranker + Monte Carlo ExtraTrees correction)",
        "opponent": "native Catanatron AB:2 (alpha-beta, depth 2, base value function)",
        "players": 2,
        "games": args.games,
        "workers": args.workers,
        "base_seed": args.seed,
        "learner_seats": "alternating and balanced; even game counts split equally",
        "weights_updated": False,
        "exploration": 0.0,
        "learner_forward_simulation": False,
        "opponent_forward_simulation": "AB:2 performs its native depth-2 search",
        "excluded_training_board_hashes": len(excluded),
        "limitations": [
            "AB:2 has a forward-search compute advantage; this compares playing strength, not equal compute",
            "one 100-game sample; report Wilson interval and seat split",
            "win rate is among completed games; unfinished games are reported separately",
        ],
    }
    write(root / "config.json", config)
    print(f"STAGE: frozen 402 vs native AB:2, {args.games} two-player games", flush=True)
    started = perf_counter()
    rows, summary, _ = play_phase(
        root,
        "evaluation",
        model,
        args.games,
        args.workers,
        training=False,
        base_seed=args.seed,
        excluded=excluded,
        opponent_policy="AB:2",
    )
    assert len(rows) == args.games
    assert all(r["players"] == 2 and r["opponent_policy"] == "AB:2" for r in rows)
    seat_counts = {str(seat): sum(r["learner_seat"] == seat for r in rows) for seat in range(2)}
    if args.games >= 2:
        assert abs(seat_counts["0"] - seat_counts["1"]) <= 1
    completed = summary["completed"]
    wins = summary["wins"]
    report = {
        **config,
        "scheduled_games": args.games,
        "completed_games": completed,
        "unfinished_games": summary["unfinished"],
        "termination_counts": {
            reason: sum(r["termination"] == reason for r in rows)
            for reason in sorted({r["termination"] for r in rows})
        },
        "wins": wins,
        "losses": completed - wins,
        "win_rate_completed": wins / completed if completed else None,
        "win_rate_all_scheduled": wins / args.games,
        "wilson_95_completed": wilson_interval(wins, completed) if completed else None,
        "seat_counts": seat_counts,
        "seat_wins": {
            str(seat): sum(r["learner_seat"] == seat and r["learner_won"] is True for r in rows)
            for seat in range(2)
        },
        "mean_actions": sum(r["actions"] for r in rows) / len(rows),
        "elapsed_seconds": perf_counter() - started,
    }
    write(root / "report.json", report)
    print("RESULT: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
