"""Compare frozen policy 402 with native MCTS at matched seeds and seat balance."""
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
DEFAULT_ROOT = REPO / "runs/402-vs-mcts-budget-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8, help="Even number of games per MCTS budget")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--opponents", default="M:100", help="Comma-separated native search specs, e.g. M:100,AB:3")
    parser.add_argument("--max-seconds", type=int, default=600, help="Per-game timeout for slower search opponents")
    parser.add_argument("--seed", type=int, default=77_000_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    opponents = [value.strip().upper() for value in args.opponents.split(",") if value.strip()]
    valid_opponents = all(
        len(parts := opponent.split(":")) == 2
        and parts[0] in ("AB", "M")
        and parts[1].isdigit()
        and int(parts[1]) > 0
        for opponent in opponents
    )
    if args.games < 2 or args.games % 2 or args.workers < 1 or args.max_seconds < 1 or not valid_opponents:
        parser.error("games must be even and >=2; workers, timeout and native AB/M opponent budgets must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    policy_doc = json.loads((PARENT / "best_model/policy.json").read_text(encoding="utf-8"))
    model = TreePolicy(
        lgb.Booster(model_file=str(PARENT / "best_model/ranker.model")),
        json.loads((PARENT / "normalizer.json").read_text(encoding="utf-8")),
        policy_doc["scale"], joblib.load(PARENT / "best_model/monte_carlo.joblib"),
        policy_doc["mix"], 402,
    )
    model.policy_name = "Tree_Ranking_MonteCarlo_402"
    excluded, _ = exclusions(REPO / "runs/online-sarsa-features-100-v4")
    config = {
        "model": "frozen TreePolicy 402",
        "opponents": opponents,
        "games_per_opponent": args.games,
        "players": 2,
        "workers": args.workers,
        "max_seconds_per_game": args.max_seconds,
        "base_seed": args.seed,
        "seats": "balanced alternating seats; same seed/index schedule for each native search opponent",
        "weights_updated": False,
        "information_protocol": "The native search opponent receives its own hand and a sampled public-belief state; actual hidden hands and dice stream are guarded.",
        "limitations": ["small pilot; use the results to select a harder benchmark, not to claim a definitive win rate"],
    }
    write(root / "config.json", config)
    started = perf_counter()
    results = {}
    for opponent in opponents:
        key, value = opponent.split(":", 1)
        name = f"{key.lower()}-{value}"
        print(f"STAGE: frozen 402 vs {opponent}, {args.games} games", flush=True)
        rows, summary, _ = play_phase(
            root, name, model, args.games, args.workers, training=False,
            base_seed=args.seed, excluded=excluded, opponent_policy=opponent,
            max_seconds=args.max_seconds,
        )
        wins = summary["wins"]
        results[opponent] = {
            "games": args.games,
            "completed": summary["completed"],
            "unfinished": summary["unfinished"],
            "wins": wins,
            "losses": summary["completed"] - wins,
            "win_rate": wins / summary["completed"] if summary["completed"] else None,
            "wilson_95": wilson_interval(wins, summary["completed"]),
            "seat_counts": {str(seat): sum(r["learner_seat"] == seat for r in rows) for seat in range(2)},
            "seat_wins": {str(seat): sum(r["learner_seat"] == seat and r["learner_won"] is True for r in rows) for seat in range(2)},
            "mean_actions": sum(r["actions"] for r in rows) / len(rows),
            "mean_game_seconds": sum(r["elapsed_seconds"] for r in rows) / len(rows),
        }
        print(f"RESULT: 402 vs {opponent}: {wins}/{args.games}; elapsed={summary['elapsed_seconds']:.1f}s", flush=True)
    report = {**config, "results": results, "elapsed_seconds": perf_counter() - started}
    write(root / "report.json", report)
    print("FINAL: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
