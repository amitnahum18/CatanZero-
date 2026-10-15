"""Frozen 402 head-to-head against the three previously ranked Catanatron bots."""
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
DEFAULT_ROOT = REPO / "runs/402-vs-core-bots-1v1-40-v1"
OPPONENTS = ("AB:2", "F", "M:10")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40, help="Games against each opponent")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=77_000_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.games < 2 or args.workers < 1:
        parser.error("games must be at least two and workers must be positive")
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
        "opponents": list(OPPONENTS),
        "games_per_opponent": args.games,
        "total_games": args.games * len(OPPONENTS),
        "workers_per_opponent": args.workers,
        "base_seed": args.seed,
        "seats": "balanced alternating seats; same seed/index schedule for each opponent",
        "weights_updated": False,
        "exploration": 0.0,
        "information_protocol": "Each native bot acts from its own private hand plus one sampled public-count belief; 402 receives its standard public observation and own private hand.",
        "limitations": [
            "40 games per opponent is a screening sample, not a definitive strength ranking",
            "search bots receive more computation than the one-step learned policy",
            "the earlier 100-game 402-vs-AB:2 run exposed engine-internal hidden hands to AB search; this run corrects that and is not directly comparable",
        ],
    }
    write(root / "config.json", config)
    started = perf_counter()
    results = {}
    for opponent in OPPONENTS:
        print(f"STAGE: frozen 402 vs {opponent}, {args.games} two-player games", flush=True)
        rows, summary, _ = play_phase(
            root,
            opponent.replace(":", "-").replace(" ", "_").lower(),
            model,
            args.games,
            args.workers,
            training=False,
            base_seed=args.seed,
            excluded=excluded,
            opponent_policy=opponent,
        )
        assert len(rows) == args.games
        assert all(r["players"] == 2 and r["opponent_policy"] == opponent for r in rows)
        assert summary["unfinished"] == 0, f"{opponent} had unfinished games"
        wins = summary["wins"]
        seat_counts = {str(seat): sum(r["learner_seat"] == seat for r in rows) for seat in range(2)}
        seat_wins = {
            str(seat): sum(r["learner_seat"] == seat and r["learner_won"] is True for r in rows)
            for seat in range(2)
        }
        results[opponent] = {
            "games": len(rows),
            "completed": summary["completed"],
            "wins": wins,
            "losses": summary["completed"] - wins,
            "win_rate": wins / summary["completed"] if summary["completed"] else None,
            "wilson_95": wilson_interval(wins, summary["completed"]),
            "seat_counts": seat_counts,
            "seat_wins": seat_wins,
            "mean_actions": sum(r["actions"] for r in rows) / len(rows),
            "mean_game_seconds": sum(r["elapsed_seconds"] for r in rows) / len(rows),
            "beats_opponent_by_raw_rate": wins > len(rows) / 2,
        }
        print(f"RESULT: 402 vs {opponent}: {wins}/{len(rows)}; seats={seat_wins}", flush=True)
    report = {
        **config,
        "results": results,
        "beat_all_by_raw_win_rate": all(row["beats_opponent_by_raw_rate"] for row in results.values()),
        "elapsed_seconds": perf_counter() - started,
    }
    write(root / "report.json", report)
    print("FINAL: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
