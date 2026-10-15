"""Paired pilot sweeping the Monte Carlo correction weight in frozen 402."""
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
DEFAULT_ROOT = REPO / "runs/402-mix-sweep-ab3-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4, help="Even games per mix; same paired seeds for every mix")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--opponent", default="AB:3")
    parser.add_argument("--mixes", default="0,0.1,0.2,0.4,0.6")
    parser.add_argument("--extra-trees-only", action="store_true",
                        help="rank actions using only the fitted ExtraTrees component")
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=78_300_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    mixes = [float(value) for value in args.mixes.split(",")]
    opponent = args.opponent.upper()
    parts = opponent.split(":", 1)
    valid_opponent = opponent == "F" or (
        len(parts) == 2 and parts[0] in ("AB", "M") and parts[1].isdigit() and int(parts[1]) > 0
    )
    if (args.games < 2 or args.games % 2 or args.workers < 1 or args.max_seconds < 1
            or not mixes or any(not 0 <= value <= 2 for value in mixes) or not valid_opponent):
        parser.error("games must be even and >=2; workers/timeout positive; mixes in [0,2]; opponent F, AB:n or M:n")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    policy_doc = json.loads((PARENT / "best_model/policy.json").read_text(encoding="utf-8"))
    model = TreePolicy(
        lgb.Booster(model_file=str(PARENT / "best_model/ranker.model")),
        json.loads((PARENT / "normalizer.json").read_text(encoding="utf-8")),
        policy_doc["scale"], joblib.load(PARENT / "best_model/monte_carlo.joblib"),
        policy_doc["mix"], 402,
    )
    model.policy_name = "Tree_Ranking_MonteCarlo_402_mix_sweep"
    excluded, _ = exclusions(REPO / "runs/online-sarsa-features-100-v4")
    config = {
        "model": "frozen TreePolicy 402; only the MC correction coefficient changes",
        "score_mode": "extra_trees_only" if args.extra_trees_only else "lightgbm_plus_weighted_extra_trees",
        "opponent": opponent,
        "mixes": mixes,
        "games_per_mix": args.games,
        "players": 2,
        "workers": args.workers,
        "base_seed": args.seed,
        "max_seconds_per_game": args.max_seconds,
        "seats": "balanced alternating seats; identical seeds and maps across all mix values",
        "weights_updated": False,
        "information_protocol": "The opponent acts from its own private hand and public information; search bots use a sampled public belief, and actual hidden hands and dice stream are guarded.",
        "decision_score": ("ExtraTrees(raw_features) only" if args.extra_trees_only else
                           "LightGBM(raw_features)/1.4757017551196938 + mix*ExtraTrees(raw_features)"),
        "limitations": ["paired small pilot; no definitive winner claim"],
    }
    write(root / "config.json", config)
    started = perf_counter()
    outcomes = {}
    rows_by_mix = {}
    for mix in mixes:
        model.mix = mix
        if args.extra_trees_only:
            model.predict = lambda raw: model.mc.predict(raw)
        label = "extra-trees-only" if args.extra_trees_only else f"mix-{mix:g}".replace(".", "p")
        mode = "ExtraTrees only" if args.extra_trees_only else f"MC mix={mix:g}"
        print(f"STAGE: {opponent} vs 402, {mode}, {args.games} paired games", flush=True)
        rows, summary, _ = play_phase(
            root, label, model, args.games, args.workers, training=False,
            base_seed=args.seed, excluded=excluded, opponent_policy=opponent,
            max_seconds=args.max_seconds,
        )
        rows_by_mix[mix] = rows
        outcomes[mix] = {r["game_index"]: r["learner_won"] if r["termination"] == "victory" else None for r in rows}
        wins = summary["wins"]
        completed = summary["completed"]
        config.setdefault("results", {})[str(mix)] = {
            "mix": mix,
            "scheduled": args.games,
            "completed": completed,
            "unfinished": summary["unfinished"],
            "wins": wins,
            "losses": completed - wins,
            "win_rate": wins / completed if completed else None,
            "wilson_95": wilson_interval(wins, completed) if completed else None,
            "seat_counts": {str(seat): sum(r["learner_seat"] == seat for r in rows) for seat in range(2)},
            "seat_wins": {str(seat): sum(r["learner_seat"] == seat and r["learner_won"] is True for r in rows) for seat in range(2)},
            "mean_game_seconds": sum(r["elapsed_seconds"] for r in rows) / len(rows),
            "model_version": 402,
        }
        write(root / "report.json", {**config, "elapsed_seconds": perf_counter() - started})
        print(f"RESULT: mix={mix:g}: {wins}/{completed}; unfinished={summary['unfinished']}", flush=True)

    baseline_mix = min(mixes, key=lambda value: abs(value - 0.2))
    paired = {}
    baseline = outcomes[baseline_mix]
    for mix in mixes:
        comparable = [(baseline.get(i), outcomes[mix].get(i)) for i in baseline if baseline.get(i) is not None and outcomes[mix].get(i) is not None]
        paired[str(mix)] = {
            "reference_mix": baseline_mix,
            "complete_pairs": len(comparable),
            "wins_where_reference_lost": sum(a is False and b is True for a, b in comparable),
            "losses_where_reference_won": sum(a is True and b is False for a, b in comparable),
            "same_outcome": sum(a == b for a, b in comparable),
        }
    report = {**config, "paired_vs_reference_mix": paired, "elapsed_seconds": perf_counter() - started}
    write(root / "report.json", report)
    print("FINAL: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
