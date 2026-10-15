"""Tune the inference mix of the RL-trained depth-16 correction vs native F."""
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

from optimize_tree_policy_vs_f import TreePolicy
from train_residual_tree_fqi import REPO, play_phase, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval


MODEL_ROOT = REPO / "runs/tree-policy-trade-aware-v3"
RL_ROOT = REPO / "runs/402-mc-rl-depth16-v1"
MIXES = (0.1, 0.2, 0.3, 0.4, 0.6)


def metrics(summary, version):
    games = summary["completed"]
    return {"completed": games, "unfinished": summary["unfinished"], "wins": summary["wins"],
            "losses": games-summary["wins"], "win_rate": summary["wins"]/games if games else None,
            "wilson_95": wilson_interval(summary["wins"], games) if games else None,
            "model_version": version}


def compare(a, b):
    pairs = [(a[i], b.get(i)) for i in a if a[i] is not None and b.get(i) is not None]
    return {"complete_pairs": len(pairs),
            "wins_where_reference_lost": sum(x is False and y is True for x, y in pairs),
            "losses_where_reference_won": sum(x is True and y is False for x, y in pairs),
            "same_outcome": sum(x == y for x, y in pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--test-games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=79_400_000)
    parser.add_argument("--output", type=Path, default=REPO / "runs/402-et16-mix-tune-v1")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.test_games < 2 or args.test_games % 2 or args.workers < 1:
        parser.error("game counts must be even and >=2; workers must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    policy = json.loads((MODEL_ROOT / "best_model/policy.json").read_text(encoding="utf-8"))
    norm = json.loads((MODEL_ROOT / "normalizer.json").read_text(encoding="utf-8"))
    teacher = lgb.Booster(model_file=str(MODEL_ROOT / "best_model/ranker.model"))
    baseline_mc = joblib.load(MODEL_ROOT / "best_model/monte_carlo.joblib")
    new_mc = joblib.load(RL_ROOT / "candidate_depth16_round_1/monte_carlo.joblib")
    excluded, _ = exclusions(REPO / "runs/online-sarsa-features-100-v4")
    config = {"opponent": "F", "players": 2, "validation_games_per_mix": args.games,
              "mixes": list(MIXES), "workers": args.workers, "validation_seed": args.seed,
              "all_mixes_share_maps_and_balanced_seats": True,
              "correction": "RL-trained ExtraTrees, 128 trees, depth 16, min_samples_leaf 16",
              "frozen_ranker": str(MODEL_ROOT / "best_model/ranker.model"),
              "test_games_per_policy": args.test_games,
              "test_seed": args.seed + 1_000_000,
              "test_1000_and_prior_200_games_excluded_from_training": True,
              "selection": "highest validation wins; final paired test run once after selection"}
    write(root / "config.json", config)
    started = perf_counter()
    models = [("402-original-mix0p2", baseline_mc, policy["mix"], 402)]
    models.extend((f"et16-mix-{m:g}".replace(".", "p"), new_mc, m, 800+i)
                  for i, m in enumerate(MIXES))
    outcomes, results = {}, {}
    for name, mc, mix, version in models:
        model = TreePolicy(teacher, norm, policy["scale"], mc, mix, version)
        model.policy_name = name
        print(f"STAGE: {name}, {args.games} paired validation games", flush=True)
        rows, summary, _ = play_phase(root, name, model, args.games, args.workers,
            training=False, base_seed=args.seed, excluded=excluded,
            opponent_policy="F", max_seconds=600)
        outcomes[name] = {r["game_index"]: r["learner_won"]
                          if r["termination"] == "victory" else None for r in rows}
        results[name] = metrics(summary, version)
        write(root / "report.json", {**config, "results": results,
                                      "elapsed_seconds": perf_counter()-started})
        print(f"RESULT: {name}: {summary['wins']}/{summary['completed']}", flush=True)

    baseline_name = models[0][0]
    reference = outcomes[baseline_name]
    validation_pairs = {name: compare(reference, data) for name, data in outcomes.items()}
    selected = max((name for name, _, _, _ in models[1:]),
                   key=lambda name: results[name]["wins"])
    selected_spec = next((mc, mix, version) for name, mc, mix, version in models if name == selected)
    candidate = TreePolicy(teacher, norm, policy["scale"], selected_spec[0], selected_spec[1], selected_spec[2])
    candidate.policy_name = selected
    candidate.save(root / "selected_candidate")
    joblib.dump(selected_spec[0], root / "selected_candidate/monte_carlo.joblib", compress=3)

    test_seed = args.seed + 1_000_000
    baseline = TreePolicy(teacher, norm, policy["scale"], baseline_mc, policy["mix"], 402)
    print(f"STAGE: final paired test, 402 vs {selected}, {args.test_games} games each", flush=True)
    before_rows, before, _ = play_phase(root, "final_test_402", baseline, args.test_games, args.workers,
        training=False, base_seed=test_seed, excluded=excluded, opponent_policy="F", max_seconds=600)
    after_rows, after, _ = play_phase(root, "final_test_selected", candidate, args.test_games, args.workers,
        training=False, base_seed=test_seed, excluded=excluded, opponent_policy="F", max_seconds=600)
    for a, b in zip(before_rows, after_rows):
        for field in ("seed", "learner_seat", "initial_map_hash", "initial_stream_hash"):
            if a[field] != b[field]:
                raise AssertionError(f"unpaired final test at {field}")
    final_outcomes_a = {r["game_index"]: r["learner_won"] if r["termination"] == "victory" else None
                        for r in before_rows}
    final_outcomes_b = {r["game_index"]: r["learner_won"] if r["termination"] == "victory" else None
                        for r in after_rows}
    report = {**config, "results": results, "validation_paired_vs_402": validation_pairs,
              "selected_mix_candidate": selected,
              "final_test_402": metrics(before, 402),
              "final_test_selected": metrics(after, selected_spec[2]),
              "final_test_paired_vs_402": compare(final_outcomes_a, final_outcomes_b),
              "final_test_seed": test_seed, "elapsed_seconds": perf_counter()-started,
              "limitations": ["candidate selected from a finite validation sweep",
                              "final test is a screening sample, not evidence of universal strength"]}
    write(root / "report.json", report)
    print("FINAL: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
